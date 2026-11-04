"""Core robustness fixes recorded as SIG-062 to SIG-070.

These were first kept as strict expected failures because every frozen benchmark
exporter bound the working tree byte for byte. Since the exporters bind the
measured source snapshot (SIG-073), the patch is applied and the tests must pass.
"""

import json
import os
import stat
import struct

import numpy as np
import pytest
from PIL import Image

from siglip_tools import cli, images
from siglip_tools.encoder import _device
from siglip_tools.images import ImageRecord, load_rgb, scan_images
from siglip_tools.index import ImageIndex, _relative_path


def test_sixteen_bit_grayscale_is_rescaled_not_clipped(tmp_path):
    gradient = np.linspace(0, 65535, 8, dtype=np.uint16).reshape(1, 8)
    path = tmp_path / "gradient.png"
    Image.fromarray(gradient).save(path)
    rgb = load_rgb(path)
    assert rgb.mode == "RGB"
    reds = [rgb.getpixel((x, 0))[0] for x in range(8)]
    assert reds[0] == 0 and reds[-1] == 255
    assert reds == sorted(reds) and len(set(reds)) == 8, "all levels must survive the rescale"


def test_distinct_sixteen_bit_images_are_not_collapsed_into_one_record(tmp_path):
    rng = np.random.default_rng(0)
    for name in ("a", "b"):
        data = rng.integers(300, 65535, size=(16, 16), dtype=np.uint16)
        Image.fromarray(data).save(tmp_path / f"{name}.png")
    records = scan_images(tmp_path)
    assert len(records) == 2 and all(not record.aliases for record in records)


def test_eight_bit_inputs_keep_their_pixel_hash_so_existing_indexes_stay_valid(tmp_path):
    image = Image.new("RGB", (3, 2), (10, 20, 30))
    assert np.array_equal(np.asarray(load_rgb(image)), np.asarray(image))
    mode_l = Image.new("L", (3, 2), 77)
    assert np.asarray(load_rgb(mode_l))[0, 0].tolist() == [77, 77, 77]


def test_float_and_32bit_integer_images_map_their_value_range(tmp_path):
    values = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
    rgb = load_rgb(Image.fromarray(values, mode="F"))
    assert [rgb.getpixel((x, 0))[0] for x in range(3)] == [0, 128, 255]
    wide = Image.fromarray(np.array([[0, 70000]], dtype=np.int32), mode="I")
    assert [load_rgb(wide).getpixel((x, 0))[0] for x in range(2)] == [0, 255]
    flat = load_rgb(Image.fromarray(np.full((1, 2), 3.5, dtype=np.float32), mode="F"))
    assert flat.getpixel((0, 0)) == (0, 0, 0)


def test_decompression_bomb_and_corrupt_exif_become_named_value_errors(tmp_path):
    bomb = tmp_path / "bomb.png"
    bomb.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", 13)
        + b"IHDR"
        + struct.pack(">IIBBBBB", 60000, 60000, 8, 2, 0, 0, 0)
        + b"\x00" * 4
    )
    with pytest.raises(ValueError, match="bomb.png"):
        load_rgb(bomb)
    corrupt = tmp_path / "exif.jpg"
    stream = __import__("io").BytesIO()
    Image.new("RGB", (4, 4), "red").save(stream, format="JPEG")
    data = stream.getvalue()
    app1 = b"Exif\x00\x00GARBAGE!" + b"\x00" * 8
    patched = data[:2] + b"\xff\xe1" + struct.pack(">H", len(app1) + 2) + app1 + data[2:]
    corrupt.write_bytes(patched)
    try:
        load_rgb(corrupt)
    except ValueError as error:
        assert "exif.jpg" in str(error)


def test_empty_or_truncated_index_files_are_value_errors(tmp_path):
    empty = tmp_path / "empty.npz"
    empty.write_bytes(b"")
    with pytest.raises(ValueError, match="Invalid index"):
        ImageIndex.load(empty)
    truncated = tmp_path / "truncated.npz"
    ImageIndex(
        [[1.0, 0.0]],
        [ImageRecord("a.png", "a" * 64, "b" * 64)],
        {"model_id": "m", "revision": "r"},
        tmp_path,
    ).save(truncated)
    truncated.write_bytes(truncated.read_bytes()[:40])
    with pytest.raises(ValueError, match="Invalid index"):
        ImageIndex.load(truncated)
    with pytest.raises(SystemExit) as error:
        cli.main(["inspect", "--index", str(empty)])
    assert error.value.code == 2


def test_saved_index_honors_the_umask_instead_of_owner_only(tmp_path):
    path = tmp_path / "index.npz"
    previous = os.umask(0o022)
    try:
        ImageIndex(
            [[1.0, 0.0]],
            [ImageRecord("a.png", "a" * 64, "b" * 64)],
            {"model_id": "m", "revision": "r"},
            tmp_path,
        ).save(path)
    finally:
        os.umask(previous)
    assert stat.S_IMODE(path.stat().st_mode) == 0o644
    with pytest.raises(FileExistsError):
        ImageIndex(
            [[1.0, 0.0]],
            [ImageRecord("a.png", "a" * 64, "b" * 64)],
            {"model_id": "m", "revision": "r"},
            tmp_path,
        ).save(path)


def test_no_clobber_publication_falls_back_when_hard_links_are_unsupported(tmp_path, monkeypatch):
    from siglip_tools import index as index_module

    def no_links(src, dst, *args, **kwargs):
        raise PermissionError("Operation not permitted")

    monkeypatch.setattr(index_module.os, "link", no_links)
    path = tmp_path / "index.npz"
    stored = ImageIndex(
        [[1.0, 0.0]],
        [ImageRecord("a.png", "a" * 64, "b" * 64)],
        {"model_id": "m", "revision": "r"},
        tmp_path,
    )
    stored.save(path)
    assert ImageIndex.load(path).fingerprint == stored.fingerprint
    with pytest.raises(FileExistsError):
        stored.save(path)


def test_records_require_string_digests_and_canonical_paths(tmp_path):
    with pytest.raises(ValueError, match="SHA-256"):
        ImageIndex(
            [[1.0, 0.0]],
            [ImageRecord("a.png", ["ab"] * 64, "b" * 64)],
            {"model_id": "m", "revision": "r"},
            tmp_path,
        )
    for bad in ("./a.png", "a//b.png", "dir/", "./", "a/./b.png"):
        with pytest.raises(ValueError):
            _relative_path(bad)
    _relative_path("dir/sub/a.png")


def test_device_parsing_rejects_garbage_and_ordinals_on_non_cuda_devices():
    for bad in ("", "   ", "gpu", "cpu:3", "mps:0", "cuda:abc", "tpu"):
        with pytest.raises(ValueError):
            _device(bad)
    assert _device("cpu").type == "cpu"
    assert _device("auto").type in {"cpu", "cuda", "mps"}


def test_cli_prints_version_and_handles_undecodable_paths(capsys, monkeypatch):
    with pytest.raises(SystemExit) as error:
        cli.main(["--version"])
    assert error.value.code == 0
    assert "siglip-tools" in capsys.readouterr().out
    printed = []

    def strict_print(text):
        if "\udcff" in text:
            raise UnicodeEncodeError("utf-8", text, 0, 1, "surrogates not allowed")
        printed.append(text)

    monkeypatch.setattr("builtins.print", strict_print)
    cli._print({"path": "bad\udcffname"})
    assert json.loads(printed[-1])["path"] == "bad\udcffname"


def test_default_translator_device_is_auto_not_mps():
    import inspect

    from siglip_tools.nllb_translation import NLLBTranslator
    from siglip_tools.translation import LocalTranslator

    assert inspect.signature(LocalTranslator.__init__).parameters["device"].default == "auto"
    assert inspect.signature(NLLBTranslator.__init__).parameters["device"].default == "auto"
    assert images.default_file_mode() & 0o600 == 0o600
