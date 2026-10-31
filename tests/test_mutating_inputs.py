import numpy as np
import pytest
from PIL import Image

from siglip_tools import cli, images


def _tiny_encoder(monkeypatch, mutate=None):
    """Replace expensive inference with actual normalized RGB features, not mock outputs."""

    class RGBEncoder:
        identity = {"model_id": "test/rgb", "revision": "fixed"}
        device = "cpu"

        def __init__(self, *args, **kwargs):
            pass

        def encode_images(self, sources, batch_size=8):
            vectors = []
            for source in sources:
                image = images.load_rgb(source)
                vector = np.asarray(image, dtype=np.float32).mean(axis=(0, 1))
                vectors.append(vector / np.linalg.norm(vector))
            if mutate is not None:
                mutate()
            return np.array(vectors, dtype=np.float32)

    monkeypatch.setattr(cli, "SigLIPEmbedder", RGBEncoder)


def _existing_index(tmp_path):
    folder = tmp_path / "images"
    folder.mkdir()
    Image.new("RGB", (2, 2), "red").save(folder / "a.png")
    Image.new("RGB", (2, 2), "red").save(folder / "b.png")
    target = tmp_path / "index.npz"
    target.write_bytes(b"existing index must survive")
    return folder, target


def test_alias_changed_during_encoding_aborts_without_replacing_index(tmp_path, monkeypatch):
    folder, target = _existing_index(tmp_path)
    _tiny_encoder(monkeypatch, lambda: Image.new("RGB", (2, 2), "blue").save(folder / "b.png"))
    with pytest.raises(SystemExit) as error:
        cli.main(["index", "--images", str(folder), "--output", str(target), "--overwrite"])
    assert error.value.code == 2
    assert target.read_bytes() == b"existing index must survive"


def test_primary_changed_between_decode_and_hash_does_not_publish_stale_pixels(
    tmp_path, monkeypatch
):
    folder, target = _existing_index(tmp_path)
    original = images.load_rgb
    changed = False

    def decode_then_replace(source):
        nonlocal changed
        result = original(source)
        if not changed:
            changed = True
            Image.new("RGB", (2, 2), "blue").save(folder / "a.png")
        return result

    monkeypatch.setattr(images, "load_rgb", decode_then_replace)
    _tiny_encoder(monkeypatch)
    with pytest.raises(SystemExit) as error:
        cli.main(["index", "--images", str(folder), "--output", str(target), "--overwrite"])
    assert error.value.code == 2
    assert target.read_bytes() == b"existing index must survive"
