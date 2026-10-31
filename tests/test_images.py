from hashlib import sha256

import pytest
from PIL import Image, PngImagePlugin

from siglip_tools.images import load_rgb, scan_images


def test_rgb_pixels_and_size_are_preserved(tmp_path):
    path = tmp_path / "rgb.png"
    Image.new("RGB", (3, 2), (12, 34, 56)).save(path)
    result = load_rgb(path)
    assert result.mode == "RGB"
    assert result.size == (3, 2)
    assert result.getpixel((0, 0)) == (12, 34, 56)


def test_exif_orientation_is_applied(tmp_path):
    path = tmp_path / "portrait.jpg"
    exif = Image.Exif()
    exif[274] = 6  # 90-degree clockwise display orientation
    Image.new("RGB", (3, 2), (12, 34, 56)).save(path, exif=exif)
    assert load_rgb(path).size == (2, 3)


def test_transparency_is_composited_on_white(tmp_path):
    path = tmp_path / "alpha.png"
    Image.new("RGBA", (2, 2), (255, 0, 0, 128)).save(path)
    assert load_rgb(path).getpixel((0, 0)) == (255, 127, 127)


def test_scan_deduplicates_pixels_and_keeps_aliases(tmp_path):
    for name in ("b.png", "a.png"):
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("source", name)
        Image.new("RGB", (2, 2), (12, 34, 56)).save(tmp_path / name, pnginfo=metadata)
    records = scan_images(tmp_path)
    assert len(records) == 1
    assert records[0].path == "a.png"
    assert records[0].aliases == ("b.png",)
    assert records[0].sha256 == sha256((tmp_path / "a.png").read_bytes()).hexdigest()
    assert len(records[0].pixel_sha256) == 64


def test_corrupt_image_reports_the_path(tmp_path):
    path = tmp_path / "broken.jpg"
    path.write_text("not an image")
    with pytest.raises(ValueError, match="broken.jpg"):
        scan_images(tmp_path)


def test_empty_directory_is_an_explicit_error(tmp_path):
    with pytest.raises(ValueError, match="[Nn]o supported images"):
        scan_images(tmp_path)


def test_scan_does_not_follow_image_symlinks(tmp_path):
    outside = tmp_path / "outside.png"
    Image.new("RGB", (2, 2)).save(outside)
    folder = tmp_path / "gallery"
    folder.mkdir()
    (folder / "link.png").symlink_to(outside)
    with pytest.raises(ValueError, match="[Nn]o supported images"):
        scan_images(folder)
