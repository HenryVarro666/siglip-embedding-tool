"""Deterministic RGB preprocessing and exact-pixel duplicate discovery."""

from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path

from PIL import Image, ImageOps

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
ImageSource = str | Path | Image.Image


@dataclass(frozen=True)
class ImageRecord:
    path: str
    sha256: str
    pixel_sha256: str
    aliases: tuple[str, ...] = ()


def load_rgb(source: ImageSource) -> Image.Image:
    """Return an independent, oriented RGB image; composite transparency on white."""
    try:
        if isinstance(source, Image.Image):
            image = source.copy()
        else:
            with Image.open(source) as opened:
                image = opened.copy()
        image = ImageOps.exif_transpose(image)
        if "A" in image.getbands() or "transparency" in image.info:
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, "white")
            image = Image.alpha_composite(background, rgba)
        return image.convert("RGB")
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not decode image {source}: {exc}") from exc


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pixel_sha256(image: Image.Image) -> str:
    """Hash dimensions and canonical RGB bytes, ignoring container metadata."""
    image = load_rgb(image)
    digest = sha256(f"{image.width}x{image.height}:RGB:".encode())
    digest.update(image.tobytes())
    return digest.hexdigest()


def scan_images(root: str | Path) -> list[ImageRecord]:
    """Scan supported files in lexical order, grouping identical RGB pixels."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"Image directory does not exist: {root}")
    candidates = sorted(
        p
        for p in root.rglob("*")
        if p.is_file()
        and not p.is_symlink()
        and p.suffix.lower() in IMAGE_SUFFIXES
        and not any(part.startswith(".") for part in p.relative_to(root).parts)
    )
    if not candidates:
        raise ValueError(f"No supported images found in {root}")
    records: dict[str, ImageRecord] = {}
    for path in candidates:
        pixels = pixel_sha256(load_rgb(path))
        relative = path.relative_to(root).as_posix()
        if pixels in records:
            previous = records[pixels]
            records[pixels] = replace(previous, aliases=(*previous.aliases, relative))
        else:
            records[pixels] = ImageRecord(relative, file_sha256(path), pixels)
    return list(records.values())
