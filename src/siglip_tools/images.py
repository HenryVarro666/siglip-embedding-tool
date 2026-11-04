"""Deterministic RGB preprocessing and exact-pixel duplicate discovery."""

import os
import struct
from dataclasses import dataclass, replace
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
ImageSource = str | Path | Image.Image


@dataclass(frozen=True)
class ImageRecord:
    path: str
    sha256: str
    pixel_sha256: str
    aliases: tuple[str, ...] = ()


_DECODE_ERRORS = (OSError, ValueError, SyntaxError, struct.error, Image.DecompressionBombError)


def _eight_bit(image: Image.Image) -> Image.Image:
    """Rescale high-bit-depth grayscale to 8 bits; Pillow's convert() would clip at 255.

    16-bit samples map linearly from 0..65535. Other integer or float modes map their
    actual value range, which is the only deterministic choice without a declared scale.
    Eight-bit inputs are returned unchanged, so existing indexes keep their pixel hashes.
    """
    if image.mode in ("I;16", "I;16L", "I;16B", "I;16N"):
        values = np.asarray(image.convert("I"), dtype=np.float64)
        scaled = np.rint(values / 257.0)
    elif image.mode in ("I", "F"):
        values = np.asarray(image, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Image contains non-finite samples")
        low, high = float(values.min()), float(values.max())
        if 0 <= low and high <= 65535 and image.mode == "I":
            scaled = np.rint(values / 257.0)
        else:
            scaled = (
                np.rint((values - low) / (high - low) * 255.0)
                if high > low
                else np.zeros_like(values)
            )
    else:
        return image
    return Image.fromarray(np.clip(scaled, 0, 255).astype(np.uint8), mode="L")


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
        return _eight_bit(image).convert("RGB")
    except _DECODE_ERRORS as exc:
        raise ValueError(f"Could not decode image {source}: {exc}") from exc


def default_file_mode() -> int:
    """Permissions an ordinary new file would get, so published artifacts honor the umask."""
    current = os.umask(0o022)
    os.umask(current)
    return 0o666 & ~current


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_image_snapshot(path: str | Path) -> tuple[Image.Image, str]:
    """Bind decoded pixels and container hash to one immutable byte snapshot."""
    try:
        content = Path(path).read_bytes()
        with Image.open(BytesIO(content)) as opened:
            rgb = load_rgb(opened)
        return rgb, sha256(content).hexdigest()
    except _DECODE_ERRORS as exc:
        raise ValueError(f"Could not decode image {path}: {exc}") from exc


def validate_records(root: Path, records: list[ImageRecord]) -> None:
    """Verify canonical files and every exact-pixel alias before publishing an index."""
    for record in records:
        image, digest = read_image_snapshot(root / record.path)
        if digest != record.sha256 or pixel_sha256(image) != record.pixel_sha256:
            raise ValueError(f"Image changed during indexing: {record.path}; retry")
        for alias in record.aliases:
            image, _ = read_image_snapshot(root / alias)
            if pixel_sha256(image) != record.pixel_sha256:
                raise ValueError(f"Duplicate image changed during indexing: {alias}; retry")


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
        image, digest = read_image_snapshot(path)
        pixels = pixel_sha256(image)
        relative = path.relative_to(root).as_posix()
        if pixels in records:
            previous = records[pixels]
            records[pixels] = replace(previous, aliases=(*previous.aliases, relative))
        else:
            records[pixels] = ImageRecord(relative, digest, pixels)
    return list(records.values())
