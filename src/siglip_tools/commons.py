"""Download byte-identical images from a frozen, attributed Commons manifest."""

import argparse
import json
import os
import re
import tempfile
from datetime import timezone
from email.utils import parsedate_to_datetime
from hashlib import sha256
from io import BytesIO
from math import isfinite
from pathlib import Path, PurePosixPath, PureWindowsPath
from time import sleep, time
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from PIL import Image

from .images import file_sha256

USER_AGENT = "SigLIPEquipmentPilot/0.1 (personal research; frozen Commons equipment images)"
DOWNLOAD_HOSTS = {"upload.wikimedia.org", "thumb.wikimedia.org"}
LICENSES = re.compile(r"(?:CC BY(?:-SA)? (?:2\.0|2\.5|3\.0|4\.0)|CC0(?: 1\.0)?)")
MAX_ATTEMPTS = 4


def _trusted_url(value, hosts):
    if not isinstance(value, str):
        return False
    try:
        parsed = urlsplit(value)
    except ValueError:
        return False
    return parsed.scheme == "https" and parsed.netloc.lower() in hosts


def validate_source_record(item: dict) -> None:
    """Validate a frozen image record's path, SHA-256, license, author and URLs.

    This validates declared source metadata; it does not independently establish
    that the attribution or license is correct. Root containment is checked by
    ``download_manifest`` because it depends on the chosen local directory.
    """
    if not isinstance(item, dict):
        raise ValueError("Every source record must be an object")
    relative = item.get("path")
    if not isinstance(relative, str) or not relative or "\\" in relative or "\0" in relative:
        raise ValueError("Image path must be a safe relative path")
    path = PurePosixPath(relative)
    windows = PureWindowsPath(relative)
    if path.is_absolute() or ".." in path.parts or path == PurePosixPath("."):
        raise ValueError("Image path must be a safe relative path")
    if windows.drive or windows.root:
        raise ValueError("Image path must be a safe relative path")
    if not isinstance(item.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]):
        raise ValueError("Every image requires a lowercase SHA-256 hash")
    if not _trusted_url(item.get("source_page"), {"commons.wikimedia.org"}):
        raise ValueError("Expected an HTTPS Commons source_page")
    if not _trusted_url(item.get("download_url"), DOWNLOAD_HOSTS):
        raise ValueError("Expected an HTTPS Commons download_url")
    if not _trusted_url(item.get("license_url"), {"creativecommons.org"}):
        raise ValueError("Expected an HTTPS Creative Commons license_url")
    name = item.get("license_name")
    if not isinstance(name, str) or not LICENSES.fullmatch(name):
        raise ValueError("Unsupported or unclear license_name")
    if not isinstance(item.get("author"), str) or not item["author"].strip():
        raise ValueError("Every image requires a nonempty author")


def _retry_after(value):
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        try:
            deadline = parsedate_to_datetime(value)
            if deadline.tzinfo is None:
                deadline = deadline.replace(tzinfo=timezone.utc)
            seconds = deadline.timestamp() - time()
        except (TypeError, ValueError, OverflowError):
            return 0.0
    return max(0.0, seconds) if isfinite(seconds) else 0.0


def _download_bytes(url, delay):
    request = Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(MAX_ATTEMPTS):
        try:
            with urlopen(request, timeout=60) as response:
                if not _trusted_url(response.geturl(), DOWNLOAD_HOSTS):
                    raise ValueError("Image download redirected outside Commons image hosts")
                return response.read()
        except HTTPError as error:
            if error.code not in {429, 503} or attempt + 1 == MAX_ATTEMPTS:
                raise
            wait = _retry_after(error.headers.get("Retry-After") if error.headers else None)
            error.close()
            sleep(max(delay, wait))


def _publish_image(content, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent, prefix=".commons-", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        # A hard link publishes the complete file atomically without clobbering
        # a file that appeared after the initial existing-file check.
        os.link(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def download_manifest(
    manifest_path: str | Path, data_root: str | Path, delay: float = 7.0
) -> dict[str, int]:
    """Download frozen image bytes sequentially; preserve all existing files.

    Matching local hashes are skipped. Other existing files cause FileExistsError.
    Responses are checked against the declared SHA-256 and Pillow ``verify``
    before atomic publication, without transforming or re-encoding the image.
    HTTP 429/503 have at most four total attempts, separated by at least ``delay``
    seconds or the server's Retry-After interval, whichever is greater.
    """
    if not isfinite(delay) or delay < 0:
        raise ValueError("delay must be finite and nonnegative")
    document = json.loads(Path(manifest_path).read_bytes())
    items = document.get("images") if isinstance(document, dict) else None
    if not isinstance(items, list) or not items:
        raise ValueError("Manifest must contain a nonempty images list")
    root = Path(data_root).expanduser().resolve()
    pending, targets = [], set()
    skipped = 0
    for item in items:
        validate_source_record(item)
        target = (root / item["path"]).resolve()
        if not target.is_relative_to(root) or target == root:
            raise ValueError("Image path escapes data root")
        if target in targets:
            raise ValueError("Manifest image paths must be unique")
        targets.add(target)
        if target.exists():
            if not target.is_file() or file_sha256(target) != item["sha256"]:
                raise FileExistsError(f"Refusing to overwrite mismatched local file: {target}")
            skipped += 1
        else:
            pending.append((item, target))

    for index, (item, target) in enumerate(pending):
        if index:
            sleep(delay)
        content = _download_bytes(item["download_url"], delay)
        if sha256(content).hexdigest() != item["sha256"]:
            raise ValueError(f"Downloaded image hash mismatch: {item['path']}")
        try:
            with Image.open(BytesIO(content)) as image:
                image.verify()
        except (OSError, ValueError, SyntaxError) as error:
            raise ValueError(f"Downloaded content is not a valid image: {item['path']}") from error
        _publish_image(content, target)
    return {"downloaded": len(pending), "skipped": skipped}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--delay", default=7.0, type=float)
    args = parser.parse_args(argv)
    result = download_manifest(args.manifest, args.data_root, args.delay)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
