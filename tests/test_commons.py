import json
from datetime import datetime, timedelta, timezone
from email.message import Message
from email.utils import format_datetime
from hashlib import sha256
from io import BytesIO
from urllib.error import HTTPError

import pytest
from PIL import Image

from siglip_tools import commons


def image_bytes():
    stream = BytesIO()
    Image.new("RGB", (3, 2), "red").save(stream, format="PNG")
    return stream.getvalue()


def source_record(content):
    return {
        "image_id": "pilot",
        "path": "images/device.png",
        "download_url": "https://thumb.wikimedia.org/wikipedia/commons/thumb/a/device.png",
        "source_page": "https://commons.wikimedia.org/wiki/File:Device.png",
        "sha256": sha256(content).hexdigest(),
        "license_name": "CC BY-SA 4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "author": "Example photographer",
    }


def manifest_file(tmp_path, items):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"images": items}), encoding="utf-8")
    return manifest


class Response(BytesIO):
    def __init__(self, content, url):
        super().__init__(content)
        self.url = url

    def geturl(self):
        return self.url


def http_error(code, retry_after=None):
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return HTTPError("https://upload.wikimedia.org/image.png", code, "busy", headers, None)


def no_network(*args, **kwargs):
    pytest.fail("This input must not cause a network request")


def test_download_preserves_frozen_image_bytes(tmp_path, monkeypatch):
    content = image_bytes()
    item = source_record(content)
    manifest = manifest_file(tmp_path, [item])
    requests = []

    def open_response(request, timeout):
        requests.append(request)
        return Response(content, request.full_url)

    monkeypatch.setattr(commons, "urlopen", open_response)
    root = tmp_path / "downloaded"

    result = commons.download_manifest(manifest, root, delay=0)

    assert result == {"downloaded": 1, "skipped": 0}
    assert (root / item["path"]).read_bytes() == content
    assert len(requests) == 1
    assert "SigLIPEquipmentPilot" in requests[0].get_header("User-agent")
    assert not list(root.rglob("*.tmp"))


def test_matching_existing_file_skips_network(tmp_path, monkeypatch):
    content = image_bytes()
    item = source_record(content)
    root = tmp_path / "downloaded"
    target = root / item["path"]
    target.parent.mkdir(parents=True)
    target.write_bytes(content)
    monkeypatch.setattr(commons, "urlopen", no_network)

    result = commons.download_manifest(manifest_file(tmp_path, [item]), root)

    assert result == {"downloaded": 0, "skipped": 1}
    assert target.read_bytes() == content


def test_mismatched_existing_file_is_preserved_without_network(tmp_path, monkeypatch):
    item = source_record(image_bytes())
    root = tmp_path / "downloaded"
    target = root / item["path"]
    target.parent.mkdir(parents=True)
    target.write_bytes(b"keep this local file")
    monkeypatch.setattr(commons, "urlopen", no_network)

    with pytest.raises(FileExistsError):
        commons.download_manifest(manifest_file(tmp_path, [item]), root)

    assert target.read_bytes() == b"keep this local file"
    assert not list(root.rglob("*.tmp"))


@pytest.mark.parametrize("failure", ["hash", "image"])
@pytest.mark.parametrize("file_appears", [False, True], ids=["absent", "concurrent-file"])
def test_invalid_response_never_creates_or_replaces_destination(
    tmp_path, monkeypatch, failure, file_appears
):
    content = b"this is not an image"
    item = source_record(image_bytes() if failure == "hash" else content)
    root = tmp_path / "downloaded"
    target = root / item["path"]

    def open_response(request, timeout):
        if file_appears:
            target.parent.mkdir(parents=True)
            target.write_bytes(b"concurrent local file")
        return Response(content, request.full_url)

    monkeypatch.setattr(commons, "urlopen", open_response)

    with pytest.raises(ValueError, match=failure):
        commons.download_manifest(manifest_file(tmp_path, [item]), root, delay=0)

    if file_appears:
        assert target.read_bytes() == b"concurrent local file"
    else:
        assert not target.exists()
    assert not list(root.rglob("*.tmp"))


def test_valid_download_cannot_overwrite_a_concurrently_created_file(tmp_path, monkeypatch):
    content = image_bytes()
    item = source_record(content)
    root = tmp_path / "downloaded"
    target = root / item["path"]

    def open_response(request, timeout):
        target.parent.mkdir(parents=True)
        target.write_bytes(b"concurrent local file")
        return Response(content, request.full_url)

    monkeypatch.setattr(commons, "urlopen", open_response)

    with pytest.raises(FileExistsError):
        commons.download_manifest(manifest_file(tmp_path, [item]), root, delay=0)

    assert target.read_bytes() == b"concurrent local file"
    assert not list(root.rglob("*.tmp"))


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_page", "http://commons.wikimedia.org/wiki/File:Device.png"),
        ("source_page", "https://commons.wikimedia.org.example.com/image"),
        ("source_page", "https://user@commons.wikimedia.org/image"),
        ("download_url", "https://example.com/image.png"),
        ("download_url", "http://upload.wikimedia.org/image.png"),
        ("download_url", "https://upload.wikimedia.org:8443/image.png"),
        ("license_name", "CC BY-NC 4.0"),
        ("license_url", "https://example.com/license"),
        ("author", "  "),
        ("path", "../image.png"),
        ("path", "/absolute/image.png"),
        ("path", "C:\\image.png"),
        ("path", "."),
        ("sha256", "not-a-sha256"),
    ],
)
def test_rejects_bad_metadata_before_network(tmp_path, monkeypatch, field, value):
    item = source_record(image_bytes())
    item[field] = value
    monkeypatch.setattr(commons, "urlopen", no_network)

    with pytest.raises(ValueError):
        commons.download_manifest(manifest_file(tmp_path, [item]), tmp_path / "data", delay=0)


@pytest.mark.parametrize("host", ["upload.wikimedia.org", "thumb.wikimedia.org"])
@pytest.mark.parametrize("license_name", ["CC BY 2.0", "CC BY-SA 2.5", "CC BY 3.0", "CC0 1.0"])
def test_source_validation_accepts_supported_hosts_and_licenses(host, license_name):
    item = source_record(image_bytes())
    item["download_url"] = f"https://{host}/image.png"
    item["license_name"] = license_name

    assert commons.validate_source_record(item) is None


def test_symlinked_parent_cannot_escape_data_root(tmp_path, monkeypatch):
    item = source_record(image_bytes())
    root = tmp_path / "data"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "images").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(commons, "urlopen", no_network)

    with pytest.raises(ValueError, match="root"):
        commons.download_manifest(manifest_file(tmp_path, [item]), root, delay=0)

    assert not list(outside.iterdir())


@pytest.mark.parametrize("kind", ["numeric", "http-date"])
def test_retry_after_is_honored(tmp_path, monkeypatch, kind):
    content = image_bytes()
    item = source_record(content)
    now = datetime(2030, 1, 1, tzinfo=timezone.utc)
    retry_after = (
        "11" if kind == "numeric" else format_datetime(now + timedelta(seconds=11), usegmt=True)
    )
    calls = []
    sleeps = []

    def open_response(request, timeout):
        calls.append(request)
        if len(calls) == 1:
            raise http_error(429, retry_after)
        return Response(content, request.full_url)

    monkeypatch.setattr(commons, "urlopen", open_response)
    monkeypatch.setattr(commons, "sleep", sleeps.append)
    monkeypatch.setattr(commons, "time", lambda: now.timestamp())

    result = commons.download_manifest(manifest_file(tmp_path, [item]), tmp_path / "data", delay=7)

    assert result == {"downloaded": 1, "skipped": 0}
    assert len(calls) == 2
    assert sleeps == [11.0]


def test_retries_are_bounded_and_respect_gentle_delay(tmp_path, monkeypatch):
    item = source_record(image_bytes())
    calls = []
    sleeps = []

    def unavailable(request, timeout):
        calls.append(request)
        raise http_error(503, "0")

    monkeypatch.setattr(commons, "urlopen", unavailable)
    monkeypatch.setattr(commons, "sleep", sleeps.append)

    with pytest.raises(HTTPError):
        commons.download_manifest(manifest_file(tmp_path, [item]), tmp_path / "data", delay=7)

    assert len(calls) == 4
    assert sleeps == [7.0, 7.0, 7.0]
    assert not (tmp_path / "data" / item["path"]).exists()


def test_sequential_downloads_have_a_gentle_gap(tmp_path, monkeypatch):
    content = image_bytes()
    first = source_record(content)
    second = {**first, "path": "images/second.png"}
    sleeps = []
    monkeypatch.setattr(
        commons, "urlopen", lambda request, timeout: Response(content, request.full_url)
    )
    monkeypatch.setattr(commons, "sleep", sleeps.append)

    result = commons.download_manifest(
        manifest_file(tmp_path, [first, second]), tmp_path / "data", delay=7
    )

    assert result == {"downloaded": 2, "skipped": 0}
    assert sleeps == [7.0]


@pytest.mark.parametrize("delay", [-1, float("inf"), float("nan")])
def test_delay_must_be_finite_and_nonnegative(tmp_path, monkeypatch, delay):
    monkeypatch.setattr(commons, "urlopen", no_network)

    with pytest.raises(ValueError, match="delay"):
        commons.download_manifest(
            manifest_file(tmp_path, [source_record(image_bytes())]), tmp_path / "data", delay=delay
        )


def test_cli_reads_the_manifest_and_reports_skipped_files(tmp_path, monkeypatch, capsys):
    content = image_bytes()
    item = source_record(content)
    root = tmp_path / "data"
    target = root / item["path"]
    target.parent.mkdir(parents=True)
    target.write_bytes(content)
    monkeypatch.setattr(commons, "urlopen", no_network)

    status = commons.main(
        [
            "--manifest",
            str(manifest_file(tmp_path, [item])),
            "--data-root",
            str(root),
            "--delay",
            "0",
        ]
    )

    assert status == 0
    assert json.loads(capsys.readouterr().out) == {"downloaded": 0, "skipped": 1}


def test_redirect_outside_commons_never_publishes_an_image(tmp_path, monkeypatch):
    content = image_bytes()
    item = source_record(content)
    root = tmp_path / "data"
    monkeypatch.setattr(
        commons,
        "urlopen",
        lambda request, timeout: Response(content, "https://example.com/image.png"),
    )

    with pytest.raises(ValueError, match="redirect"):
        commons.download_manifest(manifest_file(tmp_path, [item]), root, delay=0)

    assert not (root / item["path"]).exists()


def test_nontransient_http_errors_are_not_retried(tmp_path, monkeypatch):
    item = source_record(image_bytes())
    calls = []
    sleeps = []

    def not_found(request, timeout):
        calls.append(request)
        raise http_error(404)

    monkeypatch.setattr(commons, "urlopen", not_found)
    monkeypatch.setattr(commons, "sleep", sleeps.append)

    with pytest.raises(HTTPError):
        commons.download_manifest(manifest_file(tmp_path, [item]), tmp_path / "data", delay=7)

    assert len(calls) == 1
    assert sleeps == []
