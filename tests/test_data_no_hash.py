"""No-hash mode preserves decoding, image identity and exact download lengths."""
import asyncio
import io
import json

import httpx
from PIL import Image
import pytest

from scripts import download_b512_corners_v3 as archives
from scripts.supply_b512_images import prepare_one


def forbidden(*args, **kwargs):
    raise AssertionError("unexpected data hash computation")


def test_no_hash_preparation_does_not_calculate_sha_or_phash(tmp_path, monkeypatch):
    path = tmp_path / "image.png"
    Image.new("RGB", (600, 512), "blue").save(path)
    monkeypatch.setattr("hashlib.sha256", forbidden)
    pixels, view = prepare_one(str(path), {"view_policy": "fit_pad", "expected_source_sha256": "wrong"}, False)
    assert view["source_sha256"] is None and view["view_sha256"] is None
    assert not view["hashes_computed"] and "perceptual_hashes" not in view
    assert Image.open(io.BytesIO(pixels)).size == (512, 512)
    path.write_bytes(b"not an image")
    with pytest.raises(OSError):
        prepare_one(str(path), {"view_policy": "fit_pad"}, False)


def test_archive_no_hash_retains_length_checks_and_strict_resume_rechecks(tmp_path, monkeypatch):
    payload = b"fixture archive bytes"
    row = {"id": "fixture", "path": str(tmp_path / "archive.tar"), "revision": "pinned",
           "bytes": len(payload), "sha256": "wrong", "md5_base64": "wrong",
           "url": "https://example.invalid/archive.tar"}
    monkeypatch.setattr(archives, "digest", forbidden)
    monkeypatch.setattr("hashlib.sha256", forbidden)

    class ByteStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield payload

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=ByteStream()))) as client:
            receipt = await archives.fetch_file(client, row, 2, {"network_bytes": 0}, compute_hashes=False)
            assert receipt["sha256"] is None and receipt["compute_hashes"] is False
            assert not receipt["checksums_verified"]
            # A size-only receipt may never silently satisfy a later strict checksum check.
            with pytest.raises(AssertionError, match="unexpected data hash"):
                await archives.fetch_file(client, row, 2, {"network_bytes": 0}, compute_hashes=True)
            with pytest.raises(ValueError, match="length differs"):
                await archives.fetch_file(client, {**row, "bytes": len(payload) + 1}, 2, {"network_bytes": 0}, False)

    asyncio.run(scenario())
    assert json.loads((tmp_path / "archive.tar.v3-verified.json").read_text())["sha256"] is None
