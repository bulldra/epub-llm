import json
import os
import tempfile

from fastapi.testclient import TestClient

import src.app as app_module


def test_markdown_stream_endpoint(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        epub_dir = os.path.join(tmpdir, "epub")
        cache_dir = os.path.join(tmpdir, "cache")
        os.makedirs(epub_dir)
        os.makedirs(cache_dir)
        open(os.path.join(epub_dir, "sample.epub"), "wb").close()

        monkeypatch.setattr(app_module, "EPUB_DIR", epub_dir)
        monkeypatch.setattr(app_module, "CACHE_DIR", cache_dir)

        def fake_stream(epub_path, cache_path, max_chars=800):
            yield {"chunk_id": 0, "text": "hello"}
            yield {"chunk_id": 1, "text": "world"}

        monkeypatch.setattr(app_module, "stream_epub_markdown", fake_stream)

        client = TestClient(app_module.app)
        resp = client.get("/book/sample.epub/markdown_stream")
        assert resp.status_code == 200
        lines = [json.loads(line) for line in resp.text.strip().split("\n")]
        assert lines[0]["chunk_id"] == 0
        assert lines[1]["text"] == "world"
