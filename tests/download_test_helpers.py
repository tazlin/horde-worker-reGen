"""Shared test doubles for exercising real model downloads against a local server.

A loopback HTTP server serves deterministic bytes (faked *files*) while the download path runs for real
(real HTTP, real disk, real checksum sidecars), so download tests can prove end-to-end behaviour without a
GPU, the inference stack, or the network.
"""

from __future__ import annotations

import hashlib
import threading
import urllib.request
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING

from horde_model_reference.on_disk_layout import file_paths_for, is_present

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from horde_model_reference.model_reference_records import GenericModelRecord

__all__ = ["FakeModelServer", "RealDownloadCompVis", "deterministic_bytes"]


def deterministic_bytes(name: str, size: int) -> bytes:
    """Reproducible pseudo-random content for *name*, so checksums are stable across runs."""
    out = bytearray()
    seed = hashlib.sha256(name.encode()).digest()
    while len(out) < size:
        seed = hashlib.sha256(seed).digest()
        out.extend(seed)
    return bytes(out[:size])


class FakeModelServer:
    """A loopback HTTP server that serves fixed bytes per path and counts requests per path."""

    def __init__(self) -> None:
        """Start empty; call :meth:`add` to seed file contents, then :meth:`start`."""
        self._content: dict[str, bytes] = {}
        self.hits: Counter[str] = Counter()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def add(self, file_name: str, data: bytes) -> None:
        """Serve *data* at ``/<file_name>``."""
        self._content[f"/{file_name}"] = data

    def start(self) -> None:
        """Bind to an ephemeral loopback port and serve in a background thread."""
        content = self._content
        hits = self.hits

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - http.server's required name
                hits[self.path] += 1
                body = content.get(self.path)
                if body is None:
                    self.send_response(404)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args: object) -> None:
                """Silence per-request stderr logging."""

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def base_url(self) -> str:
        """The ``http://host:port`` the server is listening on (must be started first)."""
        assert self._server is not None
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def stop(self) -> None:
        """Shut the server down and release the socket."""
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()


class RealDownloadCompVis:
    """A stand-in compvis manager that performs genuine HTTP downloads to the canonical on-disk layout.

    Mirrors the parts of hordelib's ``BaseModelManager`` the download core/process touch
    (``download_model`` / ``validate_model`` / ``is_model_available`` / ``available_models`` /
    ``model_folder_path``). Presence is delegated to the canonical ``is_present`` so it matches the system;
    files already on disk are skipped, as the real download engine does.
    """

    def __init__(self, weights_root: Path, records: dict[str, GenericModelRecord]) -> None:
        """Resolve files under *weights_root* for the given name->record mapping."""
        self._weights_root = weights_root
        self._records = records
        self.model_folder_path = weights_root / "compvis"

    @property
    def available_models(self) -> list[str]:
        """Every configured model whose declared files all exist on disk."""
        return [name for name, record in self._records.items() if is_present(record, self._weights_root)]

    def is_model_available(self, model_name: str) -> bool:
        """Whether *model_name*'s declared files are all present (existence-only)."""
        return is_present(self._records[model_name], self._weights_root)

    def download_model(
        self,
        model_name: str,
        *,
        callback: Callable[[int, int], None] | None = None,
        connections: int = 1,
    ) -> bool:
        """Fetch each declared file over HTTP to its canonical path, skipping any already present.

        ``connections`` mirrors the real manager's signature and is accepted but ignored: this stand-in
        always fetches single-stream.
        """
        record = self._records[model_name]
        for download, dest in zip(record.config.download, file_paths_for(record, self._weights_root), strict=True):
            if dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(download.file_url) as response:  # noqa: S310 - loopback test server
                total = int(response.headers.get("Content-Length", 0))
                downloaded = 0
                with dest.open("wb") as handle:
                    while True:
                        chunk = response.read(1024)
                        if not chunk:
                            break
                        handle.write(chunk)
                        downloaded += len(chunk)
                        if callback is not None:
                            callback(downloaded, total)
            dest.with_suffix(".sha256").write_text(hashlib.sha256(dest.read_bytes()).hexdigest())
        return True

    def validate_model(self, model_name: str) -> bool:
        """Existence-based validity for the stand-in (the canonical presence check)."""
        return is_present(self._records[model_name], self._weights_root)
