"""Read, write, hash, locking, and diff helpers."""

from __future__ import annotations

import difflib
import hashlib
import os
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from alpha_agent.tools.files.errors import FileToolError
from alpha_agent.tools.files.paths import reject_device_path

_LOCKS: dict[Path, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def text_sha256(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def read_file_bytes(path: Path, *, max_file_bytes: int) -> bytes:
    reject_device_path(path)
    if path.is_symlink():
        raise FileToolError("symlink files are not followed")
    if not path.is_file():
        raise FileToolError("path must be a file")
    size = path.stat().st_size
    if size > max_file_bytes:
        raise FileToolError("file is too large to read")
    return path.read_bytes()


def decode_text(data: bytes) -> str:
    if b"\x00" in data:
        raise FileToolError("binary files are not allowed")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FileToolError("binary files are not allowed") from exc


def read_text_file(path: Path, *, max_file_bytes: int) -> tuple[bytes, str]:
    data = read_file_bytes(path, max_file_bytes=max_file_bytes)
    return data, decode_text(data)


@contextmanager
def path_locks(paths: list[Path]) -> Iterator[None]:
    locks = [_lock_for(path.resolve(strict=False)) for path in sorted(paths)]
    for lock in locks:
        lock.acquire()
    try:
        yield
    finally:
        for lock in reversed(locks):
            lock.release()


def atomic_write_text(path: Path, content: str, *, max_file_bytes: int) -> tuple[int, str]:
    data = content.encode("utf-8")
    decode_text(data)
    if len(data) > max_file_bytes:
        raise FileToolError("file is too large to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
            temp_name = handle.name
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        temp_name = None
        readback = path.read_bytes()
        if readback != data:
            raise FileToolError("post-write verification failed")
        return len(data), sha256_bytes(data)
    finally:
        if temp_name is not None:
            try:
                Path(temp_name).unlink()
            except OSError:
                pass


def bounded_unified_diff(
    before_text: str,
    after_text: str,
    *,
    path: str,
    max_chars: int,
) -> str:
    diff = "".join(
        difflib.unified_diff(
            before_text.splitlines(keepends=True),
            after_text.splitlines(keepends=True),
            fromfile=f"{path}\tbefore",
            tofile=f"{path}\tafter",
            n=3,
        )
    )
    limit = max(200, max_chars - 1000)
    if len(diff) <= limit:
        return diff
    return diff[:limit] + "\n... diff truncated ...\n"


def _lock_for(path: Path) -> threading.RLock:
    with _LOCKS_GUARD:
        lock = _LOCKS.get(path)
        if lock is None:
            lock = threading.RLock()
            _LOCKS[path] = lock
        return lock
