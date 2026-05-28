"""Local foreground process backend for the opt-in bash tool."""

from __future__ import annotations

import os
import selectors
import shutil
import signal
import subprocess
import time
from pathlib import Path

from alpha_agent.tools.base import ToolExecutionContext
from alpha_agent.tools.shell.backend import ShellRequest, ShellResult

READ_CHUNK_BYTES = 8192
POLL_SECONDS = 0.05
TERM_GRACE_SECONDS = 0.5


class LocalShellBackend:
    """Execute a foreground command through bash or sh in a local process group."""

    def __init__(self, shell_path: str | None = None):
        self.shell_path = shell_path

    def execute(self, request: ShellRequest, context: ToolExecutionContext) -> ShellResult:
        """Run the command with timeout and cooperative cancellation checks."""

        shell_path = self._shell_path()
        if shell_path is None:
            return ShellResult(
                status="error",
                exit_code=None,
                stdout="",
                stderr="No bash or sh executable was found",
                elapsed_ms=0,
            )
        shell_name = Path(shell_path).name
        start = time.monotonic()
        try:
            process = subprocess.Popen(
                self._argv(shell_path, request.command),
                cwd=request.workdir,
                env=dict(request.env),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except Exception as exc:
            return ShellResult(
                status="error",
                exit_code=None,
                stdout="",
                stderr=str(exc),
                elapsed_ms=_elapsed_ms(start),
                shell=shell_name,
                error=str(exc),
            )

        selector = selectors.DefaultSelector()
        stdout_capture = _ByteCapture(request.output_capture_bytes)
        stderr_capture = _ByteCapture(request.output_capture_bytes)
        if process.stdout is not None:
            os.set_blocking(process.stdout.fileno(), False)
            selector.register(process.stdout, selectors.EVENT_READ, stdout_capture)
        if process.stderr is not None:
            os.set_blocking(process.stderr.fileno(), False)
            selector.register(process.stderr, selectors.EVENT_READ, stderr_capture)

        status = "completed"
        termination_started_at: float | None = None
        forced_pipe_close = False
        while True:
            now = time.monotonic()
            if status == "completed":
                try:
                    context.check_canceled("during_tool")
                except Exception:
                    status = "canceled"
                    termination_started_at = now
                    self._terminate_process_group(process)
                else:
                    if now - start >= request.timeout_seconds:
                        status = "timeout"
                        termination_started_at = now
                        self._terminate_process_group(process)
            elif termination_started_at is not None:
                elapsed_since_stop = now - termination_started_at
                if elapsed_since_stop >= TERM_GRACE_SECONDS:
                    self._kill_process_group(process)
                if elapsed_since_stop >= TERM_GRACE_SECONDS * 2 and selector.get_map():
                    forced_pipe_close = True
                    self._close_selector(selector)

            if selector.get_map():
                keys = selector.select(POLL_SECONDS)
                if keys:
                    self._read_keys(selector, keys)
            else:
                time.sleep(POLL_SECONDS)

            if process.poll() is not None and not selector.get_map():
                break
            if forced_pipe_close:
                break

        try:
            process.wait(timeout=TERM_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            self._kill_process_group(process)
            process.wait(timeout=TERM_GRACE_SECONDS)
        return ShellResult(
            status=status,
            exit_code=process.returncode,
            stdout=stdout_capture.text(),
            stderr=stderr_capture.text(),
            elapsed_ms=_elapsed_ms(start),
            shell=shell_name,
        )

    def _shell_path(self) -> str | None:
        if self.shell_path:
            return self.shell_path
        return shutil.which("bash") or shutil.which("sh")

    def _argv(self, shell_path: str, command: str) -> list[str]:
        shell_name = Path(shell_path).name
        if shell_name == "bash":
            return [shell_path, "-lc", command]
        return [shell_path, "-c", command]

    def _read_keys(
        self,
        selector: selectors.DefaultSelector,
        keys: list[tuple[selectors.SelectorKey, int]],
    ) -> None:
        for key, _mask in keys:
            stream = key.fileobj
            capture = key.data
            try:
                chunk = os.read(stream.fileno(), READ_CHUNK_BYTES)
            except BlockingIOError:
                continue
            if not chunk:
                selector.unregister(stream)
                continue
            capture.append(chunk)

    def _terminate_process_group(self, process: subprocess.Popen[bytes]) -> None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except PermissionError:
            if process.poll() is None:
                process.terminate()

    def _kill_process_group(self, process: subprocess.Popen[bytes]) -> None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        except PermissionError:
            if process.poll() is None:
                process.kill()

    def _close_selector(self, selector: selectors.DefaultSelector) -> None:
        for key in list(selector.get_map().values()):
            stream = key.fileobj
            try:
                selector.unregister(stream)
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass


class _ByteCapture:
    def __init__(self, limit: int):
        self.limit = max(1, limit)
        self.data = bytearray()

    def append(self, chunk: bytes) -> None:
        remaining = self.limit - len(self.data)
        if remaining <= 0:
            return
        self.data.extend(chunk[:remaining])

    def text(self) -> str:
        return self.data.decode("utf-8", errors="replace")


def _elapsed_ms(start: float) -> int:
    return max(0, int((time.monotonic() - start) * 1000))
