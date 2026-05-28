"""Shell execution support for the opt-in bash tool."""

from alpha_agent.tools.shell.backend import ShellRequest, ShellResult
from alpha_agent.tools.shell.local import LocalShellBackend
from alpha_agent.tools.shell.policy import BashExecutionPolicy

__all__ = [
    "BashExecutionPolicy",
    "LocalShellBackend",
    "ShellRequest",
    "ShellResult",
]
