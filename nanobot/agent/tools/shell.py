"""Shell execution tool."""

import asyncio
import os
import re
from pathlib import Path
from typing import Any

from nanobot.agent.tools.base import Tool


class ExecTool(Tool):
    """Tool to execute shell commands."""

    def __init__(
        self,
        timeout: int = 60,
        working_dir: str | None = None,
        deny_patterns: list[str] | None = None,
        allow_patterns: list[str] | None = None,
        restrict_to_workspace: bool = False,
    ):
        self.timeout = timeout
        self.working_dir = working_dir
        self.deny_patterns = deny_patterns or [
            r"\brm\s+-[rf]{1,2}\b",  # rm -r, rm -rf, rm -fr
            r"\bdel\s+/[fq]\b",  # del /f, del /q
            r"\brmdir\s+/s\b",  # rmdir /s
            r"\b(format|mkfs|diskpart)\b",  # disk operations
            r"\bdd\s+if=",  # dd
            r">\s*/dev/sd",  # write to disk
            r"\b(shutdown|reboot|poweroff)\b",  # system power
            r":\(\)\s*\{.*\};\s*:",  # fork bomb
        ]
        self.allow_patterns = allow_patterns or []
        self.restrict_to_workspace = restrict_to_workspace

    @property
    def name(self) -> str:
        return "exec"

    @property
    def description(self) -> str:
        return "Execute a shell command and return its output. Use with caution."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute",
                },
                "force": {"type": "boolean", "description": "Override safety checks."},
                "reason": {"type": "string", "description": "Required if force=True."},
            },
            "required": ["command"],
        }

    def _is_soft_failure(self, command: str, stdout: str, stderr: str) -> bool:
        """Scan for errors, but whitelist 'read' commands from stdout scanning."""
        fatal = ["permission denied", "not found", "fatal error", "syntax error"]
        if any(p in stderr.lower() for p in fatal):
            return True

        # Don't fail if we are just reading a log file containing 'error'
        readers = ["cat", "grep", "head", "tail", "less", "awk"]
        parts = command.strip().split()
        if parts and parts[0] in readers:
            return False

        return any(p in stdout.lower() for p in fatal)

    async def execute(self, **kwargs: Any) -> str:
        command = kwargs.get("command")
        working_dir = kwargs.get("working_dir")
        force = kwargs.get("force", False)
        reason = kwargs.get("reason")

        if not command:
            return "STATUS: FAILED\nError: 'command' parameter is required."

        if force and not reason:
            return "STATUS: FAILED\nError: 'reason' is required when 'force' is True."

        cwd = working_dir or self.working_dir or os.getcwd()

        if not force:
            guard_error = self._guard_command(command, cwd)
            if guard_error:
                return f"STATUS: FAILED\n{guard_error}"

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=self.timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                return f"STATUS: FAILED\nError: Command timed out after {self.timeout} seconds"

            stdout_str = stdout.decode("utf-8", errors="replace") if stdout else ""
            stderr_str = stderr.decode("utf-8", errors="replace") if stderr else ""

            is_hard_fail = process.returncode != 0
            is_soft_fail = self._is_soft_failure(command, stdout_str, stderr_str)
            is_quiet = not stdout_str.strip() and not stderr_str.strip()

            if is_hard_fail or is_soft_fail:
                header = "STATUS: FAILED"
                body = f"EXIT: {process.returncode}\n{stderr_str or stdout_str}"
            else:
                header = "STATUS: SUCCESS"
                body = stdout_str if not is_quiet else "(Success. No output.)"

            result = f"{header}\n{body}"

            # Truncate very long output
            max_len = 10000
            if len(result) > max_len:
                result = (
                    result[:max_len]
                    + f"\n... (truncated, {len(result) - max_len} more chars)"
                )

            return result

        except Exception as e:
            return f"STATUS: FAILED\nError executing command: {str(e)}"

    def _guard_command(self, command: str, cwd: str) -> str | None:
        """Best-effort safety guard for potentially destructive commands."""
        cmd = command.strip()
        lower = cmd.lower()

        for pattern in self.deny_patterns:
            if re.search(pattern, lower):
                return "Error: Command blocked by safety guard (dangerous pattern detected)"

        if self.allow_patterns:
            if not any(re.search(p, lower) for p in self.allow_patterns):
                return "Error: Command blocked by safety guard (not in allowlist)"

        if self.restrict_to_workspace:
            if "..\\" in cmd or "../" in cmd:
                return "Error: Command blocked by safety guard (path traversal detected)"

            cwd_path = Path(cwd).resolve()

            win_paths = re.findall(r"[A-Za-z]:\\[^\\\"']+", cmd)
            posix_paths = re.findall(r"/[^\s\"']+", cmd)

            for raw in win_paths + posix_paths:
                try:
                    p = Path(raw).resolve()
                except Exception:
                    continue
                if cwd_path not in p.parents and p != cwd_path:
                    return "Error: Command blocked by safety guard (path outside working dir)"

        return None
