"""Shell execution tool."""

import asyncio
import os
import re
import shlex
from pathlib import Path
from typing import Any

from nanobot.agent.tools.base import STATUS_FAILED, STATUS_SUCCESS, Tool


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

    def _is_soft_failure(self, command: str, stdout: str, stderr: str, exit_code: int) -> bool:
        """Scan for errors with refined rules, considering exit code and whitelisted readers.

        - Always scan stderr for fatal patterns.
        - Only inspect stdout for fatal patterns when the exit_code indicates failure.
        - Commands known as "readers" (e.g., cat, ls) are exempt from stdout scanning.
        """
        # More specific fatal patterns (regex) to reduce false positives
        fatal_patterns = [
            r"\bpermission denied\b",
            r"\bnot found\b",
            r"^(?:error|failed|fatal)[:\s]",
            r"\bsyntax error\b",
        ]

        # Check stderr first (case-insensitive, multiline)
        for p in fatal_patterns:
            if re.search(p, stderr or "", flags=re.I | re.M):
                return True

        # If process succeeded, don't treat stdout contents as a soft failure
        if exit_code == 0:
            return False

        # Don't fail if we are just running a reader command that may legitimately
        # output the word 'error' or similar as content
        readers = ["cat", "grep", "head", "tail", "less", "awk", "find", "echo", "ls", "locate"]
        try:
            tokens = shlex.split(command.strip())
            if not tokens:
                return any(re.search(p, stdout or "", flags=re.I | re.M) for p in fatal_patterns)

            executable_token = None
            for token in tokens:
                if "=" not in token:  # Not an env var assignment
                    executable_token = token
                    break

            if executable_token:
                executable_name = os.path.basename(executable_token).lower()
                if executable_name in readers:
                    return False
        except ValueError:
            parts = command.strip().split()
            if parts and os.path.basename(parts[0]).lower() in readers:
                return False

        # Finally, check stdout for fatal patterns (since exit_code != 0)
        for p in fatal_patterns:
            if re.search(p, stdout or "", flags=re.I | re.M):
                return True
        return False
    async def execute(self, **kwargs: Any) -> str:
        command = kwargs.get("command")
        working_dir = kwargs.get("working_dir")
        force = kwargs.get("force", False)
        reason = kwargs.get("reason")

        if not command:
            return f"{STATUS_FAILED}\nError: 'command' parameter is required."

        if force and not reason:
            return f"{STATUS_FAILED}\nError: 'reason' is required when 'force' is True."

        cwd = working_dir or self.working_dir or os.getcwd()

        if not force:
            guard_error = self._guard_command(command, cwd)
            if guard_error:
                return f"{STATUS_FAILED}\n{guard_error}"

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
            except TimeoutError:
                # Attempt graceful termination before forcing a kill and always
                # wait for the process to avoid zombies.
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass

                try:
                    await asyncio.wait_for(process.wait(), timeout=5)
                except TimeoutError:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                    else:
                        try:
                            await asyncio.wait_for(process.wait(), timeout=5)
                        except TimeoutError:
                            pass

                return (
                    f"{STATUS_FAILED}\nError: Command timed out after"
                    f" {self.timeout} seconds"
                )

            stdout_str = stdout.decode("utf-8", errors="replace") if stdout else ""
            stderr_str = stderr.decode("utf-8", errors="replace") if stderr else ""

            is_hard_fail = process.returncode != 0
            is_soft_fail = self._is_soft_failure(command, stdout_str, stderr_str, process.returncode)
            is_quiet = not stdout_str.strip() and not stderr_str.strip()

            if is_hard_fail or is_soft_fail:
                header = STATUS_FAILED
                body = f"EXIT: {process.returncode}\n{stderr_str or stdout_str}"
            else:
                header = STATUS_SUCCESS
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
            return f"{STATUS_FAILED}\nError executing command: {str(e)}"

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
