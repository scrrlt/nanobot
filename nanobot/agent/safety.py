import json
import re
import shlex
from collections import deque
from pathlib import Path
from typing import Any


class OscillationDetector:
    def __init__(self, workspace_root: Path, window_size: int = 6, threshold: int = 3):
        self.history: deque[str] = deque(maxlen=window_size)
        self.threshold = threshold
        self.workspace_root = workspace_root

        # Tools/Commands that change state (State-Breakers)
        self.write_tools = {"write_file", "edit_file", "mkdir", "update_plan"}
        self.write_cmds = [
            "touch",
            "mv",
            "cp",
            "rm",
            "mkdir",
            "git",
            "npm i",
            "pip install",
        ]
        self._write_cmd_patterns = [cmd.lower().split() for cmd in self.write_cmds]

    def _extract_command_tokens(self, tokens: list[str]) -> list[str]:
        """Normalize command tokens by removing wrappers and returning lowercase tokens."""
        idx = 0
        length = len(tokens)

        while idx < length:
            token = tokens[idx]
            lower = token.lower()

            # Skip leading environment variable assignments (e.g., VAR=value)
            if "=" in token and not token.startswith("-"):
                idx += 1
                continue

            # Skip common privilege or command wrappers
            if lower in {"sudo", "doas", "command"}:
                idx += 1
                continue

            if lower == "env":
                idx += 1
                while idx < length:
                    candidate = tokens[idx]
                    if candidate.startswith("-") or "=" in candidate:
                        idx += 1
                        continue
                    break
                continue

            if lower.startswith("python") or lower in {"pythonw", "pypy"}:
                # Handle `python -m module` style invocations
                if idx + 1 < length and tokens[idx + 1] == "-m":
                    module_idx = idx + 2
                    if module_idx < length:
                        module_and_args = tokens[module_idx:]
                        return [tok.lower() for tok in module_and_args]
                    return []
                # Handle inline script execution via -c flags
                if idx + 1 < length and tokens[idx + 1] in {"-c", "-lc"}:
                    code_idx = idx + 2
                    if code_idx < length:
                        remainder = " ".join(tokens[code_idx:])
                        try:
                            nested_tokens = shlex.split(remainder)
                        except ValueError:
                            nested_tokens = remainder.split()
                        return [tok.lower() for tok in nested_tokens]
                    return []
                return [tok.lower() for tok in tokens[idx:]]

            if lower in {"sh", "bash", "zsh", "ksh"}:
                idx += 1
                if idx < length and tokens[idx].lower() in {"-c", "-lc"}:
                    idx += 1
                    if idx < length:
                        remainder = " ".join(tokens[idx:])
                        try:
                            nested_tokens = shlex.split(remainder)
                        except ValueError:
                            nested_tokens = remainder.split()
                        return [tok.lower() for tok in nested_tokens]
                    return []
                return [tok.lower() for tok in tokens[idx - 1 :]]

            break

        filtered_tokens = tokens[idx:]
        if filtered_tokens and len(filtered_tokens) == 1 and " " in filtered_tokens[0]:
            joined = filtered_tokens[0]
            try:
                filtered_tokens = shlex.split(joined)
            except ValueError:
                filtered_tokens = joined.split()

        return [tok.lower() for tok in filtered_tokens]

    def normalize_args(self, tool_name: str, args: dict[str, Any]) -> str:
        """Resolves relative paths to absolute to prevent syntax evasion."""
        normalized = args.copy()
        if tool_name == "exec" and "command" in normalized:
            # Simple heuristic to resolve ./ paths
            cmd = normalized["command"]

            def resolve(m: re.Match[str]) -> str:
                try:
                    return str((self.workspace_root / m.group(0)).resolve())
                except Exception:
                    return m.group(0)

            normalized["command"] = re.sub(r"\.{1,2}/[^\s]+", resolve, cmd)
        return json.dumps(normalized, sort_keys=True)

    def check(self, tool_name: str, args: dict[str, Any]) -> str | None:
        """Returns error message if unsafe, else None."""
        # 1. Check for State-Breaking (Write) Action
        is_write = tool_name in self.write_tools

        if not is_write and tool_name == "exec":
            command_value = args.get("command")
            if isinstance(command_value, str):
                try:
                    tokens = shlex.split(command_value)
                except ValueError:
                    tokens = command_value.split()

                normalized_tokens = self._extract_command_tokens(tokens)
                for pattern in self._write_cmd_patterns:
                    if (
                        len(normalized_tokens) >= len(pattern)
                        and normalized_tokens[: len(pattern)] == pattern
                    ):
                        is_write = True
                        break

                if not is_write:
                    lowered_command = command_value.lower()
                    for pattern in self._write_cmd_patterns:
                        pattern_str = " ".join(pattern)
                        if pattern_str and pattern_str in lowered_command:
                            is_write = True
                            break

        if is_write:
            self.history.clear()  # Environment changed, reset read history
            return None

        # 2. Check for Read-Only Oscillation
        sig = f"{tool_name}:{self.normalize_args(tool_name, args)}"
        self.history.append(sig)

        if self.history.count(sig) >= self.threshold:
            return (
                "STATUS: OSCILLATION DETECTED\n"
                "SYSTEM ALERT: You are checking the same state repeatedly without changing it.\n"
                "PROTOCOL: Perform a write action, change your query, or retry "
                "with force=true and a reason."
            )
        return None
