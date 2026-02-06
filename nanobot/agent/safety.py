from collections import deque
from pathlib import Path
import json
import re


class OscillationDetector:
    def __init__(self, workspace_root: Path, window_size: int = 6, threshold: int = 3):
        self.history = deque(maxlen=window_size)
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

    def normalize_args(self, tool_name: str, args: dict) -> str:
        """Resolves relative paths to absolute to prevent syntax evasion."""
        normalized = args.copy()
        if tool_name == "exec" and "command" in normalized:
            # Simple heuristic to resolve ./ paths
            cmd = normalized["command"]

            def resolve(m):
                try:
                    return str((self.workspace_root / m.group(0)).resolve())
                except Exception:
                    return m.group(0)

            normalized["command"] = re.sub(r"\.{1,2}/[^\s]+", resolve, cmd)
        return json.dumps(normalized, sort_keys=True)

    def check(self, tool_name: str, args: dict) -> str | None:
        """Returns error message if unsafe, else None."""
        # 1. Check for State-Breaking (Write) Action
        is_write = (tool_name in self.write_tools) or (
            tool_name == "exec"
            and any(c in args.get("command", "") for c in self.write_cmds)
        )

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
