"""Base class for agent tools."""

from abc import ABC, abstractmethod
from typing import Any, cast

STATUS_SUCCESS = "STATUS: SUCCESS"
STATUS_FAILED = "STATUS: FAILED"


class Tool(ABC):
    """
    Abstract base class for agent tools.

    Tools are capabilities that the agent can use to interact with
    the environment, such as reading files, executing commands, etc.
    """

    _TYPE_MAP = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
        "null": type(None),
    }

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name used in function calls."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what the tool does."""
        pass

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """JSON Schema for tool parameters."""
        pass

    @abstractmethod
    async def execute(self, **kwargs: Any) -> str:
        """
        Execute the tool with given parameters.

        Args:
            **kwargs: Tool-specific parameters.

        Returns:
            String result of the tool execution.
        """
        pass

    def validate_params(self, params: dict[str, Any]) -> list[str]:
        """Validate tool parameters against JSON schema. Returns error list (empty if valid)."""
        schema = self.parameters or {}
        if schema.get("type", "object") != "object":
            raise ValueError(f"Schema must be object type, got {schema.get('type')!r}")
        return self._validate(params, {**schema, "type": "object"}, "")

    def _validate(self, val: Any, schema: dict[str, Any], path: str) -> list[str]:
        label = path or "parameter"
        schema_type = schema.get("type")

        if isinstance(schema_type, list):
            first_failure: list[str] = []
            for candidate in schema_type:
                candidate_schema = {k: v for k, v in schema.items() if k != "type"}
                candidate_schema["type"] = candidate
                candidate_errors = self._validate(val, candidate_schema, path)
                if not candidate_errors:
                    return []
                if not first_failure:
                    first_failure = candidate_errors
            allowed = ", ".join(str(t) for t in schema_type)
            error = f"{label} should match one of types: {allowed}"
            return [error, *first_failure]

        if schema_type in ("integer", "number") and isinstance(val, bool):
            return [f"{label} should be {schema_type}"]

        if schema_type in self._TYPE_MAP and not isinstance(
            val, self._TYPE_MAP[schema_type]
        ):
            return [f"{label} should be {schema_type}"]

        errors: list[str] = []
        if "enum" in schema and val not in schema["enum"]:
            errors.append(f"{label} must be one of {schema['enum']}")
        if schema_type in ("integer", "number"):
            if "minimum" in schema and val < schema["minimum"]:
                errors.append(f"{label} must be >= {schema['minimum']}")
            if "maximum" in schema and val > schema["maximum"]:
                errors.append(f"{label} must be <= {schema['maximum']}")
        if schema_type == "string" and isinstance(val, str):
            if "minLength" in schema and len(val) < schema["minLength"]:
                errors.append(f"{label} must be at least {schema['minLength']} chars")
            if "maxLength" in schema and len(val) > schema["maxLength"]:
                errors.append(f"{label} must be at most {schema['maxLength']} chars")
        if schema_type == "object" and isinstance(val, dict):
            raw_props = schema.get("properties", {})
            props: dict[str, Any] = {}
            if isinstance(raw_props, dict):
                props = cast(dict[str, Any], raw_props)
            raw_required = schema.get("required", [])
            required_keys: list[str] = []
            if isinstance(raw_required, list):
                required_list = cast(list[Any], raw_required)
                required_keys = [str(item) for item in required_list]
            val_dict = cast(dict[str, Any], val)
            for key in required_keys:
                if key not in val_dict:
                    missing = f"{path}.{key}" if path else key
                    errors.append(f"missing required {missing}")
            for key, value in val_dict.items():
                child_schema = props.get(key)
                if isinstance(child_schema, dict):
                    child_path = f"{path}.{key}" if path else key
                    child_schema_dict = cast(dict[str, Any], child_schema)
                    errors.extend(self._validate(value, child_schema_dict, child_path))
        if (
            schema_type == "array"
            and isinstance(val, list)
            and isinstance(schema.get("items"), dict)
        ):
            items_schema = cast(dict[str, Any], schema["items"])
            val_list = cast(list[Any], val)
            for index, item in enumerate(val_list):
                child_path = f"{path}[{index}]" if path else f"[{index}]"
                errors.extend(self._validate(item, items_schema, child_path))

        any_of = schema.get("anyOf")
        if any_of:
            matched_any = False
            option_messages: list[str] = []
            for variant in any_of:
                variant_errors = self._validate(val, variant, path)
                if not variant_errors:
                    matched_any = True
                    break
                option_messages.append(", ".join(variant_errors))
            if not matched_any:
                detail = "; ".join(msg for msg in option_messages if msg)
                message = f"{label} did not match any allowed schema option (anyOf)"
                if detail:
                    message = f"{message}: {detail}"
                errors.append(message)

        one_of = schema.get("oneOf")
        if one_of:
            match_count = 0
            mismatch_details: list[str] = []
            for variant in one_of:
                variant_errors = self._validate(val, variant, path)
                if not variant_errors:
                    match_count += 1
                else:
                    mismatch_details.append(", ".join(variant_errors))
            if match_count == 0:
                detail = "; ".join(msg for msg in mismatch_details if msg)
                message = f"{label} did not match any allowed schema option (oneOf)"
                if detail:
                    message = f"{message}: {detail}"
                errors.append(message)
            elif match_count > 1:
                errors.append(
                    f"{label} matches multiple schema options but oneOf expects "
                    "exactly one"
                )
        return errors

    def to_schema(self) -> dict[str, Any]:
        """Convert tool to OpenAI function schema format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }
