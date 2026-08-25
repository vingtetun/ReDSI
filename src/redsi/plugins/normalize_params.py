from types import SimpleNamespace

from omegaconf import DictConfig, ListConfig, OmegaConf

# ------------------------------------------------------------------------------
# Error messages
# ------------------------------------------------------------------------------

ERR_PARAM_NOT_DICT = "[{identifier}] params must be a dict."
ERR_PARAM_UNKNOWN = "[{identifier}] unknown param '{param}'. Allowed: {allowed}."
ERR_PARAM_MISSING = "[{identifier}] missing required param '{param}'."
ERR_PARAM_TYPE = "[{identifier}] param '{param}' must be '{expected}', got '{got}'."
ERR_PARAM_CHOICE = "[{identifier}] param '{param}' must be one of {choices}, got '{got}'."

# Map string types used in Param() to actual Python types
TYPE_MAP = {
    "str": str,
    "int": int,
    "bool": bool,
    "float": float,
    "list": list,
    "dict": dict,
}


def normalize_params(identifier, plugin, user_params):
    """
    Apply defaults and validate parameters according to plugin.params schema.

    Returns: SimpleNamespace with resolved parameters.
    """

    if user_params is None:
        user_params = {}

    if not isinstance(user_params, DictConfig):
        # Accept SimpleNamespace transparently
        if hasattr(user_params, "__dict__") and not isinstance(user_params, dict):
            user_params = dict(user_params.__dict__)

        if not isinstance(user_params, dict):
            raise TypeError(ERR_PARAM_NOT_DICT.format(identifier=identifier))

    schema = getattr(plugin, "params", []) or []

    # Build schema index
    schema_by_name = {spec["name"]: spec for spec in schema}

    # --------------------------------------------------------------------------
    # Reject unknown parameters
    # --------------------------------------------------------------------------
    skip_params = ["_metadata", "_parent", "_flags_cache", "_content"]
    for param_name in user_params.keys():
        if param_name in skip_params:
            continue

        if param_name not in schema_by_name:
            raise ValueError(
                ERR_PARAM_UNKNOWN.format(
                    identifier=identifier,
                    param=param_name,
                    allowed=sorted(schema_by_name.keys()),
                )
            )

    resolved = {}

    # --------------------------------------------------------------------------
    # Apply defaults + validate each param
    # --------------------------------------------------------------------------
    for spec in schema:
        name = spec["name"]
        required = bool(spec.get("required", False))

        if name in user_params:
            value = user_params[name]
        else:
            if "default" in spec:
                value = spec["default"]
            elif required:
                raise ValueError(ERR_PARAM_MISSING.format(identifier=identifier, param=name))
            else:
                value = None

        if isinstance(value, (DictConfig, ListConfig)):
            value = OmegaConf.to_container(value, resolve=True)

        # Validate type
        if value is not None:
            expected_type_name = spec.get("type")
            expected_type = TYPE_MAP.get(expected_type_name)

            if expected_type_name == "float" and isinstance(value, int) and not isinstance(value, bool):
                value = float(value)

            if expected_type and not isinstance(value, expected_type):
                raise TypeError(
                    ERR_PARAM_TYPE.format(
                        identifier=identifier,
                        param=name,
                        expected=expected_type.__name__,
                        got=type(value).__name__,
                    )
                )

            # Validate choices
            choices = spec.get("choices")
            if choices is not None and value not in choices:
                raise ValueError(
                    ERR_PARAM_CHOICE.format(
                        identifier=identifier,
                        param=name,
                        choices=choices,
                        got=value,
                    )
                )

        resolved[name] = value

    return SimpleNamespace(**resolved)
