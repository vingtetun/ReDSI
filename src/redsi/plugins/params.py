ERR_PARAM_NAME_REQUIRED = "Param must have a non-empty 'name'."
ERR_PARAM_TYPE_INVALID = "Param '{name}' has invalid type '{type}'."
ERR_PARAM_CHOICES_INVALID = "Param '{name}' choices must be a list or tuple."


_ALLOWED_TYPES = {"str", "int", "bool", "float", "list", "dict"}


def Param(name, type, default=None, *, choices=None, required=False, help=""):
    if not name:
        raise ValueError(ERR_PARAM_NAME_REQUIRED)

    if type not in _ALLOWED_TYPES:
        raise ValueError(ERR_PARAM_TYPE_INVALID.format(name=name, type=type))

    if choices is not None and not isinstance(choices, (list, tuple)):
        raise ValueError(ERR_PARAM_CHOICES_INVALID.format(name=name))

    spec = {
        "name": name,
        "type": type,
        "help": help or "",
    }

    if default is not None:
        spec["default"] = default

    if choices is not None:
        spec["choices"] = list(choices)

    if required:
        spec["required"] = True

    return spec
