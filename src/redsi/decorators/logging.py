import functools
import inspect


def log_function_start(*paths):
    def decorator(fn):
        sig = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()

            values = []
            for path in paths:
                val = _find_value(path, bound.arguments)
                if val is not None:
                    values.append(str(val))

            extra = "[" + "/".join(values) + "]" if values else ""
            print(f"{fn.__name__} {extra}")
            return fn(*args, **kwargs)

        return wrapper

    return decorator


def _find_value(path, bound_args):
    """
    bound_args: ordered dict mapping parameter name -> value
    path: can be "split_config" or "datasets.plugin.name" etc.
    """
    # 1) Direct param name match
    if path in bound_args:
        return bound_args[path]

    parts = path.split(".")

    # 2) Try resolving dotted path against each bound arg value
    for value in bound_args.values():
        out = _get_dotted(value, parts)
        if out is not None:
            return out

        # also try value.args.<...>
        if hasattr(value, "args"):
            out = _get_dotted(value.args, parts)
            if out is not None:
                return out

    return None


def _get_dotted(obj, parts):
    cur = obj
    for name in parts:
        if cur is None:
            return None

        # object / SimpleNamespace
        if hasattr(cur, name):
            cur = getattr(cur, name)
            continue

        # dict
        if isinstance(cur, dict) and name in cur:
            cur = cur[name]
            continue

        return None

    return cur
