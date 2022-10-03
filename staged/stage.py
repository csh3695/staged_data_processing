import copy

from .promise import Promise


def stage_decorator(promise: Promise=Promise()):
    assert isinstance(promise, Promise), "promise must be an instance of Promise"

    def decorator(fn):
        def wrapper(*args, **kwargs):
            return copy.deepcopy(promise).bind_function(fn, args, kwargs)
        return wrapper

    return decorator

stage = stage_decorator
