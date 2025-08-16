import random
import time
from functools import wraps
from typing import Callable, Type, Tuple


def retry(
    exceptions: Tuple[Type[BaseException], ...],
    tries: int = 3,
    delay: float = 0.5,
    backoff: float = 2.0,
    jitter: float = 0.1,
):
    """Retry decorator with exponential backoff and jitter."""

    def deco(fn: Callable):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            _tries, _delay = tries, delay
            while _tries > 1:
                try:
                    return fn(*args, **kwargs)
                except exceptions:
                    time.sleep(_delay + random.random() * jitter)
                    _tries -= 1
                    _delay *= backoff
            return fn(*args, **kwargs)

        return wrapper

    return deco
