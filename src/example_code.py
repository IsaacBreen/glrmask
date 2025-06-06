# Top-level comment, challenging parser start
import os, sys # Multiple imports on one line
from collections import (defaultdict,
                         deque) # Multi-line import with parens

GLOBAL_VAR: int = 100
ANOTHER_GLOBAL = r"C:\raw\string\path" + \
                 " and continued" # Line continuation

def outer_decorator(arg_param):
    """Outer decorator docstring."""
    def middle_decorator(func):
        async def wrapper(*args, **kwargs):
            print(f"Before {func.__name__} with {arg_param}")
            # A comment inside a nested function
            pass
