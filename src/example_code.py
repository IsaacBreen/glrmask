from typing import (
    Any,
    List,
    Dict,
    Tuple,
    Optional,
    Union,
)

def example_function(param1: int, param2: str = "default") -> Optional[List[Union[int, str]]]:
    return [param1, param2, 123, "test"]
