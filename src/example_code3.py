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
            print("Before {} with {}".format(func.__name__, arg_param))
            # A comment inside a nested function
            if (result := await func(*args, **kwargs)) is not None:
                if isinstance(result, (list, tuple)) and len(result) > 0:
                    for i, item in enumerate(result):
                        if i % 2 == 0:
                            try:
                                # Deeply nested expression
                                complex_val = (lambda x: x * x * arg_param)(i) + \
                                              (GLOBAL_VAR // (len(str(item)) + 1))
                                print("Item {}: {}, complex_val: {}".format(i, item, complex_val))
                                if complex_val > 500:
                                    with open("log_{}.txt".format(i), "w") as f:
                                        f.write(str(item) + "\n" + """
                                        Multi-line
                                        string in with block.
                                        """)
                                else: # else for inner if
                                    pass # Pass statement
                            except ZeroDivisionError:
                                print("Oops, division by zero!")
                            except Exception as e:
                                print("Another error: {!r}".format(e)) # f-string with repr
                            finally:
                                # Finally block, could be tricky with indentation
                                print("Processed item {}".format(i))
                        else: # else for if i % 2 == 0
                            continue # Continue statement
                return [x for x in (result if result else []) if x is not None] # Nested comprehension
            print("After {}".format(func.__name__))
            return None # Explicit return None
        return wrapper
    return middle_decorator

@outer_decorator(arg_param=42) # Decorator with arguments
@outer_decorator(arg_param=10) # Multiple decorators
async def challenging_function(data: list[dict[str, any]], threshold: float = 0.5) -> list[str]:
    """
    A function designed to be challenging to parse.
    It includes various Python constructs.
    """
    processed_items: list[str] = []
    if not data: # Check for empty data
        return [] # Early return

    # This is a comment before a loop
    for index, record in enumerate(data):
        # Another comment
        if 'values' in record and (count := len(record['values'])) > 0:
            current_max = -float('inf')
            temp_list = []
            # Nested loop
            for i in range(count): # Loop
                value = record['values'][i]
                if isinstance(value, (int, float)) and value > threshold * (index + 1):
                    # Deeply nested conditional expression within an f-string
                    # and a complex calculation
                    type_str = 'integer' if isinstance(value, int) else \
                               ('float' if isinstance(value, float) else 'other')
                    message = "Processing {} (type: {}) at index {}".format(
                        value, type_str, i
                    )
                    print(message)
                    current_max = max(current_max, value)
                    temp_list.append(
                        (value ** 2) / (current_max + 1e-9) if current_max != 0 else 0.0
                    )
                elif isinstance(value, str) and \
                     len(value) > 3: # Line continuation in condition
                    processed_items.append(value.upper()[:3] + "...")
                else:
                    # Yet another level of nesting
                    if value is None:
                        # This block might be tricky due to its depth
                        print("Skipping None value")
                        break # Break statement
                    else:
                        processed_items.append(str(value)[::-1]) # String slicing

            if temp_list: # Check if temp_list is not empty
                avg_val = sum(temp_list) / len(temp_list)
                processed_items.append("Avg for record {}: {:.2f}".format(index, avg_val))
        # Comment at the end of a loop block
    # Comment after loop, before return
    return processed_items

async def main():
    sample_data = [
        {'values': [1, 2, 0.6, "hello", None, 5.0]},
        {'values': [10, "world", 0.1, 30]},
        {}
    ]
    results = await challenging_function(sample_data, threshold=0.2)
    print("\nFinal Results:")
    for res in results:
        print(res)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())