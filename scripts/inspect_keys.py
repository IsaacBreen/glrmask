import json
import sys

try:
    with open("python/serialized_compiled_grammar.json", "r") as f:
        data = json.load(f)
        print("Keys:", list(data.keys()))
except Exception as e:
    print(f"Error: {e}")
