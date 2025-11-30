import json
import sys

try:
    with open("python/serialized_grammar_constraint.json", "r") as f:
        data = json.load(f)
        if "llm_token_map" in data:
            print(f"llm_token_map type: {type(data['llm_token_map'])}")
            print(f"llm_token_map length: {len(data['llm_token_map'])}")
            print(f"Sample: {list(data['llm_token_map'])[:5]}")
        else:
            print("llm_token_map not found")
except Exception as e:
    print(f"Error: {e}")
