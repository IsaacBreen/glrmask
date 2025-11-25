import json
import sys
import _sep1 as ffi

def main():
    # 1. Load CompiledGrammar
    print("Loading CompiledGrammar...")
    with open("python/serialized_compiled_grammar.json", "r") as f:
        compiled_grammar_json = f.read()
    
    compiled_grammar = ffi.CompiledGrammar.from_json_string(compiled_grammar_json)
    
    # 2. Load Vocab from serialized_grammar_constraint.json
    print("Loading Vocab...")
    with open("python/serialized_grammar_constraint.json", "r") as f:
        constraint_data = json.load(f)
        llm_token_map_list = constraint_data["llm_token_map"]
        # Convert list [[bytes, id], ...] to dict {bytes: id}
        # Note: bytes in JSON are list of ints
        token_to_id = {}
        max_id = 0
        for token_bytes_list, token_id in llm_token_map_list:
            token_bytes = bytes(token_bytes_list)
            token_to_id[token_bytes] = token_id
            max_id = max(max_id, token_id)
            
    print(f"Vocab size: {len(token_to_id)}")
    print(f"Max Token ID: {max_id}")

    # 3. Create GrammarConstraint
    print("Creating GrammarConstraint...")
    # GrammarConstraint.new(py, grammar, token_to_id, max_id)
    # Note: token_to_id expects bytes keys. In Python dict, bytes keys are fine.
    constraint = ffi.GrammarConstraint(compiled_grammar, token_to_id, max_id)
    
    # 4. Serialize to JSON
    print("Serializing GrammarConstraint...")
    constraint_json = constraint.to_json_string()
    
    # 5. Save
    output_path = "benchmarking/constraint.json"
    print(f"Saving to {output_path}...")
    with open(output_path, "w") as f:
        f.write(constraint_json)
        
    print("Done!")

if __name__ == "__main__":
    main()
