# Requirements:
# pip install requests numpy
#
# You may also need `pip install sortedcontainers` for the Python-side replication.
#
# You also need to build the Python bindings first:
# cd python
# maturin develop
# You also need to build the Python bindings first:
# cd python
# maturin develop
# cd ..

import _sep1
import json
import os
import requests
import numpy as np
from sortedcontainers import SortedDict
from pathlib import Path

# --- Helper Functions ---

def load_or_download_gpt2_vocab(cache_dir, file_name, url):
    """Loads a vocabulary from a JSON file, downloading it if not present."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / file_name

    if cache_path.exists():
        print(f"Loading GPT-2 vocab from cache: {cache_path}")
        with open(cache_path, 'r', encoding='utf-8') as f:
            vocab_map = json.load(f)
    else:
        print(f"Downloading GPT-2 vocab from: {url}")
        response = requests.get(url)
        response.raise_for_status()
        content = response.text
        
        with open(cache_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Saved GPT-2 vocab to cache: {cache_path}")
        vocab_map = json.loads(content)
        
    return vocab_map

def greedy_tokenizer(text_bytes, id_to_token):
    """
    A simple greedy tokenizer that finds the longest matching token at each position.
    This is for demonstration purposes; a more efficient implementation (like a Trie)
    would be used in a real application.
    """
    token_to_id = {v: k for k, v in id_to_token.items()}
    
    # Sort tokens by length (descending) to ensure the longest match is found first.
    sorted_tokens = sorted(token_to_id.keys(), key=len, reverse=True)
    
    token_ids = []
    token_strs = []
    
    pos = 0
    while pos < len(text_bytes):
        match_found = False
        for token_bytes in sorted_tokens:
            if text_bytes[pos:].startswith(token_bytes):
                token_ids.append(token_to_id[token_bytes])
                token_strs.append(token_bytes.decode('utf-8', errors='replace'))
                pos += len(token_bytes)
                match_found = True
                break
        if not match_found:
            # This error indicates that the vocabulary cannot fully tokenize the input text.
            raise ValueError(f"Failed to tokenize. No token found for prefix: {text_bytes[pos:pos+20]!r}")
            
    return token_ids, token_strs

def replicate_get_mask3(constraint_state):
    """
    Replicates the logic of `get_mask3` in Python using the exposed bindings.
    This is for demonstration and testing purposes.
    """
    print("\n--- Replicating get_mask3 in Python ---")
    
    # 1. Get initial states and precomputed data
    grammar_constraint = constraint_state.inner.constraint
    initial_states = constraint_state.get_initial_trie3_states() # dict[trie_idx, PyGSSNode]
    
    precomputed_data_str = grammar_constraint.precomputed3_to_json_string()
    precomputed_data = json.loads(precomputed_data_str)
    
    # The 'values' in the arena JSON are [index, value] pairs. We convert this to a dict.
    trie3_god = {item[0]: item[1] for item in precomputed_data['trie3_god']['values']}

    # 2. Set up the traversal state
    final_mask_internal = _sep1.HybridBitset.zeros()
    
    # `todo` is a priority queue mapping depth to a set of trie indices
    todo = SortedDict()
    # `values` maps a trie index to its aggregated GSSNode
    values = {}

    # 3. Seed the traversal with initial states
    for trie_idx, gss_node in initial_states.items():
        if trie_idx in values:
            # Merge if another tokenizer state maps to the same trie root
            existing_node = values[trie_idx]
            values[trie_idx] = _sep1.GSSNode.merge_many([existing_node, gss_node], 1)
        else:
            values[trie_idx] = gss_node
        
        depth = trie3_god[trie_idx]['max_depth']
        if depth not in todo:
            todo[depth] = set()
        todo[depth].add(trie_idx)

    # 4. Main traversal loop (emulating special_map_grouped)
    while todo:
        depth, node_indices = todo.popitem(0) # Pop lowest depth
        
        for node_idx in node_indices:
            if node_idx not in values:
                continue
            
            gss_node = values.pop(node_idx)
            node_data = trie3_god[node_idx]

            # Process step: check if this node is a terminal state in the trie
            if node_data['value']['end']:
                final_mask_internal |= gss_node.allowed_llm_tokens()
            
            if not gss_node.is_ok():
                continue

            # Step to children
            for edge_key_json, dest_map_json in node_data['children']:
                pop, llm_token_bv_json = edge_key_json
                
                # Step function logic: pop from GSS
                popped_items = gss_node.popn(pop)
                
                for dest_idx, state_id_bv_json in dest_map_json:
                    # Create a bitset for the allowed state IDs on this edge
                    state_id_ranges = [tuple(r) for r in state_id_bv_json]
                    state_id_bv = _sep1.HybridBitset.from_ranges(state_id_ranges)
                    
                    valid_gss_parents = []
                    for popper_item in popped_items:
                        for peek in popper_item.peek_iter():
                            if state_id_bv.contains(peek.edge_value()):
                                valid_gss_parents.append(peek.isolated_parent())
                    
                    if not valid_gss_parents:
                        continue
                        
                    merged_gss = _sep1.GSSNode.merge_many(valid_gss_parents, 1)
                    
                    # Create a bitset for the LLM tokens allowed on this edge
                    llm_token_bv_ranges = [tuple(r) for r in llm_token_bv_json]
                    llm_token_bv = _sep1.HybridBitset.from_ranges(llm_token_bv_ranges)
                    
                    # Filter the GSS to only include paths compatible with these LLM tokens
                    merged_gss.allow_only_llm_tokens_and_prune(llm_token_bv)
                    
                    if merged_gss.is_ok():
                        # Merge step: add the new GSS to the values for the destination node
                        if dest_idx in values:
                            values[dest_idx] = _sep1.GSSNode.merge_many([values[dest_idx], merged_gss], 1)
                        else:
                            values[dest_idx] = merged_gss
                        
                        # Add destination to the todo queue
                        child_depth = trie3_god[dest_idx]['max_depth']
                        if child_depth not in todo:
                            todo[child_depth] = set()
                        todo[child_depth].add(dest_idx)

    return final_mask_internal

# --- Main Script ---

def main():
    print("--- Running JavaScript Grammar Constraint Example ---")

    # 1. Load the JS grammar from the EBNF file.
    # This script assumes it is run from the root of the `sep1` project.
    grammar_path = "../src/js.ebnf"
    if not os.path.exists(grammar_path):
        print(f"Error: Grammar file not found at '{grammar_path}'.")
        print("Please run this script from the root directory of the project.")
        return
        
    print(f"Loading grammar from: {grammar_path}")
    grammar_def = _sep1.GrammarDefinition.from_ebnf_file(grammar_path)
    
    # 2. Compile the grammar into a format usable by the constraint.
    print("Compiling grammar...")
    compiled_grammar = grammar_def.compile()
    print("Grammar compiled successfully.")

    # 3. Load a vocabulary. Here, we use the standard GPT-2 vocabulary.
    print("\nLoading GPT-2 vocabulary...")
    cache_dir = ".cache/py_example_vocabs"
    vocab_url = "https://huggingface.co/openai-community/gpt2/raw/main/vocab.json"
    vocab_file_name = "gpt2_vocab.json"
    gpt2_vocab_map = load_or_download_gpt2_vocab(cache_dir, vocab_file_name, vocab_url)

    # Process the vocabulary:
    # - Handle special characters used by the BPE tokenizer (e.g., 'Ġ' for space).
    # - Create mappings from token bytes to integer IDs and vice-versa.
    token_to_id = {}
    id_to_token = {}
    max_token_id = 0
    for token_str, token_id in gpt2_vocab_map.items():
        # The GPT-2 tokenizer uses 'Ġ' to represent a space prefix and 'Ċ' for newlines.
        # We need to convert these back to their raw byte representations.
        processed_str = token_str.replace("Ġ", " ").replace("Ċ", "\n")
        token_bytes = processed_str.encode('utf-8')
        
        token_to_id[token_bytes] = token_id
        id_to_token[token_id] = token_bytes
        if token_id > max_token_id:
            max_token_id = token_id
            
    print(f"GPT-2 vocab loaded and processed ({len(token_to_id)} tokens, max_id: {max_token_id}).")

    # 4. Construct the GrammarConstraint object. This precomputes the constraint logic.
    print("\nConstructing GrammarConstraint (this may take a moment)...")
    # The constraint needs a Python dictionary mapping bytes to integers.
    py_token_to_id = {k: v for k, v in token_to_id.items()}
    grammar_constraint = _sep1.GrammarConstraint(compiled_grammar, py_token_to_id, max_token_id)
    print("GrammarConstraint constructed successfully.")

    # 5. Load and tokenize the example JS code using our simple greedy tokenizer.
    example_code_path = "../src/example_code.js"
    print(f"\nLoading and tokenizing example code from: {example_code_path}")
    with open(example_code_path, 'rb') as f:
        js_code_bytes = f.read()
    
    token_ids, token_strs = greedy_tokenizer(js_code_bytes, id_to_token)
    print(f"Tokenized into {len(token_ids)} tokens.")

    # 6. Step through the token sequence, checking the mask at each step.
    print("\nStepping through the token sequence...")
    constraint_state = _sep1.GrammarConstraintState(grammar_constraint)
    
    for i, (token_id, token_str) in enumerate(zip(token_ids, token_strs)):
        print(f"--- Step {i+1}/{len(token_ids)} ---")
        print(f"Next token: {token_str!r} (ID: {token_id})")

        # Get the mask of allowed tokens from the current state.
        allowed_mask = constraint_state.get_mask()
        
        # Replicate the mask calculation in Python and verify
        # Note: This is slow and for validation only.
        # replicated_mask_internal = replicate_get_mask3(constraint_state)
        
        # Check if the next token in our sequence is allowed by the mask.
        if not allowed_mask[token_id]:
            print("\n--- ERROR ---")
            print(f"Token {token_str!r} (ID: {token_id}) is NOT allowed by the grammar at this position.")
            allowed_ids = [idx for idx, is_allowed in enumerate(allowed_mask) if is_allowed]
            print(f"Mask allows {len(allowed_ids)} tokens. Some allowed tokens:")
            for allowed_id in allowed_ids[:10]:
                print(f"  - ID {allowed_id}: {id_to_token.get(allowed_id, b'<unknown>')!r}")
            raise AssertionError(f"Validation failed at token {i+1}")
        
        print("Token is allowed by the mask.")
        
        # Commit the token to advance the constraint's internal state.
        constraint_state.commit(token_id)
        print("Committed token.")

    # Final check after all tokens are committed
    print("\n--- Final Mask Check ---")
    final_allowed_mask = constraint_state.get_mask()
    final_allowed_ids = [idx for idx, is_allowed in enumerate(final_allowed_mask) if is_allowed]
    print(f"Final mask allows {len(final_allowed_ids)} tokens.")
    
    # Replicate the final mask calculation and assert equality
    print("Replicating final mask calculation in Python for verification...")
    replicated_mask_internal = replicate_get_mask3(constraint_state)
    
    # The replicated mask is on internal token IDs. We need to convert it back to original IDs.
    orig_to_internal = grammar_constraint.original_to_internal_llm_token_map()
    internal_to_orig = {v: k for k, v in orig_to_internal.items()}
    replicated_mask_internal_bools = replicated_mask_internal.iter_bits()
    replicated_mask_orig = np.zeros_like(final_allowed_mask)
    for internal_id, is_allowed in enumerate(replicated_mask_internal_bools):
        if is_allowed and internal_id in internal_to_orig:
            orig_id = internal_to_orig[internal_id]
            if orig_id < len(replicated_mask_orig):
                replicated_mask_orig[orig_id] = True
    
    print("Asserting that Rust mask and Python-replicated mask are identical...")
    np.testing.assert_array_equal(final_allowed_mask, replicated_mask_orig)
    print("Assertion passed!")

    print("\n--- SUCCESS ---")
    print("Successfully processed the entire token sequence according to the grammar.")

if __name__ == "__main__":
    main()
