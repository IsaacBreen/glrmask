from __future__ import annotations

import io
import time
# import tokenize # No longer needed for grammar parsing
from pathlib import Path
from typing import Any

import numpy as np
# import pegen # No longer needed for grammar parsing
from _sep1 import PyRegexExpr as Regex, PyGrammar, PyGrammarExpr as ge, PyGrammarConstraint, PyGrammarConstraintState
from transformers import AutoTokenizer # To get vocabulary and token IDs

# Function to convert simple strings to Regex sequence of bytes
def eat_string(s: str) -> Regex:
    # Ensure input is string
    if not isinstance(s, str):
        raise TypeError(f"eat_string expects a string, got {type(s)}")
    return Regex.seq([Regex.eat_u8(b) for b in s.encode('utf-8')])

# Define ONLY the tokens required by the MINIMAL grammar
def define_tokens() -> list[tuple[str, Any]]:
    tokens = {}
    choice = Regex.choice
    eat_u8 = Regex.eat_u8
    # eat_u8_negation = Regex.eat_u8_negation # Not needed for minimal tokens
    seq = Regex.seq
    rep = Regex.rep
    eps = Regex.eps # Epsilon (empty) regex

    # --- Token Definitions ---
    # CORRECTED: Use Regex.choice with list comprehension
    digit = choice([eat_u8(ord(c)) for c in "0123456789"])
    alph_lower = choice([eat_u8(ord(c)) for c in "abcdefghijklmnopqrstuvwxyz"])
    alph_upper = choice([eat_u8(ord(c)) for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"])

    name_start = choice([alph_lower, alph_upper, eat_u8(ord("_"))])
    name_middle = choice([name_start, digit])
    tokens["NAME"] = seq([name_start, rep(name_middle)])

    integer = seq([digit, rep(digit)])
    tokens["NUMBER"] = integer

    tokens["ENDMARKER"] = eps() # Represents end of input

    # --- Final List ---
    token_exprs = []
    for name, regex_pattern in tokens.items():
         # Wrap the raw Regex pattern in a grammar expression using ge.regex()
         token_exprs.append((name, ge.regex(regex_pattern)))

    return token_exprs

# Define the minimal grammar structure directly using sep1 expressions
def define_minimal_grammar_directly() -> PyGrammar:
    print("Defining minimal grammar directly in Python using sep1 expressions...")
    exprs_map = {} # Use a dictionary to build rules before ordering

    # Define Tokens first and add to map
    token_definitions = define_tokens()
    for name, expr in token_definitions:
        exprs_map[name] = expr

    # Define Rules using ge objects and references (ge.ref)
    # atom: NAME | NUMBER | '(' expr ')'
    exprs_map["atom"] = ge.choice([
        ge.ref("NAME"),
        ge.ref("NUMBER"),
        ge.sequence([
            ge.regex(eat_string("(")), # Literal '(' handled by regex
            ge.ref("expr"),
            ge.regex(eat_string(")"))  # Literal ')' handled by regex
        ])
    ])

    # term: atom ('*' atom)*
    exprs_map["term"] = ge.sequence([
        ge.ref("atom"),
        ge.repeat(ge.sequence([
            ge.regex(eat_string("*")), # Literal '*' handled by regex
            ge.ref("atom")
        ]))
    ])

    # expr: term ('+' term)*
    exprs_map["expr"] = ge.sequence([
        ge.ref("term"),
        ge.repeat(ge.sequence([
            ge.regex(eat_string("+")), # Literal '+' handled by regex
            ge.ref("term")
        ]))
    ])

    # start: expr ENDMARKER
    exprs_map["start"] = ge.sequence([
        ge.ref("expr"),
        ge.ref("ENDMARKER")
    ])

    # --- Create the final list for PyGrammar ---
    # Ensure 'start' rule is first if required by sep1, otherwise order doesn't matter as refs are used.
    # Let's assume order matters and put 'start' first.
    final_exprs_list = []
    if "start" in exprs_map:
        final_exprs_list.append(("start", exprs_map["start"]))
    else:
        raise ValueError("Start rule 'start' not defined in grammar map.")

    for name, expr in exprs_map.items():
        if name != "start":
            final_exprs_list.append((name, expr))

    print("Creating PyGrammar object from direct definition...")
    py_grammar = PyGrammar(final_exprs_list)
    # py_grammar.print() # Optional: Print the grammar structure
    return py_grammar

# Helper for timing functions
def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"[Time taken for {func.__name__}: {(end_time - start_time) * 1000:.2f} ms]")
        return result
    return wrapper

# Function to create the grammar constraint object
@timeit
def create_grammar_constraint(grammar, llm_token_to_id, max_llm_token_id):
    print("Initializing PyGrammarConstraint...")
    # Note: Ensure PyGrammarConstraint handles whitespace between tokens appropriately.
    # If it relies on an explicit ignore pattern, that needs to be defined and passed.
    # Assuming default handling for now.
    grammar_constraint = PyGrammarConstraint(grammar, llm_token_to_id, max_llm_token_id)
    return grammar_constraint

if __name__ == "__main__":
    # --- Configuration ---
    tokenizer_name = "gpt2" # Using gpt2 for its common vocab
    # grammar_file = Path(__file__).parent / "simplified_python.gram" # No longer reading file

    # --- Load Tokenizer Vocab ---
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Create mapping from byte sequences to token IDs
    llm_token_to_id = {}
    processed_tokens = 0
    skipped_tokens = 0
    for token_str, token_id in tokenizer.vocab.items():
        try:
            # Handle potential special characters or byte representations like GPT-2's 'Ġ' for space
            processed_token_str = token_str.replace("Ġ", " ")
            # Encode to bytes, assuming UTF-8
            token_bytes = processed_token_str.encode('utf-8')
            llm_token_to_id[token_bytes] = token_id
            processed_tokens += 1
        except Exception as e:
            # print(f"Warning: Could not process token '{token_str}' (ID: {token_id}): {e}")
            skipped_tokens += 1
            pass # Skip tokens that cause encoding issues

    print(f"Processed {processed_tokens} tokens into byte mapping, skipped {skipped_tokens}.")

    if not llm_token_to_id:
        raise ValueError("Failed to create token byte mapping. No tokens processed.")
    max_token_id = max(llm_token_to_id.values())
    print(f"Max token ID in mapping: {max_token_id}")
    # eos_token_id = tokenizer.eos_token_id

    # --- Define Grammar Directly ---
    grammar = define_minimal_grammar_directly() # CALL THE NEW FUNCTION

    # --- Initialize Grammar Constraint ---
    grammar_constraint = create_grammar_constraint(grammar, llm_token_to_id, max_token_id)

    # --- Get Initial Mask ---
    print("\nInitializing grammar constraint state for initial mask...")
    grammar_constraint_state = PyGrammarConstraintState(grammar_constraint)

    print("Getting initial mask (allowed tokens at the start)...")
    initial_mask = timeit(grammar_constraint_state.get_mask)()

    # Ensure mask length matches expected size (max_token_id + 1)
    expected_mask_len = max_token_id + 1
    print(f"Initial mask received (length: {len(initial_mask)}, expected: {expected_mask_len})")
    if len(initial_mask) < expected_mask_len:
        print("Warning: Mask length is less than expected. Padding...")
        padding = np.zeros(expected_mask_len - len(initial_mask), dtype=bool)
        initial_mask = np.concatenate((initial_mask, padding))
    elif len(initial_mask) > expected_mask_len:
        print("Warning: Mask length is greater than expected. Truncating...")
        initial_mask = initial_mask[:expected_mask_len]


    # --- Analyze and Print the Mask ---
    allowed_token_ids = np.where(initial_mask)[0].tolist()
    print(f"Number of allowed tokens initially: {len(allowed_token_ids)}")

    allowed_tokens = []
    id_to_token_str = {v: k for k, v in tokenizer.vocab.items()} # For display

    for token_id in allowed_token_ids:
        # Get the original token string representation from the tokenizer's vocab
        token_str = id_to_token_str.get(token_id, f"[Unknown ID: {token_id}]")
        # Attempt to decode the ID using the tokenizer for a cleaner representation
        try:
            # Use clean_up_tokenization_spaces=False to see raw token boundaries
            decoded_str = tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            display_str = f"'{decoded_str}' (Raw: '{token_str}', ID: {token_id})"
        except Exception:
            display_str = f"(Raw: '{token_str}', ID: {token_id})" # Fallback if decode fails

        allowed_tokens.append(display_str)

    print("\nAllowed tokens at the start (Minimal Grammar - Defined Directly):")
    if not allowed_tokens:
        print("(None)")
    else:
        max_to_print = 100 # Print more tokens if available
        allowed_tokens.sort() # Sort for consistent output
        for i, token_info in enumerate(allowed_tokens):
            if i < max_to_print:
                print(f"- {token_info}")
            elif i == max_to_print:
                print(f"... and {len(allowed_tokens) - max_to_print} more.")
                break

    print("\nScript finished.")