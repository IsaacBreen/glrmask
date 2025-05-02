from __future__ import annotations

import io
import time
import tokenize # For grammar file parsing
from pathlib import Path
from typing import Any

import numpy as np
import pegen.grammar
import pegen.grammar_parser
import pegen.tokenizer
from _sep1 import PyRegexExpr as Regex, PyGrammar, PyGrammarExpr as ge, PyGrammarConstraint, PyGrammarConstraintState
from transformers import AutoTokenizer # To get vocabulary and token IDs

# Function to convert simple strings to Regex sequence of bytes
def eat_string(s: str) -> Regex:
    return Regex.seq([Regex.eat_u8(b) for b in s.encode('utf-8')])

# Function to convert pegen grammar nodes to sep1 grammar expressions
def pegen_to_sep1_regex(item: pegen.grammar.BaseGrammar, memo: dict) -> Any:
    if item in memo:
        return memo[item]

    result: Any

    if isinstance(item, pegen.grammar.NameLeaf):
        result = ge.ref(item.value)
    elif isinstance(item, pegen.grammar.StringLeaf):
        value = item.value
        if value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        else:
            raise ValueError(f"Invalid string literal in grammar: {item.value}")
        result = ge.regex(eat_string(value))
    elif isinstance(item, pegen.grammar.Opt):
        result = ge.optional(pegen_to_sep1_regex(item.node, memo))
    elif isinstance(item, pegen.grammar.Gather):
        expr = pegen_to_sep1_regex(item.node, memo)
        sep = pegen_to_sep1_regex(item.separator, memo)
        result = ge.sequence([expr, ge.repeat(ge.sequence([sep, expr]))])
    elif isinstance(item, pegen.grammar.Repeat0):
        result = ge.repeat(pegen_to_sep1_regex(item.node, memo))
    elif isinstance(item, pegen.grammar.Repeat1):
        expr = pegen_to_sep1_regex(item.node, memo)
        result = ge.sequence([expr, ge.repeat(expr)])
    elif isinstance(item, pegen.grammar.Group):
        result = pegen_to_sep1_regex(item.rhs, memo)
    elif isinstance(item, pegen.grammar.Rhs):
        if len(item.alts) == 1:
            result = pegen_to_sep1_regex(item.alts[0], memo)
        else:
            result = ge.choice([pegen_to_sep1_regex(alt, memo) for alt in item.alts])
    elif isinstance(item, pegen.grammar.Alt):
        if not item.items:
             result = ge.sequence([]) # Epsilon
        elif len(item.items) == 1:
            result = pegen_to_sep1_regex(item.items[0], memo)
        else:
            result = ge.sequence([pegen_to_sep1_regex(named_item, memo) for named_item in item.items])
    elif isinstance(item, pegen.grammar.NamedItem):
        result = pegen_to_sep1_regex(item.item, memo)
    # --- Lookaheads and Cut --- Approximated as epsilon
    elif isinstance(item, (pegen.grammar.PositiveLookahead, pegen.grammar.NegativeLookahead, pegen.grammar.Cut)):
        result = ge.sequence([]) # Epsilon
    elif isinstance(item, pegen.grammar.Forced):
         result = pegen_to_sep1_regex(item.node, memo)
    else:
        raise TypeError(f"Unsupported grammar item type: {type(item)}")

    memo[item] = result
    return result

# Define basic tokens required by the simplified grammar
def define_tokens() -> list[tuple[str, Any]]:
    tokens = {}
    choice = Regex.choice
    eat_u8 = Regex.eat_u8
    eat_u8_negation = Regex.eat_u8_negation
    seq = Regex.seq
    rep = Regex.rep
    eps = Regex.eps # Epsilon (empty) regex

    # --- Whitespace and Comments (to be ignored between tokens) ---
    ws_char = eat_u8_choice(" \t")
    whitespace = seq([rep(ws_char)])
    comment = seq([eat_u8(ord("#")), rep(eat_u8_negation(ord("\n")))])
    # Define ignore pattern - PyGrammarConstraint might need this separately
    ignore_pattern = rep(choice([ws_char, comment]))

    # --- Token Definitions ---
    digit = eat_u8_choice("0123456789")
    alph_lower = eat_u8_choice("abcdefghijklmnopqrstuvwxyz")
    alph_upper = eat_u8_choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    name_start = choice([alph_lower, alph_upper, eat_u8(ord("_"))])
    name_middle = choice([name_start, digit])
    tokens["NAME"] = seq([name_start, rep(name_middle)])

    integer = seq([digit, rep(digit)])
    float_num = seq([rep(digit), eat_u8(ord(".")), digit, rep(digit)])
    tokens["NUMBER"] = choice([float_num, integer])

    string_dq = seq([eat_u8(ord('"')), rep(eat_u8_negation(ord('"'))), eat_u8(ord('"'))])
    string_sq = seq([eat_u8(ord("'")), rep(eat_u8_negation(ord("'"))), eat_u8(ord("'"))])
    tokens["STRING"] = choice([string_dq, string_sq])

    tokens["NEWLINE"] = eat_u8(ord("\n"))
    tokens["INDENT"] = eps() # Handled by parser logic, not regex alone
    tokens["DEDENT"] = eps() # Handled by parser logic, not regex alone
    tokens["ENDMARKER"] = eps() # Represents end of input

    # --- Final List ---
    # Return list of (token_name, grammar_expression)
    token_exprs = []
    for name, regex_pattern in tokens.items():
         # Associate the raw Regex pattern using ge.regex()
         token_exprs.append((name, ge.regex(regex_pattern)))

    # Note: The 'ignore_pattern' might need to be passed to PyGrammarConstraint initialization.
    # This example assumes default handling or internal mechanisms.

    return token_exprs

# Function to parse the grammar file and convert to sep1 format
def pegen_to_sep1_grammar(grammar_path: Path) -> PyGrammar:
    print(f"Parsing grammar file: {grammar_path}")
    with grammar_path.open("r", encoding="utf-8") as f:
        grammar_text = f.read()

    try:
        grammar_bytes = grammar_text.encode('utf-8')
        token_stream = tokenize.tokenize(io.BytesIO(grammar_bytes).readline)
        # Use pegen's tokenizer wrapper - ENABLE VERBOSE MODE HERE
        pegen_tokenizer_inst = pegen.tokenizer.Tokenizer(token_stream, verbose=True) # <--- SET verbose=True
        parser = pegen.grammar_parser.GeneratedParser(pegen_tokenizer_inst)
        grammar = parser.start()
        if not grammar:
             # If grammar is None or parsing failed before returning, raise error
             raise ValueError("Failed to parse grammar file using pegen (parser.start() returned None or failed).")
    except tokenize.TokenError as e:
        print(f"Token Error parsing grammar file: {e}")
        raise
    except Exception as e:
        # Catch potential errors from parser.start() itself
        print(f"Error during pegen grammar parsing: {e}")
        # Optionally re-raise or wrap in a custom exception
        raise ValueError(f"Failed to parse grammar file using pegen. Details: {e}")


    print("Converting pegen grammar to sep1 format...")
    memo = {}
    exprs: list[tuple[str, Any]] = [] # List of (rule_name, sep1_expression)

    # Ensure grammar object is valid before accessing rules
    if not grammar or not grammar.rules:
         raise ValueError("Pegen parsing resulted in an invalid or empty grammar object.")

    start_rule_name = next(iter(grammar.rules)) # Get the first rule as start
    print(f"Identified start rule: {start_rule_name}")
    exprs.append(("start", ge.ref(start_rule_name))) # Define 'start' entry point

    for rule_name, rule in grammar.rules.items():
        sep1_expr = pegen_to_sep1_regex(rule.rhs, memo)
        exprs.append((rule_name, sep1_expr))

    print("Defining tokens...")
    tokens = define_tokens()
    exprs.extend(tokens)

    print("Creating PyGrammar object...")
    py_grammar = PyGrammar(exprs)
    # py_grammar.print() # Optional: Print the converted grammar structure
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
    # Note: May need to pass ignore_pattern regex if required by PyGrammarConstraint constructor
    grammar_constraint = PyGrammarConstraint(grammar, llm_token_to_id, max_llm_token_id)
    return grammar_constraint

if __name__ == "__main__":
    # --- Configuration ---
    tokenizer_name = "gpt2" # Using gpt2 for its common vocab
    grammar_file = Path(__file__).parent / "simplified_python.gram"

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

    # Determine max token ID actually used in the mapping
    if not llm_token_to_id:
        raise ValueError("Failed to create token byte mapping. No tokens processed.")
    max_token_id = max(llm_token_to_id.values())
    print(f"Max token ID in mapping: {max_token_id}")
    # eos_token_id = tokenizer.eos_token_id # Not needed for initial mask, but good to know

    # --- Load and Convert Grammar ---
    grammar = pegen_to_sep1_grammar(grammar_file)

    # --- Initialize Grammar Constraint ---
    # This step compiles the grammar and token regexes
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
            decoded_str = tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            display_str = f"'{decoded_str}' (Raw: '{token_str}', ID: {token_id})"
        except Exception:
            display_str = f"(Raw: '{token_str}', ID: {token_id})" # Fallback if decode fails

        allowed_tokens.append(display_str)

    print("\nAllowed tokens at the start:")
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