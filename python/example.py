"""
Simple example demonstrating grammar constraint usage with sep1.

This creates an arithmetic expression grammar and demonstrates mask generation.
"""

import _sep1
import numpy as np

# Define regexes for terminals
plus_regex = _sep1.RegexExpr.eat_u8(ord('+'))
times_regex = _sep1.RegexExpr.eat_u8(ord('*'))
open_paren_regex = _sep1.RegexExpr.eat_u8(ord('('))
close_paren_regex = _sep1.RegexExpr.eat_u8(ord(')'))
i_regex = _sep1.RegexExpr.eat_u8(ord('i'))

# Define grammar rules
rules = [
    ("E", _sep1.GrammarExpr.choice([
        _sep1.GrammarExpr.sequence([
            _sep1.GrammarExpr.ref("E"),
            _sep1.GrammarExpr.ref("PLUS"),
            _sep1.GrammarExpr.ref("T"),
        ]),
        _sep1.GrammarExpr.ref("T"),
    ])),
    ("T", _sep1.GrammarExpr.choice([
        _sep1.GrammarExpr.sequence([
            _sep1.GrammarExpr.ref("T"),
            _sep1.GrammarExpr.ref("TIMES"),
            _sep1.GrammarExpr.ref("F"),
        ]),
        _sep1.GrammarExpr.ref("F"),
    ])),
    ("F", _sep1.GrammarExpr.choice([
        _sep1.GrammarExpr.sequence([
            _sep1.GrammarExpr.ref("LPAREN"),
            _sep1.GrammarExpr.ref("E"),
            _sep1.GrammarExpr.ref("RPAREN"),
        ]),
        _sep1.GrammarExpr.ref("I"),
    ])),
]

# Define terminals (regexes that map to terminal symbols)
terminals = [
    ("PLUS", plus_regex),
    ("TIMES", times_regex),
    ("LPAREN", open_paren_regex),
    ("RPAREN", close_paren_regex),
    ("I", i_regex),
]

# Create grammar definition
grammar_def = _sep1.GrammarDefinition(rules, terminals)
print("Grammar definition created successfully!")

# Compile the grammar
compiled_grammar = grammar_def.compile()
print("Grammar compiled successfully!")

# Define LLM tokens
llm_tokens = [b"i", b"+", b"*", b"(", b")", b"(i", b"+i"]
llm_token_to_id = {token: i for i, token in enumerate(llm_tokens)}

# Create grammar constraint
grammar_constraint = _sep1.GrammarConstraint(compiled_grammar, llm_token_to_id)
grammar_constraint_state = _sep1.GrammarConstraintState(grammar_constraint)

def llm_tokens_to_ids(tokens):
    return [llm_token_to_id[token] for token in tokens]

# Initial mask check
mask = grammar_constraint_state.get_mask()
expected_mask = set(llm_tokens_to_ids([b"i", b"(", b"(i"]))
print(f"Initial Mask: {mask}")
print(f"Valid initial tokens: {[llm_tokens[i] for i in np.where(mask)[0]]}")
assert set(np.where(mask)[0]) == expected_mask, f"Mask: {set(int(x) for x in np.where(mask)[0])}, Expected: {expected_mask}"
print("✓ Initial mask correct!")

# Commit token "i"
grammar_constraint_state.commit(llm_token_to_id[b"i"])

# Mask check after committing "i"
mask = grammar_constraint_state.get_mask()
expected_mask = set(llm_tokens_to_ids([b"+", b"*", b"+i"]))
print(f"\nMask after 'i': {mask}")
print(f"Valid tokens after 'i': {[llm_tokens[i] for i in np.where(mask)[0]]}")
assert set(np.where(mask)[0]) == expected_mask, f"Mask: {set(int(x) for x in np.where(mask)[0])}, Expected: {expected_mask}"
print("✓ Mask after 'i' correct!")

print("\n✓ All tests passed!")