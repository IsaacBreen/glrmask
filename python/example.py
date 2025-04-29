import _sep1
import numpy as np

# Define regexes using PyRegexExpr
plus_regex = _sep1.PyRegexExpr.eat_u8(ord('+'))
times_regex = _sep1.PyRegexExpr.eat_u8(ord('*'))
open_paren_regex = _sep1.PyRegexExpr.eat_u8(ord('('))
close_paren_regex = _sep1.PyRegexExpr.eat_u8(ord(')'))
i_regex = _sep1.PyRegexExpr.eat_u8(ord('i'))

# Define grammar rules using the regexes
exprs = [
    ("E", _sep1.PyGrammarExpr.choice([
        _sep1.PyGrammarExpr.sequence([
            _sep1.PyGrammarExpr.ref("E"),
            _sep1.PyGrammarExpr.regex(plus_regex),
            _sep1.PyGrammarExpr.ref("T"),
        ]),
        _sep1.PyGrammarExpr.ref("T"),
    ])),
    ("T", _sep1.PyGrammarExpr.choice([
        _sep1.PyGrammarExpr.sequence([
            _sep1.PyGrammarExpr.ref("T"),
            _sep1.PyGrammarExpr.regex(times_regex),
            _sep1.PyGrammarExpr.ref("F"),
        ]),
        _sep1.PyGrammarExpr.ref("F"),
    ])),
    ("F", _sep1.PyGrammarExpr.choice([
        _sep1.PyGrammarExpr.sequence([
            _sep1.PyGrammarExpr.regex(open_paren_regex),
            _sep1.PyGrammarExpr.ref("E"),
            _sep1.PyGrammarExpr.regex(close_paren_regex),
        ]),
        _sep1.PyGrammarExpr.regex(i_regex),
    ])),
]

grammar = _sep1.PyGrammar(exprs)

# Define LLM tokens
llm_tokens = [b"i", b"+", b"*", b"(", b")", b"(i", b"+i"]
llm_token_to_id = {token: i for i, token in enumerate(llm_tokens)}

# Create grammar constraint
grammar_constraint = _sep1.PyGrammarConstraint(grammar, llm_token_to_id, len(llm_tokens))
grammar_constraint_state = _sep1.PyGrammarConstraintState(grammar_constraint)

def llm_tokens_to_ids(tokens):
    return [llm_token_to_id[token] for token in tokens]

# Initial mask check
mask = grammar_constraint_state.get_mask()
expected_mask = set(llm_tokens_to_ids([b"i", b"(", b"(i"]))  # Use set for unordered comparison
print(f"Initial Mask: {mask}")
assert set(np.where(mask)[0]) == expected_mask, f"Mask: {set(int(x) for x in np.where(mask)[0])}, Expected: {expected_mask}"

# Commit prefill tokens
prefill = llm_tokens_to_ids([b"(i", b"+", b"i", b"*", b"i"])
for token_id in prefill:
    grammar_constraint_state.commit(token_id)

# Mask check after prefill
mask = grammar_constraint_state.get_mask()
expected_mask = set(llm_tokens_to_ids([b"+", b"*", b")", b"+i"]))
print(f"Mask after committing prefill: {mask}")
assert set(np.where(mask)[0]) == expected_mask, f"Mask: {set(int(x) for x in np.where(mask)[0])}, Expected: {expected_mask}"