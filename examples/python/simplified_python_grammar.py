from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Any

import numpy as np
from _sep1 import PyRegexExpr as Regex, PyGrammar, PyGrammarExpr as ge, PyGrammarConstraint, PyGrammarConstraintState
from transformers import AutoTokenizer

# Function to convert simple strings to Regex sequence of bytes
def eat_string(s: str) -> Regex:
    if not isinstance(s, str):
        raise TypeError(f"eat_string expects a string, got {type(s)}")
    return Regex.seq([Regex.eat_u8(b) for b in s.encode('utf-8')])

# Define tokens required by the SIMPLIFIED PYTHON grammar
def define_python_tokens() -> list[tuple[str, Any]]:
    tokens = {}
    choice = Regex.choice
    eat_u8 = Regex.eat_u8
    eat_u8_negation = Regex.eat_u8_negation
    seq = Regex.seq
    rep = Regex.rep
    eps = Regex.eps # Epsilon (empty) regex

    # --- Basic Token Patterns ---
    digit = choice([eat_u8(ord(c)) for c in "0123456789"])
    alph_lower = choice([eat_u8(ord(c)) for c in "abcdefghijklmnopqrstuvwxyz"])
    alph_upper = choice([eat_u8(ord(c)) for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"])

    name_start = choice([alph_lower, alph_upper, eat_u8(ord("_"))])
    name_middle = choice([name_start, digit])
    tokens["NAME"] = seq([name_start, rep(name_middle)])

    integer = seq([digit, rep(digit)])
    # Basic float support
    float_num = choice([
        seq([rep(digit), eat_u8(ord(".")), digit, rep(digit)]),
        seq([digit, rep(digit), eat_u8(ord("."))]),
        seq([eat_u8(ord(".")), digit, rep(digit)])
    ])
    tokens["NUMBER"] = choice([float_num, integer])

    # Basic string literals (no escapes handled here, assumes tokenizer provides full string token)
    string_dq = seq([eat_u8(ord('"')), rep(eat_u8_negation(ord('"'))), eat_u8(ord('"'))])
    string_sq = seq([eat_u8(ord("'")), rep(eat_u8_negation(ord("'"))), eat_u8(ord("'"))])
    tokens["STRING"] = choice([string_dq, string_sq])

    # Special Tokens (often handled implicitly or by parser logic, but define for completeness)
    tokens["NEWLINE"] = eat_u8(ord("\n")) # Actual newline character
    tokens["INDENT"] = eps() # Placeholder, indentation is structural
    tokens["DEDENT"] = eps() # Placeholder, indentation is structural
    tokens["ENDMARKER"] = eps() # Represents end of input stream

    # --- Final List ---
    token_exprs = []
    for name, regex_pattern in tokens.items():
         token_exprs.append((name, ge.regex(regex_pattern)))

    # Keywords and Operators will be handled as literals within rules via ge.regex(eat_string(...))

    return token_exprs

# Define the simplified Python grammar structure directly using sep1 expressions
def define_simplified_python_grammar_directly() -> PyGrammar:
    print("Defining simplified Python grammar directly using sep1 expressions...")
    exprs_map = {} # Use a dictionary to build rules before ordering

    # --- Define Tokens ---
    token_definitions = define_python_tokens()
    for name, expr in token_definitions:
        exprs_map[name] = expr

    # --- Helper for Keywords/Operators ---
    def kw(s): return ge.regex(eat_string(s))

    # --- Define Rules ---

    # # Start rule
    # file: statements? ENDMARKER
    exprs_map["file"] = ge.sequence([
        ge.optional(ge.ref("statements")),
        ge.ref("ENDMARKER")
    ])

    # # Statements
    # statements: statement+
    exprs_map["statements"] = ge.repeat1(ge.ref("statement")) # repeat1 means 1 or more

    # statement: compound_stmt | simple_stmts
    exprs_map["statement"] = ge.choice([
        ge.ref("compound_stmt"),
        ge.ref("simple_stmts")
    ])

    # simple_stmts: ';'.simple_stmt+ [';'] NEWLINE | simple_stmt !';' NEWLINE
    # Note: Lookahead !';' is tricky. Let's simplify for direct definition.
    # We'll approximate with two main cases:
    # 1. One simple_stmt followed by NEWLINE
    # 2. Multiple simple_stmts separated by ';', ending in optional ';' and NEWLINE
    # This might slightly over-accept but avoids complex lookahead translation.
    simple_stmt_list = ge.sequence([
        ge.ref("simple_stmt"),
        ge.repeat(ge.sequence([kw(";"), ge.ref("simple_stmt")])),
        ge.optional(kw(";")),
        ge.ref("NEWLINE")
    ])
    single_simple_stmt = ge.sequence([ge.ref("simple_stmt"), ge.ref("NEWLINE")])
    # Prioritize the multi-statement version if possible
    exprs_map["simple_stmts"] = ge.choice([simple_stmt_list, single_simple_stmt])


    # simple_stmt:
    #     | assignment
    #     | expression_stmt
    #     | return_stmt
    #     | import_stmt
    #     | 'pass'
    #     | 'break'
    #     | 'continue'
    exprs_map["simple_stmt"] = ge.choice([
        ge.ref("assignment"),
        ge.ref("expression_stmt"),
        ge.ref("return_stmt"),
        ge.ref("import_stmt"),
        kw("pass"),
        kw("break"),
        kw("continue")
    ])

    # expression_stmt: expression
    exprs_map["expression_stmt"] = ge.ref("expression")

    # compound_stmt:
    #     | function_def
    #     | if_stmt
    #     | class_def
    #     | for_stmt
    #     | while_stmt
    exprs_map["compound_stmt"] = ge.choice([
        ge.ref("function_def"),
        ge.ref("if_stmt"),
        ge.ref("class_def"),
        ge.ref("for_stmt"),
        ge.ref("while_stmt")
    ])

    # # Simple Statements Details
    # assignment: target '=' expression
    exprs_map["assignment"] = ge.sequence([ge.ref("target"), kw("="), ge.ref("expression")])

    # return_stmt: 'return' expression?
    exprs_map["return_stmt"] = ge.sequence([kw("return"), ge.optional(ge.ref("expression"))])

    # import_stmt: import_name | import_from
    exprs_map["import_stmt"] = ge.choice([ge.ref("import_name"), ge.ref("import_from")])

    # import_name: 'import' dotted_as_names
    exprs_map["import_name"] = ge.sequence([kw("import"), ge.ref("dotted_as_names")])

    # import_from: 'from' dotted_name 'import' import_from_targets
    exprs_map["import_from"] = ge.sequence([
        kw("from"), ge.ref("dotted_name"), kw("import"), ge.ref("import_from_targets")
    ])

    # import_from_targets: '(' import_from_as_names [','] ')' | import_from_as_names !',' | '*'
    # Simplified: paren case, non-paren case, star case
    paren_import = ge.sequence([
        kw("("), ge.ref("import_from_as_names"), ge.optional(kw(",")), kw(")")
    ])
    exprs_map["import_from_targets"] = ge.choice([
        paren_import,
        ge.ref("import_from_as_names"), # Assumes it won't be followed by comma if not in parens
        kw("*")
    ])

    # import_from_as_names: ','.import_from_as_name+
    # Translates to: item (',' item)*
    exprs_map["import_from_as_names"] = ge.sequence([
        ge.ref("import_from_as_name"),
        ge.repeat(ge.sequence([kw(","), ge.ref("import_from_as_name")]))
    ])

    # import_from_as_name: NAME ['as' NAME]
    exprs_map["import_from_as_name"] = ge.sequence([
        ge.ref("NAME"),
        ge.optional(ge.sequence([kw("as"), ge.ref("NAME")]))
    ])

    # dotted_as_names: ','.dotted_as_name+
    exprs_map["dotted_as_names"] = ge.sequence([
        ge.ref("dotted_as_name"),
        ge.repeat(ge.sequence([kw(","), ge.ref("dotted_as_name")]))
    ])

    # dotted_as_name: dotted_name ['as' NAME]
    exprs_map["dotted_as_name"] = ge.sequence([
        ge.ref("dotted_name"),
        ge.optional(ge.sequence([kw("as"), ge.ref("NAME")]))
    ])

    # dotted_name: dotted_name '.' NAME | NAME
    # Left recursion: Rewrite as NAME ('.' NAME)*
    exprs_map["dotted_name"] = ge.sequence([
        ge.ref("NAME"),
        ge.repeat(ge.sequence([kw("."), ge.ref("NAME")]))
    ])

    # # Compound Statements Details
    # block: NEWLINE INDENT statements DEDENT | simple_stmts
    # Indentation is tricky. Approximate as NEWLINE + statements or just simple_stmts
    # The actual enforcement often happens outside the pure grammar matching.
    exprs_map["block"] = ge.choice([
        ge.sequence([ge.ref("NEWLINE"), ge.ref("INDENT"), ge.ref("statements"), ge.ref("DEDENT")]), # Ideal case
        ge.ref("simple_stmts") # Fallback for simple blocks
    ])

    # class_def: 'class' NAME ['(' arguments? ')'] ':' block
    exprs_map["class_def"] = ge.sequence([
        kw("class"), ge.ref("NAME"),
        ge.optional(ge.sequence([kw("("), ge.optional(ge.ref("arguments")), kw(")")])),
        kw(":"), ge.ref("block")
    ])

    # function_def: 'def' NAME '(' params? ')' ':' block
    exprs_map["function_def"] = ge.sequence([
        kw("def"), ge.ref("NAME"), kw("("), ge.optional(ge.ref("params")), kw(")"),
        kw(":"), ge.ref("block")
    ])

    # params: ','.param+
    exprs_map["params"] = ge.sequence([
        ge.ref("param"),
        ge.repeat(ge.sequence([kw(","), ge.ref("param")]))
    ])

    # param: NAME
    exprs_map["param"] = ge.ref("NAME")

    # # Revised If/Elif/Else structure
    # if_stmt: 'if' expression ':' block else_suite?
    exprs_map["if_stmt"] = ge.sequence([
        kw("if"), ge.ref("expression"), kw(":"), ge.ref("block"),
        ge.optional(ge.ref("else_suite"))
    ])

    # else_suite: 'elif' expression ':' block else_suite? | 'else' ':' block
    # Recursive definition needs careful handling or unfolding if sep1 struggles. Let's try direct translation.
    exprs_map["else_suite"] = ge.choice([
        ge.sequence([kw("elif"), ge.ref("expression"), kw(":"), ge.ref("block"), ge.optional(ge.ref("else_suite"))]),
        ge.sequence([kw("else"), kw(":"), ge.ref("block")])
    ])

    # else_block defined separately for while/for reuse
    # else_block: 'else' ':' block
    exprs_map["else_block"] = ge.sequence([kw("else"), kw(":"), ge.ref("block")])

    # while_stmt: 'while' expression ':' block [else_block]
    exprs_map["while_stmt"] = ge.sequence([
        kw("while"), ge.ref("expression"), kw(":"), ge.ref("block"),
        ge.optional(ge.ref("else_block"))
    ])

    # for_stmt: 'for' target 'in' expression ':' block [else_block]
    exprs_map["for_stmt"] = ge.sequence([
        kw("for"), ge.ref("target"), kw("in"), ge.ref("expression"), kw(":"), ge.ref("block"),
        ge.optional(ge.ref("else_block"))
    ])

    # # Expressions (Simplified Precedence, added lambda, ternary)
    # expressions: expression_list
    exprs_map["expressions"] = ge.ref("expression_list")

    # expression: boolean_expr | 'lambda' params? ':' expression
    exprs_map["expression"] = ge.choice([
        ge.ref("boolean_expr"),
        ge.sequence([kw("lambda"), ge.optional(ge.ref("params")), kw(":"), ge.ref("expression")])
    ])

    # boolean_expr: logical_or ( 'if' logical_or 'else' expression )?
    exprs_map["boolean_expr"] = ge.sequence([
        ge.ref("logical_or"),
        ge.optional(ge.sequence([kw("if"), ge.ref("logical_or"), kw("else"), ge.ref("expression")]))
    ])

    # logical_or: logical_and ('or' logical_and)*
    exprs_map["logical_or"] = ge.sequence([
        ge.ref("logical_and"),
        ge.repeat(ge.sequence([kw("or"), ge.ref("logical_and")]))
    ])

    # logical_and: logical_not ('and' logical_not)*
    exprs_map["logical_and"] = ge.sequence([
        ge.ref("logical_not"),
        ge.repeat(ge.sequence([kw("and"), ge.ref("logical_not")]))
    ])

    # logical_not: 'not' logical_not | comparison
    exprs_map["logical_not"] = ge.choice([
        ge.sequence([kw("not"), ge.ref("logical_not")]),
        ge.ref("comparison")
    ])

    # comparison: arith_expr (cmp_op arith_expr)*
    exprs_map["comparison"] = ge.sequence([
        ge.ref("arith_expr"),
        ge.repeat(ge.sequence([ge.ref("cmp_op"), ge.ref("arith_expr")]))
    ])

    # cmp_op: '=='|'!='|'<='|'<'|'>='|'>'|'in'|'not' 'in'|'is'|'is' 'not'
    exprs_map["cmp_op"] = ge.choice([
        kw("=="), kw("!="), kw("<="), kw("<"), kw(">="), kw(">"),
        kw("in"), ge.sequence([kw("not"), kw("in")]), # Sequence for 'not in'
        kw("is"), ge.sequence([kw("is"), kw("not")])  # Sequence for 'is not'
    ])

    # arith_expr: term (('+'|'-') term)*
    exprs_map["arith_expr"] = ge.sequence([
        ge.ref("term"),
        ge.repeat(ge.sequence([ge.choice([kw("+"), kw("-")]), ge.ref("term")]))
    ])

    # term: factor (('*'|'/'|'//'|'%') factor)*
    exprs_map["term"] = ge.sequence([
        ge.ref("factor"),
        ge.repeat(ge.sequence([ge.choice([kw("*"), kw("/"), kw("//"), kw("%")]), ge.ref("factor")]))
    ])

    # factor: ('+'|'-'|'~') factor | power
    exprs_map["factor"] = ge.choice([
        ge.sequence([ge.choice([kw("+"), kw("-"), kw("~")]), ge.ref("factor")]),
        ge.ref("power")
    ])

    # power: primary '**' factor | primary
    exprs_map["power"] = ge.choice([
        ge.sequence([ge.ref("primary"), kw("**"), ge.ref("factor")]),
        ge.ref("primary")
    ])

    # primary:
    #     | primary '.' NAME
    #     | primary '(' arguments? ')'
    #     | primary '[' slices ']'
    #     | atom
    # Left recursion: Rewrite as atom ( ( '.' NAME ) | ( '(' arguments? ')' ) | ( '[' slices ']' ) )*
    primary_suffix = ge.choice([
        ge.sequence([kw("."), ge.ref("NAME")]),
        ge.sequence([kw("("), ge.optional(ge.ref("arguments")), kw(")")]),
        ge.sequence([kw("["), ge.ref("slices"), kw("]")])
    ])
    exprs_map["primary"] = ge.sequence([ge.ref("atom"), ge.repeat(primary_suffix)])


    # # Revised slices structure
    # slices: slice | expression_list
    exprs_map["slices"] = ge.choice([ge.ref("slice"), ge.ref("expression_list")])

    # slice: expression? ':' expression? [':' expression?]
    exprs_map["slice"] = ge.sequence([
        ge.optional(ge.ref("expression")), kw(":"), ge.optional(ge.ref("expression")),
        ge.optional(ge.sequence([kw(":"), ge.optional(ge.ref("expression"))]))
    ])

    # expression_list: expression (',' expression)* [',']
    exprs_map["expression_list"] = ge.sequence([
        ge.ref("expression"),
        ge.repeat(ge.sequence([kw(","), ge.ref("expression")])),
        ge.optional(kw(","))
    ])

    # atom:
    #     | NAME | 'True' | 'False' | 'None' | STRING | NUMBER
    #     | '(' expressions? ')' # Group or Tuple
    #     | '[' expressions? ']' # List
    #     | '{' kvpairs? '}' # Dict
    #     | '...'
    exprs_map["atom"] = ge.choice([
        ge.ref("NAME"), kw("True"), kw("False"), kw("None"), ge.ref("STRING"), ge.ref("NUMBER"),
        ge.sequence([kw("("), ge.optional(ge.ref("expressions")), kw(")")]),
        ge.sequence([kw("["), ge.optional(ge.ref("expressions")), kw(")")]), # Corrected closing bracket
        ge.sequence([kw("{"), ge.optional(ge.ref("kvpairs")), kw("}")]),
        kw("...")
    ])

    # kvpairs: ','.kvpair+ [',']
    exprs_map["kvpairs"] = ge.sequence([
        ge.ref("kvpair"),
        ge.repeat(ge.sequence([kw(","), ge.ref("kvpair")])),
        ge.optional(kw(","))
    ])

    # kvpair: expression ':' expression
    exprs_map["kvpair"] = ge.sequence([ge.ref("expression"), kw(":"), ge.ref("expression")])

    # # Function Call Arguments
    # arguments: args?
    exprs_map["arguments"] = ge.optional(ge.ref("args"))

    # args: ','.positional_arg+ [',' kwargs] | kwargs
    positional_args_seq = ge.sequence([
        ge.ref("positional_arg"),
        ge.repeat(ge.sequence([kw(","), ge.ref("positional_arg")]))
    ])
    exprs_map["args"] = ge.choice([
        ge.sequence([positional_args_seq, ge.optional(ge.sequence([kw(","), ge.ref("kwargs")]))]),
        ge.ref("kwargs")
    ])

    # positional_arg: expression
    exprs_map["positional_arg"] = ge.ref("expression")

    # kwargs: ','.kwarg+
    exprs_map["kwargs"] = ge.sequence([
        ge.ref("kwarg"),
        ge.repeat(ge.sequence([kw(","), ge.ref("kwarg")]))
    ])

    # kwarg: NAME '=' expression
    exprs_map["kwarg"] = ge.sequence([ge.ref("NAME"), kw("="), ge.ref("expression")])

    # # Assignment Targets (Simplified)
    # target: NAME | subscription | attribute | '(' targets? ')' | '[' targets? ']'
    exprs_map["target"] = ge.choice([
        ge.ref("NAME"),
        ge.ref("subscription"),
        ge.ref("attribute"),
        ge.sequence([kw("("), ge.optional(ge.ref("targets")), kw(")")]),
        ge.sequence([kw("["), ge.optional(ge.ref("targets")), kw("]")])
    ])

    # targets: target (',' target)* [',']
    exprs_map["targets"] = ge.sequence([
        ge.ref("target"),
        ge.repeat(ge.sequence([kw(","), ge.ref("target")])),
        ge.optional(kw(","))
    ])

    # subscription: primary '[' slices ']'
    exprs_map["subscription"] = ge.sequence([ge.ref("primary"), kw("["), ge.ref("slices"), kw("]")])

    # attribute: primary '.' NAME
    exprs_map["attribute"] = ge.sequence([ge.ref("primary"), kw("."), ge.ref("NAME")])


    # --- Create the final list for PyGrammar ---
    # Order: Start rule, then others alphabetically for clarity (sep1 uses refs anyway)
    final_exprs_list = []
    start_rule_name = "file" # Set the main entry point rule name
    if start_rule_name in exprs_map:
        final_exprs_list.append((start_rule_name, exprs_map[start_rule_name]))
    else:
        raise ValueError(f"Start rule '{start_rule_name}' not defined in grammar map.")

    # Add other rules sorted by name
    for name, expr in sorted(exprs_map.items()):
        if name != start_rule_name:
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
    grammar_constraint = PyGrammarConstraint(grammar, llm_token_to_id, max_llm_token_id)
    return grammar_constraint

if __name__ == "__main__":
    # --- Configuration ---
    tokenizer_name = "gpt2"

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
            processed_token_str = token_str.replace("Ġ", " ") # Handle GPT-2 space
            token_bytes = processed_token_str.encode('utf-8')
            llm_token_to_id[token_bytes] = token_id
            processed_tokens += 1
        except Exception as e:
            skipped_tokens += 1
            pass

    print(f"Processed {processed_tokens} tokens into byte mapping, skipped {skipped_tokens}.")

    if not llm_token_to_id:
        raise ValueError("Failed to create token byte mapping. No tokens processed.")
    max_token_id = max(llm_token_to_id.values())
    print(f"Max token ID in mapping: {max_token_id}")

    # --- Define Grammar Directly ---
    grammar = define_simplified_python_grammar_directly() # CALL THE NEW FUNCTION

    # --- Initialize Grammar Constraint ---
    grammar_constraint = create_grammar_constraint(grammar, llm_token_to_id, max_token_id)

    # --- Get Initial Mask ---
    print("\nInitializing grammar constraint state for initial mask...")
    grammar_constraint_state = PyGrammarConstraintState(grammar_constraint)

    print("Getting initial mask (allowed tokens at the start)...")
    initial_mask = timeit(grammar_constraint_state.get_mask)()

    expected_mask_len = max_token_id + 1
    print(f"Initial mask received (length: {len(initial_mask)}, expected: {expected_mask_len})")
    # Adjust mask length if necessary
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
    id_to_token_str = {v: k for k, v in tokenizer.vocab.items()}

    for token_id in allowed_token_ids:
        token_str = id_to_token_str.get(token_id, f"[Unknown ID: {token_id}]")
        try:
            decoded_str = tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            display_str = f"'{decoded_str}' (Raw: '{token_str}', ID: {token_id})"
        except Exception:
            display_str = f"(Raw: '{token_str}', ID: {token_id})"
        allowed_tokens.append(display_str)

    print("\nAllowed tokens at the start (Simplified Python Grammar - Defined Directly):")
    if not allowed_tokens:
        print("(None)")
    else:
        max_to_print = 200 # Show more tokens for the richer grammar
        allowed_tokens.sort()
        for i, token_info in enumerate(allowed_tokens):
            if i < max_to_print:
                print(f"- {token_info}")
            elif i == max_to_print:
                print(f"... and {len(allowed_tokens) - max_to_print} more.")
                break

    print("\nScript finished.")