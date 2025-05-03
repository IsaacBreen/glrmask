from __future__ import annotations

import io
import time
import tokenize
from pathlib import Path
from typing import Any

import numpy as np
import pegen.grammar
import pegen.grammar_parser
import pegen.tokenizer
import torch
from _sep1 import PyRegexExpr as Regex, PyGrammar, PyGrammarExpr as ge, PyGrammarConstraint, PyGrammarConstraintState
from transformers import LogitsProcessor, AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def eat_string(s: bytes) -> Regex:
#     return Regex.seq([Regex.eat_u8(ord(c)) for c in s])
    # TODO: delete this
    return Regex.eat_u8(ord("0"))

def rule_name_is_valid(name: str) -> bool:
    return name.startswith("invalid_")


def pegen_to_sep1_regex(item: pegen.grammar.BaseGrammar, memo: dict) -> Regex:
    if isinstance(item, pegen.grammar.NameLeaf):
        if not rule_name_is_valid(item.value):
            return ge.ref(item.value)
    elif isinstance(item, pegen.grammar.StringLeaf):
        value = item.value
        if value[0] == value[-1] in {'"', "'"}:
            value = value[1:-1]
        else:
            raise ValueError(f"Invalid string literal: {value}")
        return ge.regex(eat_string(value))
    elif isinstance(item, pegen.grammar.Opt):
        return ge.optional(pegen_to_sep1_regex(item.node, memo))
    elif isinstance(item, pegen.grammar.Gather):
        expr = pegen_to_sep1_regex(item.node, memo)
        sep = pegen_to_sep1_regex(item.separator, memo)
        return ge.sequence([expr, ge.repeat(ge.sequence([sep, expr]))])
    elif isinstance(item, pegen.grammar.Repeat0):
        return ge.repeat(pegen_to_sep1_regex(item.node, memo))
    elif isinstance(item, pegen.grammar.Repeat1):
        expr = pegen_to_sep1_regex(item.node, memo)
        return ge.sequence([expr, ge.repeat(expr)])
    elif isinstance(item, pegen.grammar.Group):
        return pegen_to_sep1_regex(item.rhs, memo)
    elif isinstance(item, pegen.grammar.Rhs):
        if len(item.alts) == 1:
            return pegen_to_sep1_regex(item.alts[0], memo)
        return ge.choice([pegen_to_sep1_regex(alt, memo) for alt in item.alts])
    elif isinstance(item, pegen.grammar.Alt):
        if len(item.items) == 1:
            return pegen_to_sep1_regex(item.items[0], memo)
        return ge.sequence([pegen_to_sep1_regex(named_item.item, memo) for named_item in item.items])
    elif isinstance(item, pegen.grammar.NamedItem):
        return pegen_to_sep1_regex(item.item, memo)
    elif isinstance(item, pegen.grammar.Forced):
        return pegen_to_sep1_regex(item.node, memo)
    elif isinstance(item, pegen.grammar.PositiveLookahead):
        # return ge.lookahead(pegen_to_sep1_regex(item.node, memo))
        return ge.sequence([])
    elif isinstance(item, pegen.grammar.NegativeLookahead):
        # return ge.negative_lookahead(pegen_to_sep1_regex(item.node, memo))
        return ge.sequence([])
    elif isinstance(item, pegen.grammar.Cut):
        # return ge.cut()
        return ge.sequence([])
    else:
        raise ValueError(f"Unknown item type: {type(item)}")

def define_tokens() -> list[tuple[str, Any]]:
    tokens = {}

    choice = Regex.choice
    eat_u8 = Regex.eat_u8
    eat_u8_negation = Regex.eat_u8_negation
    seq = Regex.seq
    rep = Regex.rep
    eps = Regex.eps

    def eat_u8_choice(s):
        return choice([eat_u8(ord(c)) for c in s])

    ignore = rep(choice([
        eat_u8(ord(" ")),
        seq([eat_u8(ord("#")), rep(eat_u8_negation(ord("\n"))), eat_u8(ord("\n"))]),
    ]))

    def regex(expr):
        return ge.regex(seq([ignore, expr]))

#     # TODO: uncomment this
#     digit = choice([eat_u8(c) for c in range(ord("0"), ord("9") + 1)])
#     alph_lower = choice([eat_u8(c) for c in range(ord("a"), ord("z") + 1)])
#     alph_upper = choice([eat_u8(c) for c in range(ord("A"), ord("Z") + 1)])
#     TODO: delete this
    digit = eat_u8(ord("1"))
    alph_lower = eat_u8(ord("a"))
    alph_upper = eat_u8(ord("a"))

    name_start = choice([
        alph_lower,
        alph_upper,
        eat_u8(ord("_"))
    ])
    name_middle = choice([
        name_start,
        digit,
    ])

    tokens["NAME"] = seq([name_start, rep(name_middle)])
    tokens["NUMBER"] = choice([
        rep(digit),
        seq([rep(digit), eat_u8(ord(".")), rep(digit)]),
    ])
    tokens["NEWLINE"] = eps()
    tokens["INDENT"] = eps()
    tokens["DEDENT"] = eps()
    tokens["STRING"] = choice([
        seq([eat_u8(ord('"')), rep(eat_u8_negation(ord('"'))), eat_u8(ord('"'))]),
        seq([eat_u8(ord("'")), rep(eat_u8_negation(ord("'"))), eat_u8(ord("'"))]),
    ])
    tokens["FSTRING_START"] = choice([
        eat_string('"""'),
        eat_string("'''"),
    ])
    tokens["FSTRING_END"] = choice([
        eat_string('"""'),
        eat_string("'''"),
    ])
    tokens["FSTRING_MIDDLE"] = rep(choice([
        eat_u8_negation(ord("{")),
        eat_string("{{"),
    ]))
    tokens["TYPE_COMMENT"] = eps()
    tokens["ENDMARKER"] = eps()
    return [(name, regex(expr)) for name, expr in tokens.items()]

def pegen_to_sep1_grammar(grammar: pegen.grammar.Grammar) -> PyGrammar:
    memo = {}
    exprs: list[tuple[str, Any]] = []

    # Make sure the start production is first
    exprs.append(("start'''", ge.ref("file")))

    for rule in grammar.rules.values():
        if not rule_name_is_valid(rule.name):
            print(f"Ignoring invalid rule name: {rule.name}")
            memo[rule.name] = ge.ref(rule.name)
            exprs.append((rule.name, ge.choice([])))
        else:
            memo[rule.name] = ge.ref(rule.name)
            exprs.append((rule.name, pegen_to_sep1_regex(rule.rhs, memo)))

    tokens = define_tokens()
    exprs.extend(tokens)

    return PyGrammar(exprs)

def define_python_grammar():
    with Path(__file__).parent / "python.gram" as f:
        grammar_text = f.read_text()

    with io.StringIO(grammar_text) as f:
        tokenizer = pegen.tokenizer.Tokenizer(tokenize.generate_tokens(f.readline))
        parser = pegen.grammar_parser.GeneratedParser(tokenizer)
        grammar = parser.start()

    return pegen_to_sep1_grammar(grammar)

def debug_print(message):
    print(message, end='; ')

def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        debug_print(f"Time taken: {(end_time - start_time) * 1000:.2f} ms")
        return result
    return wrapper

class GrammarConstrainedLogitsProcessor(LogitsProcessor):
    def __init__(self, grammar_constraint_state, llm_token_to_id):
        self.grammar_constraint_state = grammar_constraint_state
        self.seen_input_ids = []
        self.llm_token_to_id = llm_token_to_id
        self.llm_token_id_to_token = {id: token for token, id in llm_token_to_id.items()}

    def __call__(self, input_ids, scores):
        current_input_ids = input_ids.view(-1).tolist()
        new_token_ids = current_input_ids[len(self.seen_input_ids):]

        for token_id in new_token_ids:
            debug_print(f"Committing token: {self.llm_token_id_to_token.get(token_id)} (ID: {token_id})")
            timeit(self.grammar_constraint_state.commit)(token_id)

        self.seen_input_ids = current_input_ids
        mask = timeit(self.grammar_constraint_state.get_mask)()

        if len(mask) < scores.shape[-1]:
            padding = np.zeros(scores.shape[-1] - len(mask), dtype=bool)
            mask = np.concatenate((mask, padding))
        elif len(mask) > scores.shape[-1]:
            mask = mask[:scores.shape[-1]]

        mask_ids = np.where(mask)[0]
        mask_id_map = {id: self.llm_token_id_to_token.get(id) for id in mask_ids}
#         debug_print(f"Mask IDs: {mask_id_map}")
        print("")

        scores = np.where(mask, scores, -np.inf)
        return torch.tensor(scores)

def generate_text(model, tokenizer, grammar_processor, input_text, max_new_tokens=50):
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    grammar_processor.seen_input_ids = input_ids[0].tolist()
    output = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        logits_processor=[grammar_processor]
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

if __name__ == "__main__":
    model_name = "Qwen/Qwen2.5-Coder-0.5B"
#     model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenizer_vocab = tokenizer.get_vocab()
    # Remove tokens that have any letter other than 'a'
    tokenizer_vocab = {k: v for k, v in tokenizer_vocab.items() if not any(c.isalpha() and c != 'a' for c in k)}
    # ...or have any number other than '1'
    tokenizer_vocab = {k: v for k, v in tokenizer_vocab.items() if not any(c.isdigit() and c != '1' for c in k)}
    # ...or have any letter or number
    tokenizer_vocab = {k: v for k, v in tokenizer_vocab.items() if not any(c.isalpha() or c.isdigit() for c in k)}
    # ...or have any character other than those in the given set
#     allowed_chars = set("÷$(¾&§.-}][¢^·/?'¼\×´¨,¡¦*¸¥»±«¤¶>+~_°¯#;½¿=!£|:%)\"{<`©®@¬")
    allowed_chars = set("_")
    tokenizer_vocab = {k: v for k, v in tokenizer_vocab.items() if not any(c not in allowed_chars for c in k)}

    # Set the vocab to just underscores
    tokenizer_vocab = {"_": 0, "__": 1}

    tokenizer_vocab = {"a": 0, "aa": 1}

    llm_token_to_id = {token.replace("Ġ", " ").encode(): i for token, i in tokenizer_vocab.items()}
    llm_tokens = list(tokenizer_vocab.keys()) # Use all tokens

    print("vocab size:", len(llm_tokens))
    # Print all characters in the vocab
    chars = set()
    for token in llm_tokens:
        chars.update(set(token))
    print(f"All characters in the vocab: {"".join(chars)}")

#     ts = ['Paris', 'London']
#     llm_tokens = [x.encode() for x in ts]
#     llm_token_to_id = {token.encode(): tokenizer.convert_tokens_to_ids(token) for token in ts}

    print("Defining grammar...")
    grammar = define_python_grammar()
    grammar.print()
    print("Initializing Parser...")
    parser = grammar.glr_parser()
    parser.print()
    print("Initializing Grammar Constraint...")
    grammar_constraint = PyGrammarConstraint(grammar, llm_token_to_id, max(llm_token_to_id.values()))
    print("Initializing grammar constraint state...")
    grammar_constraint_state = PyGrammarConstraintState(grammar_constraint)
    print("Initializing grammar processor...")
    grammar_processor = GrammarConstrainedLogitsProcessor(grammar_constraint_state, llm_token_to_id)

    model = AutoModelForCausalLM.from_pretrained(model_name)

    print("Generating text...")
#     input_text = "i^10=i*"
    input_text = "5*6 + 7*2 = 5+5+5+"

    # DEMO: Get the mask
    grammar_constraint_state = PyGrammarConstraintState(grammar_constraint)
    tokens = tokenizer.encode(input_text, return_tensors="pt")
    tokens: list[int] = tokens.tolist()[0]
#     for token_id in tokens:
#         grammar_constraint_state.commit(token_id)
    mask = grammar_constraint_state.get_mask()
    print("Got mask")
    print(f"Mask: {mask}")
    mask_ids = np.where(mask)[0].tolist()
    mask_token_ids = [tokenizer.convert_ids_to_tokens(id) for id in mask_ids]
    print(f"Mask Tokens: {mask_token_ids}")

    # DEMO: Generate text.
    grammar_constraint_state = PyGrammarConstraintState(grammar_constraint)
#     output_text = generate_text(model, tokenizer, grammar_processor, input_text)
#     print(output_text)