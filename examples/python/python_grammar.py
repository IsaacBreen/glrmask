from __future__ import annotations

import io
import json
import jsonyx
import textwrap
import time
import tokenize
from pathlib import Path
from typing import Any

import numpy as np
import pegen.grammar
import pegen.grammar_parser
import pegen.tokenizer # type: ignore
import torch # type: ignore
from _sep1 import RegexExpr as Regex,  CompiledGrammar,  GrammarExpr as ge,  GrammarConstraint,  GrammarConstraintState, GrammarDefinition
from transformers import LogitsProcessor, AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def regex(expr, name=None):
    if not isinstance(expr, ge):
        expr = ge.regex(expr)
    if name == "IGNORE":
        print(f"Ignoring ignore for IGNORE rule: {expr}")
        return name, expr
    if name is None:
        return ge.sequence([ge.ref("IGNORE"), expr])
#         return expr
#     return name, ge.regex(seq([ignore, expr]))
    return name, ge.sequence([ge.ref("IGNORE"), expr])
#     return name, expr

def eat(s: bytes) -> Regex:
    if len(s) == 1:
        return Regex.eat_u8(ord(s[0]))
    else:
        return Regex.seq([Regex.eat_u8(ord(c)) for c in s])

def rule_name_is_valid(name: str) -> bool:
    return not name.startswith("invalid_")
    # TODO: delete this
#     return True

def pegen_to_sep1_regex(item: pegen.grammar.BaseGrammar, memo: dict) -> Regex:
    if isinstance(item, pegen.grammar.NameLeaf):
        return ge.ref(item.value)
    elif isinstance(item, pegen.grammar.StringLeaf):
        value = item.value
        if value[0] == value[-1] in {'"', "'"}:
            value = value[1:-1]
        else:
            raise ValueError(f"Invalid string literal: {value}")
#         # TODO: delete this
        return regex(ge.literal(value.encode()))
#         return ge.literal("1".encode())
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
    rep1 = Regex.rep1
    eps = Regex.eps
    opt = Regex.opt

    def eat_u8_choice(s):
        return choice([eat_u8(ord(c)) for c in s])

    def eat_range(start: char, end: char) -> Regex:
        return choice([Regex.eat_u8(c) for c in range(ord(start), ord(end) + 1)])

    # TODO: Use eg eat("a") instead of eat_u8(ord("a")). It's a bit more readable.

    ignore = ge.optional(ge.regex(rep1(choice([
        eat_u8(ord(" ")),
        # TODO: delete this?
        eat_u8(ord("\n")),
        seq([eat_u8(ord("#")), rep(eat_u8_negation(ord("\n"))), eat_u8(ord("\n"))]),
    ]))))
    tokens["IGNORE"] = ignore
#     # TODO: delete this
#     tokens["IGNORE"] = eps()
#     tokens["IGNORE"] = ge.optional(ge.regex(eat(" ")))

    # TODO: uncomment this
    digit = eat_range('0', '9')
    alph_lower = eat_range('a', 'z')
    alph_upper = eat_range('A', 'Z')
# #     TODO: delete this
#     digit = eat_u8(ord("1"))
#     alph_lower = eat_u8(ord("a"))
#     alph_upper = eat_u8(ord("a"))

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
        rep1(digit),
        seq([rep(digit), eat_u8(ord(".")), rep(digit)]),
    ])
#     # TODO: delete this
#     tokens["NAME"] = eps()
#     tokens["NUMBER"] = eps()
# #     tokens["NAME"] = eat("f")
# #     tokens["NUMBER"] = rep(eat("1"))
    tokens["NEWLINE"] = eat("\n")
    tokens["INDENT"] = rep1(eat(" "))
#     tokens["DEDENT"] = eps()

    tokens["STRING"] = choice([
        seq([eat_u8(ord('"')), rep(choice([eat_u8_negation(ord('"')), eat('\"')])), eat_u8(ord('"'))]),
        seq([eat_u8(ord("'")), rep(choice([eat_u8_negation(ord("'")), eat('\'')])), eat_u8(ord("'"))]),
    ])
    tokens["FSTRING_START"] = choice([
        eat('"""'),
        eat("'''"),
    ])
    tokens["FSTRING_END"] = choice([
        eat('"""'),
        eat("'''"),
    ])
    tokens["FSTRING_MIDDLE"] = rep1(choice([
        eat_u8_negation(ord("{")),
        eat("{{"),
    ]))
#     # TODO: delete this
#     tokens["STRING"] = eps()
#     tokens["FSTRING_START"] = eps()
#     tokens["FSTRING_END"] = eps()
#     tokens["FSTRING_MIDDLE"] = rep(Regex.eat_any())
#     tokens["FSTRING_MIDDLE"] = eps()

    tokens["TYPE_COMMENT"] = seq([
        eat("#"),
        rep(eat_u8_negation(ord("\n"))),
        opt(eat_u8(ord("\n"))),
    ])
    tokens["ENDMARKER"] = eps()
    return [regex(expr, name) for name, expr in tokens.items()]
#     # TODO: delete this
#     return []
#     assert len(tokens) == len(set(tokens.keys()))
#     return [(name, ge.regex(expr)) for name, expr in tokens.items()]

def pegen_to_sep1_grammar(grammar: pegen.grammar.Grammar) -> CompiledGrammar: # Changed Grammar to CompiledGrammar
    memo = {}
    exprs: list[tuple[str, Any]] = []

    # Make sure the start production is first
    exprs.append(("start'''", ge.ref("file")))

#     # TODO: delete this
#     choice = Regex.choice
#     eat_u8 = Regex.eat_u8
#     eat_u8_negation = Regex.eat_u8_negation
#     seq = Regex.seq
#     rep = Regex.rep
#     eps = Regex.eps
#     exprs.append(("start'''", ge.regex(seq([eat_u8(ord("#")), rep(eat_u8_negation(ord("\n"))), eat_u8(ord("\n"))]))))
#     exprs.append(("start'''", ge.regex(seq([eat_u8(ord("#")), seq([eat_u8(ord(c)) for c in " This"]), eat_u8(ord("\n"))]))))
#     exprs.append(("start'''", ge.sequence([ge.ref("NAME"), ge.regex(eat_u8(ord("$")))])))
#     exprs.append(("start'''", ge.ref("IGNORE")))
#     exprs.append(("start'''", ge.ref("FSTRING_MIDDLE")))

#     exprs.append(("start'''", ge.sequence([regex(eat("def")), ge.ref("IGNORE"), ge.ref("NAME"), ge.regex(eat("(")), ge.ref("NAME"), ge.regex(eat(")"))])))
#     exprs.append(("start'''", regex(eat("def"))))
#     exprs.append(("start'''", ge.sequence([regex(eat("def")), ge.ref("IGNORE"), ge.ref("NAME"), ge.regex(eat("(")), ge.ref("NAME"), ge.regex(eat(")"))])))

#     # TODO: delete this
#     # Add a rule for "hello=world$" to the start rule
#     exprs.append(("start'''", ge.choice([ge.ref("file"), ge.sequence([ge.regex(eat("hello")), ge.regex(eat("=")), ge.regex(eat("world")), ge.regex(eat("$"))])])))
#     exprs.append(("start'''", ge.sequence([ge.regex(eat("hello")), ge.regex(eat("=")), ge.regex(eat("world")), ge.regex(eat("$"))])))
#     exprs.append(("file", ge.sequence([ge.regex(eat("fk")), ge.regex(eat("ing"))])))

#     # TODO: delete this
#     exprs.append(("start'''", ge.choice([ge.ref("file"), ge.regex(eat("hello"))])))
#     SOFT_KEYWORDS = ["a", "b", "c", "d", "e", "f", "g", "h"]
#     exprs.append(("file", ge.choice([ge.regex(eat(soft_keyword)) for soft_keyword in SOFT_KEYWORDS])))

#     # TODO: delete this
#     def eat_range(start: char, end: char) -> Regex:
#         return Regex.choice([Regex.eat_u8(c) for c in range(ord(start), ord(end) + 1)])
#     digit = eat_range('0', '9')
#     alph_lower = eat_range('a', 'z')
#     alph_upper = eat_range('A', 'Z')
# #     exprs.append(("NUM", ge.regex(Regex.rep(digit))))
#     exprs.append(("NUM", ge.regex(Regex.seq([digit, digit, digit]))))
#     exprs.append(("file", ge.sequence([ge.ref("NUM"), ge.regex(eat("+")), ge.ref("NUM"), ge.regex(eat("+")), ge.ref("NUM")])))

#     exprs.append(("file", ge.sequence([ge.optional(ge.ref("IGNORE")), ge.literal("def".encode())])))

#     exprs = [("start", ge.sequence([ge.regex(Regex.rep(Regex.eat_u8(ord(" ")))), ge.literal(b"f")]))]

    for rule in grammar.rules.values():
        memo[rule.name] = ge.ref(rule.name)
        if not rule_name_is_valid(rule.name):
            print(f"Ignoring invalid rule name: {rule.name}")
            rhs = ge.choice([])
        else:
            rhs = pegen_to_sep1_regex(rule.rhs, memo)
       # TODO: uncomment this
        exprs.append((rule.name, rhs))


    tokens = define_tokens()
    exprs.extend(tokens)

    return GrammarDefinition(exprs) # Changed Grammar to CompiledGrammar

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
#         print(f"Current input IDs: {current_input_ids}")
#         print(f"New token IDs: {new_token_ids}")
        current_full_string = "".join(self.llm_token_id_to_token[id].decode() for id in current_input_ids)
        print(f"Current full string: {current_full_string}")

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
        mask_tokens = [id_to_llm_token[id].decode() for id in mask_ids]
        mask_id_map = {id: self.llm_token_id_to_token.get(id) for id in mask_ids}
        print(f"Mask Token IDs: {textwrap.shorten(str(mask_ids), width=100)}")
        print(f"Mask Tokens: {mask_tokens[:1000]}")
        print(f"Mask Tokens (first chars): {"".join(sorted(list({m[0] for m in mask_tokens})))!r}")

        print("")

        scores = np.where(mask, scores, -np.inf)
        return torch.tensor(scores)

def generate_text(model, tokenizer, grammar_processor, pre_input_text, input_text, max_new_tokens=10):
    # TODO: We want pre_input_text to be input to the LLM that isn't passed into the grammar constraint.
    # .to(torch.int64) shouldn't be necessary here, but tokenizer.encode seems to return a torch.float32 tensor for 0-length inputs for some reason :')
    pre_input_ids = tokenizer.encode(pre_input_text, return_tensors="pt").to(torch.int64)
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(torch.int64)
    print(f"pre_input_ids: {pre_input_ids}, dtype: {pre_input_ids.dtype}")
    print(f"input_ids: {input_ids}, dtype: {input_ids.dtype}")
    full_input_ids = torch.cat([pre_input_ids, input_ids], dim=1)
    print(f"full_input_ids after cat: {full_input_ids}, dtype: {full_input_ids.dtype}")
    grammar_processor.seen_input_ids = pre_input_ids[0].tolist()
    output = model.generate(
        full_input_ids,
        max_new_tokens=max_new_tokens,
        logits_processor=[grammar_processor]
    )
    print(f"{len(output[0])} tokens generated")
    return tokenizer.decode(output[0], skip_special_tokens=True)

if __name__ == "__main__":
    from _sep1 import IncrementalParser # Import here after module is built
    model_name = "Qwen/Qwen2.5-Coder-0.5B"
#     model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenizer_vocab = tokenizer.get_vocab()
    # Ensure there are no spaces ' ' before we replace 'Ġ' with ' '
    assert not any(c == ' ' for c in tokenizer_vocab.keys())
    tokenizer_vocab = {k.replace("Ġ", " ").replace("ą", "\n"): v for k, v in tokenizer_vocab.items()}

#     # TODO: delete this
#     # Remove tokens that have any letter other than 'a'
#     tokenizer_vocab = {k: v for k, v in tokenizer_vocab.items() if not any(c.isalpha() and c != 'a' for c in k)}
#     # ...or have any number other than '1'
#     tokenizer_vocab = {k: v for k, v in tokenizer_vocab.items() if not any(c.isdigit() and c != '1' for c in k)}
#     # ...or have any letter or number
#     tokenizer_vocab = {k: v for k, v in tokenizer_vocab.items() if not any(c.isalpha() or c.isdigit() for c in k)}
#     # ...or have any character other than those in the given set
# #     allowed_chars = set("÷$(¾&§.-}][¢^·/?'¼\×´¨,¡¦*¸¥»±«¤¶>+~_°¯#;½¿=!£|:%)\"{<`©®@¬")
#     allowed_chars = set("_")
#     tokenizer_vocab = {k: v for k, v in tokenizer_vocab.items() if not any(c not in allowed_chars for c in k)}
#
#     # Set the vocab to just underscores
#     tokenizer_vocab = {"_": 0, "__": 1}
#
#     # Set the vocab to just "a" and "aa"
#     tokenizer_vocab = {"a": 0, "aa": 1}
#
#     # Set the vocab to "hello" "=" "world"
#     tokenizer_vocab = {"hello": 0, "=": 1, "world": 2}

    # Set the vocab to digits and arithmetic operators
#     tokenizer_vocab = {k: v for k, v in tokenizer.get_vocab().items() if k.isdigit() or k in "+-*/"}

#     tokenizer_vocab = {"def": 0, "$": 1}
#     tokenizer_vocab = {"def": 0}
#     tokenizer_vocab = {"def": 0, " f": 1, "(": 2, ")": 3}
#     tokenizer_vocab = {"def": 0, " f": 1, "(": 2, ")": 3, "de": 4}
#     tokenizer_vocab = {"def": 0, "de": 1}

#     # Exclude tokens that have more than _ hyphens
#     tokenizer_vocab = {k: v for k, v in tokenizer_vocab.items() if k.count("-") <= 1}

    # Exclude tokens of length more than 3
#     tokenizer_vocab = {k: v for k, v in tokenizer_vocab.items() if len(k) <= 2}
#     tokenizer_vocab = {k: v for k, v in tokenizer_vocab.items() if len(k) <= 10}
#     tokenizer_vocab = {k: v for k, v in tokenizer_vocab.items() if len(k) <= 80}
#     # Exclude tokens where any character appears more than once
#     tokenizer_vocab = {k: v for k, v in tokenizer_vocab.items() if len(set(k)) == len(k)}
    # Exclude tokens that have any letter other than 'a'
#     tokenizer_vocab = {k: v for k, v in tokenizer_vocab.items() if not any(c.isalpha() and c != 'a' for c in k)}
    # Exclude tokens that have any digit other than '1'
#     tokenizer_vocab = {k: v for k, v in tokenizer_vocab.items() if not any(c.isdigit() and c != '1' for c in k)}
    # Exclude any non-alphanumeric non-whitespace character
    # Allow only ...
#     tokenizer_vocab = {k: v for k, v in tokenizer_vocab.items() if all(c.isalnum() or c==' ' for c in k)}
#     tokenizer_vocab = {k: v for k, v in tokenizer_vocab.items() if all(c.isalpha() or c==' ' for c in k)}
#     tokenizer_vocab = {k: v for k, v in tokenizer_vocab.items() if all(c in 'a ' for c in k)}
#     tokenizer_vocab = {k: v for k, v in tokenizer_vocab.items() if all(c=='a' for c in k)}
#     tokenizer_vocab = {k: v for k, v in tokenizer_vocab.items() if all(c==' ' for c in k)}

    # Really just using these three rn (19 May 2025)
#     tokenizer_vocab = {k: v for k, v in tokenizer_vocab.items() if len(k) <= 2}
#     tokenizer_vocab = {k: v for k, v in tokenizer_vocab.items() if not any(c.isdigit() and c != '1' for c in k)}
#     tokenizer_vocab = {k: v for k, v in tokenizer_vocab.items() if all(c==' ' for c in k)}


    # Exclude tokens that have any character other than ...
#     tokenizer_vocab = {k: v for k, v in tokenizer_vocab.items() if not any(c not in 'a1# ' for c in k)}

#     tokenizer_vocab = {"def": 0, " f": 1}
#     tokenizer_vocab = {k: v for k, v in tokenizer_vocab.items() if len(k) <= 3}
#     tokenizer_vocab = {k: v for k, v in tokenizer_vocab.items() if len(k) == 1 or k in ["def", " f"]}
#     tokenizer_vocab = {k: v for k, v in tokenizer_vocab.items() if k in ["def", " f"] or k in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"}
#     tokenizer_vocab = {k: v for k, v in tokenizer_vocab.items() if len(k) <= 2 and all(c in " a" for c in k) or k in ["def", " f"]}
#     tokenizer_vocab = {k: v for k, v in tokenizer_vocab.items() if k in [" f", " "]}

    if len(tokenizer_vocab) <= 1000:
        print("Tokenizer vocab:")
    else:
        print("Tokenizer vocab (first 1000):")
    for token, id in list(sorted(tokenizer_vocab.items(), key=lambda x: x[0]))[:1000]:
        print(f"  {token!r}: {id}")

    # Map the remaining tokens to their proper IDs.
    actual_vocab = tokenizer.get_vocab()
    tokenizer_vocab = {token.replace(" ", "Ġ").replace("\n", "ą"): i for token, i in tokenizer_vocab.items()}
    tokenizer_vocab = {token.replace("Ġ", " ").replace("ą", "\n"): actual_vocab[token] for token in tokenizer_vocab}
    print(f"tokenizer_vocab: {textwrap.shorten(str(tokenizer_vocab), width=100)}")

    llm_token_to_id = {token.encode(): i for token, i in tokenizer_vocab.items()}
    id_to_llm_token = {i: token for token, i in llm_token_to_id.items()}
    llm_tokens = list(tokenizer_vocab.keys()) # Use all tokens

    print("vocab size:", len(llm_tokens))
    # Print all characters in the vocab
    chars = set()
    for token in llm_tokens:
        chars.update(set(token))
    print(f"All characters in the vocab: {"".join(chars)}")

#     ts = ['Paris', 'London']
#     llm_tokens = [x.encode() for x in ts]
#     llm_token_to_id = {token.encode(): llm_token_to_id[token] for token in ts}

    print("Defining grammar...")
    grammar_definition = define_python_grammar()

#     grammar_definition.simplify()
    grammar = grammar_definition.compile()
    grammar.print()

    # Serialize the compiled grammar to JSON string
    print("Serializing compiled grammar to JSON...")
    json_string = grammar.to_json_string()
    print(f"Serialized CompiledGrammar JSON (length: {len(json_string)}):")
    # Indent it.
    json_string = json.dumps(json.loads(json_string), indent=4)
    grammar_path = Path(__file__).parent.parent / "src" / "serialized_compiled_grammar.json"
    with open("serialized_compiled_grammar.json", "w") as f:
        f.write(json_string)

#     # TODO: delete this
#     # Define a dummy grammar that only accepts "hello=world"
#     exprs = [("S", ge.sequence([ge.regex(eat("hello")), ge.regex(eat("=")), ge.regex(eat("world")), ge.regex(eat("$"))]))]
#     grammar = CompiledGrammar(exprs) # Changed Grammar to CompiledGrammar
#     grammar.print()

#     print("Initializing parser...")
#     parser = grammar.glr_parser()
#     parser.print()
    # Accessing glr_parser directly from CompiledGrammar might not be exposed.
    # If you need to print it, you'd typically do it via the Rust `Debug` impl,
    # which `grammar.print()` already does for the whole `CompiledGrammar`.
    # If a separate `GLRParser` object with its own print was intended,
    # `CompiledGrammar` would need a method to return it.
    # For now, `grammar.print()` includes GLR parser info.
    # grammar.glr_parser().print() # This line would error if glr_parser() doesn't return a printable GLRParser


#     pre_input_text = ""
#     input_text = "i^10=i*"
#     input_text = "5*6 + 7*2 = 5+5+5+"
#     input_text = "123+456+"
#     expected_next_token = "789"

#     pre_input_text = ""
#     input_text = "hello="
#     expected_next_token = "world"

    pre_input_text = ""
#     input_text = 'from typing import Any, List, Tuple, Union'
    input_text = 'def'
#     input_text = 'NAME'
    expected_next_token = ""

    if expected_next_token:
        expected_next_token = id_to_llm_token[grammar_constraint_state.llm_token_map[expected_next_token]]

    # DEMO: Incremental Parser
    parser_state = IncrementalParser(grammar) # Use the imported class
    print(f"Initial valid: {parser_state.is_valid()}")
    assert parser_state.is_valid()
    parser_state.feed(input_text.encode("utf-8"))
    print(f"After '{input_text}': valid={parser_state.is_valid()}")
    assert parser_state.is_valid()
    print("--- End Incremental Parser Demo ---")


    print("Initializing grammar constraint...")
    grammar_constraint = GrammarConstraint(grammar, llm_token_to_id, max(llm_token_to_id.values()))

#     # Serialize grammar constraint to JSON string
#     print("Serializing grammar constraint to JSON...")
#     json_string = grammar_constraint.to_json_string()
#     print(f"Serialized GrammarConstraint JSON (length: {len(json_string)}):")
#     # Indent it.
#     json_string = json.dumps(json.loads(json_string), indent=4)
#     # Optionally print a snippet or save to file if too long
#     # print(textwrap.shorten(json_string, width=200, placeholder="..."))
#     with open("serialized_grammar_constraint.json", "w") as f:
#         f.write(json_string)
#
#     # Deserialize from JSON string
#     print("Deserializing grammar constraint from JSON...")
#     grammar_constraint_from_json = GrammarConstraint.from_json_string(json_string)
#     print("Grammar constraint deserialized successfully.")
#
#     # Use the deserialized constraint for subsequent operations
#     grammar_constraint_to_use = grammar_constraint_from_json
#     # To test with the original, uncomment the line below and comment out the line above
    grammar_constraint_to_use = grammar_constraint

    print("Initializing grammar constraint state...")
    grammar_constraint_state = GrammarConstraintState(grammar_constraint_to_use)
    print("Initializing grammar processor...")
    grammar_processor = GrammarConstrainedLogitsProcessor(grammar_constraint_state, llm_token_to_id)

    model = AutoModelForCausalLM.from_pretrained(model_name)

    print("Generating text...")

    # DEMO: Get the mask
    grammar_constraint_state = GrammarConstraintState(grammar_constraint_to_use)

    tokens = tokenizer.encode(input_text, return_tensors="pt")
    tokens: list[int] = tokens.tolist()[0]
    print(f"Committing tokens: {tokens}")
    for i, token_id in enumerate(tokens):
        print(f"Ensuring token {id_to_llm_token[token_id].decode()!r} (id: {token_id}) is in mask")
        mask = grammar_constraint_state.get_mask()
        print("Got mask")
        mask_ids = np.where(mask)[0].tolist()
        mask_tokens = [id_to_llm_token[id].decode() for id in mask_ids]
        print(f"Mask Token IDs: {textwrap.shorten(str(mask_ids), width=100)}")
        print(f"Mask Tokens: {textwrap.shorten(str(mask_tokens), width=300)}")
        print(f"Mask Tokens (first chars): {"".join(sorted(list({m[0] for m in mask_tokens})))!r}")
        assert token_id in mask_ids, f"Expected token {id_to_llm_token[token_id].decode()!r} (id: {token_id}) in mask"
        print(f"--- Committing token {id_to_llm_token[token_id].decode()!r} (id: {token_id}) ---")
        grammar_constraint_state.commit(token_id)
    print("--- End Committing Tokens ---")

    print("Getting final mask")
    mask = grammar_constraint_state.get_mask()
    print("Got mask")
    print(mask)
    mask_ids = np.where(mask)[0].tolist()
    mask_tokens = [id_to_llm_token[id].decode() for id in mask_ids]
    print(f"Mask Token IDs: {textwrap.shorten(str(mask_ids), width=100)}")
    print(f"Mask Tokens: {textwrap.shorten(str(mask_tokens), width=300)}")
    print(f"Mask Tokens (first chars): {"".join(sorted(list({m[0] for m in mask_tokens})))!r}")

    if expected_next_token:
        assert expected_next_token in mask_tokens, f"Expected '{expected_next_token}' in mask"

    # DEMO: Generate text.
    grammar_constraint_state = GrammarConstraintState(grammar_constraint_to_use)
    # The line below this one already uses grammar_constraint_state, so it's fine:
    # output_text = timeit(generate_text)(model, tokenizer, grammar_processor, pre_input_text, input_text)
    # However, grammar_processor was initialized with the *original* grammar_constraint_state.
    # For a full test of the deserialized object, grammar_processor should also be re-initialized
    # if you want the generation itself to use the deserialized constraint.
    # Let's re-initialize grammar_processor here for consistency:
    print("Re-initializing grammar processor with deserialized constraint state...")
    grammar_processor = GrammarConstrainedLogitsProcessor(grammar_constraint_state, llm_token_to_id)

    output_text = timeit(generate_text)(model, tokenizer, grammar_processor, pre_input_text, input_text, max_new_tokens=100)
    print(output_text)