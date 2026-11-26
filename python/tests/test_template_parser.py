"""Internal Python table-free tooling: masks are checked against literal languages."""


def _bound_snapshot(source, name, value):
    """Preserve fixture snapshots while exercising public single-item mutation."""
    inner_val = value.grammar if isinstance(value, GrammarCalls) else value
    if isinstance(source, GrammarCalls):
        source = source.grammar
    m = glrmask.Module(source) if isinstance(source, glrmask.Grammar) else source.copy()
    assert m.bind(name, inner_val) is None
    return GrammarCalls(m)

import copy
import gc
import json

import numpy as np
import pytest

import glrmask

from mask_helpers import token_allowed, unpack_mask

class GrammarCalls:
    """Keep public default paths and explicit internal controls separate."""
    def __init__(self, grammar): self.grammar = grammar
    def compile(self, vocab, **options): return compile_grammar(self.grammar, vocab, **options)
    def prepare(self, vocab):
        m = self.grammar if isinstance(self.grammar, glrmask.Module) else glrmask.Module(self.grammar)
        m.precompile(vocab)
        return GrammarCalls(m)
    def save(self): return self.grammar.save()

def compile_grammar(grammar, vocab, **options):
    if isinstance(grammar, GrammarCalls): grammar = grammar.grammar
    if "optimization" in options:
        options["mask_strategy"] = options.pop("optimization")
    if "parser_backend" in options or "merge_equivalent_tokens" in options:
        raise TypeError("Python does not expose engine tuning")
    return grammar.compile(vocab, **options)

def link_module(module, vocab, **options):
    if isinstance(module, GrammarCalls): module = module.grammar
    if "optimization" in options:
        options["mask_strategy"] = options.pop("optimization")
    end = options.pop("end_tokens", None)
    if end is not None:
        root = glrmask.Module(glrmask.Grammar.from_glrm("start s extern grammar BODY extern token END nt s = BODY END"))
        root.bind(BODY=module, END=glrmask.TokenSet(end))
        module = root
    return module.compile(vocab, **options)

def grammar_from_lark(source): return GrammarCalls(glrmask.Grammar.from_lark(source))
def grammar_from_glrm(source): return GrammarCalls(glrmask.Grammar.from_glrm(source))
def grammar_from_json_schema(source): return GrammarCalls(glrmask.Grammar.from_json_schema(source))



def symbol(value):
    return {"Symbol": value}


def graph(accepting=False):
    return {"start": 0, "states": [{"accepting": accepting, "transitions": []}]}


def template():
    return {"pop": graph(), "read": graph(), "push": graph(),
            "pop_to_read": [], "pop_to_push": [], "read_to_push": []}


def identity():
    value = template()
    value["pop"]["states"][0]["accepting"] = True
    return value


def rewrite_top(mapping):
    """Literal relation: pop one top symbol and append its mapped output."""
    value = template()
    value["pop_to_push"] = [None]
    for label, pushes in mapping.items():
        pop_id = len(value["pop"]["states"])
        value["pop"]["states"][0]["transitions"].append(
            {"label": symbol(label), "target": pop_id})
        value["pop"]["states"].append({"accepting": not pushes, "transitions": []})
        if not pushes:
            value["pop_to_push"].append(None)
            continue
        start = len(value["push"]["states"])
        value["pop_to_push"].append(start)
        for offset, pushed in enumerate(pushes):
            value["push"]["states"].append({"accepting": False, "transitions": [
                {"label": symbol(pushed), "target": start + offset + 1}]})
        value["push"]["states"].append({"accepting": True, "transitions": []})
    return value


def balanced_parentheses():
    return {"stack_symbol_count": 2,
            "terminals": [rewrite_top({0: [0, 1], 1: [1, 1]}), rewrite_top({1: []}), identity()],
            "completion": rewrite_top({0: []})}


def words_and_vocab():
    words = [b"(", b")", b"()", b"((", b"))", b"(()", b"())", b"()()", b" ", b"( )", b"a", b"()"]
    return words, glrmask.Vocab(dict(enumerate(words)))


def depth_after(word, depth=0, ignore_whitespace=True):
    for byte in word:
        if byte == ord("("):
            depth += 1
        elif byte == ord(")") and depth:
            depth -= 1
        elif byte != ord(" ") or not ignore_whitespace:
            return None
    return depth


MODES = [
    {},
    {"mask_strategy": glrmask.MaskStrategy.FAST_MASKS},
    {"mask_strategy": glrmask.MaskStrategy.FAST_COMPILE},
]


def representations(constraint, vocab):
    raw = constraint.save()
    loaded = glrmask.Constraint.load(raw, vocab)
    assert loaded.save() == raw
    external = constraint.save()
    return [constraint, loaded, glrmask.Constraint.load(external, vocab)]


@pytest.mark.parametrize("mode", MODES)
def test_python_builtin_template_backend_is_default_and_survives_reload(mode):
    words, vocab = words_and_vocab()
    grammar = grammar_from_lark('start: "(" start ")" start | ""')
    reference = compile_grammar(grammar, vocab, **mode)
    candidate = compile_grammar(
        grammar,
        vocab,
        **mode,
    )
    reference_backend = glrmask._internal.parser_backend(reference)
    reference_bytes = reference.save()
    for compiled in representations(reference, vocab) + representations(candidate, vocab):
        assert glrmask._internal.parser_backend(compiled) == reference_backend
        for prefix in [b"", b"(", b"()", b"(()", b"((()))", b"()("]:
            left, right = reference.start(), compiled.start()
            assert left.commit_bytes(prefix)
            assert right.commit_bytes(prefix)
            assert np.array_equal(left.mask(), right.mask())
            assert left.is_accepting() == right.is_accepting()
            depth = depth_after(prefix, ignore_whitespace=False)
            # Source nullability must survive compilation and both artifact forms.
            # The empty balanced prefix accepts; incomplete prefixes do not.
            assert right.is_accepting() == (depth == 0)
            for token_id, word in enumerate(words):
                expected = depth_after(word, depth, ignore_whitespace=False)
                assert bool(token_allowed(right.mask(), token_id)) == (expected is not None), (prefix, word)
                if expected is not None:
                    branch = compiled.start()
                    assert branch.commit_bytes(prefix)
                    assert branch.commit_token(token_id)
                    assert branch.is_accepting() == (expected == 0)
    assert reference.save() == reference_bytes


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("as_json", [False, True])
def test_python_data_only_program_matches_literal_language(mode, as_json):
    definition = balanced_parentheses()
    program = glrmask._internal.ParserProgram(json.dumps(definition) if as_json else definition)
    assert program.terminal_count == 3
    # The implementation owns the validated program, not the mutable mapping.
    definition["terminals"].clear()
    words, vocab = words_and_vocab()
    compiled = program.compile(vocab, [b"(", b")", "[ ]+"], ignore_terminal=2, mask_strategy=mode.get("mask_strategy", glrmask.MaskStrategy.AUTO), end_tokens=[64])
    del program, definition
    gc.collect()
    for compiled in representations(compiled, vocab):
        assert glrmask._internal.parser_backend(compiled) == glrmask._internal.ParserBackend.TEMPLATE_DFA
        prefixes = [b""]
        while prefixes:
            prefix = prefixes.pop()
            depth = depth_after(prefix)
            state = compiled.start()
            assert state.commit_bytes(prefix)
            assert not state.is_accepting()
            mask = unpack_mask(state.mask())
            for token_id, word in enumerate(words):
                assert bool(mask[token_id]) == (depth_after(word, depth) is not None), (prefix, word)
            assert bool(mask[64]) == (depth == 0)
            if len(prefix) < 6:
                for byte in [b"(", b")"]:
                    if depth_after(byte, depth) is not None:
                        prefixes.append(prefix + byte)


def test_python_template_validation_is_early_and_does_not_retain_python_callbacks():
    definition = balanced_parentheses()
    cyclic = copy.deepcopy(definition)
    cyclic["terminals"][0]["pop"]["states"][0]["transitions"][0]["target"] = 0
    with pytest.raises(ValueError, match="(?i)cycl"):
        glrmask._internal.ParserProgram(cyclic)
    with pytest.raises(ValueError):
        glrmask._internal.ParserProgram("not valid JSON")
    with pytest.raises(TypeError):
        glrmask._internal.ParserProgram(lambda: definition)
    program = glrmask._internal.ParserProgram(definition)
    _, vocab = words_and_vocab()
    for bad in [[b"("], [b"(", b")", "[ ]+", b"extra"]]:
        with pytest.raises(ValueError):
            program.compile(vocab, bad)
    with pytest.raises(TypeError):
        program.compile(vocab, [object(), b")", "[ ]+"])
    with pytest.raises(ValueError):
        program.compile(vocab, [b"(", b")", "[ ]+"], ignore_terminal=0)
    with pytest.raises(ValueError):
        program.compile(vocab, [b"", b")", "[ ]+"])
    with pytest.raises(ValueError):
        program.compile(vocab, ["(", b")", "[ ]+"])


@pytest.mark.parametrize("mode", [{"mask_strategy": glrmask.MaskStrategy.AUTO}] + MODES)
def test_python_prepared_composition_uses_native_defaults(mode):
    tokens = {0: b"a", 1: b"aa", 2: b"b"}
    vocab = glrmask.Vocab(tokens)
    body = grammar_from_lark('start: "a"')
    # Standalone representations and composed representations are both native;
    # executable roots are not implicitly converted into reusable components.
    for root in [body.compile(vocab, **mode), body.compile(vocab, mask_strategy=glrmask.MaskStrategy.FAST_MASKS)]:
        for restored in representations(root, vocab):
            assert restored.start().mask().tolist() == [1]
    prepared = body.prepare(vocab)
    parent = grammar_from_glrm('extern grammar C start root nt root = C').prepare(vocab)
    for source in [prepared, glrmask.Module.load(prepared.save()),
                   glrmask.Module.load(prepared.save())]:
        child_bytes = source.save()
        for selection in [{}, {}]:
            linked = link_module(_bound_snapshot(parent, "C", source), vocab, **mode)
            assert source.save() == child_bytes
            for compiled in representations(linked, vocab):
                assert glrmask._internal.parser_backend(compiled) == glrmask._internal.ParserBackend.TEMPLATE_DFA
                for prefix in [b"", b"a"]:
                    state = compiled.start()
                    assert state.commit_bytes(prefix)
                    assert state.is_accepting() == (prefix == b"a")
                    for token_id, word in tokens.items():
                        expected = b"a".startswith(prefix + word)
                        assert bool(token_allowed(state.mask(), token_id)) == expected, (prefix, word)
                        if expected:
                            branch = compiled.start()
                            assert branch.commit_bytes(prefix)
                            assert branch.commit_token(token_id)
                            assert branch.is_accepting() == (prefix + word == b"a")


@pytest.mark.parametrize("mode", MODES)
def test_python_precompiled_nullable_composition_preserves_crossings_and_root_end(mode):
    vocab = glrmask.Vocab({0: b"x", 1: b"a", 2: b"y", 3: b"xay",
                                          4: b"xy", 5: b"xx", 6: b""})
    child = grammar_from_lark('start: "a"?').prepare(vocab)
    parent = grammar_from_glrm(
        'glrm 1; start root; extern grammar C; nt root = "x" C "y";').prepare(vocab)
    linked = link_module(_bound_snapshot(parent, "C", child), vocab, **mode,
        end_tokens=[6])
    for compiled in representations(linked, vocab):
        state = compiled.start()
        assert np.array_equal(state.mask(), np.array([25], dtype=np.uint32))
        for token in [3, 4]:
            state = compiled.start()
            assert state.commit_token(token)
            assert not state.is_accepting()
            assert token_allowed(state.mask(), 6)
            assert state.commit_token(6)
            assert state.is_accepting() and not state.mask().any()
            assert not state.mask().any()


def test_python_backend_selection_does_not_accept_untyped_flags():
    vocab = glrmask.Vocab({0: b"a"})
    grammar = grammar_from_lark('start: "a"')
    for value in ["TEMPLATE_DFA", True, 1, object()]:
        with pytest.raises(TypeError):
            compile_grammar(grammar, vocab, parser_backend=value)


def test_python_explicit_lr_merged_request_is_not_exposed():
    vocab = glrmask.Vocab({0: b"a"})
    grammar = glrmask.Grammar.from_lark('start: "a"')
    with pytest.raises(TypeError):
        grammar.compile(vocab, parser_backend="lr-table", merge_equivalent_tokens=True)


@pytest.mark.parametrize("mode", MODES)
def test_python_projected_virtual_child_preserves_exact_limits_and_reload(mode):
    tokens = {0: b"p", 1: b'"x:a"', 2: b"q", 3: b'p"x:a"q',
              4: b'a"q', 5: b"a", 6: b"aaa", 7: b'"q', 8: b""}
    vocab = glrmask.Vocab(tokens)
    child = grammar_from_json_schema(json.dumps({
        "type": "string", "format": "uri", "minLength": 1, "maxLength": 5000,
    })).prepare(vocab)
    child = glrmask.Module.load(child.save())
    parent = grammar_from_glrm(
        'glrm 1; start root; extern grammar C; nt root = "p" C "q";').prepare(vocab)
    compiled = link_module(_bound_snapshot(parent, "C", child), vocab, **mode,
        end_tokens=[8])
    for compiled in representations(compiled, vocab):
        assert glrmask._internal.parser_backend(compiled) == glrmask._internal.ParserBackend.TEMPLATE_DFA
        state = compiled.start()
        assert token_allowed(state.mask(), 3)
        assert state.commit_token(3)
        assert not state.is_accepting() and token_allowed(state.mask(), 8)
        assert state.commit_token(8)
        assert state.is_accepting() and not state.mask().any()
        assert not state.mask().any()

        for count in [0, 1, 31, 4996, 4997, 4998]:
            state = compiled.start()
            assert state.commit_bytes(b'p"x:' + b"a" * count)
            mask = unpack_mask(state.mask())
            for token, payload_size in [(5, 1), (6, 3)]:
                assert bool(mask[token]) == (2 + count + payload_size <= 5000), (count, token)
            assert mask[7]  # Completing the URI and returning to the parent.
            assert state.commit_token(7)
            assert not state.is_accepting()
        state = compiled.start()
        assert not state.commit_bytes(b'p"x:' + b"a" * 4999)
@pytest.mark.parametrize("mode", MODES)
def test_python_nullable_lexical_body_survives_compiled_child_binding(mode):
    tokens = {0: b"x", 1: b"a", 2: b"y", 3: b"xay", 4: b"xy",
              5: b"ay", 6: b"aa", 7: b"yx", 8: b""}
    vocab = glrmask.Vocab(tokens)
    prepared = grammar_from_glrm(
        'start root t A = /a?/ nt root = A').prepare(vocab)
    parent = grammar_from_glrm(
        'glrm 1; start root; extern grammar C; nt root = "x" C "y";').prepare(vocab)
    language = [b"xy", b"xay"]
    for child in [prepared, glrmask.Module.load(prepared.save()),
                  glrmask.Module.load(prepared.save())]:
        child_bytes = child.save()
        linked = link_module(_bound_snapshot(parent, "C", child), vocab,
            **mode, end_tokens=[8])
        assert child.save() == child_bytes
        for compiled in representations(linked, vocab):
            assert glrmask._internal.parser_backend(compiled) == glrmask._internal.ParserBackend.TEMPLATE_DFA
            for prefix in [b"", b"x", b"xa", b"xy", b"xay"]:
                state = compiled.start()
                assert state.commit_bytes(prefix)
                mask = unpack_mask(state.mask())
                assert not state.is_accepting()
                assert bool(mask[8]) == (prefix in language)
                for token_id, word in tokens.items():
                    if token_id == 8:
                        continue
                    expected = any(complete.startswith(prefix + word) for complete in language)
                    assert bool(mask[token_id]) == expected, (mode, prefix, word)
                    if expected:
                        branch = compiled.start()
                        assert branch.commit_bytes(prefix)
                        assert branch.commit_token(token_id)
                        assert not branch.is_accepting()
                if prefix in language:
                    assert state.commit_token(8)
                    assert state.is_accepting() and not state.mask().any()
                    assert not state.mask().any()
