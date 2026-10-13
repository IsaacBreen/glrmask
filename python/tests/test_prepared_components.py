"""Typed, immutable source/prepared components and Lark compatibility profiles."""
import gc

import numpy as np
import pytest

import glrmask

MODES = [glrmask.Optimization.AUTO, glrmask.Optimization.FAST_BUILD,
         glrmask.Optimization.BALANCED, glrmask.Optimization.FAST_RUNTIME]
LEAF = 'extern token MARK start body nt body="x" MARK'
PAIR = 'extern grammar LEFT extern grammar RIGHT start root nt root=LEFT "|" RIGHT'


def vocab():
    return glrmask.Vocab.from_id_to_bytes({0: b"x", 1: b"|", 2: b"x", 4: b"", 5: b""})


def accepts(constraint, ids):
    state = constraint.start()
    for token in ids:
        mask = state.mask()
        if token >= len(mask) or not mask[token]:
            return False
        state.commit_token(token)
    return state.is_accepting()


@pytest.mark.parametrize("mode", MODES)
def test_owned_source_and_independent_open_prepared_children(mode):
    v = vocab()
    child = glrmask.Grammar.from_glrm("".join([LEAF]))
    open_child = child.prepare(v)
    parent = glrmask.Grammar.from_glrm(PAIR).prepare(v)
    package = parent.bind("LEFT", open_child).bind("RIGHT", open_child)
    assert isinstance(package, glrmask.PreparedGrammar)
    assert [(r.path, r.kind) for r in package.requirements] == [
        (["LEFT", "MARK"], "token"), (["RIGHT", "MARK"], "token")]
    assert [r.name for r in open_child.requirements] == ["MARK"]
    loaded = glrmask.PreparedGrammar.load(package.save(), vocab=v)
    complete = loaded.bind("LEFT::MARK", v.tokens([4])).bind("RIGHT::MARK", v.tokens([5]))
    del child, parent, open_child
    gc.collect()
    compiled = complete.link(optimization=mode)
    assert accepts(compiled, [0, 4, 1, 2, 5])
    assert not accepts(compiled, [0, 5, 1, 2, 4])
    assert len(package.requirements) == 2
    with pytest.raises(ValueError):
        complete.bind("LEFT::MARK", v.tokens([5]))


def test_typed_bindings_and_explicit_event_adapter():
    v = vocab()
    leaf = glrmask.Grammar.from_glrm(LEAF)
    pair = glrmask.Grammar.from_glrm(PAIR)
    with pytest.raises(ValueError):
        leaf.bind("MARK", glrmask.Grammar.from_glrm('start x nt x="x"'))
    with pytest.raises(ValueError):
        pair.bind("LEFT", v.tokens([4]))
    event = glrmask.Grammar.exact(v.tokens([2, 4]))
    g = glrmask.Grammar.from_glrm('extern grammar EVENT start root nt root=EVENT').bind("EVENT", event)
    mask = g.compile(v).start().mask()
    assert not mask[0] and mask[2] and mask[4]
    source = pair.bind("LEFT", leaf).bind("RIGHT", leaf)
    complete = source.bind("LEFT::MARK", v.tokens([4])).bind("RIGHT::MARK", v.tokens([5]))
    assert len(source.requirements) == 2 and complete.requirements == []


@pytest.mark.parametrize("mode", MODES)
def test_lark_llm_numeric_names_and_prepared_bindings(mode):
    v = vocab()
    with pytest.raises(ValueError):
        glrmask.Grammar.from_lark('start: <[2]>').compile(v)
    g = glrmask.Grammar.from_lark('start: <|begin|> @payload <[^0-2]>', profile="llm", token_names={"begin": 2})
    assert [r.name for r in g.requirements] == ["payload"]
    child = glrmask.Grammar.from_glrm('start root nt root="|"')
    compiled = g.prepare(v).bind("payload", child.prepare(v)).link(optimization=mode)
    assert accepts(compiled, [2, 1, 4]) and accepts(compiled, [2, 1, 5])
    assert not accepts(compiled, [0, 1, 4])
    with pytest.raises(ValueError):
        glrmask.Grammar.from_lark('start: <|missing|>', profile="llm").compile(v)


@pytest.mark.parametrize("mode", MODES)
def test_inline_json_and_host_bound_json_are_mask_equivalent(mode):
    v = glrmask.Vocab.from_id_to_bytes({0: b'<"a b">', 1: b'<"ab">', 2: b"<", 3: b'"a b"', 4: b">"})
    schema = {"type": "string", "const": "a b"}
    inline = glrmask.Grammar.from_lark('start: "<" %json {"type":"string","const":"a b"} ">"', profile="llm")
    bound = glrmask.Grammar.from_lark('start: "<" @json ">"', profile="llm").bind("json", glrmask.Grammar.from_json_schema(schema))
    left = inline.compile(v, optimization=mode)
    right = bound.compile(v, optimization=mode)
    assert np.array_equal(left.start().mask(), right.start().mask())
    assert accepts(left, [0]) and accepts(left, [2, 3, 4]) and not accepts(left, [1])


def test_original_requirement_source_span():
    source = "// α\nextern token MARK\nstart root\nnt root=MARK"
    v = vocab()
    g = glrmask.Grammar.from_glrm(source)
    req = g.requirements[0]
    start, end, line, column = req.source_span
    assert source.encode("utf-8")[start:end] == b"MARK"
    assert (line, column) == (2, 14)
    saved = glrmask.PreparedGrammar.load(g.prepare(v).save(), vocab=v)
    assert saved.requirements[0].source_span == req.source_span
    lark = glrmask.Grammar.from_lark("start: @payload", profile="llm")
    assert lark.requirements[0].source_span == (7, 15, 1, 8)
