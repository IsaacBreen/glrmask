"""Installed-extension tests of the end-to-end configured module contract."""
import copy

import pytest

import glrmask


def vocab():
    return glrmask.Vocab.from_id_to_bytes(
        {0: b"a", 1: b"b", 2: b"ab", 3: b",", 4: b"a,b", 5: b""}
    )


def source(text):
    return glrmask.Grammar.from_ebnf(f'start ::= "{text}"')


def test_open_precompiled_parent_late_children_and_clone_independence():
    v = vocab()
    parent = glrmask.Module(glrmask.Grammar.from_glrm(
        "extern grammar LEFT extern grammar RIGHT start s nt s=LEFT RIGHT"
    ))
    assert parent.precompile(v) is None
    persisted = parent.save()
    first = glrmask.Module.load(persisted, vocab=v)
    second = copy.copy(first)
    assert first.bind("LEFT", source("a")) is None
    assert first.bind("RIGHT", source("b")) is None
    second.bind("LEFT", source("b"))
    second.bind("RIGHT", source("a"))
    assert [x.name for x in parent.unbound] == ["LEFT", "RIGHT"]
    assert parent.save() == persisted
    assert first.compile(v).start().mask().tolist() != second.compile(v).start().mask().tolist()
    state = first.compile(v).start()
    assert state.mask()[2]
    state.commit_token(2)
    assert state.is_accepting()
    assert not state.is_rejected()


@pytest.mark.parametrize("optimization", [
    glrmask.Optimization.AUTO,
    glrmask.Optimization.BALANCED,
    glrmask.Optimization.FAST_RUNTIME,
])
def test_python_keywords_configure_only_module_and_do_not_discard_work(optimization):
    v = vocab()
    g = source("a")
    m = glrmask.Module(g, optimization=optimization)
    m.precompile(v)
    original = m.save()
    # Identical settings are accepted as a no-op after work is retained.
    m.set_options(optimization=optimization)
    assert m.save() == original
    other = (glrmask.Optimization.FAST_RUNTIME
             if optimization == glrmask.Optimization.BALANCED
             else glrmask.Optimization.BALANCED)
    with pytest.raises(ValueError):
        m.set_options(optimization=other)
    assert m.save() == original
    assert g.compile(v).start().mask().tolist() == m.compile(v).start().mask().tolist()


def test_expr_source_and_configured_leaves_use_distinct_paths():
    v = vocab()
    open_source = glrmask.Grammar.from_glrm("extern grammar VALUE start s nt s=VALUE")
    expr = glrmask.Expr.sequence([
        glrmask.Expr.sequence([open_source]), source(","), open_source,
    ])
    m = glrmask.Module(expr)
    assert [x.name for x in m.unbound] == ["leaf0::VALUE", "leaf2::VALUE"]
    before = m.save()
    with pytest.raises(ValueError):
        m.bind("leaf1::VALUE", source("a"))
    assert m.save() == before
    m.bind("leaf0::VALUE", source("a"))
    m.bind("leaf2::VALUE", source("b"))
    state = m.compile(v).start()
    assert state.mask()[4]
    state.commit_token(4)
    assert state.is_accepting()

    child = glrmask.Module(source("a"), optimization=glrmask.Optimization.BALANCED)
    child.precompile(v)
    configured = glrmask.Expr.choice([child, source("b")])
    with pytest.raises(ValueError):
        glrmask.Grammar.from_expr(configured)
    restored = glrmask.Module.load(glrmask.Module(configured).save(), vocab=v)
    mask = restored.compile(v).start().mask()
    assert mask[0] and mask[1]


def test_one_event_widens_to_grammar_slot_but_not_the_reverse():
    v = vocab()
    grammar_slot = glrmask.Module(glrmask.Grammar.from_glrm(
        "extern grammar EVENT start s nt s=EVENT"
    ))
    grammar_slot.bind("EVENT", v.tokens([5]))
    state = glrmask.Module.load(grammar_slot.save(), vocab=v).compile(v).start()
    assert state.mask()[5]
    state.commit_token(5)
    assert state.is_accepting()
    token_slot = glrmask.Module(glrmask.Grammar.from_glrm(
        "extern token EVENT start s nt s=EVENT"
    ))
    with pytest.raises(ValueError):
        token_slot.bind("EVENT", source("a"))
    assert [x.name for x in token_slot.unbound] == ["EVENT"]


def test_invalid_target_and_artifact_are_not_accepted_silently():
    v = vocab()
    m = glrmask.Module(source("a"))
    m.precompile(v)
    data = m.save()
    different = glrmask.Vocab.from_id_to_bytes({0: b"b", 1: b"a"})
    with pytest.raises(ValueError):
        glrmask.Module.load(data, vocab=different)
    with pytest.raises(ValueError):
        m.compile(different)
    with pytest.raises(ValueError):
        glrmask.Module.load(b"not a module")
    assert m.save() == data
