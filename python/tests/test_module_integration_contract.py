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


def test_open_precompiled_parent_late_children_and_copy_independence():
    v = vocab()
    parent = glrmask.Module(glrmask.Grammar.from_glrm(
        "glrm 1; start s; extern grammar LEFT; extern grammar RIGHT; nt s=LEFT RIGHT;"
    ))
    assert parent.precompile(v) is None
    persisted = parent.save()
    first = glrmask.Module.load(persisted, vocab=v)
    second = copy.copy(first)
    assert first.bind("LEFT", source("a")) is None
    assert first.bind("RIGHT", source("b")) is None
    second.bind("LEFT", source("b"))
    second.bind("RIGHT", source("a"))
    assert list(parent.externs.unbound) == ["LEFT", "RIGHT"]
    assert parent.save() == persisted
    assert first.compile(v).start().mask().tolist() != second.compile(v).start().mask().tolist()
    state = first.compile(v).start()
    assert state.mask()[2]
    state.commit_token(2)
    assert state.is_accepting()
    assert not state.is_rejected()


@pytest.mark.parametrize("mask_strategy", [
    glrmask.MaskStrategy.AUTO,
    glrmask.MaskStrategy.TOKEN_CLASS_TRIE,
    glrmask.MaskStrategy.PARSER_DWA,
])
def test_python_compile_options_do_not_discard_work(mask_strategy):
    v = vocab()
    g = source("a")
    m = glrmask.Module(g)
    m.precompile(v, mask_strategy=mask_strategy)
    assert g.compile(v).start().mask().tolist() == m.compile(v, mask_strategy=mask_strategy).start().mask().tolist()
    assert g.compile(v).start().mask().tolist() == m.compile(v).start().mask().tolist()


def test_one_event_widens_to_grammar_slot_but_not_the_reverse():
    v = vocab()
    grammar_slot = glrmask.Module(glrmask.Grammar.from_glrm(
        "glrm 1; start s; extern grammar EVENT; nt s=EVENT;"
    ))
    grammar_slot.bind("EVENT", glrmask.TokenSet([5]))
    state = glrmask.Module.load(grammar_slot.save(), vocab=v).compile(v).start()
    assert state.mask()[5]
    state.commit_token(5)
    assert state.is_accepting()
    token_slot = glrmask.Module(glrmask.Grammar.from_glrm(
        "glrm 1; start s; extern token EVENT; nt s=EVENT;"
    ))
    with pytest.raises(ValueError):
        token_slot.bind("EVENT", source("a"))
    assert list(token_slot.externs.unbound) == ["EVENT"]


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
