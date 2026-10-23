"""Target public API contract tests for v1 API.

No live tokenizer/model, network, or large vocabulary is required.
"""

import gc
import numpy as np
import pytest
import glrmask


TOKENS = {0: b"x", 1: b"a", 2: b"b", 3: b"y", 4: b"xay", 7: b"a", 63: b"", 127: b""}
HOST = 'glrm 1; start start; extern grammar child; nt start = "x" child "y";'
MARK = 'glrm 1; start start; extern token MARK; nt start = MARK;'


def vocab(entries=None):
    return glrmask.Vocab.from_id_to_bytes(TOKENS if entries is None else entries)


def accepting(state):
    value = state.is_accepting
    return bool(value() if callable(value) else value)


def _bound_snapshot(source, name, value):
    """Preserve fixture snapshots while exercising public single-item mutation."""
    m = glrmask.Module(source) if isinstance(source, glrmask.Grammar) else source.copy()
    assert m.bind(name, value) is None
    return m


def test_public_types_exist_without_exposing_backend_selection():
    for name in ["Grammar", "Module", "Constraint", "ConstraintState", "Vocab", "TokenSet", "MaskStrategy", "BoundaryMaskStrategy"]:
        assert hasattr(glrmask, name), name
    for name in ["AUTO", "FAST_BUILD", "FAST_RUNTIME", "PARSER_DWA", "TOKEN_TRIE", "TOKEN_CLASS_TRIE"]:
        assert hasattr(glrmask.MaskStrategy, name), name
    assert glrmask.MaskStrategy.FAST_BUILD is glrmask.MaskStrategy.TOKEN_TRIE
    assert glrmask.MaskStrategy.FAST_RUNTIME is glrmask.MaskStrategy.PARSER_DWA
    for name in ["AUTO", "FAST_BUILD", "FAST_RUNTIME", "PARSER_DWA", "TOKEN_TRIE"]:
        assert hasattr(glrmask.BoundaryMaskStrategy, name), name
    assert glrmask.BoundaryMaskStrategy.FAST_BUILD is glrmask.BoundaryMaskStrategy.TOKEN_TRIE
    assert glrmask.BoundaryMaskStrategy.FAST_RUNTIME is glrmask.BoundaryMaskStrategy.PARSER_DWA


def test_source_child_uses_immutable_bind_and_compiled_child_is_rejected():
    v = vocab()
    parent = glrmask.Grammar.from_glrm(HOST)
    child = glrmask.Grammar.from_ebnf('start ::= "a"')
    c = _bound_snapshot(parent, "child", child).compile(v)
    state = c.start()
    mask = np.asarray(state.mask())
    assert mask.dtype == np.bool_
    assert mask[4]
    state.commit_token(4)
    assert accepting(state)
    with pytest.raises((ValueError, RuntimeError)):
        parent.compile(v)
    with pytest.raises(TypeError):
        _bound_snapshot(parent, "child", child.compile(v))


def test_module_save_load_remains_open_and_immutable():
    v = vocab()
    host = glrmask.Module(glrmask.Grammar.from_glrm(HOST))
    host.precompile(v)
    artifact = host.save()
    assert isinstance(artifact, bytes)
    loaded = glrmask.Module.load(artifact)
    child = glrmask.Module(glrmask.Grammar.from_ebnf('start ::= "a"'))
    bound = _bound_snapshot(loaded, "child", child).compile(v)
    assert bound.start().mask()[4]
    for open_mod in [host, loaded]:
        with pytest.raises((ValueError, RuntimeError)):
            open_mod.compile(v)
    with pytest.raises((ValueError, RuntimeError)):
        glrmask.Constraint.load(artifact)


def test_exact_values_survive_original_vocab_lifetime_and_keep_ids():
    value = glrmask.TokenSet([63, 7])
    gc.collect()
    v = vocab()
    assert repr(value) == "TokenSet[7..=7, 63..=63]"
    c = _bound_snapshot(glrmask.Grammar.from_glrm(MARK), "MARK", value).compile(v)
    mask = c.start().mask()
    assert mask[7] and mask[63]
    assert not mask[1]  # Same bytes as token 7, not the same token ID.
    with pytest.raises((ValueError, RuntimeError)):
        _bound_snapshot(glrmask.Grammar.from_glrm(MARK), "MARK", glrmask.TokenSet([126])).compile(v)
    empty = glrmask.TokenSet([])
    assert empty == glrmask.TokenSet.from_ranges([])
    for mode in [glrmask.MaskStrategy.AUTO, glrmask.MaskStrategy.FAST_BUILD,
                 glrmask.MaskStrategy.TOKEN_CLASS_TRIE, glrmask.MaskStrategy.FAST_RUNTIME]:
        m = _bound_snapshot(glrmask.Grammar.from_glrm(MARK), "MARK", empty)
        impossible = m.compile(v, end_tokens=[127], mask_strategy=mode)
        state = impossible.start()
        assert not accepting(state) and not state.mask().any()
        assert state.forced_tokens() == [] and state.forced_tokens() == []
    assert glrmask.TokenSet([7, 7]) == glrmask.TokenSet([7])


def test_nested_path_and_open_child_are_rejected():
    v = vocab()
    child = glrmask.Module(glrmask.Grammar.from_glrm(MARK))
    parent = glrmask.Module(glrmask.Grammar.from_glrm(HOST))
    with pytest.raises(ValueError, match="unbound externs"):
        parent.bind("child", child)
    with pytest.raises(ValueError, match="qualified paths"):
        parent.bind("child::MARK", glrmask.TokenSet([7]))
    child.bind("MARK", glrmask.TokenSet([7]))
    parent.bind("child", child)
    c = parent.compile(v)
    state = c.start()
    assert not state.mask()[4]
    state.commit_token(0)
    assert state.mask()[7] and not state.mask()[1]
    state.commit_token(7)
    state.commit_token(3)
    assert accepting(state)


def test_precompiled_module_with_precompiled_child():
    v = vocab()
    child = glrmask.Module(glrmask.Grammar.from_glrm(MARK))
    child.bind("MARK", glrmask.TokenSet([7]))
    child.precompile(v)
    parent = glrmask.Module(glrmask.Grammar.from_glrm(HOST))
    parent.precompile(v)
    parent.bind("child", child)
    state = parent.compile(v).start()
    for token in [0, 7, 3]:
        assert state.mask()[token]
        state.commit_token(token)
    assert accepting(state)


def test_state_retains_constraint_after_python_owner_is_collected():
    v = vocab()
    constraint = glrmask.Grammar.from_ebnf('start ::= "a"').compile(v)
    state = constraint.start()
    del constraint, v
    gc.collect()
    assert state.mask()[1]
    state.commit_token(1)
    assert accepting(state)


def test_token_membership_is_checked_against_target_without_identity_guarantees():
    v = vocab()
    different = dict(TOKENS)
    different[7] = b"changed"
    desc = _bound_snapshot(glrmask.Grammar.from_glrm(MARK), "MARK", glrmask.TokenSet([7]))
    for target in [v, vocab(different)]:
        mask = desc.compile(target).start().mask()
        assert mask[7] and not mask[1]
    with pytest.raises(ValueError):
        _bound_snapshot(desc, "missing", glrmask.TokenSet([7]))


def test_packed_numpy_fill_mask_agrees_with_boolean_mask():
    v = vocab()
    c = _bound_snapshot(glrmask.Module(glrmask.Grammar.from_glrm(MARK)), "MARK", glrmask.TokenSet([7, 63, 127])).compile(v)
    state = c.start()
    words = np.full(6, -1, dtype=np.int32)
    state.fill_mask(words)
    actual = words.view(np.uint32)
    assert actual.tolist() == [1 << 7, 1 << 31, 0, 1 << 31, 0, 0]
    assert np.flatnonzero(state.mask()).tolist() == [7, 63, 127]


@pytest.mark.parametrize("mode_name", ["AUTO", "FAST_BUILD", "FAST_RUNTIME"])
def test_final_keyword_options_and_child_root_end_isolation(mode_name):
    v = vocab()
    mode = getattr(glrmask.MaskStrategy, mode_name)
    child_mod = glrmask.Module(glrmask.Grammar.from_ebnf('start ::= "a"'))
    child = child_mod.compile(v, end_tokens=[127], mask_strategy=mode)
    child = glrmask.Constraint.load(child.save())
    own_state = child.start()
    assert not own_state.mask()[127]
    own_state.commit_token(1)
    assert not accepting(own_state) and own_state.mask()[127]
    parent_mod = glrmask.Module(glrmask.Grammar.from_glrm(HOST))
    parent_mod.bind("child", child_mod)
    c = parent_mod.compile(v, end_tokens=[63], mask_strategy=mode)
    state = c.start()
    assert state.mask()[4]
    state.commit_token(4)
    assert not accepting(state)
    assert state.mask()[63] and not state.mask()[127]
    state.commit_token(63)
    assert accepting(state)
    assert not np.any(state.mask())
