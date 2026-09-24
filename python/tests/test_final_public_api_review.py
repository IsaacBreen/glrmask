"""Target public API contract tests; run after the Python facade is migrated.

These exercise semantics rather than prescribing property-versus-method
spelling for state status (the existing binding and preview differ there).
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


def test_public_types_exist_without_exposing_backend_selection():
    for name in ["Grammar", "UnlinkedConstraint", "Constraint", "ConstraintState", "Vocab", "ExactToken", "ExactTokens", "Optimization"]:
        assert hasattr(glrmask, name), name
    for name in ["AUTO", "FAST_BUILD", "FAST_RUNTIME"]:
        assert hasattr(glrmask.Optimization, name), name


def test_source_child_uses_immutable_bind_and_compiled_child_is_rejected():
    v = vocab()
    parent = glrmask.Grammar.from_glrm(HOST)
    child = glrmask.Grammar.from_ebnf('start ::= "a"')
    c = parent.bind("child", child).compile(v)
    state = c.start()
    mask = np.asarray(state.mask())
    assert mask.dtype == np.bool_
    assert mask[4]
    state.commit_token(4)
    assert accepting(state)
    with pytest.raises((ValueError, RuntimeError)):
        parent.compile(v)
    with pytest.raises(TypeError):
        parent.bind("child", child.compile(v))


def test_unlinked_constraint_save_load_remains_open_and_immutable():
    v = vocab()
    host = glrmask.Grammar.from_glrm(HOST).compile_unlinked(v)
    artifact = host.save()
    assert isinstance(artifact, bytes)
    loaded = glrmask.UnlinkedConstraint.load(artifact)
    child = glrmask.Grammar.from_ebnf('start ::= "a"').compile(v)
    bound = loaded.bind("child", child).link()
    assert bound.start().mask()[4]
    for open_unlinked in [host, loaded]:
        with pytest.raises((ValueError, RuntimeError)):
            open_unlinked.link()
    with pytest.raises((ValueError, RuntimeError)):
        glrmask.Constraint.load(artifact)


def test_exact_values_survive_original_vocab_lifetime_and_keep_ids():
    value = vocab().tokens([63, 7])
    gc.collect()
    v = vocab()
    assert list(value.ids) == [7, 63]
    c = glrmask.Grammar.from_glrm(MARK).bind("MARK", value).compile(v)
    mask = c.start().mask()
    assert mask[7] and mask[63]
    assert not mask[1]  # Same bytes as token 7, not the same token ID.
    with pytest.raises((ValueError, RuntimeError)):
        v.token(126)
    with pytest.raises((ValueError, RuntimeError)):
        v.tokens([])
    with pytest.raises((ValueError, RuntimeError)):
        v.tokens([7, 7])


def test_source_nested_open_token_survives_unlinked_save_and_late_binding():
    v = vocab()
    child = glrmask.Grammar.from_glrm(MARK)
    parent = glrmask.Grammar.from_glrm(HOST).bind("child", child)
    nested = glrmask.UnlinkedConstraint.load(parent.compile_unlinked(v).save())
    with pytest.raises((ValueError, RuntimeError)):
        nested.link()
    c = nested.bind("child.MARK", v.token(7)).link()
    state = c.start()
    assert not state.mask()[4]
    state.commit_token(0)
    assert state.mask()[7] and not state.mask()[1]
    state.commit_token(7)
    state.commit_token(3)
    assert accepting(state)


def test_unlinked_constraint_rejects_unlinked_child():
    v = vocab()
    child = glrmask.Grammar.from_ebnf('start ::= "a"').compile_unlinked(v)
    parent = glrmask.Grammar.from_glrm(HOST).compile_unlinked(v)
    with pytest.raises(TypeError):
        parent.bind("child", child)


def test_state_retains_constraint_after_python_owner_is_collected():
    v = vocab()
    constraint = glrmask.Grammar.from_ebnf('start ::= "a"').compile(v)
    state = constraint.start()
    del constraint, v
    gc.collect()
    assert state.mask()[1]
    state.commit_token(1)
    assert accepting(state)


def test_vocab_compatibility_checks_the_entire_mapping():
    v = vocab()
    different = dict(TOKENS)
    different[2] = b"changed"
    wrong_value = vocab(different).token(7)
    desc = glrmask.Grammar.from_glrm(MARK).bind("MARK", wrong_value)
    with pytest.raises((ValueError, RuntimeError)):
        desc.compile(v)


def test_packed_numpy_fill_mask_agrees_with_boolean_mask():
    v = vocab()
    c = glrmask.Grammar.from_glrm(MARK).compile_unlinked(v).bind("MARK", v.tokens([7, 63, 127])).link()
    state = c.start()
    words = np.full(6, -1, dtype=np.int32)  # Extra words must be cleared.
    state.fill_mask(words)
    actual = words.view(np.uint32)
    assert actual.tolist() == [1 << 7, 1 << 31, 0, 1 << 31, 0, 0]
    assert np.flatnonzero(state.mask()).tolist() == [7, 63, 127]


@pytest.mark.parametrize("mode_name", ["AUTO", "FAST_BUILD", "FAST_RUNTIME"])
def test_final_keyword_options_and_child_root_end_isolation(mode_name):
    v = vocab()
    mode = getattr(glrmask.Optimization, mode_name)
    child = glrmask.Grammar.from_ebnf('start ::= "a"').compile(v, end_tokens=[127], optimization=mode)
    child = glrmask.Constraint.load(child.save())
    own_state = child.start()
    assert not own_state.mask()[127]
    own_state.commit_token(1)
    assert accepting(own_state) and own_state.mask()[127]
    c = glrmask.Grammar.from_glrm(HOST).compile_unlinked(v).bind("child", child).link(
        end_tokens=[63], optimization=mode
    )
    state = c.start()
    assert state.mask()[4]  # Child EOS must not be embedded into its body.
    state.commit_token(4)
    assert accepting(state)
    assert state.mask()[63] and not state.mask()[127]
    state.commit_token(63)
    assert not np.any(state.mask())
