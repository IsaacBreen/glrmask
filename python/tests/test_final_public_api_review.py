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
    for name in ["Grammar", "PreparedGrammar", "Constraint", "ConstraintState", "Vocab", "Token", "ExactTokens", "Optimization"]:
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
    host = glrmask.Grammar.from_glrm(HOST).prepare(v)
    artifact = host.save()
    assert isinstance(artifact, bytes)
    loaded = glrmask.PreparedGrammar.load(artifact)
    child = glrmask.Grammar.from_ebnf('start ::= "a"').prepare(v)
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
        glrmask.Grammar.from_glrm(MARK).bind("MARK", glrmask.Token(126)).compile(v)
    empty = v.tokens([])
    assert len(empty) == 0 and empty.ids == [] and empty.ranges == []
    for mode in [glrmask.Optimization.AUTO, glrmask.Optimization.FAST_BUILD,
                 glrmask.Optimization.BALANCED, glrmask.Optimization.FAST_RUNTIME]:
        impossible = glrmask.Grammar.from_glrm(MARK).bind("MARK", empty).compile(
            v, optimization=mode, end_tokens=[127])
        state = impossible.start()
        assert not accepting(state) and not state.mask().any()
        assert state.forced() == [] and state.forced_token_ids() == []
    with pytest.raises((ValueError, RuntimeError)):
        v.tokens([7, 7])


def test_source_nested_open_token_survives_unlinked_save_and_late_binding():
    v = vocab()
    child = glrmask.Grammar.from_glrm(MARK)
    parent = glrmask.Grammar.from_glrm(HOST).bind("child", child)
    nested = glrmask.PreparedGrammar.load(parent.prepare(v).save())
    with pytest.raises((ValueError, RuntimeError)):
        nested.link()
    c = nested.bind("child::MARK", glrmask.Token(7)).link()
    state = c.start()
    assert not state.mask()[4]
    state.commit_token(0)
    assert state.mask()[7] and not state.mask()[1]
    state.commit_token(7)
    state.commit_token(3)
    assert accepting(state)


def test_prepared_constraint_accepts_prepared_child_without_linking_it():
    v = vocab()
    child = glrmask.Grammar.from_glrm(MARK).prepare(v)
    parent = glrmask.Grammar.from_glrm(HOST).prepare(v)
    open_parent = parent.bind("child", child)
    assert [r.name for r in open_parent.requirements] == ["child::MARK"]
    with pytest.raises(ValueError):
        open_parent.link()
    state = open_parent.bind("child::MARK", glrmask.Token(7)).link().start()
    for token in [0, 7, 3]:
        assert state.mask()[token]
        state.commit_token(token)
    assert accepting(state)
    assert [r.name for r in child.requirements] == ["MARK"]


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
    desc = glrmask.Grammar.from_glrm(MARK).bind("MARK", glrmask.Token(7))
    for target in [v, vocab(different)]:
        mask = desc.compile(target).start().mask()
        assert mask[7] and not mask[1]
    with pytest.raises(ValueError):
        desc.bind("missing", glrmask.Token(7))


def test_packed_numpy_fill_mask_agrees_with_boolean_mask():
    v = vocab()
    c = glrmask.Grammar.from_glrm(MARK).prepare(v).bind("MARK", v.tokens([7, 63, 127])).link()
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
    prepared = glrmask.Grammar.from_ebnf('start ::= "a"').prepare(v)
    child = prepared.link(end_tokens=[127], optimization=mode)
    child = glrmask.Constraint.load(child.save())
    own_state = child.start()
    assert not own_state.mask()[127]
    own_state.commit_token(1)
    assert accepting(own_state) and own_state.mask()[127]
    c = glrmask.Grammar.from_glrm(HOST).prepare(v).bind("child", prepared).link(
        end_tokens=[63], optimization=mode
    )
    state = c.start()
    assert state.mask()[4]  # Child EOS must not be embedded into its body.
    state.commit_token(4)
    assert accepting(state)
    assert state.mask()[63] and not state.mask()[127]
    state.commit_token(63)
    assert not np.any(state.mask())
