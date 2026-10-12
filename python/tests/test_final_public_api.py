"""Additional lifecycle, validation and ownership regressions for the final API."""
import gc
import numpy as np
import pytest
import glrmask


def vocab():
    return glrmask.Vocab.from_id_to_bytes({0: b'a', 1: b'b', 7: b'a', 8: b'', 63: b''})


def test_top_level_surface_hides_engine_and_experimental_policy_types():
    for name in [
        "Grammar",
        "UnlinkedConstraint",
        "Token",
        "ExactTokens",
        "Optimization",
        "Vocab",
        "Constraint",
        "ConstraintState",
    ]:
        assert hasattr(glrmask, name), name
    for name in [
        "Expr",
        "ExactToken",
        "DynamicConstraint",
        "DynamicConstraintState",
        "VocabPartition",
        "AutoConstraint",
    ]:
        assert not hasattr(glrmask, name), name
    for name in [
        "from_json_schema",
        "from_glrm_grammar",
        "from_lark",
        "from_ebnf",
        "bind_grammar",
    ]:
        assert not hasattr(glrmask.Constraint, name), name


def test_json_schema_constructor_accepts_text_and_objects():
    v = glrmask.Vocab.from_id_to_bytes({0: b'1', 1: b'2'})
    for schema in ['{"const":1}', {'const': 1}]:
        c = glrmask.Grammar.from_json_schema(schema).compile(v)
        s = c.start()
        assert s.mask()[0] and not s.mask()[1]
        s.commit_token(0)
        assert s.is_accepting()


def test_bind_validation_is_early_and_unlinked_constraints_reject_source():
    v = vocab()
    g = glrmask.Grammar.from_glrm('glrm 1; start start; extern T; extern C; nt start = T C;')
    child = glrmask.Grammar.from_ebnf('start ::= "a"')
    with pytest.raises(TypeError):
        g.bind('T', 7)
    assert g.bind('T', child)
    assert g.bind('C', glrmask.Token(7))
    bound = g.bind('T', glrmask.Token(7))
    with pytest.raises(ValueError):
        bound.bind('T', glrmask.Token(8))
    with pytest.raises(ValueError):
        bound.bind('missing', child)
    m = g.compile_unlinked(v)
    with pytest.raises(TypeError):
        m.bind('C', child)
    with pytest.raises(TypeError):
        g.bind('C', child.compile(v))


def test_oversized_and_undersized_mask_buffers_and_state_termination():
    v = vocab()
    c = glrmask.Grammar.from_ebnf('start ::= "a"').compile(v, end_tokens=[130])
    s = c.start()
    with pytest.raises(ValueError):
        s.fill_mask(np.zeros(1, dtype=np.int32))
    words = np.full(7, -1, dtype=np.int32)
    s.fill_mask(words)
    assert words[-2:].tolist() == [0, 0]
    assert not s.is_accepting() and not s.is_rejected()
    assert s.mask()[0]
    s.commit_token(0)
    assert s.mask()[130]
    s.commit_token(130)
    assert s.is_accepting() and not s.is_rejected()
    assert not np.any(s.mask())
    with pytest.raises(ValueError):
        s.commit_token(0)
    with pytest.raises(ValueError):
        s.commit_bytes(b'a')


def test_description_owns_source_bindings_after_collection():
    v = vocab()
    parent = glrmask.Grammar.from_glrm('glrm 1; start start; extern C; nt start = C;')
    child_source = glrmask.Grammar.from_ebnf('start ::= "a"')
    a = parent.bind('C', child_source)
    del parent, child_source, v
    gc.collect()
    assert a.compile(vocab()).start().mask()[0]


def test_artifact_ownership_and_existing_vocab_validation():
    v = vocab()
    c = glrmask.Grammar.from_ebnf('start ::= "a"').compile(v, end_tokens=[130])
    raw = c.save()
    assert isinstance(raw, bytes)
    del c
    gc.collect()
    c = glrmask.Constraint.load(raw, vocab=v)
    s = c.start()
    s.commit_token(0)
    assert s.mask()[130]
    with pytest.raises(ValueError):
        glrmask.Constraint.load(raw[:-1])
    other = glrmask.Vocab.from_id_to_bytes({0: b'changed'})
    with pytest.raises(ValueError):
        glrmask.Constraint.load(raw, vocab=other)


@pytest.mark.parametrize("id", [-1, 2**32])
def test_token_constructor_rejects_out_of_u32_range(id):
    with pytest.raises(OverflowError):
        glrmask.Token(id)


def test_token_constructor_and_sparse_membership_validation():
    assert not hasattr(glrmask.Vocab, "token")
    assert not hasattr(glrmask, "Expr")
    token = glrmask.Token(id=7)
    assert token.id == 7
    with pytest.raises(AttributeError):
        token.id = 8
    v = glrmask.Vocab.from_id_to_bytes({2: b"a", 7: b"a", 63: b""})
    g = glrmask.Grammar.from_glrm("glrm 1; start s; extern item; nt s = item;")
    m = glrmask.UnlinkedConstraint.load(g.compile_unlinked(v).save())
    for id in [0, 3, 62, 64, 2**32-1]:
        value = glrmask.Token(id)
        bound = g.bind("item", value)
        with pytest.raises(ValueError, match="not present"):
            bound.compile(v)
        with pytest.raises(ValueError, match="not present"):
            m.bind("item", value)
    for id in [7, 63]:
        c = m.bind("item", glrmask.Token(id)).link()
        assert np.flatnonzero(c.start().mask()).tolist() == [id]
    text = glrmask.Grammar.from_ebnf('s ::= "a"')
    assert np.flatnonzero(g.bind("item", text).compile(v).start().mask()).tolist() == [2, 7]
