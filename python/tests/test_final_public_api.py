"""Additional lifecycle, validation and ownership regressions for the final API."""

import gc
import numpy as np
import pytest
import glrmask


def vocab():
    return glrmask.Vocab.from_id_to_bytes({0: b'a', 1: b'b', 2: b'a', 3: b'x', 7: b'a', 8: b'', 63: b''})


def _bound_snapshot(source, name, value):
    """Preserve fixture snapshots while exercising public single-item mutation."""
    m = glrmask.Module(source) if isinstance(source, glrmask.Grammar) else source.copy()
    assert m.bind(name, value) is None
    return m


def test_top_level_surface_hides_engine_and_experimental_policy_types():
    for name in [
        "Grammar",
        "Module",
        "Expr",
        "Extern",
        "Externs",
        "TokenSet",
        "MaskStrategy",
        "BoundaryMaskStrategy",
        "Vocab",
        "Constraint",
        "ConstraintState",
    ]:
        assert hasattr(glrmask, name), name
    for name in [
        "ModuleOptions",
        "CompileOptions",
        "UnboundExtern",
        "PreparedGrammar",
        "Requirement",
        "Optimization",
        "BoundaryTriggerDetail",
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
    g = glrmask.Grammar.from_glrm('glrm 1; start start; extern token T; extern grammar C; nt start = T C;')
    child = glrmask.Grammar.from_ebnf('start ::= "a"')
    with pytest.raises(TypeError):
        _bound_snapshot(g, 'T', 7)
    with pytest.raises(ValueError):
        _bound_snapshot(g, 'T', child)
    open_child = glrmask.Grammar.from_glrm("glrm 1; start s; extern token X; nt s = X;")
    with pytest.raises(ValueError, match="unbound externs"):
        _bound_snapshot(g, 'C', open_child)
    bound = _bound_snapshot(g, 'T', glrmask.TokenSet([7]))
    assert list(bound.externs.unbound) == ['C']
    assert set(bound.externs) == {'T', 'C'}
    with pytest.raises(ValueError, match="already bound"):
        _bound_snapshot(bound, 'T', glrmask.TokenSet([8]))
    with pytest.raises(ValueError, match="no external"):
        _bound_snapshot(bound, 'missing', child)
    with pytest.raises(ValueError, match="qualified paths"):
        _bound_snapshot(bound, 'C::sub', child)
    complete = _bound_snapshot(bound, 'C', child)
    assert list(complete.externs.unbound) == []
    assert complete.compile(v).start().mask()[7]


def test_oversized_and_undersized_mask_buffers_and_state_termination():
    v = vocab()
    c = glrmask.Grammar.from_ebnf('start ::= "a"').compile(v, end_tokens=[130])
    s = c.start()
    with pytest.raises(ValueError):
        s.fill_mask(np.zeros(1, dtype=np.int32))
    words = np.full(7, -1, dtype=np.int32)
    s.fill_mask(words)
    assert words[-2:].tolist() == [0, 0]
    assert not s.is_accepting() and not (not s.mask().any())
    assert s.mask()[0]
    s.commit_token(0)
    assert s.mask()[130]
    s.commit_token(130)
    assert s.is_accepting() and not (not s.mask().any())
    assert not np.any(s.mask())
    with pytest.raises(ValueError):
        s.commit_token(0)
    with pytest.raises(ValueError):
        s.commit_bytes(b'a')


def test_description_owns_source_bindings_after_collection():
    v = vocab()
    parent = glrmask.Grammar.from_glrm('glrm 1; start start; extern grammar C; nt start = C;')
    child_source = glrmask.Grammar.from_ebnf('start ::= "a"')
    a = _bound_snapshot(parent, 'C', child_source)
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


def test_token_set_has_no_introspection_accessors():
    assert not hasattr(glrmask.Vocab, "tokens")
    assert not hasattr(glrmask.Vocab, "token_ranges")
    value = glrmask.TokenSet([7, 7, 3])
    assert repr(value) == "TokenSet[3..=3, 7..=7]"
    assert value == glrmask.TokenSet.from_ranges([range(3, 4), range(7, 8)])
    for name in ["ids", "ranges", "size", "__len__"]:
        assert not hasattr(value, name), name


def test_token_set_and_sparse_membership_validation():
    assert hasattr(glrmask, "Expr")
    v = glrmask.Vocab.from_id_to_bytes({2: b"a", 7: b"a", 63: b""})
    g = glrmask.Grammar.from_glrm("glrm 1; start s; extern token item; nt s = item;")
    m = glrmask.Module.load(glrmask.Module(g).save())
    for id in [0, 3, 62, 64, 2**32-1]:
        value = glrmask.TokenSet([id])
        bound = _bound_snapshot(g, "item", value)
        with pytest.raises(ValueError, match="unknown exact token ID|not present"):
            bound.compile(v)
        with pytest.raises(ValueError, match="unknown exact token ID|not present"):
            _bound_snapshot(m, "item", value).compile(v)
    for id in [7, 63]:
        c = _bound_snapshot(m, "item", glrmask.TokenSet([id])).compile(v)
        assert np.flatnonzero(c.start().mask()).tolist() == [id]
    text = glrmask.Grammar.from_ebnf('s ::= "a"')
    with pytest.raises(ValueError):
        _bound_snapshot(g, "item", text)
    general = glrmask.Grammar.from_glrm("glrm 1; start s; extern grammar item; nt s = item;")
    assert np.flatnonzero(_bound_snapshot(general, "item", text).compile(v).start().mask()).tolist() == [2, 7]
