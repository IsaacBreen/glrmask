"""Additional lifecycle, validation and ownership regressions for the final API."""
import gc
import numpy as np
import pytest
import glrmask


def vocab():
    return glrmask.Vocab.from_id_to_bytes({0: b'a', 1: b'b', 7: b'a', 8: b'', 63: b''})


def test_json_schema_constructor_accepts_text_and_objects():
    v = glrmask.Vocab.from_id_to_bytes({0: b'1', 1: b'2'})
    for schema in ['{"const":1}', {'const': 1}]:
        c = glrmask.Grammar.from_json_schema(schema).compile(v)
        s = c.start()
        assert s.mask()[0] and not s.mask()[1]
        s.commit_token(0)
        assert s.is_accepting()


def test_bind_validation_is_early_and_compiled_modules_reject_source():
    v = vocab()
    g = glrmask.Grammar.from_glrm('glrm 1; start start; extern token T; extern grammar C; nt start = T C;')
    child = glrmask.Grammar.from_ebnf('start ::= "a"')
    with pytest.raises(TypeError):
        g.bind('T', 7)
    with pytest.raises(ValueError):
        g.bind('T', child)
    with pytest.raises(ValueError):
        g.bind('C', v.token(7))
    bound = g.bind('T', v.token(7))
    with pytest.raises(ValueError):
        bound.bind('T', v.token(8))
    with pytest.raises(ValueError):
        bound.bind('missing', child)
    m = g.compile_module(v)
    with pytest.raises(TypeError):
        m.bind('C', child)


def test_oversized_and_undersized_mask_buffers_and_state_termination():
    v = vocab()
    c = glrmask.Grammar.from_ebnf('start ::= "a"').compile(v, end_tokens=[130])
    s = c.start()
    with pytest.raises(ValueError):
        s.fill_mask(np.zeros(1, dtype=np.int32))
    words = np.full(7, -1, dtype=np.int32)
    s.fill_mask(words)
    assert words[-2:].tolist() == [0, 0]
    assert not s.is_terminated()
    s.commit_token(0)
    assert s.mask()[130]
    s.commit_token(130)
    assert s.is_terminated() and s.is_accepting()
    assert not np.any(s.mask())
    with pytest.raises(ValueError):
        s.commit_token(0)
    with pytest.raises(ValueError):
        s.commit_bytes(b'a')


def test_description_owns_compiled_and_source_bindings_after_collection():
    v = vocab()
    parent = glrmask.Grammar.from_glrm('glrm 1; start start; extern grammar C; nt start = C;')
    child_source = glrmask.Grammar.from_ebnf('start ::= "a"')
    child_compiled = child_source.compile(v)
    a = parent.bind('C', child_source)
    b = parent.bind('C', child_compiled)
    del parent, child_source, child_compiled, v
    gc.collect()
    assert a.compile(vocab()).start().mask()[0]
    assert b.compile(vocab()).start().mask()[0]


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
