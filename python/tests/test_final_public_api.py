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
        "ParserBackend",
        "TokenTrie",
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
    assert not hasattr(glrmask.Grammar, "from_ebnf")


def test_json_schema_constructor_accepts_text_and_objects():
    v = glrmask.Vocab.from_id_to_bytes({0: b'1', 1: b'2'})
    for schema in ['{"const":1}', {'const': 1}]:
        c = glrmask.Grammar.from_json_schema(schema).compile(v)
        s = c.start()
        assert s.mask()[0] and not s.mask()[1]
        assert s.commit_token(0)
        assert s.is_accepting()


def test_bind_validation_is_early_and_unlinked_constraints_reject_source():
    v = vocab()
    g = glrmask.Grammar.from_glrm('glrm 1; start start; extern token T; extern grammar C; nt start = T C;')
    child = glrmask.Grammar.from_lark('start: "a"')
    with pytest.raises(TypeError):
        _bound_snapshot(g, 'T', 'not a binding')
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
    with pytest.raises(ValueError, match="there is no .*extern named"):
        _bound_snapshot(bound, 'missing', child)
    with pytest.raises(ValueError, match="qualified paths"):
        _bound_snapshot(bound, 'C::sub', child)
    complete = _bound_snapshot(bound, 'C', child)
    assert list(complete.externs.unbound) == []
    assert complete.compile(v).start().mask()[7]


def test_oversized_and_undersized_mask_buffers_and_state_termination():
    v = vocab()
    c = glrmask.Grammar.from_expr(glrmask.Expr.sequence(glrmask.Grammar.from_lark('start: "a"'), glrmask.TokenSet([130]))).compile(v)
    s = c.start()
    with pytest.raises(ValueError):
        s.fill_mask(np.zeros(1, dtype=np.int32))
    words = np.full(7, -1, dtype=np.int32)
    s.fill_mask(words)
    assert words[-2:].tolist() == [0, 0]
    assert not s.is_accepting() and not (not s.mask().any())
    assert s.mask()[0]
    assert s.commit_token(0)
    assert s.mask()[130]
    assert s.commit_token(130)
    assert s.is_accepting() and not np.any(s.mask())
    assert not s.commit_token(0)
    assert not s.commit_bytes(b'a')
def test_description_owns_source_bindings_after_collection():
    v = vocab()
    parent = glrmask.Grammar.from_glrm('glrm 1; start start; extern grammar C; nt start = C;')
    child_source = glrmask.Grammar.from_lark('start: "a"')
    a = _bound_snapshot(parent, 'C', child_source)
    del parent, child_source, v
    gc.collect()
    assert a.compile(vocab()).start().mask()[0]


def test_artifact_ownership_and_existing_vocab_validation():
    v = vocab()
    c = glrmask.Grammar.from_expr(glrmask.Expr.sequence(glrmask.Grammar.from_lark('start: "a"'), glrmask.TokenSet([130]))).compile(v)
    raw = c.save()
    assert isinstance(raw, bytes)
    del c
    gc.collect()
    c = glrmask.Constraint.load(raw, vocab=v)
    s = c.start()
    assert s.commit_token(0)
    assert s.mask()[130]
    with pytest.raises(ValueError):
        glrmask.Constraint.load(raw[:-1], v)
    other = glrmask.Vocab.from_id_to_bytes({0: b'changed'})
    with pytest.raises(ValueError):
        glrmask.Constraint.load(raw, vocab=other)


def test_token_set_has_no_introspection_accessors():
    assert not hasattr(glrmask.Vocab, "tokens")
    assert not hasattr(glrmask.Vocab, "token_ranges")
    value = glrmask.TokenSet([7, 7, 3])
    assert repr(value) == "TokenSet.from_ranges([range(3, 4), range(7, 8)])"
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
        with pytest.raises(ValueError, match="is not in the vocabulary"):
            bound.compile(v)
        with pytest.raises(ValueError, match="is not in the vocabulary"):
            _bound_snapshot(m, "item", value).compile(v)
    for id in [7, 63]:
        c = _bound_snapshot(m, "item", glrmask.TokenSet([id])).compile(v)
        assert np.flatnonzero(c.start().mask()).tolist() == [id]
    text = glrmask.Grammar.from_lark('start: "a"')
    with pytest.raises(ValueError):
        _bound_snapshot(g, "item", text)
    general = glrmask.Grammar.from_glrm("glrm 1; start s; extern grammar item; nt s = item;")
    assert np.flatnonzero(_bound_snapshot(general, "item", text).compile(v).start().mask()).tolist() == [2, 7]


def test_source_constructors_reject_invalid_source_at_construction():
    for factory, source in [
        (glrmask.Grammar.from_lark, "start: ("),
        (glrmask.Grammar.from_json_schema, "{"),
        (glrmask.Grammar.from_glrm, "invalid source"),
    ]:
        with pytest.raises(ValueError):
            factory(source)


def test_token_trie_settings_value_semantics_and_validation():
    default = glrmask.TokenTrie()
    assert default.parser_backend is glrmask.ParserBackend.LR_TABLE
    assert default.merge_equivalent_tokens is False
    assert default == glrmask.TokenTrie()
    assert hash(default) == hash(glrmask.TokenTrie())
    assert repr(default) == (
        "TokenTrie(parser_backend=ParserBackend.LR_TABLE, "
        "merge_equivalent_tokens=False)"
    )
    native = glrmask.TokenTrie(
        parser_backend=glrmask.ParserBackend.TEMPLATE_DFA,
        merge_equivalent_tokens=True,
    )
    assert native.parser_backend is glrmask.ParserBackend.TEMPLATE_DFA
    assert native.merge_equivalent_tokens is True
    assert native != default
    assert hash(native) != hash(default)
    assert repr(native) == (
        "TokenTrie(parser_backend=ParserBackend.TEMPLATE_DFA, "
        "merge_equivalent_tokens=True)"
    )
    assert len({default, glrmask.TokenTrie(), native}) == 2

    v = vocab()
    grammar = glrmask.Grammar.from_lark('start: "a"')
    compiled = grammar.compile(v, mask_strategy=native)
    assert compiled.start().mask()[0]
    for bad in ["token-trie", 2, object()]:
        with pytest.raises(TypeError):
            grammar.compile(v, mask_strategy=bad)
    with pytest.raises(TypeError):
        glrmask.TokenTrie(parser_backend="lr-table")
    with pytest.raises(TypeError):
        glrmask.TokenTrie(merge_equivalent_tokens=1)


def test_version_metadata_and_module_attributes_agree():
    import importlib.metadata

    assert glrmask.__version__ == importlib.metadata.version("glrmask")
    v = vocab()
    g = glrmask.Grammar.from_lark('start: "a"')
    c = g.compile(v)
    assert type(v).__module__ == "glrmask"
    assert type(c).__module__ == "glrmask"
    assert type(c.start()).__module__ == "glrmask"


def test_module_load_takes_only_bytes_and_from_expr_is_positional():
    v = vocab()
    module = glrmask.Module(glrmask.Grammar.from_lark('start: "a"'))
    data = module.save()
    assert glrmask.Module.load(data) is not None
    with pytest.raises(TypeError):
        glrmask.Module.load(data, v)
    grammar = glrmask.Grammar.from_expr(glrmask.TokenSet([0]))
    assert grammar.compile(v).start().mask()[0]


def test_public_reprs_are_stable_and_python_quoted():
    one = glrmask.Vocab.from_id_to_bytes({0: b"a"})
    assert repr(one) == "<glrmask.Vocab: 1 token>"
    two = glrmask.Vocab.from_id_to_bytes({0: b"a", 1: b"b"})
    assert repr(two) == "<glrmask.Vocab: 2 tokens>"

    assert repr(glrmask.TokenSet([])) == "TokenSet.from_ranges([])"
    assert repr(glrmask.TokenSet.from_ranges([range(0, 10), range(20, 30)])) == (
        "TokenSet.from_ranges([range(0, 10), range(20, 30)])"
    )
    wide = glrmask.TokenSet.from_ranges([range(index, index + 1) for index in range(0, 18, 2)])
    assert repr(wide) == "<glrmask.TokenSet: 9 tokens in 9 ranges>"

    grammar = glrmask.Grammar.from_glrm(
        "glrm 1; start root; extern token TOK; extern grammar child; nt root = TOK;"
    )
    externs = grammar.externs
    assert repr(externs["TOK"]) == "Extern(name='TOK', kind='token')"
    assert repr(externs) == (
        "Externs({'TOK': Extern(name='TOK', kind='token'), "
        "'child': Extern(name='child', kind='grammar')})"
    )
    assert repr(glrmask.Grammar.from_lark('start: "a"').externs) == "Externs({})"


def token_extern_grammar():
    return glrmask.Grammar.from_glrm(
        "glrm 1; start root; extern token T; nt root = T;"
    )


def test_bind_accepts_a_token_id_for_a_token_extern():
    v = vocab()
    grammar = token_extern_grammar()
    assert grammar.bind("T", 0) is None
    assert len(grammar.externs.unbound) == 0
    state = grammar.compile(v).start()
    assert state.mask()[0] and state.commit_token(0) and state.is_accepting()

    module = glrmask.Module(token_extern_grammar())
    assert module.bind("T", 0) is None
    assert len(module.externs.unbound) == 0
    state = module.compile(v).start()
    assert state.mask()[0] and state.commit_token(0) and state.is_accepting()

    with pytest.raises(ValueError):
        token_extern_grammar().bind("T", -1)
    with pytest.raises(ValueError):
        glrmask.Module(token_extern_grammar()).bind("T", 2**32)


def test_grammar_copy_binds_independently():
    grammar = token_extern_grammar()
    copy = grammar.copy()
    assert copy.bind("T", 0) is None
    assert len(copy.externs.unbound) == 0
    assert len(grammar.externs.unbound) == 1
    assert "T" in grammar.externs.unbound


def test_grammar_batch_bind_is_atomic():
    grammar = glrmask.Grammar.from_glrm(
        "glrm 1; start root; extern token a; extern token b; nt root = a b;"
    )
    with pytest.raises(ValueError):
        grammar.bind(a=0, b=glrmask.Grammar.from_lark('start: "a"'))
    assert len(grammar.externs.unbound) == 2
    assert "a" in grammar.externs.unbound
    assert "b" in grammar.externs.unbound

    with pytest.raises(TypeError):
        grammar.bind("a", 0, b=1)
    with pytest.raises(TypeError):
        grammar.bind()
    with pytest.raises(TypeError):
        glrmask.Module(grammar).bind()

    assert grammar.bind(a=0, b=1) is None
    assert len(grammar.externs.unbound) == 0


def test_equality_with_other_types_is_false_not_an_error():
    assert (glrmask.TokenSet([1]) == 1) is False
    assert (glrmask.TokenSet([1]) != "x") is True
    assert glrmask.TokenSet([1]) == glrmask.TokenSet([1])


def test_internal_module_stays_importable_but_is_not_public():
    assert "_internal" not in glrmask.__all__
    assert hasattr(glrmask._internal, "compiler_cache_stats")
