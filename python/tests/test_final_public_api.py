"""Additional lifecycle, validation and ownership regressions for the final API."""

import gc
import numpy as np
import pytest
import glrmask


from mask_helpers import token_allowed, unpack_mask

def vocab():
    return glrmask.Vocab({0: b'a', 1: b'b', 2: b'a', 3: b'x', 7: b'a', 8: b'', 63: b''})


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
    v = glrmask.Vocab({0: b'1', 1: b'2'})
    for schema in ['{"const":1}', {'const': 1}]:
        c = glrmask.Grammar.from_json_schema(schema).compile(v)
        s = c.start()
        assert token_allowed(s.mask(), 0) and not token_allowed(s.mask(), 1)
        assert s.commit_token(0)
        assert s.is_accepting()


def test_bind_validation_is_early_and_unlinked_constraints_reject_source():
    v = vocab()
    g = glrmask.Grammar.from_glrm('glrm 1; start start; extern token T; extern grammar C; nt start = T C;')
    child = glrmask.Grammar.from_lark('start: "a"')
    with pytest.raises(TypeError):
        _bound_snapshot(g, 'T', 'not a binding')
    with pytest.raises(glrmask.GLRMaskError):
        _bound_snapshot(g, 'T', child)
    open_child = glrmask.Grammar.from_glrm("glrm 1; start s; extern token X; nt s = X;")
    with pytest.raises(glrmask.GLRMaskError, match="unbound externs"):
        _bound_snapshot(g, 'C', open_child)
    bound = _bound_snapshot(g, 'T', glrmask.TokenSet([7]))
    assert list(bound.externs.unbound) == ['C']
    assert set(bound.externs) == {'T', 'C'}
    with pytest.raises(glrmask.GLRMaskError, match="already bound"):
        _bound_snapshot(bound, 'T', glrmask.TokenSet([8]))
    with pytest.raises(glrmask.GLRMaskError, match="there is no .*extern named"):
        _bound_snapshot(bound, 'missing', child)
    with pytest.raises(glrmask.GLRMaskError, match="there is no extern named"):
        _bound_snapshot(bound, 'C::sub', child)
    complete = _bound_snapshot(bound, 'C', child)
    assert list(complete.externs.unbound) == []
    assert token_allowed(complete.compile(v).start().mask(), 7)


def test_oversized_and_undersized_mask_buffers_and_state_termination():
    v = vocab()
    c = _module_expr(glrmask.Expr.sequence(glrmask.Grammar.from_lark('start: "a"'), glrmask.TokenSet([130]))).compile(v)
    s = c.start()
    with pytest.raises(glrmask.GLRMaskError):
        s.fill_mask(np.zeros(1, dtype=np.uint32))
    words = np.full(7, 0xffffffff, dtype=np.uint32)
    s.fill_mask(words)
    assert words[-2:].tolist() == [0, 0]
    assert not s.is_accepting() and not (not s.mask().any())
    assert token_allowed(s.mask(), 0)
    assert s.commit_token(0)
    assert token_allowed(s.mask(), 130)
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
    assert token_allowed(a.compile(vocab()).start().mask(), 0)


def test_artifact_ownership_and_existing_vocab_validation():
    v = vocab()
    c = _module_expr(glrmask.Expr.sequence(glrmask.Grammar.from_lark('start: "a"'), glrmask.TokenSet([130]))).compile(v)
    raw = c.save()
    assert isinstance(raw, bytes)
    del c
    gc.collect()
    c = glrmask.Constraint.load(raw, v)
    s = c.start()
    assert s.commit_token(0)
    assert token_allowed(s.mask(), 130)
    with pytest.raises(glrmask.GLRMaskError):
        glrmask.Constraint.load(raw[:-1], v)
    other = glrmask.Vocab({0: b'changed'})
    with pytest.raises(glrmask.GLRMaskError):
        glrmask.Constraint.load(raw, other)


def test_token_set_has_no_introspection_accessors():
    assert not hasattr(glrmask.Vocab, "tokens")
    assert not hasattr(glrmask.Vocab, "token_ranges")
    value = glrmask.TokenSet([7, 7, 3])
    assert repr(value) == "TokenSet([range(3, 4), range(7, 8)])"
    assert value == glrmask.TokenSet([range(3, 4), range(7, 8)])
    for name in ["ids", "ranges", "size", "__len__"]:
        assert not hasattr(value, name), name


def test_token_set_and_sparse_membership_validation():
    assert hasattr(glrmask, "Expr")
    v = glrmask.Vocab({2: b"a", 7: b"a", 63: b""})
    g = glrmask.Grammar.from_glrm("glrm 1; start s; extern token item; nt s = item;")
    m = glrmask.Module.load(glrmask.Module(g).save())
    # A token set is IDs only; IDs the vocabulary does not define are allowed in
    # the set and simply never match any text.
    for id in [0, 3, 62, 64]:
        value = glrmask.TokenSet([id])
        bound = _bound_snapshot(g, "item", value)
        assert bound.compile(v) is not None
        assert _bound_snapshot(m, "item", value).compile(v) is not None
    # The largest token ID is accepted; one above it is an error, not a huge mask.
    assert _bound_snapshot(g, "item", glrmask.TokenSet([2**24 - 1])) is not None
    with pytest.raises(glrmask.GLRMaskError, match="maximum token ID"):
        glrmask.TokenSet([2**24])
    for id in [7, 63]:
        c = _bound_snapshot(m, "item", glrmask.TokenSet([id])).compile(v)
        assert np.flatnonzero(unpack_mask(c.start().mask())).tolist() == [id]
    text = glrmask.Grammar.from_lark('start: "a"')
    with pytest.raises(glrmask.GLRMaskError):
        _bound_snapshot(g, "item", text)
    general = glrmask.Grammar.from_glrm("glrm 1; start s; extern grammar item; nt s = item;")
    assert np.flatnonzero(unpack_mask(_bound_snapshot(general, "item", text).compile(v).start().mask())).tolist() == [2, 7]


def test_source_constructors_reject_invalid_source_at_construction():
    for factory, source in [
        (glrmask.Grammar.from_lark, "start: ("),
        (glrmask.Grammar.from_json_schema, "{"),
        (glrmask.Grammar.from_glrm, "invalid source"),
    ]:
        with pytest.raises(glrmask.GLRMaskError):
            factory(source)


def test_strategy_presets_are_distinct_and_legacy_tuning_is_not_public():
    for strategy in (glrmask.MaskStrategy,):
        assert len({strategy.AUTO, strategy.FAST_COMPILE, strategy.FAST_MASKS}) == 3
        assert not hasattr(strategy, "PARSER_DWA")
        assert not hasattr(strategy, "TOKEN_TRIE")
    assert not hasattr(glrmask, "TokenTrie")
    assert not hasattr(glrmask, "ParserBackend")
    grammar = glrmask.Grammar.from_lark('start: "a"')
    for mode in (glrmask.MaskStrategy.AUTO, glrmask.MaskStrategy.FAST_COMPILE, glrmask.MaskStrategy.FAST_MASKS):
        c = grammar.compile(vocab(), mask_strategy=mode)
        assert token_allowed(c.start().mask(), 0)
    for bad in ["token-trie", "parser-dwa", 2, object(), {"kind": "token-trie"}]:
        with pytest.raises(TypeError):
            grammar.compile(vocab(), mask_strategy=bad)


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
    grammar = _module_expr(glrmask.TokenSet([0]))
    assert token_allowed(grammar.compile(v).start().mask(), 0)


def test_public_reprs_are_stable_and_python_quoted():
    one = glrmask.Vocab({0: b"a"})
    assert repr(one) == "<glrmask.Vocab: 1 token>"
    two = glrmask.Vocab({0: b"a", 1: b"b"})
    assert repr(two) == "<glrmask.Vocab: 2 tokens>"

    assert repr(glrmask.TokenSet([])) == "TokenSet([])"
    assert repr(glrmask.TokenSet([range(0, 10), range(20, 30)])) == (
        "TokenSet([range(0, 10), range(20, 30)])"
    )
    wide = glrmask.TokenSet([range(index, index + 1) for index in range(0, 18, 2)])
    assert repr(wide) == "<glrmask.TokenSet: 9 tokens in 9 ranges>"

    grammar = glrmask.Grammar.from_glrm(
        "glrm 1; start root; extern token TOK; extern grammar child; nt root = TOK;"
    )
    externs = grammar.externs
    assert repr(externs["TOK"]) == "Extern(name='TOK', kind='token')"
    assert repr(externs) == (
        "Externs({'TOK': Extern(name='TOK', kind='token'), "
        "'child': Extern(name='child', kind='grammar')}, unbound=['TOK', 'child'])"
    )
    assert repr(glrmask.Grammar.from_lark('start: "a"').externs) == "Externs({}, unbound=[])"


def token_extern_grammar():
    return glrmask.Grammar.from_glrm(
        "glrm 1; start root; extern token T; nt root = T;"
    )


def test_bind_requires_a_token_set_for_a_token_extern():
    v = vocab()
    grammar = token_extern_grammar()
    with pytest.raises(glrmask.GLRMaskError, match="token sets can only be bound to a Module"):
        grammar.bind("T", glrmask.TokenSet([0]))
    assert len(grammar.externs.unbound) == 1

    module = glrmask.Module(token_extern_grammar())
    assert module.bind("T", glrmask.TokenSet([0])) is None
    assert len(module.externs.unbound) == 0
    state = module.compile(v).start()
    assert token_allowed(state.mask(), 0) and state.commit_token(0) and state.is_accepting()

    for token_id in (0, -1, 2**32):
        with pytest.raises(TypeError):
            token_extern_grammar().bind("T", token_id)
        with pytest.raises(TypeError):
            glrmask.Module(token_extern_grammar()).bind("T", token_id)


def test_grammar_copy_binds_independently():
    grammar = glrmask.Grammar.from_glrm("extern grammar T start root nt root = T")
    copy = grammar.copy()
    assert copy.bind("T", glrmask.Expr.literal("a")) is None
    assert len(copy.externs.unbound) == 0
    assert len(grammar.externs.unbound) == 1
    assert "T" in grammar.externs.unbound




def test_vocab_requires_bytes_values():
    from collections import UserDict
    assert isinstance(glrmask.Vocab(UserDict({0: b"a"})), glrmask.Vocab)
    for value in ("a", bytearray(b"a"), memoryview(b"a")):
        with pytest.raises(TypeError, match="vocab values must be bytes"):
            glrmask.Vocab({7: value})



def test_equality_with_other_types_is_false_not_an_error():
    assert (glrmask.TokenSet([1]) == 1) is False
    assert (glrmask.TokenSet([1]) != "x") is True
    assert glrmask.TokenSet([1]) == glrmask.TokenSet([1])


def test_internal_module_stays_importable_but_is_not_public():
    assert "_internal" not in glrmask.__all__
    assert hasattr(glrmask._internal, "compiler_cache_stats")

def test_fill_mask_dtype_and_layout():
    state = glrmask.Grammar.from_lark('start: "a"').compile(vocab()).start()
    words = np.zeros(4, dtype=np.uint32)
    state.fill_mask(words)
    signed = np.zeros(4, dtype=np.int32)
    state.fill_mask(signed.view(np.uint32))
    assert np.array_equal(words, signed.view(np.uint32))
    readonly = words.copy()
    readonly.flags.writeable = False
    for array in (signed, np.zeros(4, dtype=np.int64), np.zeros((2, 2), dtype=np.uint32), words[::2], readonly):
        with pytest.raises(TypeError) as exc:
            state.fill_mask(array)
        assert str(exc.value) == "fill_mask needs a contiguous, writable 1-D uint32 NumPy array; for an int32 array pass arr.view(np.uint32)"


@pytest.mark.parametrize("strategy", [glrmask.MaskStrategy.AUTO, glrmask.MaskStrategy.FAST_COMPILE, glrmask.MaskStrategy.FAST_MASKS])
def test_precompile_requires_same_vocab(strategy):
    v = vocab()
    wrong = glrmask.Vocab({0: b"z"})
    module = glrmask.Module(glrmask.Grammar.from_glrm('start s extern grammar item nt s = item'))
    module.precompile(v, mask_strategy=strategy)
    with pytest.raises(glrmask.GLRMaskError, match="vocabular"):
        module.precompile(wrong, mask_strategy=strategy)
    module.bind("item", glrmask.Grammar.from_lark('start: "a"'))
    with pytest.raises(glrmask.GLRMaskError, match="vocabular"):
        module.compile(wrong, mask_strategy=strategy)
    module.compile(v, mask_strategy=strategy)


@pytest.mark.parametrize("strategy", [glrmask.MaskStrategy.AUTO, glrmask.MaskStrategy.FAST_COMPILE, glrmask.MaskStrategy.FAST_MASKS])
def test_precompile_vocab_lock_survives_end_token_delegation_and_roundtrip(strategy):
    v = glrmask.Vocab({0: b"a", 1: b"", 2: b"b"})
    wrong = glrmask.Vocab({0: b"b", 1: b"", 2: b"a"})
    root = glrmask.Module(glrmask.Grammar.from_glrm(
        "start s extern grammar BODY extern token END nt s = BODY END"))
    root.precompile(v, mask_strategy=strategy)
    for module in (root.copy(), glrmask.Module.load(root.save())):
        with pytest.raises(glrmask.GLRMaskError, match="vocabular"):
            module.precompile(wrong, mask_strategy=strategy)
        module.bind("BODY", glrmask.Grammar.from_lark('start: "a"'))
        module.bind("END", glrmask.TokenSet([1]))
        outer = glrmask.Module(glrmask.Grammar.from_glrm("start s extern grammar INNER nt s = INNER"))
        outer.bind("INNER", module)
        for nested in (module, glrmask.Module.load(module.save()), outer, glrmask.Module.load(outer.save())):
            with pytest.raises(glrmask.GLRMaskError, match="vocabular"):
                nested.compile(wrong, mask_strategy=strategy)
            with pytest.raises(glrmask.GLRMaskError, match="vocabular"):
                nested.precompile(wrong, mask_strategy=strategy)
            state = nested.compile(v, mask_strategy=strategy).start()
            assert state.commit_token(0)
            assert state.commit_token(1)
            assert state.is_accepting()


def test_signed_mask_sign_bit_and_repr():
    v = glrmask.Vocab({31: b"a", 63: b"a", 255: b"b"})
    constraint = glrmask.Grammar.from_lark('start: "a"').compile(v)
    state = constraint.start()
    assert repr(constraint) == "<glrmask.Constraint mask_bit_len=256>"
    assert repr(state) == "<glrmask.ConstraintState accepting=False terminated=False>"
    expected = state.mask()
    signed = np.zeros((2, len(expected)), dtype=np.int32)
    state.fill_mask(signed[1].view(np.uint32))
    assert np.array_equal(signed[1].view(np.uint32), expected)
    assert signed[1, 0] < 0 and signed[1, 1] < 0
    assert not signed[0].any()
    import importlib.util
    if importlib.util.find_spec("torch") is not None:
        import torch
        tensor = torch.zeros(len(expected), dtype=torch.int32, device="cpu")
        state.fill_mask(tensor.numpy().view(np.uint32))
        assert np.array_equal(tensor.numpy().view(np.uint32), expected)
    assert state.commit_token(31)
    assert repr(state) == f"<glrmask.ConstraintState accepting={state.is_accepting()} terminated={state.is_terminated()}>"


def test_hashable_public_values_and_token_set_expr_parts():
    for value in (glrmask.MaskStrategy.AUTO, glrmask.MaskStrategy.FAST_COMPILE, glrmask.MaskStrategy.FAST_MASKS):
        assert {value: 1}[value] == 1
        assert hash(value) == hash(int(value))
    assert hash(glrmask.TokenSet([1])) == hash(glrmask.TokenSet([1]))
    grammar = glrmask.Grammar.from_glrm('extern token A; start s; nt s = A;')
    assert len(set(grammar.externs.values())) == 1
    assert hash(grammar.externs["A"]) == hash(grammar.externs["A"])
    with pytest.raises(TypeError):
        glrmask.Expr.sequence(5, 6)
    with pytest.raises(TypeError):
        glrmask.Expr.choice(5, 6)
    state = _module_expr(glrmask.Expr.sequence(glrmask.TokenSet([5]), glrmask.TokenSet([6]))).compile(glrmask.Vocab({})).start()
    assert state.commit_token(5)
    assert state.commit_token(6)
    assert state.is_accepting()
    choice = _module_expr(glrmask.Expr.choice(glrmask.TokenSet([5]), glrmask.TokenSet([6]))).compile(glrmask.Vocab({})).start()
    assert choice.commit_token(6)
    assert choice.is_accepting()

def test_grammar_rejects_modules_in_expressions_but_module_bind_accepts():
    v = vocab()
    source = glrmask.Grammar.from_lark('start: "a"')
    def parent():
        return glrmask.Grammar.from_glrm('start s extern grammar item nt s = item')
    prepared = glrmask.Module(source)
    prepared.precompile(v)
    for m in (glrmask.Module(source), prepared):
        with pytest.raises(TypeError):
            glrmask.Grammar.from_expr(m)
        for expr in (glrmask.Expr.sequence(m), glrmask.Expr.sequence(glrmask.Expr.choice(m), glrmask.Expr.literal(""))):
            with pytest.raises(glrmask.GLRMaskError) as exc:
                glrmask.Grammar.from_expr(expr)
            assert exc.value.kind == "invalid_input"
            assert str(exc.value) == "a Module can't be part of a Grammar; bind it to a Module instead"
            g = parent()
            with pytest.raises(glrmask.GLRMaskError) as exc:
                g.bind("item", expr)
            assert exc.value.kind == "invalid_input"
            assert len(g.externs.unbound) == 1
            outer = glrmask.Module(parent())
            outer.bind("item", expr)
            state = outer.compile(v).start()
            assert state.commit_bytes(b"a") and state.is_accepting()


def _module_expr(value):
    module = glrmask.Module(glrmask.Grammar.from_glrm("extern grammar VALUE start root nt root = VALUE"))
    module.bind("VALUE", value)
    return module


def test_grammar_rejects_tokens_transitively_and_module_accepts_them():
    token = glrmask.TokenSet([5])
    for value in (token, glrmask.Expr.choice(token, glrmask.Expr.literal("x")),
                  glrmask.Expr.sequence(glrmask.Expr.literal(""), token),
                  glrmask.Expr.sequence(glrmask.Expr.choice(glrmask.Expr.literal("x"), token), glrmask.Expr.literal(""))):
        grammar = glrmask.Grammar.from_glrm("extern grammar VALUE start root nt root = VALUE")
        if isinstance(value, glrmask.TokenSet):
            with pytest.raises(TypeError):
                glrmask.Grammar.from_expr(value)
        attempts = [lambda: grammar.bind("VALUE", value)]
        if isinstance(value, glrmask.Expr):
            attempts.append(lambda: glrmask.Grammar.from_expr(value))
        for attempt in attempts:
            with pytest.raises(glrmask.GLRMaskError, match="token sets can only be bound to a Module") as exc:
                attempt()
            assert exc.value.kind == "invalid_input"
        assert len(grammar.externs.unbound) == 1
        state = _module_expr(value).compile(glrmask.Vocab({0: b"x"})).start()
        assert state.commit_token(5) and state.is_accepting()


def test_removed_api_forms_and_explicit_strategy_default():
    grammar = glrmask.Grammar.from_lark('start: "a"')
    v = vocab()
    module = glrmask.Module(grammar)
    assert grammar.compile(v).start().mask().tolist() == grammar.compile(v, mask_strategy=glrmask.MaskStrategy.AUTO).start().mask().tolist()
    for op in (grammar.compile, module.compile, module.precompile):
        with pytest.raises(TypeError):
            op(v, mask_strategy=None)
    for obj in (grammar, module):
        with pytest.raises(TypeError):
            obj.bind(T=glrmask.Expr.literal("a"))
    assert not hasattr(grammar.compile(v).start(), "commit_tokens")
    assert not hasattr(glrmask.TokenSet, "from_ranges")


def test_mixed_token_set_constructor_and_bool_rejection():
    assert glrmask.TokenSet([0, range(1, 4), range(8, 2, -2), 9]) == glrmask.TokenSet([0, 1, 2, 3, 4, 6, 8, 9])
    assert glrmask.TokenSet(range(2**24)) == glrmask.TokenSet([range(2**24)])
    with pytest.raises(TypeError):
        glrmask.TokenSet([True])
    with pytest.raises(TypeError):
        glrmask.TokenSet([False])
