"""Canonical lifecycle regressions, exercised against an installed wheel."""
import copy
import gc

import glrmask as g
import pytest

HOST = 'glrm 1; start start; extern grammar child; nt start = "x" child "y";'
MARK = 'glrm 1; start start; extern token MARK; nt start = MARK;'


def vocab(changed=False):
    return g.Vocab.from_id_to_bytes({0:b'x',1:b'a',2:b'b',3:b'y',4:b'xay',5:b'ab',6:b'xa',7:b'ay',8:b'z' if changed else b'a',63:b''})


def source(text='a'):
    return g.Grammar.from_ebnf(f'start ::= "{text}"')


def accepts(module, v, ids):
    state = module.compile(v).start()
    for token in ids:
        assert state.mask()[token]
        state.commit_token(token)
    assert state.is_accepting()


def test_new_types_and_frozen_options():
    v = vocab()
    options = g.ModuleOptions(optimization=g.Optimization.BALANCED, boundary_trigger=g.BoundaryTriggerDetail.EXACT)
    module = g.Module(source(), options=options)
    assert module.options == options
    assert module.options is not module.options
    with pytest.raises(AttributeError):
        options.optimization = g.Optimization.FAST_BUILD
    with pytest.raises(TypeError):
        g.Module(source(), options=options, optimization=g.Optimization.AUTO)
    with pytest.raises(TypeError):
        g.Module(source(), options=options, boundary_trigger=g.BoundaryTriggerDetail.NONE)
    with pytest.raises(AttributeError):
        source().options
    assert g.CompileOptions(end_tokens=[63]).end_tokens == [63]
    accepts(module, v, [1])


def test_local_options_and_root_only_compile():
    v = vocab()
    child = g.Module(source(), optimization=g.Optimization.BALANCED)
    assert child.precompile(v) is None
    parent = g.Module(g.Grammar.from_glrm(HOST), optimization=g.Optimization.FAST_RUNTIME)
    assert parent.bind('child', child) is None
    assert parent.options.optimization == g.Optimization.FAST_RUNTIME
    assert child.options.optimization == g.Optimization.BALANCED
    state = parent.compile(v, end_tokens=[63]).start()
    assert not state.mask()[63]
    state.commit_token(4)
    assert state.is_accepting() and state.mask()[63]
    state.commit_token(63)
    assert not state.mask().any()
    for keyword in ['optimization', 'boundary_trigger']:
        with pytest.raises(TypeError):
            parent.compile(v, **{keyword: None})


@pytest.mark.parametrize('clone', [lambda m:m.clone(), copy.copy, copy.deepcopy])
def test_independent_mutable_clones(clone):
    v = vocab()
    original = g.Grammar.from_glrm(HOST).module()
    other = clone(original)
    other.bind('child', source())
    assert [r.name for r in original.unbound] == ['child']
    assert other.unbound == []
    original.bind('child', source('b'))
    accepts(other, v, [4])
    accepts(original, v, [0,2,3])


def test_with_options_is_independent_and_failures_preserve_prepared_work():
    v = vocab()
    module = source().module()
    configured = module.with_options(g.ModuleOptions(optimization=g.Optimization.BALANCED))
    assert module.options.optimization == g.Optimization.AUTO
    assert configured.options.optimization == g.Optimization.BALANCED
    configured.precompile(v)
    saved = configured.save()
    for change in [configured.with_options, configured.set_options]:
        with pytest.raises(ValueError, match='precompiled'):
            change(g.ModuleOptions(optimization=g.Optimization.FAST_RUNTIME))
        assert configured.save() == saved
        accepts(configured, v, [1])
    assert configured.set_options(configured.options) is None
    assert configured.save() == saved


@pytest.mark.parametrize('prepared_parent', [False, True])
@pytest.mark.parametrize('prepared_child', [False, True])
def test_saved_mixed_graph_and_late_exact_token_binding(prepared_parent, prepared_child):
    v = vocab()
    parent = g.Grammar.from_glrm(HOST).module()
    child = g.Module(g.Grammar.from_glrm(MARK), optimization=g.Optimization.BALANCED)
    if prepared_parent:
        parent.precompile(v)
    if prepared_child:
        child.precompile(v)
    parent.bind('child', child)
    loaded = g.Module.load(parent.save(), v)
    assert [r.name for r in loaded.unbound] == ['child::MARK']
    assert loaded.bind('child::MARK', v.tokens([8])) is None
    accepts(loaded, v, [0,8,3])
    assert not loaded.compile(v).start().mask()[4]
    assert [r.name for r in parent.unbound] == ['child::MARK']
    assert [r.name for r in child.unbound] == ['MARK']
    with pytest.raises(ValueError):
        g.Constraint.load(parent.save())


def test_prepared_parent_accepts_cold_raw_child_after_load():
    v = vocab()
    parent = g.Grammar.from_glrm(HOST).module()
    parent.precompile(v)
    loaded = g.Module.load(parent.save())
    loaded.bind('child', source())
    accepts(loaded, v, [4])


def test_wrong_paths_kinds_duplicates_and_vocab_are_atomic():
    v = vocab()
    parent = g.Grammar.from_glrm(HOST).module()
    parent.precompile(v)
    before = parent.save()
    incompatible = source().module()
    incompatible.precompile(vocab(True))
    for name, child in [('missing', source()), ('child::MARK', g.Token(8)), ('child', incompatible)]:
        with pytest.raises(ValueError):
            parent.bind(name, child)
        assert parent.save() == before
    parent.bind('child', source())
    before = parent.save()
    with pytest.raises(ValueError):
        parent.bind('child', source('b'))
    assert parent.save() == before
    accepts(parent, v, [4])
    token_parent = g.Grammar.from_glrm(MARK).module()
    with pytest.raises(ValueError):
        token_parent.bind('MARK', source())
    assert [r.name for r in token_parent.unbound] == ['MARK']
    token_parent.bind('MARK', g.Token(62))
    with pytest.raises(ValueError):
        token_parent.compile(v)


def test_cross_vocab_precompile_load_and_exact_events_preserve_state():
    v = vocab()
    parent = source().module()
    parent.precompile(v)
    before = parent.save()
    with pytest.raises(ValueError):
        parent.precompile(vocab(True))
    assert parent.save() == before
    with pytest.raises(ValueError):
        g.Module.load(before, vocab(True))
    for data in [b'', b'GLRASM01bad', before[:-1]]:
        with pytest.raises(ValueError):
            g.Module.load(data)
    target = g.Grammar.from_glrm(MARK).module()
    target.bind('MARK', vocab(True).tokens([8]))
    with pytest.raises(ValueError):
        target.compile(v)
    accepts(parent, v, [1])


def test_expression_paths_union_sequence_and_source_binding():
    v = vocab()
    open_leaf = g.Grammar.from_glrm(MARK)
    expr = g.Expr.choice([g.Expr.sequence([source('x'), open_leaf]), source('b')])
    for result in [g.Grammar.from_expr(expr), g.Module(expr)]:
        assert [r.path for r in result.unbound] == [['leaf1', 'MARK']]
        result.bind('leaf1::MARK', g.Token(8))
        state = result.compile(v).start()
        assert state.mask()[0] and state.mask()[2]
        state.commit_token(2)
        assert state.is_accepting()
    host = g.Grammar.from_glrm(HOST)
    host.bind('child', g.Expr.choice([source(), source('b')]))
    assert host.compile(v).start().mask()[4]
    module = g.Grammar.from_glrm(HOST).module()
    module.bind('child', g.Expr.choice([source(), source('b')]))
    accepts(module, v, [4])


def test_expression_whole_token_crosses_boundaries_and_singletons():
    v = vocab()
    expr = g.Expr.sequence(value for value in [source('x'), g.Expr.sequence([source(), source('y')])])
    for result in [g.Module(expr), g.Grammar.from_expr(expr)]:
        state = result.compile(v).start()
        assert state.mask()[4]
        state.commit_token(4)
        assert state.is_accepting()
    accepts(g.Module(g.Expr.choice([source()])), v, [1])
    accepts(g.Module(g.Expr.sequence([source()])), v, [1])
    assert g.Module(g.Expr.sequence([])).compile(v).start().is_accepting()
    empty = g.Expr.choice([])
    for factory in [g.Module, g.Grammar.from_expr]:
        with pytest.raises(ValueError, match='empty'):
            factory(empty)


def test_expression_snapshots_and_configured_source_conversion_rejection():
    v = vocab()
    leaf = g.Grammar.from_glrm(MARK).module()
    expr = g.Expr.sequence([leaf])
    leaf.bind('MARK', g.Token(8))
    assembled = g.Module(expr)
    assert [r.name for r in assembled.unbound] == ['leaf0::MARK']
    assembled.bind('leaf0::MARK', g.Token(1))
    accepts(assembled, v, [1])
    configured = g.Module(source(), optimization=g.Optimization.BALANCED)
    prepared = source().module()
    prepared.precompile(v)
    for leaf in [configured, prepared]:
        expr = g.Expr.choice([leaf])
        with pytest.raises(ValueError):
            g.Grammar.from_expr(expr)
        host = g.Grammar.from_glrm(HOST)
        with pytest.raises(ValueError):
            host.bind('child', expr)
        assert [r.name for r in host.unbound] == ['child']
        accepts(g.Module(expr), v, [1])
    assert g.Grammar.from_expr(g.Expr.sequence([source().module()])).compile(v).start().mask()[1]


def test_self_attachment_and_canonical_introspection():
    v = vocab()
    module = g.Grammar.from_glrm(HOST).module()
    extern = module.unbound[0]
    assert isinstance(extern, g.UnboundExtern)
    assert extern.name == 'child' and extern.path == ['child'] and extern.kind == 'grammar'
    assert extern.source_span is not None
    assert module.bind('child', module) is None
    assert [r.name for r in module.unbound] == ['child::child']
    assert {r.name for r in module.externs} == {'child', 'child::child'}
    module.bind('child::child', source())
    accepts(module, v, [0,4,3])


def test_iterator_reentrancy_and_ownership_after_collection():
    v = vocab()
    leaf = g.Grammar.from_glrm(MARK).module()
    def operands():
        yield leaf
        leaf.bind('MARK', g.Token(8))
        yield source('b')
    expr = g.Expr.sequence(operands())
    module = g.Module(expr)
    assert [r.name for r in module.unbound] == ['leaf0::MARK']
    module.bind('leaf0::MARK', g.Token(1))
    state = module.compile(v).start()
    del module, expr, leaf, v
    gc.collect()
    # Exact events retain token identity and cannot be absorbed into a byte token.
    assert not state.mask()[5]
    state.commit_token(1)
    state.commit_token(2)
    assert state.is_accepting()


def test_standalone_fastbuild_and_explicit_attachment_error():
    v = vocab()
    child = g.Module(source(), optimization=g.Optimization.FAST_BUILD)
    child.precompile(v)
    loaded = g.Module.load(child.save(), v)
    assert loaded.options.optimization == g.Optimization.FAST_BUILD
    accepts(loaded, v, [1])
    parent = g.Grammar.from_glrm(HOST).module()
    parent.bind('child', loaded)
    with pytest.raises(ValueError, match='FastBuild|LR'):
        parent.compile(v)
    assert parent.unbound == []
    assert parent.options.optimization == g.Optimization.AUTO
