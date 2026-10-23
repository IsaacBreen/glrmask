"""Canonical module and expression regressions for v1 API."""
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


def test_compile_options_and_helpers():
    v = vocab()
    module = g.Module(source())
    assert not hasattr(module, "mask_strategy")
    assert not hasattr(module, "boundary_mask_strategy")
    assert not hasattr(g, "ModuleOptions")
    assert not hasattr(g, "CompileOptions")
    assert not hasattr(source(), "mask_strategy")
    assert g.MaskStrategy.FAST_BUILD == g.MaskStrategy.TOKEN_TRIE
    assert g.BoundaryMaskStrategy.FAST_RUNTIME == g.BoundaryMaskStrategy.PARSER_DWA

    assert module.precompile(
        v,
        mask_strategy=g.MaskStrategy.TOKEN_CLASS_TRIE,
        boundary_mask_strategy=g.BoundaryMaskStrategy.PARSER_DWA,
    ) is None
    state = module.compile(
        v,
        mask_strategy=g.MaskStrategy.TOKEN_CLASS_TRIE,
        boundary_mask_strategy=g.BoundaryMaskStrategy.TOKEN_TRIE,
    ).start()
    assert state.mask()[1]
    state.commit_token(1)
    assert state.is_accepting()
    accepts(module, v, [1])


def test_local_precompilation_and_root_only_compile():
    v = vocab()
    child = g.Module(source())
    assert child.precompile(v, mask_strategy=g.MaskStrategy.TOKEN_CLASS_TRIE) is None
    parent = g.Module(g.Grammar.from_glrm(HOST))
    assert parent.bind('child', child) is None
    state = parent.compile(v, end_tokens=[63], mask_strategy=g.MaskStrategy.TOKEN_CLASS_TRIE).start()
    assert not state.mask()[63]
    state.commit_token(4)
    assert state.is_accepting() and state.mask()[63]
    state.commit_token(63)
    assert not state.mask().any()
    for keyword in ['optimization', 'boundary_trigger']:
        with pytest.raises(TypeError):
            parent.compile(v, **{keyword: None})


@pytest.mark.parametrize('copy_fn', [lambda m: m.copy(), copy.copy, copy.deepcopy])
def test_independent_mutable_copies(copy_fn):
    v = vocab()
    original = g.Module(g.Grammar.from_glrm(HOST))
    other = copy_fn(original)
    other.bind('child', source())
    assert list(original.externs.unbound) == ['child']
    assert list(other.externs.unbound) == []
    original.bind('child', source('b'))
    accepts(other, v, [4])
    accepts(original, v, [0, 2, 3])


def test_precompilation_persists_across_save_and_retained_work_preserved():
    v = vocab()
    module = g.Module(source())
    module.precompile(v, mask_strategy=g.MaskStrategy.TOKEN_CLASS_TRIE)
    saved = module.save()
    loaded = g.Module.load(saved, vocab=v)
    accepts(loaded, v, [1])


def test_prepared_parent_accepts_cold_raw_child_after_load():
    v = vocab()
    parent = g.Module(g.Grammar.from_glrm(HOST))
    parent.precompile(v)
    loaded = g.Module.load(parent.save())
    loaded.bind('child', source())
    accepts(loaded, v, [4])


def test_wrong_paths_kinds_duplicates_and_vocab_are_atomic():
    v = vocab()
    parent = g.Module(g.Grammar.from_glrm(HOST))
    parent.precompile(v)
    before = parent.save()
    incompatible = g.Module(source())
    incompatible.precompile(vocab(True))
    for name, child in [('missing', source()), ('child::MARK', g.Token(8)), ('child', incompatible)]:
        with pytest.raises(ValueError):
            parent.bind(name, child)
        assert parent.save() == before
    parent.bind('child', source())
    before = parent.save()
    with pytest.raises(ValueError, match="already bound"):
        parent.bind('child', source('b'))
    assert parent.save() == before
    accepts(parent, v, [4])
    token_parent = g.Module(g.Grammar.from_glrm(MARK))
    with pytest.raises(ValueError):
        token_parent.bind('MARK', source())
    assert list(token_parent.externs.unbound) == ['MARK']
    token_parent.bind('MARK', g.Token(62))
    with pytest.raises(ValueError):
        token_parent.compile(v)


def test_cross_vocab_precompile_load_and_exact_events_preserve_state():
    v = vocab()
    parent = g.Module(source())
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
    target = g.Module(g.Grammar.from_glrm(MARK))
    target.bind('MARK', vocab(True).tokens([8]))
    with pytest.raises(ValueError):
        target.compile(v)
    accepts(parent, v, [1])


def test_expression_union_sequence_and_module_binding():
    v = vocab()
    closed_leaf = g.Module(g.Grammar.from_glrm(MARK))
    closed_leaf.bind('MARK', g.Token(8))
    expr = g.Expr.choice(g.Expr.sequence(source('x'), closed_leaf), source('b'))

    parent = g.Module(g.Grammar.from_glrm('glrm 1; start start; extern grammar E; nt start = E;'))
    parent.bind('E', expr)
    state = parent.compile(v).start()
    assert state.mask()[0] and state.mask()[2]
    state.commit_token(2)
    assert state.is_accepting()

    host = g.Module(g.Grammar.from_glrm(HOST))
    host.bind('child', g.Expr.choice(source(), source('b')))
    assert host.compile(v).start().mask()[4]


def test_expression_whole_token_crosses_boundaries():
    v = vocab()
    expr = g.Expr.sequence(source('x'), g.Expr.sequence(source(), source('y')))
    parent = g.Module(g.Grammar.from_glrm('glrm 1; start start; extern grammar S; nt start = S;'))
    parent.bind('S', expr)
    state = parent.compile(v).start()
    assert state.mask()[4]
    state.commit_token(4)
    assert state.is_accepting()


def test_open_child_rejection_at_bind_and_expr():
    open_leaf = g.Module(g.Grammar.from_glrm(MARK))
    with pytest.raises(ValueError, match="unbound externs"):
        g.Expr.choice(open_leaf)
    with pytest.raises(ValueError, match="unbound externs"):
        g.Expr.sequence(open_leaf)
    parent = g.Module(g.Grammar.from_glrm(HOST))
    with pytest.raises(ValueError, match="unbound externs"):
        parent.bind('child', open_leaf)


def test_self_attachment_rejected_when_open():
    module = g.Module(g.Grammar.from_glrm(HOST))
    with pytest.raises(ValueError, match="unbound externs"):
        module.bind('child', module)
