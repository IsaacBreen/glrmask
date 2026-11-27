"""Canonical module and expression regressions for v1 API."""
import copy
import gc

import glrmask as g
from mask_helpers import token_allowed, unpack_mask

import pytest

HOST = 'glrm 1; start start; extern grammar child; nt start = "x" child "y";'
MARK = 'glrm 1; start start; extern token MARK; nt start = MARK;'


def vocab(changed=False):
    return g.Vocab({0:b'x',1:b'a',2:b'b',3:b'y',4:b'xay',5:b'ab',6:b'xa',7:b'ay',8:b'z' if changed else b'a',63:b''})


def source(text='a'):
    return g.Grammar.from_lark(f'start: "{text}"')


def accepts(module, v, ids):
    state = module.compile(v).start()
    for token in ids:
        assert token_allowed(state.mask(), token)
        assert state.commit_token(token)
    assert state.is_accepting()


def test_compile_options_and_helpers():
    v = vocab()
    module = g.Module(source())
    assert not hasattr(module, "mask_strategy")
    assert not hasattr(module, "boundary_mask_strategy")
    assert not hasattr(g, "ModuleOptions")
    assert not hasattr(g, "CompileOptions")
    assert not hasattr(source(), "mask_strategy")
    assert g.MaskStrategy.FAST_COMPILE != g.MaskStrategy.FAST_MASKS
    assert not hasattr(g, "BoundaryMaskStrategy")

    assert module.precompile(
        v,
        mask_strategy=g.MaskStrategy.FAST_MASKS,
    ) is None
    state = module.compile(
        v,
        mask_strategy=g.MaskStrategy.FAST_MASKS,
    ).start()
    assert token_allowed(state.mask(), 1)
    assert state.commit_token(1)
    assert state.is_accepting()
    accepts(module, v, [1])


def test_local_precompilation_and_root_only_compile():
    v = vocab()
    child = g.Module(source())
    assert child.precompile(v, mask_strategy=g.MaskStrategy.FAST_MASKS) is None
    parent = g.Module(g.Grammar.from_glrm(HOST))
    assert parent.bind('child', child) is None
    root = g.Module(g.Grammar.from_glrm("start s extern grammar BODY extern token END nt s = BODY END"))
    root.bind('BODY', parent)
    root.bind('END', g.TokenSet([63]))
    state = root.compile(v, mask_strategy=g.MaskStrategy.FAST_MASKS).start()
    assert not token_allowed(state.mask(), 63)
    assert state.commit_token(4)
    assert not state.is_accepting() and token_allowed(state.mask(), 63)
    assert state.commit_token(63)
    assert state.is_accepting() and not state.mask().any()
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
    module.precompile(v, mask_strategy=g.MaskStrategy.FAST_MASKS)
    saved = module.save()
    loaded = g.Module.load(saved)
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
    for name, child in [('missing', source()), ('child::MARK', g.TokenSet([8])), ('child', incompatible)]:
        with pytest.raises(g.GLRMaskError):
            parent.bind(name, child)
        assert parent.save() == before
    parent.bind('child', source())
    before = parent.save()
    with pytest.raises(g.GLRMaskError, match="already bound"):
        parent.bind('child', source('b'))
    assert parent.save() == before
    accepts(parent, v, [4])
    token_parent = g.Module(g.Grammar.from_glrm(MARK))
    with pytest.raises(g.GLRMaskError):
        token_parent.bind('MARK', source())
    assert list(token_parent.externs.unbound) == ['MARK']
    token_parent.bind('MARK', g.TokenSet([62]))
    assert token_parent.compile(v) is not None


def test_module_precompile_retains_work():
    v = vocab()

    # A grammar precompiles into a bindable Module, including while its own
    # externs are still open.
    open_grammar = g.Grammar.from_glrm(HOST)
    assert not hasattr(open_grammar, "precompile")
    module = g.Module(open_grammar)
    module.precompile(v)
    child = g.Module(source())
    child.precompile(v, mask_strategy=g.MaskStrategy.FAST_MASKS)
    module.bind('child', child)
    state = module.compile(v).start()
    assert token_allowed(state.mask(), 4)
    assert state.commit_token(4)
    assert state.is_accepting()

    # Masks equal the explicit Module(grammar) + bind + precompile path.
    reference = g.Module(g.Grammar.from_glrm(HOST))
    reference.bind('child', g.Module(source()))
    reference.precompile(v)
    assert list(module.compile(v).start().mask()) == list(reference.compile(v).start().mask())
    assert module.compile(v).start().is_accepting() == reference.compile(v).start().is_accepting()

    # The returned Module retains work: its artifact carries the static
    # preparation, survives a round-trip, and a TokenTrie compile keeps it.
    retained = g.Module(source())
    retained.precompile(v, mask_strategy=g.MaskStrategy.FAST_MASKS)
    assert len(retained.save()) > len(g.Module(source()).save())
    restored = g.Module.load(retained.save())
    expected = list(g.Module(source()).compile(v).start().mask())
    assert list(restored.compile(v, mask_strategy=g.MaskStrategy.FAST_COMPILE).start().mask()) == expected


def test_cross_vocab_precompile_load_and_exact_events_preserve_state():
    v = vocab()
    parent = g.Module(source())
    parent.precompile(v)
    before = parent.save()
    with pytest.raises(g.GLRMaskError):
        parent.precompile(vocab(True))
    assert parent.save() == before
    assert g.Module.load(before) is not None
    for data in [b'', b'GLRASM01bad', before[:-1]]:
        with pytest.raises(g.GLRMaskError):
            g.Module.load(data)
    target = g.Module(g.Grammar.from_glrm(MARK))
    target.bind('MARK', g.TokenSet([99]))
    assert target.compile(v) is not None
    accepts(parent, v, [1])


def test_expression_union_sequence_and_module_binding():
    v = vocab()
    closed_leaf = g.Module(g.Grammar.from_glrm(MARK))
    closed_leaf.bind('MARK', g.TokenSet([8]))
    expr = g.Expr.choice(g.Expr.sequence(source('x'), closed_leaf), source('b'))

    parent = g.Module(g.Grammar.from_glrm('glrm 1; start start; extern grammar E; nt start = E;'))
    parent.bind('E', expr)
    state = parent.compile(v).start()
    assert token_allowed(state.mask(), 0) and token_allowed(state.mask(), 2)
    assert state.commit_token(2)
    assert state.is_accepting()

    host = g.Module(g.Grammar.from_glrm(HOST))
    host.bind('child', g.Expr.choice(source(), source('b')))
    assert token_allowed(host.compile(v).start().mask(), 4)


def test_expression_whole_token_crosses_boundaries():
    v = vocab()
    expr = g.Expr.sequence(source('x'), g.Expr.sequence(source(), source('y')))
    parent = g.Module(g.Grammar.from_glrm('glrm 1; start start; extern grammar S; nt start = S;'))
    parent.bind('S', expr)
    state = parent.compile(v).start()
    assert token_allowed(state.mask(), 4)
    assert state.commit_token(4)
    assert state.is_accepting()


def test_open_child_rejection_at_bind_and_expr():
    open_leaf = g.Module(g.Grammar.from_glrm(MARK))
    with pytest.raises(g.GLRMaskError, match="unbound externs"):
        g.Expr.choice(open_leaf)
    with pytest.raises(g.GLRMaskError, match="unbound externs"):
        g.Expr.sequence(open_leaf)
    parent = g.Module(g.Grammar.from_glrm(HOST))
    with pytest.raises(g.GLRMaskError, match="unbound externs"):
        parent.bind('child', open_leaf)


def test_self_attachment_rejected_when_open():
    module = g.Module(g.Grammar.from_glrm(HOST))
    with pytest.raises(g.GLRMaskError, match="unbound externs"):
        module.bind('child', module)

def test_externs_are_snapshots():
    grammar = g.Grammar.from_glrm(HOST)
    before = grammar.externs
    del grammar
    assert "child" in before.unbound
    for owner in [g.Module(g.Grammar.from_glrm(HOST))]:
        before = owner.externs
        assert "child" in before.unbound
        assert before == owner.externs
        assert before == before.unbound
        assert before.unbound == before.unbound.unbound
        assert "unbound=['child']" in repr(before)
        assert before["child"] == owner.externs["child"]
        with pytest.raises(TypeError):
            hash(before)
        owner.bind("child", source())
        after = owner.externs
        assert "child" in before.unbound
        assert "child" not in after.unbound
        assert list(before) == list(after)
        assert before != after
        assert after == owner.externs
        assert "unbound=[]" in repr(after)
        assert after["child"] == before["child"]

def test_stronger_preparation_survives_fast_compile():
    v = vocab()
    child = g.Module(source())
    child.precompile(v, mask_strategy=g.MaskStrategy.FAST_MASKS)
    before = child.save()
    child.precompile(v, mask_strategy=g.MaskStrategy.FAST_COMPILE)
    assert child.save() == before
    parent = g.Module(g.Grammar.from_glrm(HOST))
    parent.bind("child", child)
    actual = parent.compile(v, mask_strategy=g.MaskStrategy.FAST_COMPILE)
    fresh = g.Module(g.Grammar.from_glrm(HOST))
    fresh.bind("child", source())
    expected = fresh.compile(v, mask_strategy=g.MaskStrategy.FAST_MASKS)
    for ids in [[], [0], [0, 1], [0, 1, 3], [4]]:
        a, b = actual.start(), expected.start()
        for token_id in ids:
            assert a.commit_token(token_id) and b.commit_token(token_id)
        assert list(a.mask()) == list(b.mask())
        assert a.is_accepting() == b.is_accepting()

def test_externs_equality_uses_name_order_and_kind():
    def externs(first, second, kind="grammar"):
        return g.Grammar.from_glrm(
            f"extern {kind} {first} extern grammar {second} start r nt r={first}"
        ).externs
    original = externs("A", "B")
    assert original == externs("A", "B")
    assert original == externs("B", "A")
    assert list(original) == ["A", "B"]
    assert original != externs("A", "C")
    assert original != externs("A", "B", "token")
    assert original != {"A": original["A"], "B": original["B"]}
