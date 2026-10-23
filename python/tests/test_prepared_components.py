"""Tests for v1 API rules and semantics."""

from collections.abc import Mapping
import copy
import numpy as np
import pytest
import glrmask

HOST = 'glrm 1; start start; extern grammar child; nt start = "x" child "y";'
TWO_EXTERNS = 'glrm 1; start start; extern token A; extern grammar B; nt start = A B;'
MARK = 'glrm 1; start start; extern token MARK; nt start = MARK;'


def vocab():
    return glrmask.Vocab.from_id_to_bytes({0: b"x", 1: b"a", 2: b"b", 3: b"y", 4: b"xay", 5: b"xb", 63: b""})


def test_atomic_kwargs_bind():
    m = glrmask.Module(glrmask.Grammar.from_glrm(TWO_EXTERNS))
    # Valid atomic keyword bind
    m.bind(A=glrmask.TokenSet([1]), B=glrmask.Grammar.from_ebnf('start ::= "b"'))
    assert len(m.externs.unbound) == 0

    # Failed atomic keyword bind: second argument is invalid
    m2 = glrmask.Module(glrmask.Grammar.from_glrm(TWO_EXTERNS))
    saved = m2.save()
    open_child = glrmask.Grammar.from_glrm("glrm 1; start s; extern token X; nt s = X;")
    with pytest.raises(ValueError, match="unbound externs"):
        m2.bind(A=glrmask.TokenSet([1]), B=open_child)
    # On error, m2 must be completely unchanged; A was not bound!
    assert m2.save() == saved
    assert "A" in m2.externs.unbound
    assert "B" in m2.externs.unbound

    # Mixing positional and keyword arguments is a TypeError
    with pytest.raises(TypeError, match="mix"):
        m2.bind("A", glrmask.TokenSet([1]), B=glrmask.Grammar.from_ebnf('start ::= "b"'))

    # Zero arguments is a TypeError
    with pytest.raises(TypeError):
        m2.bind()


def test_rebinding_error():
    m = glrmask.Module(glrmask.Grammar.from_glrm(TWO_EXTERNS))
    m.bind("A", glrmask.TokenSet([1]))
    with pytest.raises(ValueError, match="already bound"):
        m.bind("A", glrmask.TokenSet([1]))


def test_qualified_paths_rejected():
    m = glrmask.Module(glrmask.Grammar.from_glrm(HOST))
    with pytest.raises(ValueError, match="qualified paths are not supported"):
        m.bind("child::sub", glrmask.Grammar.from_ebnf('start ::= "a"'))


def test_open_child_rejected():
    open_child = glrmask.Module(glrmask.Grammar.from_glrm(MARK))
    parent = glrmask.Module(glrmask.Grammar.from_glrm(HOST))
    # Binding open child to parent raises ValueError
    with pytest.raises(ValueError, match="unbound externs"):
        parent.bind("child", open_child)
    # Expr.choice / sequence with open child raises ValueError immediately
    with pytest.raises(ValueError, match="unbound externs"):
        glrmask.Expr.choice(open_child, glrmask.Grammar.from_ebnf('start ::= "a"'))
    with pytest.raises(ValueError, match="unbound externs"):
        glrmask.Expr.sequence(open_child)


def test_externs_mapping_semantics():
    m = glrmask.Module(glrmask.Grammar.from_glrm(TWO_EXTERNS))
    externs = m.externs
    assert isinstance(externs, Mapping)
    assert len(externs) == 2
    assert "A" in externs
    assert "B" in externs
    assert "C" not in externs

    # Declaration order preserved
    assert list(externs) == ["A", "B"]
    assert externs.keys() == ["A", "B"]

    # Item access
    a = externs["A"]
    assert a.name == "A"
    assert a.kind == "token"

    b = externs["B"]
    assert b.name == "B"
    assert b.kind == "grammar"

    with pytest.raises(KeyError):
        _ = externs["missing"]

    assert externs.get("A").name == "A"
    assert externs.get("missing") is None
    assert externs.get("missing", 42) == 42

    # Unbound view
    assert list(externs.unbound) == ["A", "B"]
    assert isinstance(externs.unbound, Mapping)
    assert externs.unbound.unbound == externs.unbound

    # After binding A, unbound only has B, but externs still has A and B
    m.bind("A", glrmask.TokenSet([1]))
    assert list(m.externs.unbound) == ["B"]
    assert list(m.externs) == ["A", "B"]


def test_copy_independence():
    m = glrmask.Module(glrmask.Grammar.from_glrm(TWO_EXTERNS))
    assert not hasattr(m, "clone")

    copied = copy.copy(m)
    method_copied = m.copy()

    # Binding on copy does not affect original
    copied.bind("A", glrmask.TokenSet([1]))
    assert "A" in m.externs.unbound
    assert "A" not in copied.externs.unbound


def test_compile_options_and_save_load_roundtrip():
    v = vocab()
    g = glrmask.Grammar.from_ebnf('start ::= "a"')
    m = glrmask.Module(g)
    m.precompile(
        v,
        mask_strategy=glrmask.MaskStrategy.PARSER_DWA,
        boundary_mask_strategy=glrmask.BoundaryMaskStrategy.FAST_RUNTIME,
    )

    # Aliases
    assert glrmask.MaskStrategy.FAST_BUILD is glrmask.MaskStrategy.TOKEN_TRIE
    assert glrmask.MaskStrategy.FAST_RUNTIME is glrmask.MaskStrategy.PARSER_DWA

    # Save / load round-trip preserves retained work; options are supplied per compile.
    saved = m.save()
    loaded = glrmask.Module.load(saved)
    assert loaded.compile(
        v,
        mask_strategy=glrmask.MaskStrategy.TOKEN_TRIE,
        boundary_mask_strategy=glrmask.BoundaryMaskStrategy.FAST_RUNTIME,
    ).start().mask().tolist() == m.compile(v).start().mask().tolist()


def test_expr_literal_str_rejection_and_empty_choice():
    v = vocab()
    # Expr.literal accepts str and bytes
    lit_str = glrmask.Expr.literal("x")
    lit_bytes = glrmask.Expr.literal(b"x")
    assert isinstance(lit_str, glrmask.Expr)
    assert isinstance(lit_bytes, glrmask.Expr)

    # Plain str in Expr.choice / sequence raises TypeError mentioning Expr.literal
    with pytest.raises(TypeError, match="Expr.literal"):
        glrmask.Expr.choice("x")
    with pytest.raises(TypeError, match="Expr.literal"):
        glrmask.Expr.sequence("x")

    # Empty choice matches nothing, does not error
    empty_choice = glrmask.Expr.choice()
    assert isinstance(empty_choice, glrmask.Expr)

    # Empty sequence matches empty string
    empty_seq = glrmask.Expr.sequence()
    assert isinstance(empty_seq, glrmask.Expr)


def test_grammar_compile_sugar():
    v = vocab()
    g = glrmask.Grammar.from_ebnf('start ::= "a"')
    sugar_constraint = g.compile(
        v,
        mask_strategy=glrmask.MaskStrategy.TOKEN_CLASS_TRIE,
        boundary_mask_strategy=glrmask.BoundaryMaskStrategy.FAST_BUILD,
    )
    module_constraint = glrmask.Module(g).compile(
        v,
        mask_strategy=glrmask.MaskStrategy.TOKEN_CLASS_TRIE,
        boundary_mask_strategy=glrmask.BoundaryMaskStrategy.FAST_BUILD,
    )
    assert np.array_equal(sugar_constraint.start().mask(), module_constraint.start().mask())

    # Grammar is source-only
    assert not hasattr(g, "bind")
    assert not hasattr(g, "prepare")
    assert not hasattr(g, "module")
    assert not hasattr(g, "unbound")
    assert not hasattr(g, "externs")
    assert not hasattr(g, "clone")
    assert not hasattr(g, "exact")
