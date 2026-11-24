import pytest
import glrmask

def test_error_base_and_syntax():
    assert issubclass(glrmask.GLRMaskError, ValueError)
    with pytest.raises(glrmask.GLRMaskError) as exc:
        glrmask.Grammar.from_glrm("start s nt s = )")
    assert exc.value.kind == "syntax"

def test_invalid_input():
    g = glrmask.Grammar.from_glrm("start s extern token E nt s = E")
    with pytest.raises(glrmask.GLRMaskError) as exc:
        g.bind("missing", 0)
    assert exc.value.kind == "invalid_input"
    with pytest.raises(glrmask.GLRMaskError) as exc:
        g.compile(glrmask.Vocab.from_id_to_bytes({0: b"a"}))
    assert exc.value.kind == "invalid_input"
    g.bind("E", 0)
    with pytest.raises(glrmask.GLRMaskError) as exc:
        g.bind("E", 0)
    assert exc.value.kind == "invalid_input"

def test_wrong_type_and_missing_key():
    g = glrmask.Grammar.from_glrm("start s extern token E nt s = E")
    with pytest.raises(TypeError):
        g.bind("E", object())
    with pytest.raises(KeyError):
        g.externs["missing"]

def test_serialization():
    with pytest.raises(glrmask.GLRMaskError) as exc:
        glrmask.Module.load(b"bad")
    assert exc.value.kind == "serialization"

def test_resource_limit():
    expr = glrmask.Expr.literal("a")
    for _ in range(70):
        expr = glrmask.Expr.sequence([expr])
    with pytest.raises(glrmask.GLRMaskError) as exc:
        glrmask.Grammar.from_expr(expr)
    assert exc.value.kind == "resource_limit"
