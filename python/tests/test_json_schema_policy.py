"""Exact-first acceptance policy through the public Python facade."""
import json
import pytest
import glrmask


def compile_schema(schema):
    vocab = glrmask.Vocab.from_id_to_bytes({b: bytes([b]) for b in range(256)})
    return glrmask.Grammar.from_json_schema(schema).compile(vocab)


def accepts(constraint, value):
    state = constraint.start()
    for byte in json.dumps(value, ensure_ascii=False).encode():
        if not state.mask()[byte]:
            return False
        state.commit_token(byte)
    return state.is_accepting()


@pytest.mark.parametrize("schema", [
    {"oneOf": [{"type": "integer"}, {"type": "number"}]},
    {"not": {"uniqueItems": True}},
    {"if": {"type": "integer"}, "then": False},
    {"unevaluatedProperties": False},
    {"$vocabulary": {"https://example.invalid/vocab": True}},
])
def test_unsupported_assertions_error_instead_of_broadening(schema):
    with pytest.raises(ValueError, match="unsupported|Unimplemented"):
        compile_schema(schema)


def test_disjoint_oneof_and_annotation_formats():
    c = compile_schema({"oneOf": [{"type": "string", "format": "email"}, {"type": "integer"}]})
    assert accepts(c, "not an email")
    assert accepts(c, 1)
    assert not accepts(c, False)


def test_metadata_is_opaque_and_constant_conditionals_exact():
    c = compile_schema({"if": True, "then": {"type": "integer", "minimum": 2}, "default": {"not": True}})
    assert accepts(c, 2)
    assert not accepts(c, 1)


def test_removed_approximation_option_errors(monkeypatch):
    monkeypatch.setenv("GLRMASK_JSON_SCHEMA_COERCE_ONE_OF_TO_ANY_OF", "1")
    with pytest.raises(ValueError, match="approximation has been removed"):
        compile_schema({"type": "string"})


def test_allof_boolean_identity_and_string_type_guards():
    c = compile_schema({"allOf": [{"type": "object"}, True]})
    assert accepts(c, {})
    assert not accepts(c, 1)
    c = compile_schema({"allOf": [{"type": "string"}, {"minLength": 2}]})
    assert accepts(c, "ab")
    assert not accepts(c, 1)
    assert not accepts(c, "a")
