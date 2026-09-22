from __future__ import annotations

import numpy as np

import glrmask
import glrmask.auto as auto


def test_auto_policy_selects_all_three_tiers() -> None:
    o1 = {
        "type": "object",
        "properties": {
            "x": {"type": "number"},
            "y": {"type": "number"},
            "z": {"type": "number"},
        },
    }
    o2 = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "count": {"type": "integer"}},
    }
    o3 = {
        "type": "object",
        "properties": {
            f"field_{i}": {"type": "string", "maxLength": 64}
            for i in range(60)
        },
    }
    assert auto.select_json_schema_auto_tier(o1) == "o1"
    assert auto.select_json_schema_auto_tier(o2) == "o2"
    assert auto.select_json_schema_auto_tier(o3) == "o3"


def test_auto_constraint_roundtrip_preserves_selected_tier() -> None:
    vocab = glrmask.Vocab.from_dict({b"{": 0, b"}": 1, b"0": 2, b"1": 3})
    schema = {
        "type": "object",
        "properties": {
            "x": {"type": "number"},
            "y": {"type": "number"},
            "z": {"type": "number"},
        },
    }
    constraint = auto.AutoConstraint.from_json_schema(schema, vocab)
    assert constraint.selected_tier == "o1"
    restored = auto.AutoConstraint.load(constraint.save(), vocab)
    assert restored.selected_tier == constraint.selected_tier
    assert restored.mask_len() == constraint.mask_len()

    # The auto wrapper returns the native state type directly.
    state = restored.start()
    mask = np.zeros(restored.mask_len(), dtype=np.int32)
    state.fill_mask(mask)


def test_auto_shape_uses_fit_corpus_schema_normalization() -> None:
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "definitions": {"n": {"type": "number"}},
        "$ref": "#/definitions/n",
        # Validation siblings of $ref are ignored by draft-07. The policy must
        # not count this deliberately huge ignored subtree when choosing a tier.
        "properties": {
            f"ignored_{i}": {"type": "string", "maxLength": 4096}
            for i in range(200)
        },
    }
    shape = auto.json_schema_auto_shape(schema)
    assert shape.properties == 0
    assert shape.max_length == 0


def test_auto_selector_short_circuits_obvious_o3_without_normalized_copy(monkeypatch) -> None:
    schema = {
        "type": "object",
        "properties": {
            f"field_{i}": {"type": "string", "maxLength": 64}
            for i in range(200)
        },
    }

    def fail_if_called(_schema):
        raise AssertionError("obvious O3 selection should not normalize/copy the full schema")

    monkeypatch.setattr(auto, "_normalize_auto_schema", fail_if_called)
    assert auto.select_json_schema_auto_tier(schema) == "o3"
