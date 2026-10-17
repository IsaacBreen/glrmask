"Adaptive JSON Schema policy inspection and persisted tier selection. For ordinary compilation, prefer `Grammar.compile(..., optimization=Optimization.AUTO)`. Policy thresholds are empirical choices, not latency guarantees. All tiers inherit the JSON Schema importer\u2019s partial assertion support."

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ._glrmask import Constraint, DynamicConstraint, Grammar, _internal as _native_internal


AUTO_POLICY_NAME = "tail200-v1"
_AUTO_MAGIC = b"GLRMAUTO"
_TIER_TO_TAG = {"o1": b"\x01", "o2": b"\x02", "o3": b"\x03"}
_TAG_TO_TIER = {tag: tier for tier, tag in _TIER_TO_TAG.items()}

_LEGACY_REF_SIBLING_DRAFT_MARKERS = (
    "draft-03",
    "draft-04",
    "draft-06",
    "draft-07",
)
_REF_PASSTHROUGH_KEYS = frozenset(
    {
        "$anchor",
        "$comment",
        "$defs",
        "$dynamicAnchor",
        "$id",
        "$ref",
        "$schema",
        "default",
        "definitions",
        "deprecated",
        "description",
        "examples",
        "id",
        "readOnly",
        "title",
        "writeOnly",
    }
)


@dataclass(frozen=True)
class AutoSchemaShape:
    "Source-shape counts used by the policy, not a schema validation result. `json_bytes` measures normalized JSON serialized with sorted keys and compact separators."
    nodes: int = 0
    dicts: int = 0
    lists: int = 0
    leaves: int = 0
    properties: int = 0
    property_name_chars: int = 0
    string_types: int = 0
    number_types: int = 0
    array_types: int = 0
    pattern_chars: int = 0
    max_length: int = 0
    json_bytes: int = 0


def _normalize_auto_schema(schema: Any) -> Any:
    """Mirror the semantics-preserving normalization used by the CFA fit corpus."""

    def normalize_legacy_ref_siblings(node: Any) -> Any:
        if isinstance(node, dict):
            ref_value = node.get("$ref")
            if isinstance(ref_value, str):
                normalized = {"$ref": ref_value}
                for key in _REF_PASSTHROUGH_KEYS:
                    if key == "$ref" or key not in node:
                        continue
                    normalized[key] = normalize_legacy_ref_siblings(node[key])
                return normalized
            return {key: normalize_legacy_ref_siblings(value) for key, value in node.items()}
        if isinstance(node, list):
            return [normalize_legacy_ref_siblings(value) for value in node]
        return node

    if isinstance(schema, str):
        schema = json.loads(schema)

    schema_uri = schema.get("$schema") if isinstance(schema, dict) else None
    if isinstance(schema_uri, str) and any(
        marker in schema_uri.lower() for marker in _LEGACY_REF_SIBLING_DRAFT_MARKERS
    ):
        schema = normalize_legacy_ref_siblings(schema)

    def normalize_empty_properties(node: Any) -> Any:
        if isinstance(node, dict):
            normalized = {
                key: normalize_empty_properties(value) for key, value in node.items()
            }
            properties = normalized.get("properties")
            pattern_properties = normalized.get("patternProperties")
            if (
                isinstance(properties, dict)
                and not properties
                and isinstance(pattern_properties, dict)
                and pattern_properties
            ):
                normalized.pop("properties", None)
            return normalized
        if isinstance(node, list):
            return [normalize_empty_properties(value) for value in node]
        return node

    return normalize_empty_properties(schema)


def json_schema_auto_shape(schema: Any) -> AutoSchemaShape:
    "Extract the policy’s source-shape features from JSON text or a JSON-serializable schema."
    schema_obj = json.loads(schema) if isinstance(schema, str) else schema
    schema_text = schema if isinstance(schema, str) else json.dumps(schema, separators=(",", ":"))
    native = _native_internal.auto_json_schema_shape_counts(schema_text)
    counts = dict(
        zip(
            (
                "nodes",
                "dicts",
                "lists",
                "leaves",
                "properties",
                "property_name_chars",
                "string_types",
                "number_types",
                "array_types",
                "pattern_chars",
                "max_length",
            ),
            native,
            strict=True,
        )
    )
    normalized = _normalize_auto_schema(schema_obj)
    counts["json_bytes"] = len(
        json.dumps(normalized, separators=(",", ":"), sort_keys=True)
    )
    return AutoSchemaShape(**counts)


def _select_json_schema_auto_tier_from_text(schema_obj: Any, schema_text: str) -> str:
    native = _native_internal.auto_json_schema_shape_counts(schema_text)
    shape = AutoSchemaShape(
        nodes=native[0],
        dicts=native[1],
        lists=native[2],
        leaves=native[3],
        properties=native[4],
        property_name_chars=native[5],
        string_types=native[6],
        number_types=native[7],
        array_types=native[8],
        pattern_chars=native[9],
        max_length=native[10],
    )

    # O1's conservative safe region collapsed to this single conjunction.
    if (
        shape.property_name_chars <= 86
        and shape.pattern_chars <= 5
        and shape.number_types >= 3
        and shape.nodes <= 43
    ):
        return "o1"

    if shape.leaves <= 106:
        if shape.dicts < 23 or shape.lists > 4 or shape.string_types > 12:
            return "o2"
        # This is the only policy branch that depends on serialized normalized
        # size. It is restricted to tiny-leaf schemas, so keep the exact Python
        # normalization/escaping semantics used to fit the policy instead of
        # approximating them in native code.
        normalized = _normalize_auto_schema(schema_obj)
        json_bytes = len(
            json.dumps(normalized, separators=(",", ":"), sort_keys=True)
        )
        return "o3" if json_bytes <= 2041 else "o2"

    return "o3" if (
        shape.properties >= 120
        or shape.max_length >= 27
        or (shape.pattern_chars >= 29 and shape.array_types >= 8)
    ) else "o2"


def select_json_schema_auto_tier(schema: Any) -> str:
    "Select `o1`, `o2` or `o3` using fitted source-shape thresholds. The choice predicts a compilation/masking trade-off; it does not guarantee a latency bound."

    if isinstance(schema, str):
        schema_text = schema
        schema_obj = json.loads(schema)
    else:
        schema_obj = schema
        schema_text = json.dumps(schema, separators=(",", ":"))
    return _select_json_schema_auto_tier_from_text(schema_obj, schema_text)


class AutoConstraint:
    "Constraint wrapper that records the selected tier and policy. Its artifact carries the tier so loading does not rerun selection. JSON Schema assertions unsupported by the importer may remain unenforced."

    def __init__(self, inner: Any, selected_tier: str):
        "Use `from_json_schema` for normal construction; unknown tier names raise `ValueError`."
        if selected_tier not in _TIER_TO_TAG:
            raise ValueError(f"unknown auto tier {selected_tier!r}")
        self._inner = inner
        self.selected_tier = selected_tier
        self.policy = AUTO_POLICY_NAME

    @classmethod
    def from_json_schema(cls, schema: str | dict[str, Any], vocab: Any) -> "AutoConstraint":
        "Choose a tier and compile. Accepts schema text or an object; inherits the importer’s partial JSON Schema support."
        schema_obj = json.loads(schema) if isinstance(schema, str) else schema
        schema_text = schema if isinstance(schema, str) else json.dumps(schema)
        tier = _select_json_schema_auto_tier_from_text(schema_obj, schema_text)
        if tier == "o1":
            inner = DynamicConstraint.from_json_schema(schema_text, vocab, vocab_partition=False)
        elif tier == "o2":
            inner = DynamicConstraint.from_json_schema(schema_text, vocab, vocab_partition=True)
        else:
            inner = Grammar.from_json_schema(schema_text).compile(vocab)
        return cls(inner, tier)

    @classmethod
    def load(cls, data: bytes, vocab: Any) -> "AutoConstraint":
        "Restore the recorded tier. Rejects missing or unknown artifact tags; the supplied vocabulary must match the original."
        if not data.startswith(_AUTO_MAGIC) or len(data) <= len(_AUTO_MAGIC):
            raise ValueError("not a GLRMask AutoConstraint artifact")
        tag = data[len(_AUTO_MAGIC) : len(_AUTO_MAGIC) + 1]
        try:
            tier = _TAG_TO_TIER[tag]
        except KeyError as exc:
            raise ValueError(f"unknown AutoConstraint tier tag {tag!r}") from exc
        payload = data[len(_AUTO_MAGIC) + 1 :]
        if tier in {"o1", "o2"}:
            inner = DynamicConstraint.load(payload, vocab)
        else:
            inner = Constraint.load(payload, vocab)
        return cls(inner, tier)

    def save(self) -> bytes:
        "Includes the tier tag and underlying constraint artifact."
        return _AUTO_MAGIC + _TIER_TO_TAG[self.selected_tier] + bytes(self._inner.save())

    def start(self) -> Any:
        # ConstraintState and DynamicConstraintState intentionally expose the
        # same decoding surface, so callers need no auto-specific state type.
        "Independent decoding state with the same mask/commit interface as `ConstraintState`."
        return self._inner.start()

    def mask_len(self) -> int:
        "Packed 32-bit word count for a mask buffer."
        return int(self._inner.mask_len())


__all__ = [
    "AUTO_POLICY_NAME",
    "AutoConstraint",
    "AutoSchemaShape",
    "json_schema_auto_shape",
    "select_json_schema_auto_tier",
]
