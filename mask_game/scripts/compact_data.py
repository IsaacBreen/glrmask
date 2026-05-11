#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import gzip
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    if path.suffix == ".gz":
        f = gzip.open(path, "wt", encoding="utf-8", compresslevel=9)
    else:
        f = open(path, "wt", encoding="utf-8")
    with f:
        json.dump(payload, f, separators=(",", ":"))


def decode_fixed_ids(encoded: str) -> list[int]:
    raw = base64.b64decode(encoded)
    if len(raw) % 4:
        raise ValueError(f"fixed internal_ids_b64 byte length {len(raw)} is not divisible by 4")
    return [int.from_bytes(raw[i : i + 4], "little") for i in range(0, len(raw), 4)]


def decode_varint_ids(encoded: str) -> list[int]:
    raw = base64.b64decode(encoded)
    ids: list[int] = []
    offset = 0
    prev = 0
    while offset < len(raw):
        value = 0
        shift = 0
        while True:
            if offset >= len(raw):
                raise ValueError("truncated internal_ids_vb64 varint")
            byte = raw[offset]
            offset += 1
            value |= (byte & 0x7F) << shift
            if byte & 0x80 == 0:
                break
            shift += 7
            if shift >= 32:
                raise ValueError("internal_ids_vb64 varint is too large")
        prev += value
        ids.append(prev)
    return ids


def encode_varint_ids(internal_ids: list[int]) -> str:
    out = bytearray()
    prev = 0
    for internal in internal_ids:
        internal = int(internal)
        if internal < prev:
            raise ValueError("internal ids must be sorted before delta-varint encoding")
        value = internal - prev
        prev = internal
        while value >= 0x80:
            out.append((value & 0x7F) | 0x80)
            value >>= 7
        out.append(value)
    return base64.b64encode(out).decode("ascii")


def internal_ids_for_case(case: dict[str, Any]) -> list[int]:
    if "internal_ids" in case:
        return [int(x) for x in case["internal_ids"]]
    if "internal_ids_vb64" in case:
        return decode_varint_ids(case["internal_ids_vb64"])
    if "internal_ids_b64" in case:
        return decode_fixed_ids(case["internal_ids_b64"])
    return []


def compact_case(case: dict[str, Any]) -> dict[str, Any]:
    compact = dict(case)
    compact["internal_ids_vb64"] = encode_varint_ids(internal_ids_for_case(case))
    compact.pop("internal_ids", None)
    compact.pop("internal_ids_b64", None)
    compact.pop("internal_dense_words", None)
    return compact


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    payload = read_json(args.input)
    payload["cases"] = [compact_case(case) for case in payload.get("cases", [])]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, payload)

    before = args.input.stat().st_size
    after = args.output.stat().st_size
    print(f"wrote {args.output} ({after} bytes; input {before} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
