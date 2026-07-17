from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def remove_function(text: str, name: str) -> str:
    marker = f"fn {name}("
    start = text.find(marker)
    if start < 0:
        return text
    line_start = text.rfind("\n", 0, start) + 1
    brace = text.find("{", start)
    if brace < 0:
        raise RuntimeError(f"missing body for {name}")
    depth = 0
    i = brace
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                while end < len(text) and text[end] == "\n":
                    end += 1
                return text[:line_start] + text[end:]
        i += 1
    raise RuntimeError(f"unterminated body for {name}")


def remove_call_statements(text: str, name: str) -> str:
    marker = f"{name}("
    while True:
        start = text.find(marker)
        if start < 0:
            return text
        line_start = text.rfind("\n", 0, start) + 1
        opening = text.find("(", start)
        depth = 0
        i = opening
        while i < len(text):
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    while end < len(text) and text[end] in " \t":
                        end += 1
                    if end >= len(text) or text[end] != ";":
                        raise RuntimeError(f"{name} call is not a statement")
                    end += 1
                    while end < len(text) and text[end] == "\n":
                        end += 1
                    text = text[:line_start] + text[end:]
                    break
            i += 1
        else:
            raise RuntimeError(f"unterminated call to {name}")


path = ROOT / "src/runtime/mask/mod.rs"
text = path.read_text()
for name in ("update_eos_mask", "eos_mask_bit"):
    text = remove_function(text, name)
text = remove_call_statements(text, "update_eos_mask")
start = text.find("        let eos_is_special = self")
if start >= 0:
    end = text.find("        let cache_start", start)
    if end < 0:
        raise RuntimeError("EOS finalization block has no cache boundary")
    text = text[:start] + text[end:]
path.write_text(text)

for forbidden in ("eos_token_id", "EOS_BYTES", "update_eos_mask", "finalize_eos_ns"):
    hits = []
    for rust in ROOT.rglob("*.rs"):
        if forbidden in rust.read_text():
            hits.append(str(rust.relative_to(ROOT)))
    if hits:
        raise RuntimeError(f"forbidden {forbidden} remains in {hits}")
