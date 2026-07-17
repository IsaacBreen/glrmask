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

old_cache = '''        if can_use_merged_cache {
            if reuse_existing_cache_dense {
                if eos_unchanged {
                    self.touch_mask_cache_generation();
                } else {
                    self.store_mask_cache_reuse_dense(buf);
                }
            } else {
                self.store_mask_cache(buf, &merged);
            }
        } else {
            self.store_mask_cache_reuse_dense(buf);
        }
'''
new_cache = '''        if can_use_merged_cache {
            if reuse_existing_cache_dense {
                self.touch_mask_cache_generation();
            } else {
                self.store_mask_cache(buf, &merged);
            }
        } else {
            self.store_mask_cache_reuse_dense(buf);
        }
'''
if old_cache in text:
    text = text.replace(old_cache, new_cache, 1)
elif "eos_unchanged" in text:
    raise RuntimeError("unexpected eos_unchanged cache layout")
path.write_text(text)

wasm_path = ROOT / "glrmask-wasm/src/lib.rs"
wasm = wasm_path.read_text()
old_wasm = '''/// Whether end-of-sequence is currently grammatically admissible.
#[unsafe(no_mangle)]
pub extern "C" fn glrmask_eos_allowed(handle: u32) -> u32 {
    with_session_mut(handle, |session| u32::from(session.inner.eos_allowed())).unwrap_or(0)
}

'''
if old_wasm in wasm:
    wasm = wasm.replace(old_wasm, "", 1)
elif "eos_allowed" in wasm:
    raise RuntimeError("unexpected WASM eos_allowed layout")
wasm_path.write_text(wasm)

for forbidden in (
    "eos_token_id",
    "EOS_BYTES",
    "update_eos_mask",
    "finalize_eos_ns",
    "eos_allowed",
):
    hits = []
    for rust in ROOT.rglob("*.rs"):
        if forbidden in rust.read_text():
            hits.append(str(rust.relative_to(ROOT)))
    if hits:
        raise RuntimeError(f"forbidden {forbidden} remains in {hits}")
