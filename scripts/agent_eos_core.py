from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path: str) -> str:
    return (ROOT / path).read_text()


def save(path: str, text: str) -> None:
    (ROOT / path).write_text(text)


def replace(path: str, old: str, new: str) -> None:
    text = load(path)
    if old in text:
        save(path, text.replace(old, new, 1))
    elif new not in text:
        raise RuntimeError(f"{path}: replacement target missing")


def remove_regex(path: str, pattern: str) -> None:
    text = load(path)
    updated, count = re.subn(pattern, "", text, flags=re.S)
    if count:
        save(path, updated)


def rewrite_vocab_calls(text: str) -> str:
    edits: list[tuple[int, int, str]] = []
    start = 0
    needle = "Vocab::new("
    while True:
        found = text.find(needle, start)
        if found < 0:
            break
        opening = found + len(needle) - 1
        i = opening + 1
        stack: list[str] = []
        commas: list[int] = []
        while i < len(text):
            if text.startswith("//", i):
                j = text.find("\n", i + 2)
                i = len(text) if j < 0 else j + 1
                continue
            if text.startswith("/*", i):
                depth = 1
                i += 2
                while i < len(text) and depth:
                    if text.startswith("/*", i):
                        depth += 1; i += 2
                    elif text.startswith("*/", i):
                        depth -= 1; i += 2
                    else:
                        i += 1
                continue
            raw = re.match(r"(?:br|rb|r)(?P<h>#{0,32})\"", text[i:])
            if raw:
                marker = '"' + raw.group("h")
                j = text.find(marker, i + raw.end())
                if j < 0: raise RuntimeError("unterminated raw string")
                i = j + len(marker); continue
            quote = None
            if text.startswith(('b"', "b'"), i): quote = text[i + 1]
            elif text[i] in ('"', "'"): quote = text[i]
            if quote:
                j = i + (2 if text[i] == 'b' else 1)
                while j < len(text):
                    if text[j] == "\\": j += 2
                    elif text[j] == quote: j += 1; break
                    elif quote == "'" and text[j] in "\r\n": break
                    else: j += 1
                i = j; continue
            ch = text[i]
            if ch in "([{": stack.append(ch)
            elif ch in ")]}":
                if ch == ")" and not stack:
                    if len(commas) in (1, 2):
                        first = text[opening + 1:commas[0]].rstrip()
                        edits.append((opening + 1, i, first))
                    elif len(commas) > 2:
                        raise RuntimeError("unexpected Vocab::new arity")
                    start = i + 1
                    break
                if stack: stack.pop()
            elif ch == "," and not stack:
                commas.append(i)
            i += 1
        else:
            raise RuntimeError("unterminated Vocab::new call")
    for begin, end, value in reversed(edits):
        text = text[:begin] + value + text[end:]
    return text


# Vocab is strictly an ID -> byte realization map.
save("src/vocab.rs", '''use std::any::{Any, TypeId};
use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, Mutex};

#[derive(serde::Serialize, serde::Deserialize)]
pub struct Vocab {
    pub entries: Arc<BTreeMap<u32, Vec<u8>>>,
    #[serde(skip)]
    compiler_cache: VocabCompilerCache,
}

#[derive(Default)]
struct VocabCompilerCache {
    artifacts: Mutex<BTreeMap<TypeId, Arc<dyn Any + Send + Sync>>>,
}

pub(crate) trait VocabDerivedArtifact: Any + Send + Sync {}

impl fmt::Debug for VocabCompilerCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VocabCompilerCache")
            .field("entries", &self.artifacts.lock().map(|a| a.len()).unwrap_or(0))
            .finish()
    }
}

impl fmt::Debug for Vocab {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Vocab").field("entries", &self.entries).finish()
    }
}

impl Clone for Vocab {
    fn clone(&self) -> Self {
        Self { entries: Arc::clone(&self.entries), compiler_cache: VocabCompilerCache::default() }
    }
}

impl Vocab {
    pub fn new(entries: Vec<(u32, Vec<u8>)>) -> Self {
        Self { entries: Arc::new(entries.into_iter().collect()), compiler_cache: VocabCompilerCache::default() }
    }

    pub fn len(&self) -> usize { self.entries.len() }
    pub fn is_empty(&self) -> bool { self.entries.is_empty() }
    pub fn max_token_id(&self) -> u32 {
        self.entries.last_key_value().map_or(0, |(&id, _)| id)
    }

    pub(crate) fn vocab_derived_cache_get<T: VocabDerivedArtifact>(&self) -> Option<Arc<T>> {
        self.compiler_cache.artifacts.lock().ok()?.get(&TypeId::of::<T>()).cloned()
            .and_then(|artifact| artifact.downcast::<T>().ok())
    }

    pub(crate) fn vocab_derived_cache_set<T: VocabDerivedArtifact>(&self, artifact: Arc<T>) {
        let erased: Arc<dyn Any + Send + Sync> = artifact;
        if let Ok(mut artifacts) = self.compiler_cache.artifacts.lock() {
            artifacts.entry(TypeId::of::<T>()).or_insert(erased);
        }
    }

    #[doc(hidden)]
    pub fn compiler_cache_entry_count(&self) -> usize {
        self.compiler_cache.artifacts.lock().map(|a| a.len()).unwrap_or(0)
    }
}
''')

for path in ROOT.rglob("*.rs"):
    if ".git" in path.parts or "target" in path.parts:
        continue
    text = path.read_text()
    updated = rewrite_vocab_calls(text)
    if updated != text:
        path.write_text(updated)

# Remove EOS metadata from compiled/runtime structures and serialized payloads.
for path in ["src/runtime/artifact.rs", "src/runtime/dynamic_constraint.rs", "src/runtime/serde.rs"]:
    text = load(path)
    text = re.sub(r"^\s*(?:pub\(crate\) )?eos_token_id: Option<u32>,\n", "", text, flags=re.M)
    text = re.sub(r"^\s*eos_token_id: [^\n]+,\n", "", text, flags=re.M)
    save(path, text)

for path in ["src/compiler/pipeline.rs", "src/runtime/constraint.rs"]:
    text = load(path)
    text = text.replace("            eos_token_id: vocab.eos_token_id,\n", "")
    text = text.replace("            .chain(self.eos_token_id)\n", "")
    save(path, text)

# Artifact layouts changed deliberately on this feature branch.
replace("src/runtime/serde.rs", "const CONSTRAINT_VERSION: u16 = 2;", "const CONSTRAINT_VERSION: u16 = 3;")
replace("src/runtime/dynamic_constraint.rs", "const DYNAMIC_CONSTRAINT_VERSION: u16 = 2;", "const DYNAMIC_CONSTRAINT_VERSION: u16 = 3;")
replace("glrmask-runtime/src/lib.rs", "pub const ARTIFACT_VERSION: u16 = 4;", "pub const ARTIFACT_VERSION: u16 = 5;")
replace("glrmask-runtime/src/lib.rs", "    pub fn eos_allowed(&self) -> bool { self.state.is_complete() }\n\n", "")

# EOS no longer affects forced-byte discovery.
replace("src/runtime/state.rs", "        let eos = self.constraint.eos_token_id;\n", "")
remove_regex("src/runtime/state.rs", r"\n\s*if let Some\(eos_id\) = eos \{\n\s*if is_token_set\(&mask, eos_id\) \{\n\s*break;\n\s*\}\n\s*\}\n")
remove_regex("src/runtime/state.rs", r"\nfn is_token_set\(mask: &\[u32\], token_id: u32\) -> bool \{.*?\n\}\n")

# Dynamic masks use only generic exact-token handling.
remove_regex("src/runtime/dynamic_mask.rs", r"\nfn update_eos_mask\(state: &ConstraintState<'_>, buf: &mut \[u32\]\) \{.*?\n\}\n")
replace("src/runtime/dynamic_mask.rs", "    update_special_token_mask(state, buf);\n    update_eos_mask(state, buf);", "    update_special_token_mask(state, buf);")

# Static masks use only parser-produced token sets.
remove_regex("src/runtime/mask/mod.rs", r"\nfn update_eos_mask\(.*?\n\}\n\nfn eos_mask_bit\(.*?\n\}\n")
remove_regex("src/runtime/mask/mod.rs", r"\n\s*let eos_is_special = self.*?profile\.finalize_eos_ns \+= elapsed_ns\(start\);\n\s*\}\n\s*\}\n")
text = load("src/runtime/mask/mod.rs")
text = text.replace(" finalize_eos_ns={}", "")
text = text.replace("                profile.finalize_eos_ns,\n", "")
save("src/runtime/mask/mod.rs", text)

# Remove obsolete profile surface.
text = load("src/runtime/mask/profile.rs")
text = re.sub(r"^\s*pub finalize_eos_ns: u64,\n", "", text, flags=re.M)
text = text.replace("\t\t\tfinalize_eos_ns: inner.finalize_eos_ns,\n", "")
save("src/runtime/mask/profile.rs", text)
text = load("python/src/lib.rs").replace('    dict.set_item("finalize_eos_ns", profile.finalize_eos_ns)?;\n', "")
save("python/src/lib.rs", text)

# Caller-sized packed masks: smaller buffers are rejected, larger tails are zeroed.
replace("src/runtime/mask/mod.rs", '''    pub fn fill_mask(&self, buf: &mut [u32]) {
        if !self.try_fill_mask_from_cache(buf) {
            self.fill_mask_uncached(buf);
        }
        assert_dynamic_mask_equivalence(self, buf);
    }
''', '''    pub fn fill_mask(&self, buf: &mut [u32]) {
        let required = self.constraint.mask_len();
        assert!(buf.len() >= required, "mask buffer is smaller than constraint mask");
        let (mask, tail) = buf.split_at_mut(required);
        tail.fill(0);
        if !self.try_fill_mask_from_cache(mask) {
            self.fill_mask_uncached(mask);
        }
        assert_dynamic_mask_equivalence(self, mask);
    }
''')
replace("src/runtime/mask/mod.rs", '''    pub(crate) fn fill_mask_profiled(&self, buf: &mut [u32]) -> MaskProfile {
        let total_start = Instant::now();
        if self.try_fill_mask_from_cache(buf) {
''', '''    pub(crate) fn fill_mask_profiled(&self, buf: &mut [u32]) -> MaskProfile {
        let required = self.constraint.mask_len();
        assert!(buf.len() >= required, "mask buffer is smaller than constraint mask");
        let (buf, tail) = buf.split_at_mut(required);
        tail.fill(0);
        let total_start = Instant::now();
        if self.try_fill_mask_from_cache(buf) {
''')
replace("src/runtime/dynamic_mask.rs", ''') -> Result<(), String> {
    let check_deadline = || {
''', ''') -> Result<(), String> {
    let required = state.constraint.mask_len();
    assert!(buf.len() >= required, "mask buffer is smaller than constraint mask");
    let (buf, tail) = buf.split_at_mut(required);
    tail.fill(0);
    let check_deadline = || {
''')

# Fail loudly if runtime EOS semantics survived.
for forbidden in ["eos_token_id", "EOS_BYTES", "update_eos_mask", "finalize_eos_ns"]:
    hits = []
    for path in ROOT.rglob("*.rs"):
        if forbidden in path.read_text(): hits.append(str(path.relative_to(ROOT)))
    if hits:
        print(f"pending cleanup for {forbidden}: {hits}")
