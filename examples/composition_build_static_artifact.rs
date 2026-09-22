//! Build one current provider-native selected10 static composition artifact.
//! Usage:
//!   composition_build_static_artifact <cache-dir> <output-file>
use glrmask::{Constraint, Vocab};
use glrmask::__private::ConstraintExt;
use std::{fs, path::Path, time::Instant};

fn read_vocab_dump(path: &Path) -> Vocab {
    let bytes = fs::read(path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    let mut offset = 0usize;
    let read_u32 = |offset: &mut usize| {
        let end = *offset + 4;
        let value = u32::from_le_bytes(bytes[*offset..end].try_into().unwrap());
        *offset = end;
        value
    };
    let count = read_u32(&mut offset) as usize;
    let mut entries = Vec::with_capacity(count);
    for _ in 0..count {
        let token = read_u32(&mut offset);
        let len = read_u32(&mut offset) as usize;
        let end = offset + len;
        entries.push((token, bytes[offset..end].to_vec()));
        offset = end;
    }
    assert_eq!(offset, bytes.len(), "vocab dump has trailing bytes");
    Vocab::new(entries)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert_eq!(args.len(), 3, "usage: build-static <cache-dir> <output-file>");
    let cache = Path::new(&args[1]);
    let output = Path::new(&args[2]);
    let vocab = read_vocab_dump(&cache.join("vocab_dump.bin"));
    let core_bytes = fs::read(cache.join("core.bin")).expect("read core.bin");
    let dispatch_bytes =
        fs::read(cache.join("dispatch-literal.bin")).expect("read dispatch-literal.bin");
    let core = Constraint::load_with_vocab(&core_bytes, &vocab).expect("load core");
    let dispatch =
        Constraint::load_with_vocab(&dispatch_bytes, &vocab).expect("load dispatch");

    let started = Instant::now();
    let composed = core
        .compose_compiled_subgrammars(&[("PROGRAMMATIC_TOOL_SUFFIX", &dispatch)], &vocab)
        .expect("static composition");
    let compose_ms = started.elapsed().as_secs_f64() * 1000.0;

    let started = Instant::now();
    let bytes = composed.save();
    let save_ms = started.elapsed().as_secs_f64() * 1000.0;
    fs::write(output, &bytes).unwrap_or_else(|error| panic!("write {}: {error}", output.display()));
    println!(
        "BUILD_RESULT compose_ms={compose_ms:.3} save_ms={save_ms:.3} artifact_bytes={} output={}",
        bytes.len(),
        output.display(),
    );
}
