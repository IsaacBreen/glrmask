//! Runtime-only static composition benchmark over a prebuilt artifact.
//! Usage:
//!   composition_loaded_static_probe <cache-dir> [composed|core] [repeats=251]
//!
//! Both A/B binaries load the exact same artifact and vocabulary. Composition
//! build/link work is therefore completely outside this runtime comparison.
//! Timed states are clones because ConstraintState::clone intentionally drops
//! the state-local mask cache.
use glrmask::{Constraint, Vocab};
use std::{fs, hint::black_box, path::Path, time::Instant};

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

fn percentile(values: &mut [f64], q: f64) -> f64 {
    values.sort_by(f64::total_cmp);
    values[((values.len() - 1) as f64 * q).round() as usize]
}

fn hash_mask(mask: &[u32]) -> u64 {
    mask.iter().fold(0xcbf29ce484222325u64, |hash, &word| {
        (hash ^ u64::from(word)).wrapping_mul(0x100000001b3)
    })
}

fn first_token(mask: &[u32]) -> Option<u32> {
    mask.iter().enumerate().find_map(|(word_index, &word)| {
        (word != 0).then(|| word_index as u32 * 32 + word.trailing_zeros())
    })
}

fn composed_prefixes() -> Vec<Vec<u8>> {
    let mut result = vec![Vec::new(), b"const x = tools".to_vec()];
    for tool in 0..10 {
        result.push(format!("const x = tools.tool_{tool}(").into_bytes());
        result.push(format!("const x = tools.tool_{tool}({{").into_bytes());
    }
    result
}

fn core_prefixes() -> Vec<Vec<u8>> {
    vec![
        Vec::new(),
        b"const x = ".to_vec(),
        b"const x = { a: ".to_vec(),
        b"function f(a, b) { return ".to_vec(),
        b"for (let i = 0; i < ".to_vec(),
        b"const s = \"hello".to_vec(),
        b"tools".to_vec(),
        b"const x = tools.".to_vec(),
    ]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert!(args.len() >= 2, "usage: probe <cache-dir> [composed|core] [repeats]");
    let cache = Path::new(&args[1]);
    let kind = args.get(2).map(String::as_str).unwrap_or("composed");
    let repeats = args
        .get(3)
        .map(|value| value.parse::<usize>().expect("positive repeats"))
        .unwrap_or(251);
    assert!(repeats > 0);

    let vocab = read_vocab_dump(&cache.join("vocab_dump.bin"));
    let artifact = match kind {
        "composed" => cache.join("composed-latest.bin"),
        "core" => cache.join("core.bin"),
        other => panic!("unknown kind {other:?}"),
    };
    let bytes = fs::read(&artifact).unwrap_or_else(|error| panic!("read {}: {error}", artifact.display()));
    let constraint = Constraint::load_with_vocab(&bytes, &vocab)
        .unwrap_or_else(|error| panic!("load {}: {error}", artifact.display()));
    let prefixes = match kind {
        "composed" => composed_prefixes(),
        "core" => core_prefixes(),
        _ => unreachable!(),
    };

    eprintln!(
        "LOADED_STATIC_PROBE kind={kind} artifact={} bytes={} repeats={repeats}",
        artifact.display(),
        bytes.len(),
    );
    println!("kind,prefix_index,prefix_hex,repeats,mask_p50_us,mask_p90_us,mask_p99_us,mask_max_us,tbm_p50_us,tbm_p90_us,tbm_p99_us,tbm_max_us,allowed,hash");

    for (prefix_index, prefix) in prefixes.iter().enumerate() {
        let mut template = constraint.start();
        if !prefix.is_empty() && template.commit_bytes(prefix).is_err() {
            eprintln!("SKIP prefix_index={prefix_index} prefix={:?}", String::from_utf8_lossy(prefix));
            continue;
        }
        let expected = template.mask();
        let selected = first_token(&expected);
        let allowed: u32 = expected.iter().map(|word| word.count_ones()).sum();
        let hash = hash_mask(&expected);
        let mut out = vec![0u32; expected.len()];
        let mut masks = Vec::with_capacity(repeats);
        let mut tbms = Vec::with_capacity(repeats);

        for sample in 0..repeats + 8 {
            let mut state = template.clone();
            let started = Instant::now();
            state.fill_mask(black_box(&mut out));
            let mask_done = Instant::now();
            if let Some(token) = selected {
                state.commit_token(black_box(token)).expect("mask/commit disagreement");
            }
            let commit_done = Instant::now();
            assert_eq!(out, expected, "unstable mask at prefix {prefix_index}");
            if sample >= 8 {
                masks.push((mask_done - started).as_secs_f64() * 1e6);
                tbms.push((commit_done - started).as_secs_f64() * 1e6);
            }
        }

        let p50 = percentile(&mut masks, 0.50);
        let p90 = percentile(&mut masks, 0.90);
        let p99 = percentile(&mut masks, 0.99);
        let max = percentile(&mut masks, 1.00);
        let tbm50 = percentile(&mut tbms, 0.50);
        let tbm90 = percentile(&mut tbms, 0.90);
        let tbm99 = percentile(&mut tbms, 0.99);
        let tbmmax = percentile(&mut tbms, 1.00);
        let prefix_hex = prefix.iter().map(|byte| format!("{byte:02x}")).collect::<String>();
        println!(
            "{kind},{prefix_index},{prefix_hex},{repeats},{p50:.3},{p90:.3},{p99:.3},{max:.3},{tbm50:.3},{tbm90:.3},{tbm99:.3},{tbmmax:.3},{allowed},{hash:016x}"
        );
    }
}
