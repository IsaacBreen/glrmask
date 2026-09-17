// Committed reproducible static-vs-dynamic mask differential for the
// prepared-static composition-linker redesign (selected10 outer link).
//
// Gate semantics: compose the selected10 outer link twice — once with the
// static backend (`compose_compiled_subgrammars`, whatever backend step 5
// wires in) and once with the DynamicDirect backend
// (`compose_compiled_subgrammars_dynamic`) — then walk a DETERMINISTIC corpus
// (fixed byte prefixes + a seeded token RNG stream) and require mask-for-mask
// equality at every position. Prints mismatch count, position count, and an
// FNV-1a checksum over the reference (dynamic) mask stream.
//
// Reproduce: `cargo run --release --features internal-api --example
// static_link_diff -- --cache-dir <v29-cache>` (plus `--steps N --seed S` to
// vary the corpus; the gate record pins the exact inputs). Exit status is 0
// on zero mismatches, 1 otherwise.
use std::fs;
use std::path::{Path, PathBuf};

use glrmask::{Constraint, Vocab};
use glrmask::__private::ConstraintExt;

const DEFAULT_CACHE_DIR: &str = "/Users/isaacbreen/Projects2/temp/2026-09/glrmask-selected10-cache-v29";
const DEFAULT_STEPS: usize = 32;
const DEFAULT_SEED: u64 = 0x9e37_79b9_7f4a_7c15;

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
    assert_eq!(entries.len(), 128_256, "selected10 expects the CFA Llama-3.1 vocabulary");
    Vocab::new(entries)
}

fn load_constraint(path: &Path) -> Constraint {
    let bytes = fs::read(path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    Constraint::load(&bytes).unwrap_or_else(|error| panic!("load {}: {error}", path.display()))
}

fn choose_token(mask: &[u32], rng: &mut u64) -> Option<u32> {
    let allowed: usize = mask.iter().map(|word| word.count_ones() as usize).sum();
    if allowed == 0 {
        return None;
    }
    *rng ^= *rng >> 12;
    *rng ^= *rng << 25;
    *rng ^= *rng >> 27;
    let draw = rng.wrapping_mul(0x2545_f491_4f6c_dd1d);
    let mut rank = draw as usize % allowed;
    for (word_index, &word) in mask.iter().enumerate() {
        let count = word.count_ones() as usize;
        if rank >= count {
            rank -= count;
            continue;
        }
        let mut live = word;
        for _ in 0..rank {
            live &= live - 1;
        }
        let bit = live.trailing_zeros() as usize;
        return Some((word_index * 32 + bit) as u32);
    }
    None
}

#[allow(clippy::too_many_arguments)]
fn check_position(
    st_static: &mut glrmask::ConstraintState<'_>,
    st_dynamic: &mut glrmask::ConstraintState<'_>,
    scenario: usize,
    consumed: usize,
    what: &str,
    checksum: &mut u64,
    positions: &mut usize,
    mismatches: &mut usize,
    first_mismatch: &mut Option<(usize, usize, String)>,
) -> Vec<u32> {
    let mask_static = st_static.mask();
    let mask_dynamic = st_dynamic.mask();
    fnv_mix(checksum, &mask_dynamic);
    *positions += 1;
    if mask_static != mask_dynamic {
        *mismatches += 1;
        if first_mismatch.is_none() {
            let count = |mask: &[u32]| mask.iter().map(|w| w.count_ones()).sum::<u32>();
            *first_mismatch = Some((
                scenario,
                consumed,
                format!(
                    "{what} static_admits={} dynamic_admits={}",
                    count(&mask_static),
                    count(&mask_dynamic),
                ),
            ));
        }
    }
    mask_dynamic
}

fn fnv_mix(checksum: &mut u64, mask: &[u32]) {
    for (index, &word) in mask.iter().enumerate() {
        *checksum ^= (word as u64).wrapping_add(index as u64);
        *checksum = checksum.wrapping_mul(0x1000_0000_01b3);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut cache_dir = PathBuf::from(DEFAULT_CACHE_DIR);
    let mut steps = DEFAULT_STEPS;
    let mut seed = DEFAULT_SEED;
    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--cache-dir" => {
                i += 1;
                cache_dir = PathBuf::from(&args[i]);
            }
            "--steps" => {
                i += 1;
                steps = args[i].parse().expect("--steps integer");
            }
            "--seed" => {
                i += 1;
                let raw = args[i].trim_start_matches("0x");
                seed = u64::from_str_radix(raw, 16).expect("--seed hex u64");
            }
            other => panic!("unknown arg {other:?} (--cache-dir|--steps|--seed)"),
        }
        i += 1;
    }

    let vocab = read_vocab_dump(&cache_dir.join("vocab_dump.bin"));
    let dispatch_name =
        if cache_dir.join("dispatch-literal.bin").exists() { "dispatch-literal.bin" } else { "dispatch.bin" };
    let core = load_constraint(&cache_dir.join("core.bin"));
    let dispatch = load_constraint(&cache_dir.join(dispatch_name));
    let static_comp =
        core.compose_compiled_subgrammars(&[("PROGRAMMATIC_TOOL_SUFFIX", &dispatch)], &vocab).unwrap();
    let core = load_constraint(&cache_dir.join("core.bin"));
    let dispatch = load_constraint(&cache_dir.join(dispatch_name));
    let dynamic_comp = core
        .compose_compiled_subgrammars_dynamic(&[("PROGRAMMATIC_TOOL_SUFFIX", &dispatch)], &vocab)
        .unwrap();

    let mut prefixes: Vec<Vec<u8>> = vec![Vec::new(), b"const x = tools".to_vec()];
    for tool in 0..10 {
        prefixes.push(format!("const x = tools.tool_{tool}(").into_bytes());
        prefixes.push(format!("const x = tools.tool_{tool}({{").into_bytes());
    }

    let mut rng = seed;
    let mut checksum: u64 = 0xcbf2_9ce4_8422_2325;
    let mut positions = 0usize;
    let mut mismatches = 0usize;
    let mut first_mismatch: Option<(usize, usize, String)> = None;
    for (scenario, prefix) in prefixes.iter().enumerate() {
        let mut st_static = static_comp.start();
        let mut st_dynamic = dynamic_comp.start();
        let mut last_dynamic_mask = check_position(
            &mut st_static,
            &mut st_dynamic,
            scenario,
            0,
            "prefix-root",
            &mut checksum,
            &mut positions,
            &mut mismatches,
            &mut first_mismatch,
        );
        let mut alive = true;
        for (consumed, &byte) in prefix.iter().enumerate() {
            let r_static = st_static.commit_bytes(&[byte]);
            let r_dynamic = st_dynamic.commit_bytes(&[byte]);
            if r_static.is_ok() != r_dynamic.is_ok() {
                mismatches += 1;
                if first_mismatch.is_none() {
                    first_mismatch = Some((
                        scenario,
                        consumed,
                        format!("prefix-commit-divergence static_ok={} dynamic_ok={}", r_static.is_ok(), r_dynamic.is_ok()),
                    ));
                }
                alive = false;
                break;
            }
            if r_dynamic.is_err() {
                alive = false;
                break;
            }
            last_dynamic_mask = check_position(
                &mut st_static,
                &mut st_dynamic,
                scenario,
                consumed + 1,
                "prefix-byte",
                &mut checksum,
                &mut positions,
                &mut mismatches,
                &mut first_mismatch,
            );
        }
        if !alive {
            continue;
        }
        for step in 0..steps {
            // Reuse the previous position's dynamic mask: masks are
            // deterministic, so the RNG stream is identical to re-masking.
            let Some(token) = choose_token(&last_dynamic_mask, &mut rng) else { break };
            let r_static = st_static.commit_token(token);
            let r_dynamic = st_dynamic.commit_token(token);
            if r_static.is_ok() != r_dynamic.is_ok() {
                mismatches += 1;
                if first_mismatch.is_none() {
                    first_mismatch = Some((
                        scenario,
                        prefix.len() + step,
                        format!("token-commit-divergence token={token} static_ok={} dynamic_ok={}", r_static.is_ok(), r_dynamic.is_ok()),
                    ));
                }
                break;
            }
            if r_dynamic.is_err() {
                break;
            }
            last_dynamic_mask = check_position(
                &mut st_static,
                &mut st_dynamic,
                scenario,
                prefix.len() + step + 1,
                "token-step",
                &mut checksum,
                &mut positions,
                &mut mismatches,
                &mut first_mismatch,
            );
        }
    }
    println!(
        "DIFF_RESULT static_backend=compose_compiled_subgrammars dynamic_backend=DynamicDirect scenarios={} steps={} seed=0x{seed:016x} positions={positions} mismatches={mismatches} checksum={checksum:016x}",
        prefixes.len(),
        steps,
    );
    if let Some((scenario, consumed, detail)) = &first_mismatch {
        println!("DIFF_FIRST_MISMATCH scenario={scenario} consumed={consumed} {detail}");
    }
    if mismatches != 0 {
        std::process::exit(1);
    }
}
