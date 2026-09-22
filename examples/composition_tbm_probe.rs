//! Fixed-state composition mask/commit and link-time probe.
//! Usage: composition_tbm_probe <CFA llama3_vocab.json> [repeats=21] [case-substring]
//! The vocabulary file maps decimal token IDs to hex-encoded bytes.
//! Timed TBM is fill_mask + commit_token; state setup and token selection are
//! outside the timer. Each sample starts a fresh state at the same byte prefix.
use glrmask::{Constraint, DynamicConstraint, Grammar, Vocab};
use std::{collections::BTreeMap, hint::black_box, time::Instant};

fn read_vocab(path: &str) -> Vocab {
    let encoded: BTreeMap<String, String> =
        serde_json::from_slice(&std::fs::read(path).expect("read vocabulary")).expect("parse vocabulary");
    let entries = encoded.into_iter().map(|(id, hex)| {
        assert_eq!(hex.len() % 2, 0, "odd hex length for token {id}");
        let bytes = (0..hex.len()).step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).expect("hex byte"))
            .collect::<Vec<_>>();
        (id.parse::<u32>().expect("token id"), bytes)
    }).collect::<Vec<_>>();
    eprintln!("VOCAB entries={}", entries.len());
    Vocab::new(entries)
}

fn percentile(values: &mut [f64], q: f64) -> f64 {
    values.sort_by(f64::total_cmp);
    values[((values.len() - 1) as f64 * q).round() as usize]
}

fn report(kind: &str, case: &str, prefix: &str, times: &mut [f64], allowed: u32, hash: u64) {
    let p50 = percentile(times, 0.5);
    let p90 = percentile(times, 0.9);
    let p99 = percentile(times, 0.99);
    let max = percentile(times, 1.0);
    println!("{kind},{case},{prefix},{},{p50:.3},{p90:.3},{p99:.3},{max:.3},{allowed},{hash:016x}", times.len());
}

macro_rules! measure {
    ($name:expr, $constraint:expr, $prefixes:expr, $repeats:expr) => {{
        let constraint = &$constraint;
        for &(prefix_name, prefix) in $prefixes {
            let mut setup = constraint.start();
            setup.commit_bytes(prefix).expect("valid fixture prefix");
            let expected = setup.mask();
            let allowed = expected.iter().map(|word| word.count_ones()).sum();
            let hash = expected.iter().fold(0xcbf29ce484222325u64, |hash, &word| {
                (hash ^ u64::from(word)).wrapping_mul(0x100000001b3)
            });
            let token = expected.iter().enumerate().find_map(|(index, &word)| {
                (word != 0).then(|| index as u32 * 32 + word.trailing_zeros())
            });
            let mut output = vec![0u32; expected.len()];
            let mut masks = Vec::new();
            let mut tbms = Vec::new();
            for i in 0..($repeats + 2) {
                let mut state = constraint.start();
                state.commit_bytes(prefix).expect("replay prefix");
                let t0 = Instant::now();
                state.fill_mask(black_box(&mut output));
                let t1 = Instant::now();
                if let Some(token) = token {
                    state.commit_token(black_box(token)).expect("mask/commit agreement");
                }
                let t2 = Instant::now();
                assert_eq!(output, expected, "stable complete mask for {} / {}", $name, prefix_name);
                if i >= 2 {
                    masks.push((t1 - t0).as_secs_f64() * 1e6);
                    tbms.push((t2 - t0).as_secs_f64() * 1e6);
                }
            }
            report("mask_us", $name, prefix_name, &mut masks, allowed, hash);
            report("tbm_us", $name, prefix_name, &mut tbms, allowed, hash);
        }
    }};
}

macro_rules! link {
    ($name:expr, $build:expr) => {{
        let mut times = Vec::new();
        let mut output = None;
        for repeat in 0..7 {
            eprintln!("LINK_START case={} repeat={repeat}", $name);
            let t0 = Instant::now();
            let built = $build.expect("compose fixture");
            let us = t0.elapsed().as_secs_f64() * 1e6;
            eprintln!("LINK_DONE case={} repeat={repeat} us={us:.3}", $name);
            times.push(us);
            output = Some(built);
        }
        report("link_us", $name, "all", &mut times, 0, 0);
        output.unwrap()
    }};
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert!(args.len() >= 2, "usage: composition_tbm_probe <llama3_vocab.json> [repeats]");
    let repeats = args.get(2).map(|s| s.parse::<usize>().expect("repeats")).unwrap_or(21);
    assert!(repeats > 0);
    let filter = args.get(3).map(String::as_str).unwrap_or("");
    let vocab = read_vocab(&args[1]);
    println!("metric,case,prefix,samples,p50_us,p90_us,p99_us,max_us,allowed,mask_hash");
    let middle_source = "glrm 1; start middle; extern grammar leaf; nt middle = \"[\" leaf \"]\";";
    let outer_source = "glrm 1; start outer; extern grammar middle; nt outer = \"X\" middle \"!\";";
    for (fixture, leaf_source, prefixes) in [
        ("literal", r#"glrm 1; start leaf; nt leaf = "a";"#, vec![
            ("root", &b""[..]), ("outer", &b"X"[..]), ("leaf", &b"X["[..]),
            ("return", &b"X[a"[..]), ("outer_return", &b"X[a]"[..]), ("done", &b"X[a]!"[..]),
        ]),
        ("text", r#"glrm 1; start leaf; t TEXT = /[a-zA-Z0-9 ]{1,64}/; nt leaf = TEXT;"#, vec![
            ("root", &b""[..]), ("outer", &b"X"[..]), ("leaf", &b"X["[..]),
            ("text", &b"X[hello"[..]), ("outer_return", &b"X[hello]"[..]), ("done", &b"X[hello]!"[..]),
        ]),
    ] {
        let leaf = Constraint::compile(Grammar::glrm(leaf_source), &vocab).unwrap();
        let middle_parent = Constraint::compile(Grammar::glrm(middle_source), &vocab).unwrap();
        let outer_parent = Constraint::compile(Grammar::glrm(outer_source), &vocab).unwrap();
        let middle = middle_parent.bind_grammar_dynamic_boundary("leaf", leaf.clone()).unwrap();
        let name = format!("{fixture}_static_components_dynamic_boundary");
        if name.contains(filter) {
            let bound = link!(&name, outer_parent.bind_grammar_dynamic_boundary("middle", middle.clone()));
            measure!(&name, bound, &prefixes, repeats);
        }

        let name = format!("{fixture}_static_outer_dynamic_middle");
        if name.contains(filter) {
            let hybrid = link!(&name, outer_parent.bind_grammar("middle", middle.clone()));
            measure!(&name, hybrid, &prefixes, repeats);
        }

        let name = format!("{fixture}_fully_static");
        if name.contains(filter) {
            let static_middle = middle_parent.bind_grammar("leaf", leaf).unwrap();
            let static_bound = link!(&name, outer_parent.bind_grammar("middle", static_middle.clone()));
            measure!(&name, static_bound, &prefixes, repeats);
        }

        let name = format!("{fixture}_fully_dynamic");
        if name.contains(filter) {
            let dynamic_leaf = DynamicConstraint::compile(Grammar::glrm(leaf_source), &vocab).unwrap();
            let dynamic_middle_parent = DynamicConstraint::compile(Grammar::glrm(middle_source), &vocab).unwrap();
            let dynamic_outer_parent = DynamicConstraint::compile(Grammar::glrm(outer_source), &vocab).unwrap();
            let dynamic_middle = dynamic_middle_parent.bind_grammar_dynamic_boundary("leaf", dynamic_leaf).unwrap();
            let dynamic_bound = link!(&name, dynamic_outer_parent.bind_grammar_dynamic_boundary("middle", dynamic_middle.clone()));
            measure!(&name, dynamic_bound, &prefixes, repeats);
        }
    }
}
