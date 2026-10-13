//! Independent small-vocabulary composition regression and timing probe.
//! Usage: composition_static_probe [repeats=51] [case-substring] [vocab-depth=2]
//! Compile against the SAME checkpoint/refactor release configuration for A/B.
//! The complete mask is checked against a monolithic constraint at every prefix.
//! Timed states are clones: ConstraintState::clone drops its last-mask cache.
//! No sampled-token selection, state cloning, replay, or assertions are timed.
use glrmask::{Constraint, DynamicConstraint, Grammar, Vocab};
use std::{hint::black_box, time::Instant};

fn vocabulary(depth: usize) -> Vocab {
    assert!((1..=4).contains(&depth), "vocabulary depth must be 1..=4");
    let mut words = Vec::new();
    let mut layer = vec![Vec::new()];
    for _ in 0..depth {
        let mut next = Vec::new();
        for prefix in layer {
            for &byte in b"abcX[]! " {
                let mut word = prefix.clone();
                word.push(byte);
                next.push(word);
            }
        }
        words.extend(next.iter().cloned());
        layer = next;
    }
    words.extend([b"X[a]!".to_vec(), b"X[abc]!".to_vec(), b"abc]!".to_vec()]);
    Vocab::new(words.into_iter().enumerate().map(|(id, b)| (id as u32, b)).collect())
}

fn percentile(times: &mut [f64], q: f64) -> f64 {
    times.sort_by(f64::total_cmp);
    times[((times.len() - 1) as f64 * q).round() as usize]
}

macro_rules! measure {
    ($name:expr, $constraint:expr, $reference:expr, $prefixes:expr, $repeats:expr) => {{
        let c = &$constraint;
        for &(prefix_name, prefix) in $prefixes {
            let mut template = c.start();
            template.commit_bytes(prefix).expect("fixture prefix");
            let mut oracle = $reference.start();
            oracle.commit_bytes(prefix).expect("oracle prefix");
            let expected = oracle.mask();
            assert_eq!(template.mask(), expected, "{} / {} exact oracle", $name, prefix_name);
            let token = expected.iter().enumerate().find_map(|(i, &w)| {
                (w != 0).then(|| (i as u32) * 32 + w.trailing_zeros())
            });
            let count: u32 = expected.iter().map(|w| w.count_ones()).sum();
            let hash = expected.iter().fold(0xcbf29ce484222325u64, |h, &w| {
                (h ^ u64::from(w)).wrapping_mul(0x100000001b3)
            });
            let mut out = vec![0u32; expected.len()];
            let mut masks = Vec::new();
            let mut tbms = Vec::new();
            for i in 0..($repeats + 5) {
                // Unlike start(), clone does not prefill this state's mask cache.
                let mut state = template.clone();
                let begin = Instant::now();
                state.fill_mask(black_box(&mut out));
                let mask_end = Instant::now();
                if let Some(token) = token {
                    state.commit_token(black_box(token)).expect("mask/commit agreement");
                }
                let commit_end = Instant::now();
                assert_eq!(out, expected, "{} / {} timed mask", $name, prefix_name);
                if i >= 5 {
                    masks.push((mask_end - begin).as_secs_f64() * 1e6);
                    tbms.push((commit_end - begin).as_secs_f64() * 1e6);
                }
            }
            let mask50 = percentile(&mut masks, 0.5);
            let mask90 = percentile(&mut masks, 0.9);
            let maskmax = percentile(&mut masks, 1.0);
            let tbm50 = percentile(&mut tbms, 0.5);
            let tbm90 = percentile(&mut tbms, 0.9);
            let tbmmax = percentile(&mut tbms, 1.0);
            println!("{},{},{},{mask50:.3},{mask90:.3},{maskmax:.3},{tbm50:.3},{tbm90:.3},{tbmmax:.3},{count},{hash:016x}", $name, prefix_name, $repeats);
        }
    }};
}

macro_rules! timed_link {
    ($name:expr, $link:expr) => {{
        let mut times = Vec::new();
        let mut result = None;
        for repeat in 0..3 {
            eprintln!("LINK_START {} repeat={repeat}", $name);
            let begin = Instant::now();
            let bound = $link.expect("bind");
            times.push(begin.elapsed().as_secs_f64() * 1e6);
            result = Some(bound);
        }
        eprintln!("LINK {},samples=3,p50_us={:.3},max_us={:.3}", $name,
                  percentile(&mut times, 0.5), percentile(&mut times, 1.0));
        result.unwrap()
    }};
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let repeats: usize = args.get(1).map(|s| s.parse().expect("positive repeats")).unwrap_or(51);
    assert!(repeats > 0);
    let filter = args.get(2).map(String::as_str).unwrap_or("");
    let depth = args.get(3).map(|s| s.parse::<usize>().expect("vocab depth")).unwrap_or(2);
    let v = vocabulary(depth);
    eprintln!("PROBE_CONFIG vocabulary_depth={depth} repeats={repeats}");
    let middle_source = "glrm 1; start middle; extern grammar leaf; nt middle = \"[\" leaf \"]\";";
    let outer_source = "glrm 1; start outer; extern grammar middle; nt outer = \"X\" middle \"!\";";
    println!("case,prefix,repeats,mask_p50_us,mask_p90_us,mask_max_us,tbm_p50_us,tbm_p90_us,tbm_max_us,allowed,hash");
    for (fixture, leaf_source, monolithic_source, prefixes) in [
        ("literal", "glrm 1; start leaf; nt leaf = \"a\";",
         "glrm 1; start outer; nt outer = \"X\" \"[\" \"a\" \"]\" \"!\";",
         vec![("root", b"".as_slice()), ("outer", b"X"), ("leaf", b"X["),
              ("return", b"X[a"), ("outer_return", b"X[a]"), ("done", b"X[a]!")]),
        ("text", "glrm 1; start leaf; t TEXT = /[a-c ]{1,8}/; nt leaf = TEXT;",
         "glrm 1; start outer; t TEXT = /[a-c ]{1,8}/; nt outer = \"X\" \"[\" TEXT \"]\" \"!\";",
         vec![("root", b"".as_slice()), ("outer", b"X"), ("leaf", b"X["),
              ("text", b"X[abc"), ("limit", b"X[abcabcab"), ("outer_return", b"X[abc]"), ("done", b"X[abc]!")]),
    ] {
        let mono = Constraint::compile(Grammar::glrm(monolithic_source), &v).unwrap();
        let ordinary_name = format!("{fixture}_ordinary_static");
        if ordinary_name.contains(filter) { measure!(&ordinary_name, mono, mono, &prefixes, repeats); }
        let composed_selected = ["fully_static", "static_outer_dynamic_middle", "dynamic_components_static_boundary"]
            .iter().any(|suffix| format!("{fixture}_{suffix}").contains(filter));
        if !composed_selected {
            continue;
        }
        eprintln!("COMPONENT_BUILD_START {fixture}");
        let leaf = Constraint::compile(Grammar::glrm(leaf_source), &v).unwrap();
        let middle_parent = Constraint::compile(Grammar::glrm(middle_source), &v).unwrap();
        let outer_parent = Constraint::compile(Grammar::glrm(outer_source), &v).unwrap();
        eprintln!("COMPONENT_BUILD_DONE {fixture}");
        let name = format!("{fixture}_fully_static");
        if name.contains(filter) {
            let middle = timed_link!(format!("{name}_inner"), middle_parent.bind_grammar("leaf", &leaf));
            let bound = timed_link!(format!("{name}_outer"), outer_parent.bind_grammar("middle", &middle));
            measure!(&name, bound, mono, &prefixes, repeats);
        }
        let name = format!("{fixture}_static_outer_dynamic_middle");
        if name.contains(filter) {
            let middle = middle_parent.bind_grammar_dynamic_boundary("leaf", &leaf).unwrap();
            let bound = timed_link!(&name, outer_parent.bind_grammar("middle", &middle));
            measure!(&name, bound, mono, &prefixes, repeats);
        }
        let name = format!("{fixture}_dynamic_components_static_boundary");
        if name.contains(filter) {
            let leaf = DynamicConstraint::compile(Grammar::glrm(leaf_source), &v).unwrap();
            let middle = middle_parent.bind_grammar("leaf", &leaf).unwrap();
            let bound = timed_link!(&name, outer_parent.bind_grammar("middle", &middle));
            measure!(&name, bound, mono, &prefixes, repeats);
        }
    }
}
