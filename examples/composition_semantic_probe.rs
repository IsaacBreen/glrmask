//! Supplemental public-API regression probe for shared static mask traversal.
//! Same source/binary configuration must be used on checkpoint and refactor.
//! Includes JSON ordinary-path controls, scoped ignores, delayed exclusions,
//! nullable returns, duplicate byte spellings, special tokens, and persistence.
use glrmask::{Constraint, ConstraintSpec, Grammar, Vocab};
use std::{hint::black_box, time::Instant};

fn vocabulary() -> (Vocab, Vec<(u32, Vec<u8>)>) {
    let mut entries: Vec<_> = (0..128).map(|id| (id, vec![id as u8])).collect();
    for bytes in [
        " double", "Quote", "X \ta!", "X\t a!", "Xa\t !", "Xa \t!",
        "Xa!", "a!", "X!", "@ doubleQuote", " x=0", "%=0", "=0",
        "{\"x\":\"ab\"}", "{\"x\":", "\"ab\"}", "\"}", "a", "X",
        "dead-special", "dead-special", "X[a]!", "a]!", "[]!", "X[]!",
    ] {
        entries.push((entries.len() as u32, bytes.as_bytes().to_vec()));
    }
    for n in 0..256 {
        entries.push((entries.len() as u32, format!("dead-branch-{n:04}").into_bytes()));
    }
    (Vocab::new(entries.clone()), entries)
}

fn token(entries: &[(u32, Vec<u8>)], text: &str) -> u32 {
    entries.iter().find(|(_, b)| b.as_slice() == text.as_bytes()).expect("fixture token").0
}

fn bit(mask: &[u32], token: u32) -> bool {
    mask.get(token as usize / 32).is_some_and(|word| word & (1 << (token % 32)) != 0)
}

fn percentile(times: &mut [f64], q: f64) -> f64 {
    times.sort_by(f64::total_cmp);
    times[((times.len() - 1) as f64 * q).round() as usize]
}

fn probe(
    name: &str, constraint: &Constraint, reference: &Constraint,
    prefixes: &[(&str, Vec<u32>)], token_ids: &[u32], repeats: usize,
) {
    for (label, tokens) in prefixes {
        let mut template = constraint.start();
        let mut oracle = reference.start();
        for &token in tokens {
            template.commit_token(token).unwrap();
            oracle.commit_token(token).unwrap();
        }
        let expected = oracle.mask();
        assert_eq!(template.mask(), expected, "{name}/{label}: complete oracle mask");
        // Independent endpoint oracle. Not included in any timing window.
        for &token in token_ids {
            let mut endpoint = template.clone();
            assert_eq!(bit(&expected, token), endpoint.commit_token(token).is_ok(),
                       "{name}/{label}: pointwise endpoint {token}");
        }
        let selected = token_ids.iter().copied().find(|&id| bit(&expected, id));
        let mut out = vec![0u32; expected.len()];
        let mut masks = Vec::new();
        let mut tbms = Vec::new();
        for sample in 0..repeats + 5 {
            // start() prefills masks, clone() clears the last-mask cache.
            let mut state = template.clone();
            let begin = Instant::now();
            state.fill_mask(black_box(&mut out));
            let mask_end = Instant::now();
            if let Some(token) = selected { state.commit_token(black_box(token)).unwrap(); }
            let commit_end = Instant::now();
            assert_eq!(out, expected, "{name}/{label}: timed mask");
            if sample >= 5 {
                masks.push((mask_end - begin).as_secs_f64() * 1e6);
                tbms.push((commit_end - begin).as_secs_f64() * 1e6);
            }
        }
        let count: u32 = expected.iter().map(|w| w.count_ones()).sum();
        let hash = expected.iter().fold(0xcbf29ce484222325u64, |h, &w| {
            (h ^ u64::from(w)).wrapping_mul(0x100000001b3)
        });
        println!("{name},{label},{repeats},{:.3},{:.3},{:.3},{:.3},{count},{hash:016x}",
                 percentile(&mut masks, 0.5), percentile(&mut masks, 0.9),
                 percentile(&mut tbms, 0.5), percentile(&mut tbms, 0.9));
    }
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let repeats: usize = args.get(1).map(|s| s.parse().expect("repeat count")).unwrap_or(31);
    assert!(repeats > 0);
    let filter = args.get(2).map(String::as_str).unwrap_or("");
    let (vocab, entries) = vocabulary();
    let ids: Vec<_> = entries.iter().map(|(id, _)| *id).collect();
    let t = |text: &str| token(&entries, text);
    let byte_tokens = |s: &str| s.bytes().map(u32::from).collect::<Vec<_>>();
    println!("case,prefix,repeats,mask_p50_us,mask_p90_us,tbm_p50_us,tbm_p90_us,allowed,hash");

    if "ordinary_json".contains(filter) {
        let c = Constraint::compile(Grammar::json_schema(
            r#"{"type":"object","properties":{"x":{"type":"string","maxLength":8}},"required":["x"],"additionalProperties":false}"#,
        ), &vocab).unwrap();
        let prefixes = [
            ("root", byte_tokens("")), ("object", byte_tokens("{")),
            ("property", byte_tokens("{\"x\":")), ("string", byte_tokens("{\"x\":\"ab")),
            ("limit", byte_tokens("{\"x\":\"abcdefgh")), ("return", byte_tokens("{\"x\":\"ab\"")),
            ("done", byte_tokens("{\"x\":\"ab\"}")),
        ];
        probe("ordinary_json", &c, &c, &prefixes, &ids, repeats);
    }

    if "scoped_ignores_static".contains(filter) {
        let parent = Constraint::compile(Grammar::glrm(
            r#"glrm 1; start document; ignore PARENT_WS; t PARENT_WS = " "+;
               extern grammar child; nt document = "X" child "!";"#,
        ), &vocab).unwrap();
        let child = Constraint::compile(Grammar::glrm(
            r#"glrm 1; start child; ignore CHILD_WS; t CHILD_WS = "\t"+; nt child = "a";"#,
        ), &vocab).unwrap();
        let reference = Constraint::compile(Grammar::glrm(
            r#"glrm 1; start document; ignore PARENT_WS; t PARENT_WS = " "+;
               g child = { start child; ignore CHILD_WS; t CHILD_WS = "\t"+; nt child = "a"; };
               nt document = "X" child "!";"#,
        ), &vocab).unwrap();
        let begin = Instant::now();
        let bound = parent.bind_grammar("child", &child).unwrap();
        eprintln!("LINK scoped_ignores_static us={:.3}", begin.elapsed().as_secs_f64() * 1e6);
        let prefixes = [("root", vec![]), ("entry", vec![t("X")]),
                        ("return", vec![t("X"), t("a")]), ("done", vec![t("X"), t("a"), t("!")])];
        probe("scoped_ignores_static", &bound, &reference, &prefixes, &ids, repeats);
        let loaded = Constraint::load(bound.save()).unwrap();
        probe("scoped_ignores_static_loaded", &loaded, &reference, &prefixes, &ids, repeats);
    }

    if "exclusion_static".contains(filter) {
        let body = Constraint::compile(Grammar::glrm(
            r#"glrm 1; start start; ignore WS;
               t WS = /[ \t\r\n]+/; t OP = "=" | "%="; t ID = /[A-Za-z_$][A-Za-z0-9_$]*/;
               nt declaration = "@" ID; nt expression = ID OP "0";
               nt start = declaration expression;"#,
        ), &vocab).unwrap();
        let parent = Constraint::compile(Grammar::glrm(
            "glrm 1; start document; extern grammar body; nt document = body;",
        ), &vocab).unwrap();
        let begin = Instant::now();
        let bound = parent.bind_grammar("body", &body).unwrap();
        eprintln!("LINK exclusion_static us={:.3}", begin.elapsed().as_secs_f64() * 1e6);
        let prefixes = [("root", vec![]), ("declaration", vec![t("@"), t(" double")]),
                        ("excluded", vec![t("@"), t(" double"), t("Quote")])];
        let mut excluded = bound.start();
        for &token in &prefixes[2].1 { excluded.commit_token(token).unwrap(); }
        assert!(!bit(&excluded.mask(), t("=")));
        assert!(!bit(&excluded.mask(), t("%")));
        probe("exclusion_static", &bound, &body, &prefixes, &ids, repeats);
        let loaded = Constraint::load(bound.save()).unwrap();
        probe("exclusion_static_loaded", &loaded, &body, &prefixes, &ids, repeats);
    }

    if "nullable_special_static".contains(filter) {
        let special_id = t("dead-special");
        let ids_with_special: Vec<_> = ids.iter().copied().chain([5000]).collect();
        let child = ConstraintSpec::builder(Grammar::glrm(
            r#"glrm 1; start child; extern token SPECIAL; nt child = ("a" | SPECIAL)?;"#,
        ), &vocab).unwrap().bind_token("SPECIAL", [special_id, 5000]).unwrap().build().unwrap().compile().unwrap();
        let parent = Constraint::compile(Grammar::glrm(
            "glrm 1; start document; extern grammar child; nt document = \"X\" child \"!\";",
        ), &vocab).unwrap();
        let reference = ConstraintSpec::builder(Grammar::glrm(
            r#"glrm 1; start document; extern token SPECIAL; nt document = "X" ("a" | SPECIAL)? "!";"#,
        ), &vocab).unwrap().bind_token("SPECIAL", [special_id, 5000]).unwrap().build().unwrap().compile().unwrap();
        let bound = parent.bind_grammar("child", &child).unwrap();
        let prefixes = [("root", vec![]), ("entry", vec![t("X")]),
                        ("special_return", vec![t("X"), special_id]),
                        ("unspelled_return", vec![t("X"), 5000]), ("empty_done", vec![t("X!")])];
        probe("nullable_special_static", &bound, &reference, &prefixes, &ids_with_special, repeats);
        let loaded = Constraint::load(bound.save()).unwrap();
        probe("nullable_special_static_loaded", &loaded, &reference, &prefixes, &ids_with_special, repeats);
    }
}
