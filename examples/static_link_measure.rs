// Measurement driver for the prepared-static composition-linker redesign.
// Subcommands: sizes | link | tbm | compile-one | dump-schema | monolithic
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use glrmask::{Constraint, Grammar, Vocab};
use glrmask::__private::{ConstraintExt, ConstraintStateExt};
use serde_json::Value;

const DEFAULT_CFA_ROOT: &str = "/Users/isaacbreen/Projects2/constraint-framework-analysis";
const DEFAULT_CACHE_DIR: &str = "/Users/isaacbreen/Projects2/temp/2026-09/glrmask-selected10-cache-v29";
const JS_PLACEHOLDER_TOKEN: u32 = 128_300;
const DISPATCH_PLACEHOLDER_BASE: u32 = 128_320;

const SELECTED10: [(&str, &str); 10] = [
    ("o31994", "Github_hard---o31994.json"),
    ("kb_620_Normalized", "Kubernetes---kb_620_Normalized.json"),
    (
        "sil-kit-participant-configuration",
        "JsonSchemaStore---sil-kit-participant-configuration.json",
    ),
    ("o9792", "Github_medium---o9792.json"),
    ("o9896", "Github_easy---o9896.json"),
    ("o16060", "Github_hard---o16060.json"),
    ("kb_678_Normalized", "Kubernetes---kb_678_Normalized.json"),
    ("o83390", "Github_ultra---o83390.json"),
    ("taurus", "JsonSchemaStore---taurus.json"),
    ("kb_1104_Normalized", "Kubernetes---kb_1104_Normalized.json"),
];

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

fn schema_source(cfa_root: &Path, file: &str) -> String {
    let path = cfa_root
        .join("data/sources/jsonschemabench/maskbench/data")
        .join(file);
    let wrapper: Value = serde_json::from_slice(
        &fs::read(&path).unwrap_or_else(|error| panic!("read {}: {error}", path.display())),
    )
    .unwrap_or_else(|error| panic!("parse {}: {error}", path.display()));
    serde_json::to_string(
        wrapper
            .get("schema")
            .unwrap_or_else(|| panic!("{} has no schema field", path.display())),
    )
    .unwrap()
}

fn schema_cache_path(cache_dir: &Path, index: usize, short_name: &str) -> PathBuf {
    cache_dir.join(format!("schema-{index:02}-{short_name}.bin"))
}

fn js_core_source(cfa_root: &Path) -> String {
    let path = cfa_root.join("data/sources/grammars/js.glrm");
    let mut source = fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    let needle = "nt member_expression_with_suffixes ::=\n    primary_expression";
    let replacement = "nt member_expression_with_suffixes ::=\n    'tools' PROGRAMMATIC_TOOL_SUFFIX\n  | primary_expression";
    assert!(source.contains(needle), "JS grammar shape changed; cannot insert programmatic-tools alternative");
    source = source.replacen(needle, replacement, 1);
    source.push_str(&format!(
        "\n// selected10 programmatic-tools external-call sentinel\nt PROGRAMMATIC_TOOL_SUFFIX ::= @token({JS_PLACEHOLDER_TOKEN});\n"
    ));
    source
}

fn dispatcher_literal_names_parent_source() -> String {
    let mut source = String::from("start suffix;\n");
    for index in 0..SELECTED10.len() {
        source.push_str(&format!(
            "t TOOL_ARGS_SLOT_{index} ::= @token({});\n",
            DISPATCH_PLACEHOLDER_BASE + SELECTED10.len() as u32 + index as u32
        ));
    }
    source.push_str("nt suffix ::=\n    ");
    for index in 0..SELECTED10.len() {
        if index != 0 { source.push_str("\n  | "); }
        source.push_str(&format!(r#"".tool_{index}(" TOOL_ARGS_SLOT_{index} ")""#));
    }
    source.push_str(";\n");
    source
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    let index = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[index]
}

fn stat_line(name: &str, bytes_len: usize, c: &Constraint) {
    println!(
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
        name,
        bytes_len,
        c.num_parser_states(),
        c.num_terminals(),
        c.num_tokenizer_states(),
        c.internal_tsid_count(),
        c.final_internal_token_count(),
        c.parser_dwa_num_states(),
        c.parser_dwa_num_transitions(),
    );
}

fn print_stat_row(name: &str, path: &Path) {
    let bytes = fs::read(path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    let c = Constraint::load(&bytes).unwrap_or_else(|error| panic!("load {}: {error}", path.display()));
    stat_line(name, bytes.len(), &c);
}

fn cmd_sizes(cache_dir: &Path, dispatch_name: &str) {
    println!("name\tartifact_bytes\tlr_states\tterminals\ttokenizer_states\tinternal_tsids\tinternal_tokens\tparser_dwa_states\tparser_dwa_transitions");
    print_stat_row("core", &cache_dir.join("core.bin"));
    print_stat_row("dispatch", &cache_dir.join(dispatch_name));
    for (index, (short_name, _)) in SELECTED10.iter().enumerate() {
        print_stat_row(
            &format!("schema-{index:02}-{short_name}"),
            &schema_cache_path(cache_dir, index, short_name),
        );
    }
    let composed = cache_dir.join("composed-latest.bin");
    if composed.exists() {
        print_stat_row("composed", &composed);
    }
}

fn cmd_link(cache_dir: &Path, vocab: &Vocab, dispatch_name: &str, runs: usize) {
    println!("name\tartifact_bytes\tlr_states\tterminals\ttokenizer_states\tinternal_tsids\tinternal_tokens\tparser_dwa_states\tparser_dwa_transitions");
    let core_bytes = fs::read(cache_dir.join("core.bin")).expect("core.bin missing");
    let dispatch_bytes = fs::read(cache_dir.join(dispatch_name)).expect("dispatch cache missing");
    let mut compose_wall = Vec::with_capacity(runs);
    let mut dyn_wall = Vec::with_capacity(runs);
    for run in 0..runs {
        // Static path A: compose_compiled_subgrammars (the bench path).
        let started = Instant::now();
        let core = Constraint::load(&core_bytes).unwrap();
        let dispatch = Constraint::load(&dispatch_bytes).unwrap();
        let load = started.elapsed();
        let started = Instant::now();
        let composed = core
            .compose_compiled_subgrammars(&[("PROGRAMMATIC_TOOL_SUFFIX", &dispatch)], vocab)
            .unwrap();
        let compose = started.elapsed();
        let composed_len = composed.save().len();
        std::hint::black_box(composed_len);
        if run == 0 {
            stat_line("composed-static#fresh", composed_len, &composed);
        }
        compose_wall.push(compose);

        // Dynamic path: same placeholder substitution, DynamicDirect backend.
        let started = Instant::now();
        let core = Constraint::load(&core_bytes).unwrap();
        let dispatch = Constraint::load(&dispatch_bytes).unwrap();
        let _ = started.elapsed();
        let started = Instant::now();
        let dyn_result = core.compose_compiled_subgrammars_dynamic(
            &[("PROGRAMMATIC_TOOL_SUFFIX", &dispatch)],
            vocab,
        );
        let dyn_ = started.elapsed();
        match &dyn_result {
            Ok(dynb) => {
                let dyn_len = dynb.save().len();
                std::hint::black_box(dyn_len);
                if run == 0 {
                    stat_line("composed-dynamic#fresh", dyn_len, dynb);
                }
                dyn_wall.push(dyn_);
            }
            Err(error) => {
                eprintln!("[link] run {}/{} DYNAMIC FAILED after {:.3} ms: {error}", run + 1, runs, dyn_.as_secs_f64() * 1000.0);
            }
        }
        println!(
            "[link] run {}/{} load_ms={:.3} compose_static_ms={:.3} compose_dynamic_ms={:.3}",
            run + 1,
            runs,
            load.as_secs_f64() * 1000.0,
            compose.as_secs_f64() * 1000.0,
            dyn_.as_secs_f64() * 1000.0,
        );
    }
    for (label, samples) in [
        ("compose_static", compose_wall),
        ("compose_dynamic", dyn_wall),
    ] {
        if samples.is_empty() {
            println!("LINK_RESULT which={label} runs=0 FAILED");
            continue;
        }
        let mut ms: Vec<f64> = samples.iter().map(|d| d.as_secs_f64() * 1000.0).collect();
        ms.sort_by(f64::total_cmp);
        println!(
            "LINK_RESULT which={label} runs={} p50_ms={:.3} p100_ms={:.3} run0_ms={:.3}",
            ms.len(),
            percentile(&ms, 0.50),
            percentile(&ms, 1.0),
            samples[0].as_secs_f64() * 1000.0,
        );
    }
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

fn cmd_tbm(cache_dir: &Path, vocab: &Vocab, which: &str, steps: usize) {
    let constraint = match which {
        "core" => load_constraint(&cache_dir.join("core.bin")),
        "static" | "dynamic" => {
            let dispatch_name = if cache_dir.join("dispatch-literal.bin").exists() {
                "dispatch-literal.bin"
            } else {
                "dispatch.bin"
            };
            let core = load_constraint(&cache_dir.join("core.bin"));
            let dispatch = load_constraint(&cache_dir.join(dispatch_name));
            if which == "static" {
                core.compose_compiled_subgrammars(&[("PROGRAMMATIC_TOOL_SUFFIX", &dispatch)], vocab)
                    .unwrap()
            } else {
                core.compose_compiled_subgrammars_dynamic(
                    &[("PROGRAMMATIC_TOOL_SUFFIX", &dispatch)],
                    vocab,
                )
                .unwrap()
            }
        }
        "mono" => load_constraint(&cache_dir.join("monolithic.bin")),
        other => panic!("unknown --which {other:?} (core|static|dynamic|mono)"),
    };
    let composed_prefixes: Vec<Vec<u8>> = {
        let mut v = vec![Vec::<u8>::new(), b"const x = tools".to_vec()];
        for tool in 0..SELECTED10.len() {
            v.push(format!("const x = tools.tool_{tool}(").into_bytes());
            v.push(format!("const x = tools.tool_{tool}({{").into_bytes());
        }
        v
    };
    let core_prefixes: Vec<Vec<u8>> = vec![
        Vec::<u8>::new(),
        b"const x = ".to_vec(),
        b"const x = { a: ".to_vec(),
        b"function f(a, b) { return ".to_vec(),
        b"for (let i = 0; i < ".to_vec(),
        b"const s = \"hello".to_vec(),
        b"tools".to_vec(),
        b"const x = tools.".to_vec(),
    ];
    let prefixes: &[Vec<u8>] = match which {
        "core" => &core_prefixes,
        _ => &composed_prefixes,
    };
    let only_scenario: Option<usize> = std::env::var("STATIC_LINK_MEASURE_ONLY_SCENARIO")
        .ok()
        .and_then(|value| value.parse().ok());
    let trace = std::env::var_os("STATIC_LINK_MEASURE_TRACE").is_some();
    let mut rng = 0x9e37_79b9_7f4a_7c15u64;
    let mut tbm_ns: Vec<f64> = Vec::new();
    let mut mask_ns: Vec<f64> = Vec::new();
    let mut committed_scenarios = 0usize;
    let mut admitted_but_rejected = 0usize;
    for (scenario, prefix) in prefixes.iter().enumerate() {
        // Keep one continuous rng stream so a filtered run reproduces the
        // unfiltered run's draws exactly; filtered-out scenarios still walk
        // (consuming identical draws) but their timings are discarded.
        let record = only_scenario.map_or(true, |only| only == scenario);
        let mut state = constraint.start();
        if !prefix.is_empty() && state.commit_bytes(prefix).is_err() {
            eprintln!("[tbm] which={which} prefix={:?} rejected; skipping scenario", String::from_utf8_lossy(prefix));
            continue;
        }
        if record {
            committed_scenarios += 1;
        }
        let mut trace_tokens: Vec<u32> = Vec::new();
        for step in 0..steps {
            let t0 = Instant::now();
            let mask = state.mask();
            let t1 = Instant::now();
            let Some(token) = choose_token(&mask, &mut rng) else {
                break;
            };
            if record && trace {
                trace_tokens.push(token);
            }
            if let Err(error) = state.commit_token(token) {
                if record {
                    admitted_but_rejected += 1;
                    eprintln!("[tbm] which={which} scenario={scenario} step={step} prefix={:?} mask-admitted token={token} COMMIT-REJECTED: {error}; ending scenario", String::from_utf8_lossy(prefix));
                    if trace {
                        eprintln!("[tbm] trace scenario={scenario} tokens={trace_tokens:?}");
                    }
                }
                break;
            }
            let t3 = Instant::now();
            if record {
                mask_ns.push((t1 - t0).as_secs_f64() * 1e9);
                tbm_ns.push((t3 - t0).as_secs_f64() * 1e9);
            }
        }
        if record && trace && committed_scenarios > 0 {
            eprintln!("[tbm] trace scenario={scenario} tokens={trace_tokens:?}");
        }
    }
    eprintln!("[tbm] which={which} admitted_but_rejected_scenarios={admitted_but_rejected}");
    mask_ns.sort_by(f64::total_cmp);
    tbm_ns.sort_by(f64::total_cmp);
    println!(
        "TBM_RESULT which={which} scenarios={} steps_per_scenario={} samples={} mask_p50_us={:.3} mask_p90_us={:.3} mask_p99_us={:.3} mask_max_us={:.3} tbm_p50_us={:.3} tbm_p90_us={:.3} tbm_p99_us={:.3} tbm_max_us={:.3}",
        committed_scenarios,
        steps,
        tbm_ns.len(),
        percentile(&mask_ns, 0.50) / 1000.0,
        percentile(&mask_ns, 0.90) / 1000.0,
        percentile(&mask_ns, 0.99) / 1000.0,
        percentile(&mask_ns, 1.0) / 1000.0,
        percentile(&tbm_ns, 0.50) / 1000.0,
        percentile(&tbm_ns, 0.90) / 1000.0,
        percentile(&tbm_ns, 0.99) / 1000.0,
        percentile(&tbm_ns, 1.0) / 1000.0,
    );
}

fn cmd_compile_one(cfa_root: &Path, vocab: &Vocab, kind: &str, repeats: usize) {
    println!("name\tartifact_bytes\tlr_states\tterminals\ttokenizer_states\tinternal_tsids\tinternal_tokens\tparser_dwa_states\tparser_dwa_transitions");
    for rep in 0..repeats {
        let started = Instant::now();
        let (label, constraint) = if kind == "core" {
            let c = Constraint::compile(Grammar::glrm(&js_core_source(cfa_root)), vocab).unwrap();
            ("core", c)
        } else if kind == "dispatch-parent" {
            let c = Constraint::compile(Grammar::glrm(&dispatcher_literal_names_parent_source()), vocab).unwrap();
            ("dispatch-parent", c)
        } else if let Some(index_str) = kind.strip_prefix("schema:") {
            let index: usize = index_str.parse().expect("schema:<index>");
            let (short_name, file) = SELECTED10[index];
            let source = schema_source(cfa_root, file);
            let c = Constraint::compile(Grammar::json_schema(&source), vocab).unwrap();
            (short_name, c)
        } else {
            panic!("unknown kind {kind:?} (core|dispatch-parent|schema:<index>)");
        };
        let wall = started.elapsed();
        let saved_len = constraint.save().len();
        println!(
            "[compile-one] kind={label} rep={}/{} wall_ms={:.3}",
            rep + 1,
            repeats,
            wall.as_secs_f64() * 1000.0,
        );
        stat_line(&format!("{label}#fresh{rep}"), saved_len, &constraint);
    }
}

fn cmd_dispatch_compose(cache_dir: &Path, vocab: &Vocab, runs: usize) {
    println!("name\tartifact_bytes\tlr_states\tterminals\ttokenizer_states\tinternal_tsids\tinternal_tokens\tparser_dwa_states\tparser_dwa_transitions");
    for run in 0..runs {
        let started = Instant::now();
        let parent =
            Constraint::compile(Grammar::glrm(&dispatcher_literal_names_parent_source()), vocab)
                .unwrap();
        let parent_ms = started.elapsed();
        let schemas = SELECTED10
            .iter()
            .enumerate()
            .map(|(index, (short_name, _))| {
                load_constraint(&schema_cache_path(cache_dir, index, short_name))
            })
            .collect::<Vec<_>>();
        let bindings = (0..SELECTED10.len())
            .map(|index| (format!("TOOL_ARGS_SLOT_{index}"), &schemas[index]))
            .collect::<Vec<_>>();
        let refs = bindings
            .iter()
            .map(|(name, child)| (name.as_str(), *child))
            .collect::<Vec<_>>();
        if run == 0 {
            let parent_len = parent.save().len();
            stat_line("dispatch-parent#fresh", parent_len, &parent);
        }
        let started = Instant::now();
        let dispatch = parent.compose_compiled_subgrammars(&refs, vocab).unwrap();
        let compose = started.elapsed();
        let saved_len = dispatch.save().len();
        println!(
            "[dispatch-compose] run={}/{} parent_ms={:.3} compose10_ms={:.3} bytes={}",
            run + 1,
            runs,
            parent_ms.as_secs_f64() * 1000.0,
            compose.as_secs_f64() * 1000.0,
            saved_len,
        );
        if run == 0 {
            stat_line("dispatch#fresh", saved_len, &dispatch);
        }
    }
}

fn mask_set_bits(mask: &[u32]) -> Vec<u32> {
    let mut out = Vec::new();
    for (word_index, &word) in mask.iter().enumerate() {
        let mut live = word;
        while live != 0 {
            let bit = live.trailing_zeros() as usize;
            out.push((word_index * 32 + bit) as u32);
            live &= live - 1;
        }
    }
    out
}

fn cmd_diff_masks(cache_dir: &Path, prefix: &str, tokens_csv: &str) {
    let composed = load_constraint(&cache_dir.join("composed-latest.bin"));
    let mut state = composed.start();
    if !prefix.is_empty() {
        state.commit_bytes(prefix.as_bytes()).expect("prefix commit");
    }
    let tokens: Vec<u32> = if tokens_csv.is_empty() {
        Vec::new()
    } else {
        tokens_csv.split(',').map(|t| t.trim().parse().unwrap()).collect()
    };
    for (step, token) in tokens.iter().enumerate() {
        if let Err(error) = state.commit_token(*token) {
            println!("[diff-masks] replay stopped: step={step} token={token} rejected: {error}");
            return;
        }
    }
    let started = Instant::now();
    let static_mask = state.mask();
    let static_ms = started.elapsed().as_secs_f64() * 1000.0;
    let started = Instant::now();
    let dynamic_mask = state.fill_mask_dynamic_vec();
    let dynamic_ms = started.elapsed().as_secs_f64() * 1000.0;
    let s: std::collections::BTreeSet<u32> = mask_set_bits(&static_mask).into_iter().collect();
    let d: std::collections::BTreeSet<u32> = mask_set_bits(&dynamic_mask).into_iter().collect();
    let static_only: Vec<u32> = s.difference(&d).copied().collect();
    let dynamic_only: Vec<u32> = d.difference(&s).copied().collect();
    println!(
        "[diff-masks] prefix={:?} tokens={} static_admits={} dynamic_admits={} static_only={} dynamic_only={} static_ms={:.3} dynamic_ms={:.3}",
        prefix, tokens.len(), s.len(), d.len(), static_only.len(), dynamic_only.len(), static_ms, dynamic_ms,
    );
    println!("[diff-masks] static_only_first64={static_only:?}");
    println!("[diff-masks] dynamic_only_first64={dynamic_only:?}");
}

fn cmd_dump_schema(cfa_root: &Path, index: usize) {
    let (short_name, file) = SELECTED10[index];
    let source = schema_source(cfa_root, file);
    let glrm = Constraint::dump_json_schema_grammar_glrm(&source).unwrap();
    eprintln!("[dump-schema] {index:02} {short_name}: {} bytes", glrm.len());
    print!("{glrm}");
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let sub = args.first().map(String::as_str).unwrap_or("sizes");
    let mut cache_dir = PathBuf::from(DEFAULT_CACHE_DIR);
    let mut cfa_root = PathBuf::from(DEFAULT_CFA_ROOT);
    let mut runs = 5usize;
    let mut steps = 24usize;
    let mut which = String::from("static");
    let mut kind = String::from("core");
    let mut index = 0usize;
    let mut dispatch_name = String::from("dispatch-literal.bin");
    let mut positional: Vec<String> = Vec::new();
    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--cache-dir" => { i += 1; cache_dir = PathBuf::from(&args[i]); }
            "--cfa-root" => { i += 1; cfa_root = PathBuf::from(&args[i]); }
            "--runs" => { i += 1; runs = args[i].parse().expect("--runs integer"); }
            "--steps" => { i += 1; steps = args[i].parse().expect("--steps integer"); }
            "--which" => { i += 1; which = args[i].clone(); }
            "--kind" => { i += 1; kind = args[i].clone(); }
            "--dispatch" => { i += 1; dispatch_name = args[i].clone(); }
            "--index" => { i += 1; index = args[i].parse().expect("--index integer"); }
            other => positional.push(other.to_owned()),
        }
        i += 1;
    }
    if sub == "dump-schema" && !positional.is_empty() {
        index = positional[0].parse().expect("index integer");
    }
    match sub {
        "sizes" => cmd_sizes(&cache_dir, &dispatch_name),
        "link" => {
            let vocab = read_vocab_dump(&cache_dir.join("vocab_dump.bin"));
            cmd_link(&cache_dir, &vocab, &dispatch_name, runs);
        }
        "tbm" => {
            let vocab = read_vocab_dump(&cache_dir.join("vocab_dump.bin"));
            cmd_tbm(&cache_dir, &vocab, &which, steps);
        }
        "compile-one" => {
            let vocab = read_vocab_dump(&cache_dir.join("vocab_dump.bin"));
            cmd_compile_one(&cfa_root, &vocab, &kind, runs);
        }
        "dump-schema" => cmd_dump_schema(&cfa_root, index),
        "diff-masks" => {
            let prefix = positional.first().cloned().unwrap_or_default();
            let tokens = positional.get(1).cloned().unwrap_or_default();
            cmd_diff_masks(&cache_dir, &prefix, &tokens);
        }
        "dispatch-compose" => {
            let vocab = read_vocab_dump(&cache_dir.join("vocab_dump.bin"));
            cmd_dispatch_compose(&cache_dir, &vocab, runs);
        }
        other => panic!("unknown subcommand {other:?} (sizes|link|tbm|compile-one|dump-schema|dispatch-compose|diff-masks)"),
    }
}
