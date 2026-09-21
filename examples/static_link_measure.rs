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

/// Sorted `schema-*.bin` cache files: exactly 10, names mapped to SELECTED10
/// index by the `schema-{index:02}-{short_name}` convention. Returns
/// (path, short_name) in index order; panics unless exactly 10 files exist.
fn sorted_schema_cache_files(cache_dir: &Path) -> Vec<(PathBuf, String)> {
    let mut names: Vec<String> = std::fs::read_dir(cache_dir)
        .expect("read cache dir")
        .map(|entry| entry.expect("cache dir entry").file_name().to_string_lossy().into_owned())
        .filter(|name| name.starts_with("schema-") && name.ends_with(".bin"))
        .collect();
    for name in &names {
        assert!(
            cache_dir.join(name).is_file(),
            "cache entry {} is not a regular file",
            cache_dir.join(name).display(),
        );
    }
    names.sort();
    assert_eq!(names.len(), 10, "cache must hold exactly 10 schema-*.bin files");
    names
        .into_iter()
        .enumerate()
        .map(|(index, name)| {
            let (short_name, _) = SELECTED10[index];
            let expected = schema_cache_path(cache_dir, index, short_name);
            assert_eq!(
                cache_dir.join(&name),
                expected,
                "schema file {name:?} does not equal expected {expected:?}",
            );
            (expected, short_name.to_string())
        })
        .collect()
}

/// matrix-link: full public-API static + dynamic compose timing per matrix
/// case. Single setup path returns (parent, child, slot, input records,
/// parent_compile_ms, cache_load_ms); each arm clones the parent OUTSIDE its
/// clock and times ONLY the compose API call. Timed outputs are kept as
/// Constraint and validated directly (never recomposed). Stage profile lines
/// go to stderr inside each timed arm only; the driver parses them later.
/// Unknown cases rejected up front.
/// NOTE: inner parent source is dispatcher_literal_names_parent_source().
/// Shape and token-id bases match link_timing_all_compositions'
/// dispatch_parent_source() (both 128_330+i); only alternation-joiner
/// whitespace differs, which the grammar parses identically — no semantic
/// change claimed or implied.
#[allow(clippy::too_many_arguments)]
fn matrix_link_setup(
    cache_dir: &Path,
    vocab: &Vocab,
    dispatch_name: &str,
    matrix_case: &str,
) -> (
    Constraint,
    Constraint,
    String,
    serde_json::Value,
    f64,
    f64,
) {
    // Resolve the case BEFORE any compose: unknown cases rejected up front.
    let inner_index: Option<usize> = if matrix_case == "outer" {
        None
    } else if let Some(n) = matrix_case.strip_prefix("inner-") {
        let n: usize = n.parse().expect("--matrix-case inner-N integer");
        assert!(n < 10, "--matrix-case inner-N must be 0..9");
        Some(n)
    } else {
        panic!("unknown --matrix-case {matrix_case:?} (inner-0..inner-9|outer)");
    };
    // Parent compile clock (inner only; outer parent is the loaded core).
    let pc0 = Instant::now();
    let parent_source: Option<String> =
        inner_index.map(|_| dispatcher_literal_names_parent_source());
    // Same lexical tokens on both parent spellings: TOOL_ARGS_SLOT_i bound to
    // 128_330+i in each. Parse-level assertion, not a semantic claim.
    if let Some(source) = parent_source.as_ref() {
        for i in 0..SELECTED10.len() {
            assert!(
                source.contains(&format!("TOOL_ARGS_SLOT_{i} ::= @token({})", 128_320 + 10 + i)),
                "matrix-link {matrix_case}: parent source lexical token mismatch at slot {i}",
            );
        }
    }
    let parent_compile_ms = pc0.elapsed().as_secs_f64() * 1000.0;
    // Cache load clock: inner loads one schema; outer loads core+dispatch ONCE.
    let ld0 = Instant::now();
    let (parent, child, slot, records) = if let Some(n) = inner_index {
        let files = sorted_schema_cache_files(cache_dir);
        let (schema_path, short_name) = &files[n];
        let schema_bytes = fs::read(schema_path)
            .unwrap_or_else(|error| panic!("read {}: {error}", schema_path.display()));
        let schema = Constraint::load(&schema_bytes)
            .unwrap_or_else(|error| panic!("load {}: {error}", schema_path.display()));
        let parent = Constraint::compile(
            Grammar::glrm(parent_source.as_ref().expect("inner source")),
            vocab,
        )
        .expect("dispatch parent compile");
        let records = serde_json::json!({
            "inputs_desc": format!("dispatch-parent+{short_name}"),
            "schema_path": schema_path.to_string_lossy(),
            "schema_short_name": short_name,
            "schema_bytes": schema_bytes.len(),
        });
        (parent, schema, format!("TOOL_ARGS_SLOT_{n}"), records)
    } else {
        let core = load_constraint(&cache_dir.join("core.bin"));
        let dispatch = load_constraint(&cache_dir.join(dispatch_name));
        let records = serde_json::json!({
            "inputs_desc": "core+dispatch",
            "core_bytes": core.save().len(),
            "dispatch_bytes": dispatch.save().len(),
        });
        (core, dispatch, String::from("PROGRAMMATIC_TOOL_SUFFIX"), records)
    };
    let cache_load_ms = ld0.elapsed().as_secs_f64() * 1000.0;
    (parent, child, slot, records, parent_compile_ms, cache_load_ms)
}

/// Serialize input stats (stat_line getters) into JSON for the per-case record.
fn constraint_stats(name: &str, bytes_len: usize, c: &Constraint) -> serde_json::Value {
    stat_line(name, bytes_len, c);
    serde_json::json!({
        "name": name,
        "artifact_bytes": bytes_len,
        "lr_states": c.num_parser_states(),
        "terminals": c.num_terminals(),
        "tokenizer_states": c.num_tokenizer_states(),
        "internal_tsids": c.internal_tsid_count(),
        "internal_tokens": c.final_internal_token_count(),
        "parser_dwa_states": c.parser_dwa_num_states(),
        "parser_dwa_transitions": c.parser_dwa_num_transitions(),
    })
}

#[allow(clippy::too_many_arguments)]
fn cmd_matrix_link_with_vocab(
    cache_dir: &Path,
    vocab: &Vocab,
    dispatch_name: &str,
    matrix_case: &str,
    out_path: &str,
    repeat_index: usize,
    vocab_ms: f64,
) {
    let (parent, child, slot, input_records, parent_compile_ms, cache_load_ms) =
        matrix_link_setup(cache_dir, vocab, dispatch_name, matrix_case);
    // Common input stats, both stdout (stat_line) and JSON record.
    let parent_stats = constraint_stats(
        &format!("matrix-input-parent#{matrix_case}"),
        parent.save().len(),
        &parent,
    );
    let child_stats = constraint_stats(
        &format!("matrix-input-child#{matrix_case}"),
        child.save().len(),
        &child,
    );
    eprintln!("[matrix-link] case={matrix_case} repeat={repeat_index} vocab_ms={vocab_ms:.3} parent_compile_ms={parent_compile_ms:.3} cache_load_ms={cache_load_ms:.3}");

    // Static arm: begin marker + flush BEFORE t0; end marker AFTER elapsed.
    let static_parent = parent.clone();
    eprintln!("[matrix-link][static-arm-begin] case={matrix_case} repeat={repeat_index}");
    {
        use std::io::Write as _;
        let _ = std::io::stderr().flush();
    }
    let t0 = Instant::now();
    let static_comp = static_parent
        .compose_compiled_subgrammars(&[(slot.as_str(), &child)], vocab)
        .unwrap_or_else(|error| panic!("{matrix_case} static compose: {error}"));
    let static_ms = t0.elapsed().as_secs_f64() * 1000.0;
    eprintln!("[matrix-link][static-arm-end] case={matrix_case} repeat={repeat_index} static_ms={static_ms:.3}");
    {
        use std::io::Write as _;
        let _ = std::io::stderr().flush();
    }
    println!("[matrix-link] case={matrix_case} repeat={repeat_index} static_ms={static_ms:.3}");

    // Dynamic arm over the same original parent (clone outside the clock).
    let dynamic_parent = parent.clone();
    let t0 = Instant::now();
    let dynamic_comp = dynamic_parent
        .compose_compiled_subgrammars_dynamic(&[(slot.as_str(), &child)], vocab)
        .unwrap_or_else(|error| panic!("{matrix_case} dynamic compose: {error}"));
    let dynamic_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("[matrix-link] case={matrix_case} repeat={repeat_index} dynamic_ms={dynamic_ms:.3}");

    // Validate the TIMED outputs directly (never recompose): owned Vec masks.
    let v0 = Instant::now();
    let static_mask: Vec<u32> = static_comp.start().mask();
    let dynamic_mask: Vec<u32> = dynamic_comp.start().mask();
    let masks_equal = static_mask == dynamic_mask;
    let masks_nonempty = static_mask.iter().any(|&w| w != 0);
    let validation_ms = v0.elapsed().as_secs_f64() * 1000.0;
    assert!(masks_equal, "matrix-link {matrix_case}: initial static/dynamic masks differ");
    assert!(masks_nonempty, "matrix-link {matrix_case}: initial mask unexpectedly empty");
    // Persist the SAME timed outputs before the JSON marker.
    let s_bytes = static_comp.save();
    let d_bytes = dynamic_comp.save();
    let out_dir = Path::new(out_path);
    let is_dir = out_path.ends_with('/') || out_path.ends_with(std::path::MAIN_SEPARATOR);
    let out_file = if is_dir {
        std::fs::create_dir_all(out_dir).expect("create out dir");
        out_dir.join(format!("{matrix_case}-rep{repeat_index}.json"))
    } else {
        if let Some(parent) = out_dir.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).expect("create out parent");
            }
        }
        out_dir.to_path_buf()
    };
    // Fresh-vs-reload per backend (separately timed validation): each
    // backend's reload mask must equal ITS OWN fresh mask, not just each other.
    let r0 = Instant::now();
    let rs = Constraint::load(&s_bytes).expect("reload static");
    let rd = Constraint::load(&d_bytes).expect("reload dynamic");
    let rs_mask: Vec<u32> = rs.start().mask();
    let rd_mask: Vec<u32> = rd.start().mask();
    let reload_static_equal = rs_mask == static_mask;
    let reload_dynamic_equal = rd_mask == dynamic_mask;
    let reload_ms = r0.elapsed().as_secs_f64() * 1000.0;
    assert!(reload_static_equal, "matrix-link {matrix_case}: static reload mask differs from fresh");
    assert!(reload_dynamic_equal, "matrix-link {matrix_case}: dynamic reload mask differs from fresh");
    let stem = out_file
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or_else(|| panic!("matrix-link {matrix_case}: out path has no stem"));
    let dir = out_file.parent().unwrap_or(Path::new("."));
    let static_path = dir.join(format!("{stem}.static.bin"));
    let dynamic_path = dir.join(format!("{stem}.dynamic.bin"));
    std::fs::write(&static_path, &s_bytes).expect("write static");
    std::fs::write(&dynamic_path, &d_bytes).expect("write dynamic");
    let doc = serde_json::json!({
        "case": matrix_case,
        "repeat": repeat_index,
        "success": true,
        "input_records": input_records,
        "parent_stats": parent_stats,
        "child_stats": child_stats,
        "vocab_ms": vocab_ms,
        "parent_compile_ms": parent_compile_ms,
        "cache_load_ms": cache_load_ms,
        "static_ms": static_ms,
        "static_bytes": s_bytes.len(),
        "static_path": static_path.to_string_lossy(),
        "dynamic_ms": dynamic_ms,
        "dynamic_bytes": d_bytes.len(),
        "dynamic_path": dynamic_path.to_string_lossy(),
        "validation_ms": validation_ms,
        "reload_ms": reload_ms,
        "initial_masks_equal": masks_equal,
        "initial_masks_nonempty": masks_nonempty,
        "reload_static_equal": reload_static_equal,
        "reload_dynamic_equal": reload_dynamic_equal,
        "diagnostic_overhead_note": "profile env fixed ON identically across all observations; stage-line cost is inside arm timers, reported as limitation not subtracted",
    });
    fs::write(&out_file, serde_json::to_string_pretty(&doc).expect("json")).expect("write json");
    println!("[matrix-link] wrote {}", out_file.display());
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
    if sorted.is_empty() {
        return f64::NAN;
    }
    let index = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[index.min(sorted.len() - 1)]
}

fn print_percentiles(name: &str, unit: &str, samples: &[f64]) {
    let mut s = samples.to_vec();
    s.sort_by(f64::total_cmp);
    println!(
        "PERCENTILE name={name} unit={unit} n={} p50={:.3} p90={:.3} p95={:.3} p99={:.3} max={:.3}",
        s.len(),
        percentile(&s, 0.50),
        percentile(&s, 0.90),
        percentile(&s, 0.95),
        percentile(&s, 0.99),
        percentile(&s, 1.0),
    );
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

/// Shared current-prefix coverage: the 22 composed byte prefixes driven by
/// `cmd_tbm` for static/dynamic/mono (NOT the historical 22 differential
/// scenarios, whose case identity is unrecovered). Factored so `cmd_tbm` and
/// the bounded `prefix-diff` command cannot drift.
fn composed_prefixes() -> Vec<Vec<u8>> {
    let mut v = vec![Vec::<u8>::new(), b"const x = tools".to_vec()];
    for tool in 0..SELECTED10.len() {
        v.push(format!("const x = tools.tool_{tool}(").into_bytes());
        v.push(format!("const x = tools.tool_{tool}({{").into_bytes());
    }
    v
}

fn rng_choose_token(mask: &[u32], rng: &mut u64) -> Option<u32> {
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
    let composed_prefixes: Vec<Vec<u8>> = composed_prefixes();
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
            let Some(token) = rng_choose_token(&mask, &mut rng) else {
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
        "TBM_RESULT which={which} scenarios={} steps_per_scenario={} samples={} mask_p50_us={:.3} mask_p90_us={:.3} mask_p95_us={:.3} mask_p99_us={:.3} mask_max_us={:.3} tbm_p50_us={:.3} tbm_p90_us={:.3} tbm_p95_us={:.3} tbm_p99_us={:.3} tbm_max_us={:.3}",
        committed_scenarios,
        steps,
        tbm_ns.len(),
        percentile(&mask_ns, 0.50) / 1000.0,
        percentile(&mask_ns, 0.90) / 1000.0,
        percentile(&mask_ns, 0.95) / 1000.0,
        percentile(&mask_ns, 0.99) / 1000.0,
        percentile(&mask_ns, 1.0) / 1000.0,
        percentile(&tbm_ns, 0.50) / 1000.0,
        percentile(&tbm_ns, 0.90) / 1000.0,
        percentile(&tbm_ns, 0.95) / 1000.0,
        percentile(&tbm_ns, 0.99) / 1000.0,
        percentile(&tbm_ns, 1.0) / 1000.0,
    );
    print_percentiles("mask_ns", "ns", &mask_ns);
    print_percentiles("tbm_ns", "ns", &tbm_ns);
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

/// Single timed compare-record step shared by the prefix and continuation
/// phases of `cmd_prefix_diff`. Takes both masks (already timed by the
/// caller), asserts full original-token-domain equality + accepting equality,
/// folds the position into `prefix-diff-checksum/1`, pushes the record, and
/// returns the static mask words for chained decisions (rng draw, admission).
/// Checksum input is exactly what the JSON records: the reported static_hash
/// value + global index + phase tag + accepting bit (NOT raw mask words —
/// those never leave the process; the driver recomputes from the recorded
/// hash identically).
#[allow(clippy::too_many_arguments)]
fn record_prefix_diff_position(
    scenario: usize,
    global_index: usize,
    phase: &'static str,
    prefix_byte_offset: Option<usize>,
    continuation_step: Option<usize>,
    mask_static: &[u32],
    mask_dynamic: &[u32],
    static_accepting: bool,
    dynamic_accepting: bool,
    static_ns: f64,
    dynamic_ns: f64,
    mask_width_words: &mut Option<usize>,
    checksum: &mut u64,
    positions: &mut Vec<serde_json::Value>,
) -> Vec<u32> {
    let width = match *mask_width_words {
        Some(width) => width,
        None => {
            assert!(!mask_static.is_empty(), "refusing zero-width mask scenario={scenario}");
            assert_eq!(
                mask_dynamic.len(),
                mask_static.len(),
                "initial mask width mismatch scenario={scenario}",
            );
            *mask_width_words = Some(mask_static.len());
            mask_static.len()
        }
    };
    assert_eq!(
        mask_static.len(),
        width,
        "static mask width drift scenario={scenario} global={global_index}",
    );
    assert_eq!(
        mask_dynamic.len(),
        width,
        "dynamic mask width drift scenario={scenario} global={global_index}",
    );
    assert_eq!(
        mask_static, mask_dynamic,
        "mask mismatch scenario={scenario} global={global_index} phase={phase}",
    );
    let empty = mask_static.iter().all(|&word| word == 0);
    let static_hash = format!("{:016x}", fnv_word_hash(0xcbf29ce484222325, mask_static));
    let dynamic_hash = format!("{:016x}", fnv_word_hash(0xcbf29ce484222325, mask_dynamic));
    assert_eq!(static_hash.len(), 16);
    assert_eq!(dynamic_hash.len(), 16);
    assert_eq!(static_hash, dynamic_hash);
    assert_eq!(
        static_accepting, dynamic_accepting,
        "accepting divergence scenario={scenario} global={global_index} phase={phase}",
    );
    // Terminal-empty rule: an empty mask FAILS unless both sides are
    // accepting (accepted terminal stop, recorded via `empty` + accepting).
    assert!(
        !empty || static_accepting,
        "unexpected non-accepting empty mask scenario={scenario} global={global_index} phase={phase}",
    );
    // prefix-diff-checksum/1: fold the RECORDED static_hash bytes (16 hex
    // chars = the u64 value's 8 bytes = 2 LE u32 words, matching the
    // driver's recompute) + global index + phase tag + accepting bit.
    // Claims nothing about legacy 6f1a9c4ce273d3c2 / 2e980a08a2cbbd4c
    // (different streams).
    positions.push(serde_json::json!({
        "index": global_index,
        "phase": phase,
        "prefix_byte_offset": prefix_byte_offset,
        "continuation_step": continuation_step,
        "empty": empty,
        "width_words": mask_static.len(),
        "static_hash": static_hash.clone(),
        "dynamic_hash": dynamic_hash,
        "static_accepting": static_accepting,
        "dynamic_accepting": dynamic_accepting,
        "static_mask_ns": static_ns,
        "dynamic_mask_ns": dynamic_ns,
    }));
    // Fold the RECORDED hash (16 hex chars -> 8 LE u32 words, exactly what
    // the driver recomputes from JSON) + index + phase + accepting.
    let hash_bytes = hex16_to_words(&static_hash);
    for (word_index, &word) in hash_bytes.iter().enumerate() {
        *checksum ^= (word as u64).wrapping_add(word_index as u64);
        *checksum = checksum.wrapping_mul(0x100000001b3);
    }
    let phase_tag: u64 = if phase == "prefix" { 0x5052_4546 } else { 0x434f_4e54 };
    *checksum ^= (global_index as u64).wrapping_add(phase_tag);
    *checksum = checksum.wrapping_mul(0x100000001b3);
    *checksum ^= (static_accepting as u64).wrapping_add(phase_tag);
    *checksum = checksum.wrapping_mul(0x100000001b3);
    mask_static.to_vec()
}

/// Decode a 16-hex-char mask hash into 2 little-endian u32 words (the exact
/// bytes of the u64 value `fnv_word_hash` returned, formatted `{:016x}`;
/// shared with the driver recompute).
fn hex16_to_words(hash: &str) -> [u32; 2] {
    assert_eq!(hash.len(), 16, "mask hash must be 16 hex chars");
    let bytes = (0..16)
        .step_by(2)
        .map(|i| u8::from_str_radix(&hash[i..i + 2], 16).expect("hash hex must parse"))
        .collect::<Vec<_>>();
    assert_eq!(bytes.len(), 8);
    [
        u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
    ]
}

/// Bounded differential over the 22 CURRENT `composed_prefixes` (current-prefix
/// coverage, NOT recovered historical scenarios): loads prepared outer
/// Static/Dynamic artifacts + exact vocab, then records TWO phases with FULL
/// original-token mask + accepting compare at every state. Phase PREFIX covers
/// the consumed prefix byte-by-byte: root is byte-count 0, after each byte is
/// byte-count 1..=len; `commit_bytes` validates raw-byte grammar progress
/// directly (ASCII bytes are NOT vocab token IDs — no token-domain admission
/// applies to bytes; a partial byte prefix needs no singleton vocab token).
/// The completed-prefix state doubles as the continuation initial state and is
/// NOT double-counted. Phase CONTINUATION runs up to max_steps deterministic
/// rng-chosen token commits drawn from the retained last-compared mask.
/// Empty masks FAIL unless both sides are accepting (accepted terminal stop).
/// Emits per-position phase/offsets/hashes/acceptance/timings, seed/prefix
/// identity, stop reason, one JSON object + `PREFIX_DIFF_OK` only after all
/// assertions. One scenario per process.
fn cmd_prefix_diff(
    vocab: &Vocab,
    static_path: &str,
    dynamic_path: &str,
    scenario: usize,
    max_steps: usize,
    seed: u64,
) {
    let prefixes = composed_prefixes();
    assert!(
        scenario < prefixes.len(),
        "scenario {scenario} out of range (0..{})",
        prefixes.len()
    );
    assert_eq!(prefixes.len(), 22, "current-prefix coverage must stay 22");
    let prefix = &prefixes[scenario];
    // Opt-in diagnostic phase markers (existing profile env only, flushed):
    // the final JSON is buffered, so a hang would otherwise leave zero
    // output. First-8 positions only for mask/commit markers; load/bind
    // always marked (one line each, no vocab spam). Completed JSON unchanged.
    let phase_mark = std::env::var_os("GLRMASK_PROFILE_COMPOSE").is_some()
        || std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some();
    let mark = |msg: &str| {
        if phase_mark {
            eprintln!("[prefix-diff-phase] scenario={scenario} {msg}");
        }
    };
    mark(&format!("input-load start static={static_path} dynamic={dynamic_path}"));
    let static_bytes =
        fs::read(static_path).unwrap_or_else(|error| panic!("read {static_path}: {error}"));
    let dynamic_bytes =
        fs::read(dynamic_path).unwrap_or_else(|error| panic!("read {dynamic_path}: {error}"));
    mark("input-load done");
    assert!(!static_bytes.is_empty(), "refusing empty static artifact");
    assert!(!dynamic_bytes.is_empty(), "refusing empty dynamic artifact");
    assert_ne!(
        static_path, dynamic_path,
        "static and dynamic artifact paths must differ",
    );
    let static_c = Constraint::load(&static_bytes).expect("load static artifact");
    let dynamic_c = Constraint::load(&dynamic_bytes).expect("load dynamic artifact");
    mark("load done");
    // Pair identity via existing public metadata APIs: same terminal /
    // tokenizer-state counts, exact-vocab bind on both. NOTE (verified
    // against matrix-vfinal outer-rep0): final_original_token_map is NOT
    // asserted equal — the static (walk-shard/compact) and dynamic
    // representations legitimately use different internal token layouts over
    // the same original vocabulary. Same-vocab is pinned by bind_vocab_exact
    // on both; map lengths are recorded for provenance, not asserted.
    assert_eq!(
        dynamic_c.num_terminals(),
        static_c.num_terminals(),
        "pair terminal counts must match",
    );
    assert_eq!(
        dynamic_c.num_tokenizer_states(),
        static_c.num_tokenizer_states(),
        "pair tokenizer-state counts must match",
    );
    let static_map_len = static_c.final_original_token_map().len();
    let dynamic_map_len = dynamic_c.final_original_token_map().len();
    assert!(static_map_len > 0, "static token map must be nonempty");
    assert!(dynamic_map_len > 0, "dynamic token map must be nonempty");
    for (label, constraint) in [("static", &static_c), ("dynamic", &dynamic_c)] {
        mark(&format!("bind start {label}"));
        constraint
            .clone()
            .bind_vocab_exact(vocab)
            .unwrap_or_else(|error| panic!("{label} must bind the exact vocab: {error}"));
        mark(&format!("bind done {label}"));
    }
    let mut st_static = static_c.start();
    let mut st_dynamic = dynamic_c.start();
    let mut rng = seed;
    // prefix-diff-checksum/1: FNV-1a mix over per-position (mask words, global
    // index, phase tag, accepting bit). Folded in record_prefix_diff_position.
    // Claims NOTHING about legacy 6f1a9c4ce273d3c2 / 2e980a08a2cbbd4c values
    // (different streams); coverage + per-position hash equality is the gate.
    let mut checksum: u64 = 0xcbf29ce484222325;
    let mut positions: Vec<serde_json::Value> = Vec::new();
    // Original-token widths: derived from the first TIMED masks (no warmup —
    // a static-only warmup would bias the first static timing), asserted
    // nonzero then consistent at every later position.
    let mut mask_width_words: Option<usize> = None;
    let mut global_index = 0usize;
    let mut stop_reason: Option<String> = None;
    // Retained last-compared STATIC mask: drives the continuation rng draw
    // and chosen-token admission without recomputing untimed masks. Valid
    // because every recorded mask pair was asserted equal at record time.
    // (Prefix bytes need no admission mask: commit_bytes validates raw-byte
    // grammar progress directly. ASCII byte values are NOT vocab token IDs —
    // no token-domain admission applies; a partial byte prefix needs no
    // singleton vocab token. Only full masks at byte boundaries are compared.)
    let mut retained_static_mask: Option<Vec<u32>> = None;
    // ---- Phase PREFIX: root (byte-count 0) + one full-compare position per
    // prefix byte (byte-count 1..=len). ----
    {
        if global_index < 8 {
            mark(&format!("mask start prefix global={global_index}"));
        }
        let t0 = Instant::now();
        let mask_static = st_static.mask();
        let static_ns = t0.elapsed().as_secs_f64() * 1e9;
        if global_index < 8 {
            mark(&format!("mask done static prefix global={global_index}"));
        }
        let t0 = Instant::now();
        let mask_dynamic = st_dynamic.mask();
        let dynamic_ns = t0.elapsed().as_secs_f64() * 1e9;
        if global_index < 8 {
            mark(&format!("mask done dynamic prefix global={global_index}"));
        }
        let root_accepting = st_static.is_accepting();
        assert_eq!(
            root_accepting, st_dynamic.is_accepting(),
            "root accepting divergence scenario={scenario}",
        );
        if mask_static.iter().all(|&word| word == 0) {
            assert!(
                root_accepting,
                "empty root mask without accepting scenario={scenario}",
            );
            stop_reason = Some("accepted-terminal-at-root".to_string());
        }
        record_prefix_diff_position(
            scenario, global_index, "prefix", Some(0), None,
            &mask_static, &mask_dynamic,
            root_accepting, root_accepting,
            static_ns, dynamic_ns,
            &mut mask_width_words, &mut checksum, &mut positions,
        );
        retained_static_mask = Some(mask_static);
        global_index = global_index.checked_add(1).expect("global index overflow");
    }
    if stop_reason.is_none() {
        for (consumed, &byte) in prefix.iter().enumerate() {
            if global_index < 8 {
                mark(&format!("commit start prefix consumed={consumed} global={global_index}"));
            }
            let r_static = st_static.commit_bytes(&[byte]);
            let r_dynamic = st_dynamic.commit_bytes(&[byte]);
            if global_index < 8 {
                mark(&format!("commit done prefix consumed={consumed} global={global_index}"));
            }
            assert_eq!(
                r_static.is_ok(),
                r_dynamic.is_ok(),
                "prefix commit divergence scenario={scenario} consumed={consumed}",
            );
            assert!(
                r_static.is_ok(),
                "prefix commit must succeed scenario={scenario} consumed={consumed}",
            );
            if global_index < 8 {
                mark(&format!("mask start prefix-after-byte consumed={consumed} global={global_index}"));
            }
            let t0 = Instant::now();
            let mask_static = st_static.mask();
            let static_ns = t0.elapsed().as_secs_f64() * 1e9;
            if global_index < 8 {
                mark(&format!("mask done static prefix-after-byte consumed={consumed} global={global_index}"));
            }
            let t0 = Instant::now();
            let mask_dynamic = st_dynamic.mask();
            let dynamic_ns = t0.elapsed().as_secs_f64() * 1e9;
            if global_index < 8 {
                mark(&format!("mask done dynamic prefix-after-byte consumed={consumed} global={global_index}"));
            }
            let accepting = st_static.is_accepting();
            assert_eq!(
                accepting, st_dynamic.is_accepting(),
                "accepting divergence scenario={scenario} after byte {consumed}",
            );
            if mask_static.iter().all(|&word| word == 0) {
                assert!(
                    accepting,
                    "non-accepting empty mask after prefix byte {consumed} scenario={scenario}",
                );
                stop_reason = Some(format!("accepted-terminal-after-prefix-byte-{consumed}"));
            }
            record_prefix_diff_position(
                scenario, global_index, "prefix", Some(consumed + 1), None,
                &mask_static, &mask_dynamic,
                accepting, accepting,
                static_ns, dynamic_ns,
                &mut mask_width_words, &mut checksum, &mut positions,
            );
            retained_static_mask = Some(mask_static);
            global_index = global_index.checked_add(1).expect("global index overflow");
            if stop_reason.is_some() {
                break;
            }
        }
    }
    let prefix_positions = positions.len();
    // max_steps budget for continuation only: completed-prefix state is the
    // continuation initial state (NOT double-counted). Continuation records at
    // most max_steps further commits, each exactly one new position.
    let max_continuation = max_steps;
    let mut continuation_steps = 0usize;
    if stop_reason.is_none() {
        loop {
            if continuation_steps >= max_continuation {
                stop_reason = Some(format!("completed-{max_continuation}-steps"));
                break;
            }
            // Draw from the RETAINED last-compared static mask (asserted equal
            // to dynamic at record time) — no untimed recompute. Admission of
            // the chosen ordinary token is checked, then BOTH commits must
            // succeed: any admitted-token commit error is a correctness bug
            // and fails loudly. No silent EOS branch exists (no exact-EOS
            // token API is exposed; only is_accepting, already compared).
            let mask_here = retained_static_mask
                .as_ref()
                .expect("retained mask always set after prefix phase");
            let Some(token) = rng_choose_token(mask_here, &mut rng) else {
                panic!(
                    "rng found no choice in nonempty retained mask scenario={scenario} step={continuation_steps}"
                );
            };
            assert!(
                mask_admits_token(mask_here, token),
                "chosen token {token} not admitted scenario={scenario} step={continuation_steps}",
            );
            // Unguarded single-line per-step markers (step + token id only, no
            // mask dump): the first-8 sampling hid the step-8 hang location.
            mark(&format!("loop top continuation step={continuation_steps} global={global_index}"));
            mark(&format!("commit start continuation step={continuation_steps} token={token}"));
            let r_static = st_static.commit_token(token);
            let r_dynamic = st_dynamic.commit_token(token);
            mark(&format!("commit done continuation step={continuation_steps} token={token} static_ok={} dynamic_ok={}", r_static.is_ok(), r_dynamic.is_ok()));
            assert_eq!(
                r_static.is_ok(),
                r_dynamic.is_ok(),
                "commit divergence scenario={scenario} step={continuation_steps} token={token}",
            );
            assert!(
                r_static.is_ok(),
                "chosen ordinary token commit must succeed scenario={scenario} step={continuation_steps} token={token}: static={r_static:?} dynamic={r_dynamic:?}",
            );
            let t0 = Instant::now();
            let mask_static = st_static.mask();
            let static_ns = t0.elapsed().as_secs_f64() * 1e9;
            mark(&format!("mask done static continuation step={continuation_steps} global={global_index}"));
            let t0 = Instant::now();
            let mask_dynamic = st_dynamic.mask();
            let dynamic_ns = t0.elapsed().as_secs_f64() * 1e9;
            mark(&format!("mask done dynamic continuation step={continuation_steps} global={global_index}"));
            mark(&format!("accepting start continuation step={continuation_steps} global={global_index}"));
            let accepting = st_static.is_accepting();
            assert_eq!(
                accepting, st_dynamic.is_accepting(),
                "accepting divergence scenario={scenario} step={continuation_steps}",
            );
            mark(&format!("accepting done continuation step={continuation_steps} global={global_index} accepting={accepting}"));
            if mask_static.iter().all(|&word| word == 0) {
                assert!(
                    accepting,
                    "non-accepting empty mask after continuation commit scenario={scenario} step={continuation_steps}",
                );
                stop_reason = Some("accepted-terminal-after-commit".to_string());
            }
            record_prefix_diff_position(
                scenario, global_index, "continuation", None, Some(continuation_steps),
                &mask_static, &mask_dynamic,
                accepting, accepting,
                static_ns, dynamic_ns,
                &mut mask_width_words, &mut checksum, &mut positions,
            );
            retained_static_mask = Some(mask_static);
            global_index = global_index.checked_add(1).expect("global index overflow");
            continuation_steps = continuation_steps.checked_add(1).expect("step overflow");
            if stop_reason.is_some() {
                break;
            }
        }
    }
    if stop_reason.is_none() {
        stop_reason = Some("completed-prefix-only".to_string());
    }
    let stop_reason = stop_reason.expect("stop reason always set");
    assert!(!positions.is_empty(), "refusing empty-position record");
    assert_eq!(
        positions.len(),
        global_index,
        "position vector must hold one record per global index",
    );
    assert_eq!(
        prefix_positions,
        prefix.len().checked_add(1).expect("prefix budget overflow"),
        "prefix phase must record root + one position per byte",
    );
    assert!(
        continuation_steps <= max_continuation,
        "continuation {continuation_steps} exceeds max_steps {max_steps}",
    );
    let doc = serde_json::json!({
        "schema": "prefix-diff/2",
        "checksum_schema": "prefix-diff-checksum/1",
        "scenario": scenario,
        "prefix": String::from_utf8_lossy(prefix).into_owned(),
        "prefix_len": prefix.len(),
        "seed": seed,
        "max_steps": max_steps,
        "actual_positions": positions.len(),
        "prefix_positions": prefix_positions,
        "continuation_steps": continuation_steps,
        "stop_reason": stop_reason,
        "checksum": format!("{checksum:016x}"),
        "vocab_entries": vocab_entries_len(vocab),
        "static_token_map_len": static_map_len,
        "dynamic_token_map_len": dynamic_map_len,
        "positions": positions,
    });
    println!("{}", serde_json::to_string(&doc).expect("serialize prefix-diff"));
    eprintln!(
        "PREFIX_DIFF_OK scenario={scenario} positions={} checksum={checksum:016x}",
        positions.len()
    );
}

/// Trace-replay semantic gate over the recovered selected10 traces.
/// Replays pre-tokenized `token_ids` (no retokenization) through freshly
/// composed static + dynamic constraints, comparing the FULL mask word sets
/// initially and after every commit. Hash is an explicitly documented
/// custom FNV-style word mix (offset basis + per-word index add + FNV prime
/// multiply); it is REPORTED, never asserted against the legacy
/// `6f1a9c4ce273d3c2` oracle (original mask representation unrecovered —
/// legacy checksum unconfirmed).
fn fnv_word_hash(mut hash: u64, mask: &[u32]) -> u64 {
    for (index, &word) in mask.iter().enumerate() {
        hash ^= (word as u64).wrapping_add(index as u64);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn mask_admits_token(mask: &[u32], token: u32) -> bool {
    mask.get(token as usize / 32)
        .is_some_and(|word| word & (1u32 << (token % 32)) != 0)
}

fn vocab_entries_len(vocab: &Vocab) -> usize {
    vocab.entries_map().len()
}

fn cmd_trace_replay(
    cache_dir: &Path,
    vocab: &Vocab,
    dispatch_name: &str,
    traces_path: &str,
    only_trace: Option<&str>,
    samples_path: Option<&str>,
    trace_progress: bool,
    max_trace_commits: Option<usize>,
    prepare_out_dir: Option<&str>,
    prepared_in_dir: Option<&str>,
    trace_start: usize,
    trace_count: Option<usize>,
) {
    // Validate the --trace selector BEFORE any compile: exactly the three
    // historical labels or 'all'. Reject unknowns and non-integer ids early.
    const LABELS: [&str; 3] = ["sequential-3", "parallel-4", "all-tools"];
    if let Some(only) = only_trace {
        if only != "all" && !LABELS.contains(&only) {
            panic!("unknown --trace {only:?} (expected one of sequential-3|parallel-4|all-tools|all)");
        }
    }
    // Chunk scope requires a single selected trace (never 'all'/None).
    let chunked = trace_count.is_some();
    if chunked {
        let only = only_trace.expect("--trace-count requires --trace <single-label>");
        assert!(LABELS.contains(&only), "--trace-count requires a single trace label, got {only:?}");
    }
    if prepare_out_dir.is_some() && prepared_in_dir.is_some() {
        panic!("--prepare-out-dir and --prepared-in-dir are mutually exclusive");
    }
    let progress = |stage: &str, label: &str, index: usize, token: u32| {
        if trace_progress || index < 16 || index % 128 == 0 {
            eprintln!("[trace-progress] stage={stage} trace={label} index={index} token={token}");
        }
    };
    // Structured per-operation timing: one line AFTER EACH static_commit /
    // dynamic_commit / static_mask / dynamic_mask with elapsed_us, so even a
    // hard kill leaves timing data isolating which operation is slow.
    let timed = |op: &str, label: &str, index: usize, token: u32, elapsed_us: f64| {
        eprintln!("[trace-op] op={op} trace={label} index={index} token={token} elapsed_us={elapsed_us:.1}");
    };
    let raw = fs::read(traces_path).unwrap_or_else(|error| panic!("read {traces_path}: {error}"));
    let traces: Value = serde_json::from_slice(&raw).expect("trace JSON must parse");
    let list = traces.as_array().expect("trace JSON must be a list");
    // Provenance contract: exactly the three historical traces / token counts.
    let expected = [("sequential-3", 2864usize), ("parallel-4", 3114usize), ("all-tools", 5786usize)];
    assert_eq!(list.len(), 3, "trace file must hold exactly 3 traces");
    for (entry, (label, n)) in list.iter().zip(expected.iter()) {
        assert_eq!(entry.get("label").and_then(Value::as_str), Some(*label));
        assert_eq!(entry.get("n_tokens").and_then(Value::as_u64), Some(*n as u64));
        assert_eq!(
            entry.get("token_ids").and_then(Value::as_array).map(Vec::len),
            Some(*n),
        );
    }
    let max_token = list
        .iter()
        .flat_map(|t| t.get("token_ids").and_then(Value::as_array).cloned().unwrap_or_default())
        .map(|v| v.as_u64().expect("trace token id must be an integer"))
        .max()
        .unwrap_or(0);
    assert!(
        max_token < 128_256,
        "trace token ids must fit the 128,256-entry CFA Llama-3.1 vocab, got max {max_token}",
    );

    eprintln!("[trace-progress] stage=load BEGIN");
    let load_started = Instant::now();
    // Prepared-artifact mode: load fresh static/dynamic constraints saved by
    // --prepare-out-dir, bypassing the link (explicit load timing, NOT a
    // link benchmark). Otherwise compose fresh from core+dispatch.
    let (static_comp, dynamic_comp, load_note) = if let Some(dir) = prepared_in_dir {
        let dir = Path::new(dir);
        let t0 = Instant::now();
        let s_bytes = fs::read(dir.join("static.bin")).expect("prepared static.bin missing");
        let d_bytes = fs::read(dir.join("dynamic.bin")).expect("prepared dynamic.bin missing");
        let s_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let t0 = Instant::now();
        let static_comp = Constraint::load_with_vocab(&s_bytes, vocab).expect("load prepared static");
        let dynamic_comp = Constraint::load_with_vocab(&d_bytes, vocab).expect("load prepared dynamic");
        let l_ms = t0.elapsed().as_secs_f64() * 1000.0;
        eprintln!("[trace-progress] stage=prepared-load BEGIN");
        (static_comp, dynamic_comp, format!("prepared_read_ms={s_ms:.3} prepared_decode_ms={l_ms:.3}"))
    } else {
        let core_bytes = fs::read(cache_dir.join("core.bin")).expect("core.bin missing");
        let dispatch_bytes = fs::read(cache_dir.join(dispatch_name)).expect("dispatch cache missing");
        let core = load_constraint(&cache_dir.join("core.bin"));
        let dispatch = load_constraint(&cache_dir.join(dispatch_name));
        let _ = (core_bytes.len(), dispatch_bytes.len());
        eprintln!("[trace-progress] stage=static-link BEGIN");
        let link_started = Instant::now();
        let static_comp = core
            .clone()
            .compose_compiled_subgrammars(&[("PROGRAMMATIC_TOOL_SUFFIX", &dispatch)], vocab)
            .expect("static compose for trace replay");
        let static_link_ms = link_started.elapsed().as_secs_f64() * 1000.0;
        eprintln!("[trace-progress] stage=dynamic-link BEGIN");
        let link_started = Instant::now();
        let dynamic_comp = core
            .compose_compiled_subgrammars_dynamic(&[("PROGRAMMATIC_TOOL_SUFFIX", &dispatch)], vocab)
            .expect("dynamic compose for trace replay");
        let dynamic_link_ms = link_started.elapsed().as_secs_f64() * 1000.0;
        println!("[trace-replay] link_static_ms={static_link_ms:.3} link_dynamic_ms={dynamic_link_ms:.3}");
        (static_comp, dynamic_comp, String::from("fresh-link"))
    };
    let load_ms = load_started.elapsed().as_secs_f64() * 1000.0;
    println!("[trace-replay] load_ms={load_ms:.3} via={load_note}");

    // --prepare-out-dir: verify fresh-vs-reload FULL-mask equality on
    // initial + first 4 commits (each backend), then authorize artifacts.
    if let Some(dir) = prepare_out_dir {
        let dir = Path::new(dir);
        std::fs::create_dir_all(dir).expect("create prepare dir");
        // Reload round-trip through save/load_with_vocab with shared vocab.
        let s_bytes = static_comp.save();
        let d_bytes = dynamic_comp.save();
        let re_static = Constraint::load_with_vocab(&s_bytes, vocab).expect("reload static");
        let re_dynamic = Constraint::load_with_vocab(&d_bytes, vocab).expect("reload dynamic");
        // Verify on the first selected trace: initial + first 4 commits.
        let first_label = only_trace.unwrap_or("sequential-3");
        let first_ids: Vec<u32> = list
            .iter()
            .find(|e| e.get("label").and_then(Value::as_str) == Some(first_label))
            .expect("first trace present")
            .get("token_ids")
            .and_then(Value::as_array)
            .expect("token_ids")
            .iter()
            .map(|v| v.as_u64().expect("integer") as u32)
            .collect();
        for (backend, fresh_c, re_c) in [("static", &static_comp, &re_static), ("dynamic", &dynamic_comp, &re_dynamic)] {
            let mut st_fresh = fresh_c.start();
            let mut st_re = re_c.start();
            let mut m_fresh = st_fresh.mask();
            let mut m_re = st_re.mask();
            assert_eq!(m_fresh, m_re, "prepare verify {backend} initial mask differs");
            for (i, token) in first_ids.iter().take(4).enumerate() {
                assert!(mask_admits_token(&m_fresh, *token), "prepare verify {backend} token {token} not admitted at {i}");
                st_fresh.commit_token(*token).expect("fresh commit");
                st_re.commit_token(*token).expect("reload commit");
                m_fresh = st_fresh.mask();
                m_re = st_re.mask();
                assert_eq!(m_fresh, m_re, "prepare verify {backend} mask differs at commit {i}");
            }
        }
        println!("[trace-prepare] reload_verified backends=static,dynamic positions=initial+4 trace={first_label}");
        std::fs::write(dir.join("static.bin"), &s_bytes).expect("write static");
        std::fs::write(dir.join("dynamic.bin"), &d_bytes).expect("write dynamic");
        std::fs::write(dir.join("traces_name.txt"), traces_path.as_bytes()).expect("write traces ref");
        println!("[trace-prepare] wrote static.bin={} dynamic.bin={} dir={}", s_bytes.len(), d_bytes.len(), dir.display());
        return;
    }

    let mut total_commits = 0usize;
    let mut total_masks = 0usize;
    let mut hash: u64 = 0xcbf29ce484222325;
    let mut static_commit_ns: Vec<f64> = Vec::new();
    let mut dynamic_commit_ns: Vec<f64> = Vec::new();
    let mut static_mask_ns: Vec<f64> = Vec::new();
    let mut dynamic_mask_ns: Vec<f64> = Vec::new();
    let mut covered_traces = 0usize;
    for entry in list {
        let label = entry.get("label").and_then(Value::as_str).unwrap_or("?");
        if let Some(only) = only_trace {
            if only != "all" && only != label {
                continue;
            }
        }
        covered_traces += 1;
        // token_ids borrowed (no Value-array clone): validate integers + vocab
        // range element-wise.
        let arr = entry
            .get("token_ids")
            .and_then(Value::as_array)
            .expect("trace token_ids array");
        let vocab_size = vocab_entries_len(vocab);
        let mut ids: Vec<u32> = Vec::with_capacity(arr.len());
        for v in arr {
            let id = v.as_u64().expect("trace token id must be an integer") as u32;
            assert!(
                (id as usize) < vocab_size,
                "trace token id {id} out of actual vocab range {vocab_size}",
            );
            ids.push(id);
        }
        eprintln!("TRACE_REPLAY_BEGIN trace={label}");
        // Chunk mode: replay the exact prefix token_ids[..start] commit-only
        // (errors fatal), THEN compare full masks at `start` and after each
        // of `count` commits. Never start at an interior position without
        // the prefix.
        if chunked {
            let count = trace_count.unwrap();
            assert!(trace_start + count <= ids.len(), "chunk [{trace_start}..{trace_start}+{count}) out of bounds len={}", ids.len());
            let mut st_static = static_comp.start();
            let mut st_dynamic = dynamic_comp.start();
            for (pi, token) in ids[..trace_start].iter().enumerate() {
                let m = st_static.mask();
                assert!(mask_admits_token(&m, *token), "prefix token {token} not admitted at {pi}");
                st_static.commit_token(*token).unwrap_or_else(|e| panic!("prefix static commit failed at {pi}: {e:?}"));
                st_dynamic.commit_token(*token).unwrap_or_else(|e| panic!("prefix dynamic commit failed at {pi}: {e:?}"));
            }
            // Mask at chunk start (after prefix), then one per chunk commit.
            let mut positions: Vec<serde_json::Value> = Vec::with_capacity(count + 1);
            let mut op_us: Vec<serde_json::Value> = Vec::new();
            let mut record = |global: usize, m_static: &[u32], m_dynamic: &[u32]| {
                assert_eq!(m_static, m_dynamic, "chunk mask vectors differ at global {global}");
                positions.push(serde_json::json!({
                    "global_index": global,
                    "static_hash": format!("{:016x}", fnv_word_hash(0xcbf29ce484222325, m_static)),
                    "dynamic_hash": format!("{:016x}", fnv_word_hash(0xcbf29ce484222325, m_dynamic)),
                }));
            };
            let t0 = Instant::now();
            let mut m_static = st_static.mask();
            let mut us = t0.elapsed().as_secs_f64() * 1e6;
            op_us.push(serde_json::json!({"op": "static-mask", "global_index": trace_start, "us": us}));
            let t0 = Instant::now();
            let m_dynamic = st_dynamic.mask();
            us = t0.elapsed().as_secs_f64() * 1e6;
            op_us.push(serde_json::json!({"op": "dynamic-mask", "global_index": trace_start, "us": us}));
            record(trace_start, &m_static, &m_dynamic);
            for (k, token) in ids[trace_start..trace_start + count].iter().enumerate() {
                let global = trace_start + k;
                assert!(mask_admits_token(&m_static, *token), "chunk token {token} not admitted at global {global}");
                let c0 = Instant::now();
                st_static.commit_token(*token).unwrap_or_else(|e| panic!("chunk static commit failed at {global}: {e:?}"));
                us = c0.elapsed().as_secs_f64() * 1e6;
                op_us.push(serde_json::json!({"op": "static-commit", "global_index": global, "us": us}));
                let c0 = Instant::now();
                st_dynamic.commit_token(*token).unwrap_or_else(|e| panic!("chunk dynamic commit failed at {global}: {e:?}"));
                us = c0.elapsed().as_secs_f64() * 1e6;
                op_us.push(serde_json::json!({"op": "dynamic-commit", "global_index": global, "us": us}));
                let t0 = Instant::now();
                m_static = st_static.mask();
                us = t0.elapsed().as_secs_f64() * 1e6;
                op_us.push(serde_json::json!({"op": "static-mask", "global_index": global + 1, "us": us}));
                let t0 = Instant::now();
                let m_dynamic = st_dynamic.mask();
                us = t0.elapsed().as_secs_f64() * 1e6;
                op_us.push(serde_json::json!({"op": "dynamic-mask", "global_index": global + 1, "us": us}));
                record(global + 1, &m_static, &m_dynamic);
            }
            let end = trace_start + count;
            println!("TRACE_CHUNK_OK trace={label} start={trace_start} end={end} commits={count} masks={}", count + 1);
            // Chunk JSON (distinct from full/partial samples): per-position
            // global indices + fresh-seed full-word hashes + per-op us.
            // Written whenever --samples-out is given (default chunk path).
            let chunk_path = samples_path.unwrap_or("/tmp/mac-audit-20260919/chunk-last.json");
            let doc = serde_json::json!({
                "trace": label,
                "start": trace_start,
                "end": end,
                "commits": count,
                "masks": count + 1,
                "positions": positions,
                "op_us": op_us,
            });
            fs::write(chunk_path, serde_json::to_string(&doc).expect("chunk JSON")).expect("write chunk");
            println!("[trace-replay] chunk_written={chunk_path}");
            continue;
        }
        // Full/partial legacy path below; chunk mode already returned above.
        let mut st_static = static_comp.start();
        let mut st_dynamic = dynamic_comp.start();
        // Initial masks (position 0 of this trace).
        progress("initial-static-mask", label, 0, 0);
        let t0 = Instant::now();
        let mut m_static = st_static.mask();
        let us = t0.elapsed().as_secs_f64() * 1e6;
        static_mask_ns.push(us * 1e3);
        timed("initial-static-mask", label, 0, 0, us);
        progress("initial-dynamic-mask", label, 0, 0);
        let t0 = Instant::now();
        let m_dynamic = st_dynamic.mask();
        let us = t0.elapsed().as_secs_f64() * 1e6;
        dynamic_mask_ns.push(us * 1e3);
        timed("initial-dynamic-mask", label, 0, 0, us);
        total_masks += 1;
        hash = fnv_word_hash(hash, &m_static);
        if m_static != m_dynamic {
            let (a, b) = first_mask_difference(&m_static, &m_dynamic);
            panic!("TRACE_MISMATCH trace={label} index=initial(first mask) static_only_first={a:?} dynamic_only_first={b:?}");
        }
        for (index, token) in ids.iter().enumerate() {
            if max_trace_commits.is_some_and(|max| total_commits >= max) {
                break;
            }
            // Each trace token must be mask-admitted before commit; on
            // violation report trace/index/token and stop (no retry).
            if !mask_admits_token(&m_static, *token) {
                panic!("TRACE_MISMATCH trace={label} index={index} token={token} not-admitted-by-static-mask");
            }
            progress("commit", label, index, *token);
            let c0 = Instant::now();
            let r_static = st_static.commit_token(*token);
            let us = c0.elapsed().as_secs_f64() * 1e6;
            static_commit_ns.push(us * 1e3);
            timed("static-commit", label, index, *token, us);
            let c0 = Instant::now();
            let r_dynamic = st_dynamic.commit_token(*token);
            let us = c0.elapsed().as_secs_f64() * 1e6;
            dynamic_commit_ns.push(us * 1e3);
            timed("dynamic-commit", label, index, *token, us);
            match (r_static, r_dynamic) {
                (Ok(()), Ok(())) => {}
                (a, b) => panic!(
                    "TRACE_MISMATCH trace={label} index={index} token={token} static={} dynamic={}",
                    a.map(|_| "ok").unwrap_or("ERR"),
                    b.map(|_| "ok").unwrap_or("ERR"),
                ),
            }
            total_commits += 1;
            progress("mask", label, index, *token);
            let t0 = Instant::now();
            m_static = st_static.mask();
            let us = t0.elapsed().as_secs_f64() * 1e6;
            static_mask_ns.push(us * 1e3);
            timed("static-mask", label, index, *token, us);
            let t0 = Instant::now();
            let m_dynamic = st_dynamic.mask();
            let us = t0.elapsed().as_secs_f64() * 1e6;
            dynamic_mask_ns.push(us * 1e3);
            timed("dynamic-mask", label, index, *token, us);
            total_masks += 1;
            hash = fnv_word_hash(hash, &m_static);
            if m_static != m_dynamic {
                let (a, b) = first_mask_difference(&m_static, &m_dynamic);
                panic!("TRACE_MISMATCH trace={label} index={index} token={token} static_only_first={a:?} dynamic_only_first={b:?}");
            }
        }
        println!("[trace-replay] trace={label} commits={} masks_so_far={total_masks} hash_so_far={hash:016x}", ids.len());
    }
    // In chunk mode every selected trace `continue`s above. `covered_traces`
    // counts selections; reaching here with chunked set and zero coverage
    // means the selector matched nothing.
    if chunked {
        assert!(covered_traces > 0, "chunk mode covered zero traces (selector matched nothing)");
        return;
    }
    assert!(covered_traces > 0, "trace selector covered zero traces");
    // Diagnostic-only partial replay: stop after N commits with an explicit
    // PARTIAL marker (never OK); the full gate below is unmodified.
    let partial = max_trace_commits.is_some_and(|max| total_commits >= max);
    if let Some(max) = max_trace_commits {
        assert_eq!(total_commits, max.min(total_commits), "partial commit accounting");
    }
    if partial {
        println!("TRACE_REPLAY_PARTIAL commits={total_commits} masks={total_masks} hash={hash:016x}");
    }
    // Exact expected commits/masks: all = 11764 commits / 11767 masks
    // (n_tokens sum + 3 initial masks); single-trace selections assert their
    // own counts below via the per-trace line.
    // Full-gate marker: only when NOT partial (partial prints PARTIAL above
    // and skips the OK line via early return of the count asserts).
    if partial {
        if let Some(path) = samples_path {
            let doc = serde_json::json!({
                "commits": total_commits,
                "masks": total_masks,
                "hash": format!("{hash:016x}"),
                "success": false,
                "partial": true,
            });
            fs::write(path, serde_json::to_string(&doc).expect("samples JSON")).expect("write samples");
        }
        return;
    }
    if only_trace.is_none_or(|only| only == "all") {
        assert_eq!(total_commits, 11_764, "all-traces commit count");
        assert_eq!(total_masks, 11_767, "all-traces mask count");
    }
    println!("TRACE_REPLAY_OK commits={total_commits} masks={total_masks} hash={hash:016x}");
    print_percentiles("trace_static_commit_ns", "ns", &static_commit_ns);
    print_percentiles("trace_dynamic_commit_ns", "ns", &dynamic_commit_ns);
    print_percentiles("trace_static_mask_ns", "ns", &static_mask_ns);
    print_percentiles("trace_dynamic_mask_ns", "ns", &dynamic_mask_ns);
    if let Some(path) = samples_path {
        let doc = serde_json::json!({
            "commits": total_commits,
            "masks": total_masks,
            "hash": format!("{hash:016x}"),
            "static_commit_ns": static_commit_ns,
            "dynamic_commit_ns": dynamic_commit_ns,
            "static_mask_ns": static_mask_ns,
            "dynamic_mask_ns": dynamic_mask_ns,
        });
        fs::write(path, serde_json::to_string(&doc).expect("samples JSON")).expect("write samples");
        println!("[trace-replay] samples_written={path}");
    }
}

fn first_mask_difference(a: &[u32], b: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let len = a.len().max(b.len());
    let mut a_only = Vec::new();
    let mut b_only = Vec::new();
    for i in 0..len {
        let (x, y) = (a.get(i).copied().unwrap_or(0), b.get(i).copied().unwrap_or(0));
        for bit in (x ^ y).trailing_zeros()..32 {
            if (x ^ y) & (1u32 << bit) == 0 {
                continue;
            }
            let token = (i * 32 + bit as usize) as u32;
            if x & (1u32 << bit) != 0 {
                a_only.push(token);
            } else {
                b_only.push(token);
            }
            if a_only.len() + b_only.len() >= 16 {
                return (a_only, b_only);
            }
        }
    }
    (a_only, b_only)
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
    let mut traces_path = String::from("/Users/isaacbreen/Projects2/temp/2026-08/glrmask-selected10-final-measure-traces.json");
    let mut only_trace: Option<String> = None;
    let mut samples_path: Option<String> = None;
    let mut trace_progress = false;
    let mut max_trace_commits: Option<usize> = None;
    let mut prepare_out_dir: Option<String> = None;
    let mut prepared_in_dir: Option<String> = None;
    let mut trace_start: usize = 0;
    let mut trace_count: Option<usize> = None;
    let mut matrix_case: Option<String> = None;
    let mut matrix_out: Option<String> = None;
    let mut matrix_repeat: usize = 0;
    let mut prefix_static: Option<String> = None;
    let mut prefix_dynamic: Option<String> = None;
    let mut prefix_scenario: usize = 0;
    let mut prefix_max_steps: usize = 32;
    let mut prefix_seed: u64 = 0x9e37_79b9_7f4a_7c15;
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
            "--traces" => { i += 1; traces_path = args[i].clone(); }
            "--trace" => { i += 1; only_trace = Some(args[i].clone()); }
            "--trace-progress" => { trace_progress = true; }
            "--max-trace-commits" => { i += 1; max_trace_commits = Some(args[i].parse().expect("--max-trace-commits integer")); }
            "--prepare-out-dir" => { i += 1; prepare_out_dir = Some(args[i].clone()); }
            "--prepared-in-dir" => { i += 1; prepared_in_dir = Some(args[i].clone()); }
            "--trace-start" => { i += 1; trace_start = args[i].parse().expect("--trace-start integer"); }
            "--trace-count" => { i += 1; trace_count = Some(args[i].parse().expect("--trace-count integer")); }
            "--samples-out" => { i += 1; samples_path = Some(args[i].clone()); }
            "--matrix-case" => { i += 1; matrix_case = Some(args[i].clone()); }
            "--matrix-out" => { i += 1; matrix_out = Some(args[i].clone()); }
            "--matrix-repeat" => { i += 1; matrix_repeat = args[i].parse().expect("--matrix-repeat integer"); }
            "--prefix-static" => { i += 1; prefix_static = Some(args[i].clone()); }
            "--prefix-dynamic" => { i += 1; prefix_dynamic = Some(args[i].clone()); }
            "--prefix-scenario" => { i += 1; prefix_scenario = args[i].parse().expect("--prefix-scenario integer"); }
            "--prefix-max-steps" => { i += 1; prefix_max_steps = args[i].parse().expect("--prefix-max-steps integer"); }
            "--prefix-seed" => { i += 1; prefix_seed = args[i].parse().expect("--prefix-seed integer"); }
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
        "trace-replay" => {
            let vocab = read_vocab_dump(&cache_dir.join("vocab_dump.bin"));
            cmd_trace_replay(
                &cache_dir,
                &vocab,
                &dispatch_name,
                &traces_path,
                only_trace.as_deref(),
                samples_path.as_deref(),
                trace_progress,
                max_trace_commits,
                prepare_out_dir.as_deref(),
                prepared_in_dir.as_deref(),
                trace_start,
                trace_count,
            );
        }
        "matrix-link" => {
            // Vocab timing is owned by the caller here (read once per process).
            let vocab_started = Instant::now();
            let vocab = read_vocab_dump(&cache_dir.join("vocab_dump.bin"));
            let vocab_ms = vocab_started.elapsed().as_secs_f64() * 1000.0;
            eprintln!("[matrix-link] vocab_ms={vocab_ms:.3}");
            cmd_matrix_link_with_vocab(
                &cache_dir,
                &vocab,
                &dispatch_name,
                matrix_case.as_deref().expect("--matrix-case required"),
                matrix_out.as_deref().expect("--matrix-out required"),
                matrix_repeat,
                vocab_ms,
            );
        }
        "prefix-diff" => {
            let vocab = read_vocab_dump(&cache_dir.join("vocab_dump.bin"));
            cmd_prefix_diff(
                &vocab,
                prefix_static.as_deref().expect("--prefix-static required"),
                prefix_dynamic.as_deref().expect("--prefix-dynamic required"),
                prefix_scenario,
                prefix_max_steps,
                prefix_seed,
            );
        }
        other => panic!("unknown subcommand {other:?} (sizes|link|tbm|compile-one|dump-schema|dispatch-compose|diff-masks|trace-replay|matrix-link|prefix-diff)"),
    }
}
