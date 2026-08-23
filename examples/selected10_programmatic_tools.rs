use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use glrmask::{Constraint as Constraint, Vocab};
use glrmask::__private::ConstraintExt;
use serde_json::Value;

const DEFAULT_CFA_ROOT: &str = "/Users/isaacbreen/Projects2/constraint-framework-analysis";
const DEFAULT_CACHE_DIR: &str = "/Users/isaacbreen/Projects2/temp/2026-08/glrmask-selected10-cache";
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Mode {
    Prepare,
    Bench,
    PrepareAndBench,
}

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

fn save_constraint(path: &Path, constraint: &Constraint) {
    let bytes = constraint.save();
    fs::write(path, &bytes).unwrap_or_else(|error| panic!("write {}: {error}", path.display()));
    eprintln!("[selected10] saved {} ({:.2} MiB)", path.display(), bytes.len() as f64 / (1024.0 * 1024.0));
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

fn name_cache_path(cache_dir: &Path, index: usize) -> PathBuf {
    cache_dir.join(format!("tool-name-{index:02}.bin"))
}

fn build_name_constraint(index: usize, vocab: &Vocab) -> Constraint {
    Constraint::compile(glrmask::Grammar::glrm(&format!(
            "start name;\nnt name ::= \"tool_{index}\";\n"
        )), vocab)
    .unwrap()
}

fn dispatcher_parent_source() -> String {
    let mut source = String::from("start suffix;\n");
    for index in 0..SELECTED10.len() {
        source.push_str(&format!(
            "t TOOL_NAME_SLOT_{index} ::= @token({});\n",
            DISPATCH_PLACEHOLDER_BASE + index as u32
        ));
        source.push_str(&format!(
            "t TOOL_ARGS_SLOT_{index} ::= @token({});\n",
            DISPATCH_PLACEHOLDER_BASE + SELECTED10.len() as u32 + index as u32
        ));
    }
    source.push_str("nt suffix ::=\n    ");
    for index in 0..SELECTED10.len() {
        if index != 0 {
            source.push_str("\n  | ");
        }
        source.push_str(&format!(
            "\".\" TOOL_NAME_SLOT_{index} \"(\" TOOL_ARGS_SLOT_{index} \")\""
        ));
    }
    source.push_str(";\n");
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

fn prepare(cache_dir: &Path, cfa_root: &Path, vocab: &Vocab, rebuild: bool) {
    fs::create_dir_all(cache_dir).unwrap();
    let upgrade_summaries =
        std::env::var_os("SELECTED10_UPGRADE_COMPOSITION_GRAMMAR_SUMMARIES").is_some();

    for (index, (short_name, file)) in SELECTED10.iter().enumerate() {
        let path = schema_cache_path(cache_dir, index, short_name);
        if rebuild || !path.exists() {
            let source = schema_source(cfa_root, file);
            let started = Instant::now();
            let constraint = Constraint::compile(glrmask::Grammar::json_schema(&source), vocab)
                .unwrap_or_else(|error| panic!("compile {short_name}: {error}"));
            eprintln!("[selected10] schema {index:02} {short_name}: {:.3} ms", started.elapsed().as_secs_f64() * 1000.0);
            save_constraint(&path, &constraint);
        } else if upgrade_summaries {
            let mut constraint = load_constraint(&path);
            constraint.prepare_composition_grammar_summary().unwrap();
            save_constraint(&path, &constraint);
        }

        let name_path = name_cache_path(cache_dir, index);
        if rebuild || !name_path.exists() {
            let constraint = build_name_constraint(index, vocab);
            save_constraint(&name_path, &constraint);
        } else if upgrade_summaries {
            let mut constraint = load_constraint(&name_path);
            constraint.prepare_composition_grammar_summary().unwrap();
            save_constraint(&name_path, &constraint);
        }
    }

    let literal_names = std::env::var_os("SELECTED10_LITERAL_NAMES").is_some();
    let dispatch_path = cache_dir.join(if literal_names { "dispatch-literal.bin" } else { "dispatch.bin" });
    if rebuild || !dispatch_path.exists() {
        let parent_started = Instant::now();
        let parent_source = if literal_names { dispatcher_literal_names_parent_source() } else { dispatcher_parent_source() };
        let parent = Constraint::compile(glrmask::Grammar::glrm(&parent_source), vocab).unwrap();
        eprintln!("[selected10] dispatcher parent compile: {:.3} ms", parent_started.elapsed().as_secs_f64() * 1000.0);

        let schemas = SELECTED10
            .iter()
            .enumerate()
            .map(|(index, (short_name, _))| load_constraint(&schema_cache_path(cache_dir, index, short_name)))
            .collect::<Vec<_>>();
        let names = (!literal_names).then(|| {
            (0..SELECTED10.len())
                .map(|index| load_constraint(&name_cache_path(cache_dir, index)))
                .collect::<Vec<_>>()
        });
        let mut bindings = Vec::<(String, &Constraint)>::with_capacity(if literal_names { SELECTED10.len() } else { SELECTED10.len() * 2 });
        for index in 0..SELECTED10.len() {
            if let Some(names) = names.as_ref() {
                bindings.push((format!("TOOL_NAME_SLOT_{index}"), &names[index]));
            }
            bindings.push((format!("TOOL_ARGS_SLOT_{index}"), &schemas[index]));
        }
        let refs = bindings.iter().map(|(name, child)| (name.as_str(), *child)).collect::<Vec<_>>();
        let started = Instant::now();
        let dispatch = parent.compose_compiled_subgrammars(&refs, vocab).unwrap();
        eprintln!("[selected10] dispatcher 20-child compose: {:.3} ms", started.elapsed().as_secs_f64() * 1000.0);
        save_constraint(&dispatch_path, &dispatch);
    } else if upgrade_summaries {
        let mut dispatch = load_constraint(&dispatch_path);
        dispatch.prepare_composition_grammar_summary().unwrap();
        save_constraint(&dispatch_path, &dispatch);
    }

    let core_path = cache_dir.join("core.bin");
    if rebuild || !core_path.exists() {
        let started = Instant::now();
        let core = Constraint::compile(glrmask::Grammar::glrm(&js_core_source(cfa_root)), vocab).unwrap();
        eprintln!("[selected10] JS core compile: {:.3} ms", started.elapsed().as_secs_f64() * 1000.0);
        save_constraint(&core_path, &core);
    } else if upgrade_summaries {
        let mut core = load_constraint(&core_path);
        core.prepare_composition_grammar_summary().unwrap();
        save_constraint(&core_path, &core);
    }
}

#[derive(Debug)]
struct Sample {
    load: Duration,
    compose: Duration,
    save: Duration,
    bytes: usize,
}

fn percentile_ms(samples: &[Duration], q: f64) -> f64 {
    let mut values = samples.iter().map(Duration::as_secs_f64).collect::<Vec<_>>();
    values.sort_by(f64::total_cmp);
    let index = ((values.len() - 1) as f64 * q).round() as usize;
    values[index] * 1000.0
}

fn bench(cache_dir: &Path, vocab: &Vocab, runs: usize, save_output: bool) {
    let core_bytes = fs::read(cache_dir.join("core.bin")).expect("core.bin missing; run --prepare");
    let literal_names = std::env::var_os("SELECTED10_LITERAL_NAMES").is_some();
    let dispatch_name = if literal_names { "dispatch-literal.bin" } else { "dispatch.bin" };
    let dispatch_bytes = fs::read(cache_dir.join(dispatch_name)).expect("dispatch cache missing; run --prepare");
    let mut samples = Vec::with_capacity(runs);
    let mut output = None;

    for run in 0..runs {
        eprintln!("[selected10/debug] run={} before_load dispatch={dispatch_name}", run + 1);
        let load_started = Instant::now();
        let core = Constraint::load(&core_bytes).unwrap();
        let dispatch = Constraint::load(&dispatch_bytes).unwrap();
        let load = load_started.elapsed();
        eprintln!("[selected10/debug] run={} after_load ms={:.3}", run + 1, load.as_secs_f64() * 1000.0);

        eprintln!("[selected10/debug] run={} before_compose", run + 1);
        let compose_started = Instant::now();
        let composed = core
            .compose_compiled_subgrammars(&[("PROGRAMMATIC_TOOL_SUFFIX", &dispatch)], vocab)
            .unwrap();
        let compose = compose_started.elapsed();

        // Exercise the exact prefixes that historically exposed selected10 boundary holes.
        let mut state = composed.start();
        for prefix in [
            b"const x = tools".as_slice(),
            b"const x = tools.tool_0".as_slice(),
            b"const x = tools.tool_0({".as_slice(),
        ] {
            let mut probe = composed.start();
            probe.commit_bytes(prefix).unwrap_or_else(|error| panic!("selected10 prefix {prefix:?} rejected: {error}"));
            std::hint::black_box(probe.mask());
        }
        if std::env::var_os("SELECTED10_EXTRA_BOUNDARY_PROBES").is_some() {
            for prefix in [
                b"const x = tool".as_slice(),
                b"const x = tools.tool_".as_slice(),
                b"const x = tools.tool_0(".as_slice(),
                b"const x = tools.tool_0({}".as_slice(),
            ] {
                let mut probe = composed.start();
                probe.commit_bytes(prefix).unwrap_or_else(|error| {
                    panic!("selected10 extra boundary prefix {prefix:?} rejected: {error}")
                });
                std::hint::black_box(probe.mask());
            }
        }
        state.commit_bytes(b"const x = tools.tool_0({})").unwrap();
        std::hint::black_box(state.mask());

        let save_started = Instant::now();
        let bytes = composed.save();
        let save = save_started.elapsed();
        if save_output && run + 1 == runs {
            output = Some(bytes.clone());
        }
        eprintln!(
            "[selected10] run {}/{} load={:.3} ms compose={:.3} ms save={:.3} ms bytes={}",
            run + 1,
            runs,
            load.as_secs_f64() * 1000.0,
            compose.as_secs_f64() * 1000.0,
            save.as_secs_f64() * 1000.0,
            bytes.len(),
        );
        samples.push(Sample { load, compose, save, bytes: bytes.len() });
    }

    if let Some(bytes) = output {
        fs::write(cache_dir.join("composed-latest.bin"), bytes).unwrap();
    }
    let loads = samples.iter().map(|sample| sample.load).collect::<Vec<_>>();
    let composes = samples.iter().map(|sample| sample.compose).collect::<Vec<_>>();
    let saves = samples.iter().map(|sample| sample.save).collect::<Vec<_>>();
    println!(
        "SELECTED10_RESULT runs={} load_p50_ms={:.3} load_p100_ms={:.3} compose_p50_ms={:.3} compose_p100_ms={:.3} save_p50_ms={:.3} save_p100_ms={:.3} artifact_bytes={}",
        runs,
        percentile_ms(&loads, 0.50),
        percentile_ms(&loads, 1.0),
        percentile_ms(&composes, 0.50),
        percentile_ms(&composes, 1.0),
        percentile_ms(&saves, 0.50),
        percentile_ms(&saves, 1.0),
        samples.last().map_or(0, |sample| sample.bytes),
    );
}

fn main() {
    let mut mode = Mode::Bench;
    let mut runs = 5usize;
    let mut rebuild = false;
    let mut save_output = false;
    let mut cache_dir = PathBuf::from(DEFAULT_CACHE_DIR);
    let mut cfa_root = PathBuf::from(DEFAULT_CFA_ROOT);
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let mut index = 0usize;
    while index < args.len() {
        match args[index].as_str() {
            "--prepare" => mode = Mode::Prepare,
            "--prepare-and-bench" => mode = Mode::PrepareAndBench,
            "--rebuild" => rebuild = true,
            "--save-output" => save_output = true,
            "--runs" => {
                index += 1;
                runs = args[index].parse().expect("--runs requires an integer");
            }
            "--cache-dir" => {
                index += 1;
                cache_dir = PathBuf::from(&args[index]);
            }
            "--cfa-root" => {
                index += 1;
                cfa_root = PathBuf::from(&args[index]);
            }
            other => panic!("unknown argument {other:?}"),
        }
        index += 1;
    }
    assert!(runs > 0);

    let vocab_path = cache_dir.join("vocab_dump.bin");
    let vocab = read_vocab_dump(&vocab_path);
    if matches!(mode, Mode::Prepare | Mode::PrepareAndBench) {
        prepare(&cache_dir, &cfa_root, &vocab, rebuild);
    }
    if matches!(mode, Mode::Bench | Mode::PrepareAndBench) {
        bench(&cache_dir, &vocab, runs, save_output);
    }
}
