use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use glrmask::probe::{
    probe_problem_vocab_equivalence,
    BranchTimingRecord,
    ProblemTimingRecord,
};
use glrmask::Vocab;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize)]
struct ManifestEntry {
    problem_id: String,
    format: String,
    source_text: String,
}

#[derive(Serialize)]
struct SummaryStats {
    count: usize,
    sum_ms: f64,
    mean_ms: f64,
    median_ms: f64,
    p90_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
    min_ms: f64,
    max_ms: f64,
}

fn calc_stats(mut vals: Vec<f64>) -> SummaryStats {
    if vals.is_empty() {
        return SummaryStats {
            count: 0,
            sum_ms: 0.0,
            mean_ms: 0.0,
            median_ms: 0.0,
            p90_ms: 0.0,
            p95_ms: 0.0,
            p99_ms: 0.0,
            min_ms: 0.0,
            max_ms: 0.0,
        };
    }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = vals.len();
    let sum: f64 = vals.iter().sum();
    let mean = sum / n as f64;
    let median = vals[n / 2];
    let p90 = vals[((n as f64 * 0.90) as usize).min(n - 1)];
    let p95 = vals[((n as f64 * 0.95) as usize).min(n - 1)];
    let p99 = vals[((n as f64 * 0.99) as usize).min(n - 1)];
    let min = vals[0];
    let max = vals[n - 1];
    SummaryStats {
        count: n,
        sum_ms: sum,
        mean_ms: mean,
        median_ms: median,
        p90_ms: p90,
        p95_ms: p95,
        p99_ms: p99,
        min_ms: min,
        max_ms: max,
    }
}

fn load_llama3_vocab(path: &Path) -> Vocab {
    let raw = std::fs::read_to_string(path)
        .unwrap_or_else(|err| panic!("failed to read Llama 3 vocab from {}: {err}", path.display()));
    let id_to_hex: BTreeMap<u32, String> = serde_json::from_str(&raw)
        .unwrap_or_else(|err| panic!("failed to parse Llama 3 vocab JSON from {}: {err}", path.display()));
    Vocab::new(
        id_to_hex
            .into_iter()
            .map(|(token_id, hex)| {
                (
                    token_id,
                    hex_to_bytes(&hex).unwrap_or_else(|err| {
                        panic!("invalid hex bytes for token {token_id} in {}: {err}", path.display())
                    }),
                )
            })
            .collect(),
    )
}

fn csv_escape(value: &str) -> String {
    value
        .replace('\r', "\\r")
        .replace('\n', "\\n")
        .replace('"', "\"\"")
}

fn hex_to_bytes(hex: &str) -> Result<Vec<u8>, String> {
    if hex.len() % 2 != 0 {
        return Err(format!("odd hex length {}", hex.len()));
    }
    (0..hex.len())
        .step_by(2)
        .map(|idx| u8::from_str_radix(&hex[idx..idx + 2], 16).map_err(|err| err.to_string()))
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut manifest_path = PathBuf::from("/root/results/vocab-equiv-20260829/corpus_manifest.json");
    let mut vocab_path = PathBuf::from("/root/src/cfa/.cache/vocab_cache/llama3_vocab.json");
    let mut output_dir = PathBuf::from("/root/results/vocab-equiv-20260829");
    let mut limit: Option<usize> = None;
    let mut filter_str: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--manifest" => {
                i += 1;
                manifest_path = PathBuf::from(&args[i]);
            }
            "--vocab" => {
                i += 1;
                vocab_path = PathBuf::from(&args[i]);
            }
            "--output-dir" => {
                i += 1;
                output_dir = PathBuf::from(&args[i]);
            }
            "--limit" => {
                i += 1;
                limit = Some(args[i].parse().expect("limit must be integer"));
            }
            "--filter" => {
                i += 1;
                filter_str = Some(args[i].clone());
            }
            _ => {}
        }
        i += 1;
    }

    std::fs::create_dir_all(&output_dir).expect("failed to create output dir");

    eprintln!("[runner] Loading vocab from {}", vocab_path.display());
    let vocab_load_start = Instant::now();
    let vocab = load_llama3_vocab(&vocab_path);
    eprintln!(
        "[runner] Vocab loaded: {} tokens in {:.2}ms",
        vocab.len(),
        vocab_load_start.elapsed().as_secs_f64() * 1000.0
    );
    let vocab_prepare_start = Instant::now();
    glrmask::probe::prepare_vocab_for_vocab_equiv_probe(&vocab);
    eprintln!(
        "[runner] Vocab prepared outside per-problem timings in {:.2}ms",
        vocab_prepare_start.elapsed().as_secs_f64() * 1000.0
    );

    eprintln!("[runner] Loading manifest from {}", manifest_path.display());
    let manifest_raw = std::fs::read_to_string(&manifest_path)
        .unwrap_or_else(|err| panic!("failed to read manifest from {}: {err}", manifest_path.display()));
    let mut manifest: Vec<ManifestEntry> = serde_json::from_str(&manifest_raw)
        .unwrap_or_else(|err| panic!("failed to parse manifest from {}: {err}", manifest_path.display()));

    if let Some(pat) = filter_str.as_ref() {
        manifest.retain(|entry| entry.problem_id.contains(pat));
    }
    if let Some(lim) = limit {
        manifest.truncate(lim);
    }
    eprintln!("[runner] Processing {} problems...", manifest.len());

    let problems_csv_path = output_dir.join("problems.csv");
    let partitions_csv_path = output_dir.join("partitions.csv");

    let mut prob_writer = BufWriter::new(File::create(&problems_csv_path).expect("failed to create problems.csv"));
    let mut part_writer = BufWriter::new(File::create(&partitions_csv_path).expect("failed to create partitions.csv"));

    // Write CSV Headers
    writeln!(
        prob_writer,
        "problem_id,problem_format,status,error_message,num_terminals,tokenizer_states,final_vocab_classes,original_vocab_tokens,reduction_pct,import_parse_wall_ms,import_parse_cpu_ms,grammar_prep_wall_ms,grammar_prep_cpu_ms,lexer_setup_wall_ms,lexer_setup_cpu_ms,grammar_analysis_wall_ms,grammar_analysis_cpu_ms,glr_table_wall_ms,glr_table_cpu_ms,classify_routing_wall_ms,classify_routing_cpu_ms,global_max_len_wall_ms,global_max_len_cpu_ms,partition_total_wall_ms,partition_total_cpu_ms,global_merge_wall_ms,global_merge_cpu_ms,equiv_ready_wall_ms,equiv_ready_cpu_ms,instrumented_setup_total_wall_ms,instrumented_setup_total_cpu_ms,total_wall_ms,total_cpu_ms"
    ).unwrap();

    writeln!(
        part_writer,
        "problem_id,partition_label,branch_type,vocab_tokens,active_terminals,source_states,kernel,prep_wall_ms,prep_cpu_ms,pre_state_wall_ms,pre_state_cpu_ms,exact_state_wall_ms,exact_state_cpu_ms,vocab_equiv_wall_ms,vocab_equiv_cpu_ms,finalize_wall_ms,finalize_cpu_ms,branch_total_wall_ms,branch_total_cpu_ms,branch_vocab_classes"
    ).unwrap();

    let mut all_problems: Vec<ProblemTimingRecord> = Vec::with_capacity(manifest.len());
    let mut all_branches: Vec<BranchTimingRecord> = Vec::new();
    let runner_start = Instant::now();

    for (idx, entry) in manifest.iter().enumerate() {
        let outcome = probe_problem_vocab_equivalence(
            &entry.problem_id,
            &entry.format,
            &entry.source_text,
            &vocab,
        );

        let p = &outcome.problem_record;
        writeln!(
            prob_writer,
            "\"{}\",\"{}\",\"{}\",\"{}\",{},{},{},{},{:.6},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}",
            csv_escape(&p.problem_id),
            csv_escape(&p.problem_format),
            csv_escape(&p.status),
            csv_escape(p.error_message.as_deref().unwrap_or("")),
            p.num_terminals,
            p.tokenizer_states,
            p.final_vocab_classes,
            vocab.len(),
            if p.status == "ok" { 100.0 * (1.0 - p.final_vocab_classes as f64 / vocab.len() as f64) } else { 0.0 },
            p.import_parse_wall_ms,
            p.import_parse_cpu_ms,
            p.grammar_prep_wall_ms,
            p.grammar_prep_cpu_ms,
            p.lexer_setup_wall_ms,
            p.lexer_setup_cpu_ms,
            p.grammar_analysis_wall_ms,
            p.grammar_analysis_cpu_ms,
            p.glr_table_wall_ms,
            p.glr_table_cpu_ms,
            p.classify_routing_wall_ms,
            p.classify_routing_cpu_ms,
            p.global_max_len_wall_ms,
            p.global_max_len_cpu_ms,
            p.partition_total_wall_ms,
            p.partition_total_cpu_ms,
            p.global_merge_wall_ms,
            p.global_merge_cpu_ms,
            p.equiv_ready_wall_ms,
            p.equiv_ready_cpu_ms,
            p.instrumented_setup_total_wall_ms,
            p.instrumented_setup_total_cpu_ms,
            p.total_wall_ms,
            p.total_cpu_ms,
        ).unwrap();

        for b in &outcome.branch_records {
            writeln!(
                part_writer,
                "\"{}\",\"{}\",\"{}\",{},{},{},\"{}\",{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{}",
                csv_escape(&b.problem_id),
                csv_escape(&b.partition_label),
                csv_escape(&b.branch_type),
                b.vocab_tokens,
                b.active_terminals,
                b.source_states,
                csv_escape(&b.kernel),
                b.prep_wall_ms,
                b.prep_cpu_ms,
                b.pre_state_wall_ms,
                b.pre_state_cpu_ms,
                b.exact_state_wall_ms,
                b.exact_state_cpu_ms,
                b.vocab_equiv_wall_ms,
                b.vocab_equiv_cpu_ms,
                b.finalize_wall_ms,
                b.finalize_cpu_ms,
                b.branch_total_wall_ms,
                b.branch_total_cpu_ms,
                b.branch_vocab_classes,
            ).unwrap();
        }

        if (idx + 1) % 50 == 0 || (idx + 1) == manifest.len() || idx < 5 {
            eprintln!(
                "[{}/{}] id={} status={} equiv_ready_wall={:.2}ms equiv_ready_cpu={:.2}ms final_classes={}",
                idx + 1,
                manifest.len(),
                p.problem_id,
                p.status,
                p.equiv_ready_wall_ms,
                p.equiv_ready_cpu_ms,
                p.final_vocab_classes,
            );
            prob_writer.flush().unwrap();
            part_writer.flush().unwrap();
        }

        all_problems.push(outcome.problem_record);
        all_branches.extend(outcome.branch_records);
    }

    prob_writer.flush().unwrap();
    part_writer.flush().unwrap();

    let total_elapsed = runner_start.elapsed().as_secs_f64();
    eprintln!(
        "[runner] Finished {} problems in {:.2}s",
        all_problems.len(),
        total_elapsed
    );

    // Compute Summaries
    let ok_problems: Vec<&ProblemTimingRecord> = all_problems.iter().filter(|p| p.status == "ok").collect();
    let equiv_ready_wall = calc_stats(ok_problems.iter().map(|p| p.equiv_ready_wall_ms).collect());
    let equiv_ready_cpu = calc_stats(ok_problems.iter().map(|p| p.equiv_ready_cpu_ms).collect());
    let setup_total_wall = calc_stats(ok_problems.iter().map(|p| p.instrumented_setup_total_wall_ms).collect());
    let setup_total_cpu = calc_stats(ok_problems.iter().map(|p| p.instrumented_setup_total_cpu_ms).collect());
    let import_wall = calc_stats(ok_problems.iter().map(|p| p.import_parse_wall_ms).collect());
    let prep_wall = calc_stats(ok_problems.iter().map(|p| p.grammar_prep_wall_ms).collect());
    let lexer_wall = calc_stats(ok_problems.iter().map(|p| p.lexer_setup_wall_ms).collect());
    let analysis_wall = calc_stats(ok_problems.iter().map(|p| p.grammar_analysis_wall_ms).collect());
    let glr_table_wall = calc_stats(ok_problems.iter().map(|p| p.glr_table_wall_ms).collect());
    let classify_wall = calc_stats(ok_problems.iter().map(|p| p.classify_routing_wall_ms).collect());
    let partition_wall = calc_stats(ok_problems.iter().map(|p| p.partition_total_wall_ms).collect());
    let merge_wall = calc_stats(ok_problems.iter().map(|p| p.global_merge_wall_ms).collect());
    let total_wall = calc_stats(ok_problems.iter().map(|p| p.total_wall_ms).collect());
    let total_cpu = calc_stats(ok_problems.iter().map(|p| p.total_cpu_ms).collect());
    let classes_stats = calc_stats(ok_problems.iter().map(|p| p.final_vocab_classes as f64).collect());
    let reduction_pct_stats = calc_stats(ok_problems.iter().map(|p| {
        100.0 * (1.0 - p.final_vocab_classes as f64 / vocab.len() as f64)
    }).collect());

    let summary_json = serde_json::json!({
        "total_problems": all_problems.len(),
        "ok_count": ok_problems.len(),
        "error_count": all_problems.len() - ok_problems.len(),
        "total_runner_seconds": total_elapsed,
        "equiv_ready_wall_ms": equiv_ready_wall,
        "equiv_ready_cpu_ms": equiv_ready_cpu,
        "instrumented_setup_total_wall_ms": setup_total_wall,
        "instrumented_setup_total_cpu_ms": setup_total_cpu,
        "import_parse_wall_ms": import_wall,
        "grammar_prep_wall_ms": prep_wall,
        "lexer_setup_wall_ms": lexer_wall,
        "grammar_analysis_wall_ms": analysis_wall,
        "glr_table_wall_ms": glr_table_wall,
        "classify_routing_wall_ms": classify_wall,
        "partition_total_wall_ms": partition_wall,
        "global_merge_wall_ms": merge_wall,
        "total_wall_ms": total_wall,
        "total_cpu_ms": total_cpu,
        "final_vocab_classes": classes_stats,
        "reduction_pct": reduction_pct_stats,
    });

    let summary_json_path = output_dir.join("summary.json");
    std::fs::write(
        &summary_json_path,
        serde_json::to_string_pretty(&summary_json).unwrap(),
    )
    .unwrap();

    let summary_md_path = output_dir.join("summary.md");
    let mut md = String::new();
    md.push_str("# Vocab Equivalence Experiment Summary\n\n");
    md.push_str(&format!("- **Total Problems**: {}\n", all_problems.len()));
    md.push_str(&format!("- **Passed**: {}\n", ok_problems.len()));
    md.push_str(&format!("- **Failed**: {}\n", all_problems.len() - ok_problems.len()));
    md.push_str(&format!("- **Total Wall Clock Time**: {:.2}s\n\n", total_elapsed));

    md.push_str("## Stage Timing Metrics (ms)\n\n");
    md.push_str("| Metric | Mean | Median | P90 | P95 | P99 | **P100** | Sum |\n");
    md.push_str("|---|---|---|---|---|---|---|---|\n");

    let add_row = |name: &str, s: &SummaryStats, md: &mut String| {
        md.push_str(&format!(
            "| **{}** | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.1} |\n",
            name, s.mean_ms, s.median_ms, s.p90_ms, s.p95_ms, s.p99_ms, s.max_ms, s.sum_ms
        ));
    };

    add_row("equiv_ready (Wall)", &equiv_ready_wall, &mut md);
    add_row("equiv_ready (Thread CPU)", &equiv_ready_cpu, &mut md);
    add_row("setup_total (Wall)", &setup_total_wall, &mut md);
    add_row("setup_total (Thread CPU)", &setup_total_cpu, &mut md);
    add_row("import_parse (Wall)", &import_wall, &mut md);
    add_row("grammar_prep (Wall)", &prep_wall, &mut md);
    add_row("lexer_setup (Wall)", &lexer_wall, &mut md);
    add_row("grammar_analysis (Wall)", &analysis_wall, &mut md);
    add_row("glr_table (Wall)", &glr_table_wall, &mut md);
    add_row("classify_routing (Wall)", &classify_wall, &mut md);
    add_row("partition_total (Wall)", &partition_wall, &mut md);
    add_row("global_merge (Wall)", &merge_wall, &mut md);
    add_row("total (Wall)", &total_wall, &mut md);
    add_row("total (Thread CPU)", &total_cpu, &mut md);

    md.push_str("\n## Vocabulary Equivalence Stats\n\n");
    md.push_str(&format!(
        "- **Vocabulary Size**: {} tokens\n- **Equivalence Classes (Mean)**: {:.1}\n- **Equivalence Classes (Median)**: {:.0}\n- **Equivalence Classes (P90/P99/P100)**: {:.0} / {:.0} / {:.0}\n- **Equivalence Classes (Min)**: {:.0}\n- **Reduction (Mean/P90/P99/P100)**: {:.3}% / {:.3}% / {:.3}% / {:.3}%\n",
        vocab.len(),
        classes_stats.mean_ms,
        classes_stats.median_ms,
        classes_stats.p90_ms,
        classes_stats.p99_ms,
        classes_stats.max_ms,
        classes_stats.min_ms,
        reduction_pct_stats.mean_ms,
        reduction_pct_stats.p90_ms,
        reduction_pct_stats.p99_ms,
        reduction_pct_stats.max_ms,
    ));

    std::fs::write(&summary_md_path, &md).unwrap();
    eprintln!("[runner] Summary written to {}", summary_md_path.display());
}
