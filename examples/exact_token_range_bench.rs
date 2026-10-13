//! Native, reproducible lifecycle measurements for compact exact-token sets.
//!
//! cargo run --release --example exact_token_range_bench -- 131072 2000
//! Prints JSON. Run without competing CPU-heavy jobs. Vocabulary preparation,
//! cold compilation, warm compilation and precompiled linking are distinct.
use std::hint::black_box;
use std::ops::RangeInclusive;
use std::time::{Duration, Instant};
use glrmask::{BuildOptions, Grammar, Optimization, Vocab};
use serde_json::{json, Value};

const SOURCE: &str = "glrm 1; start start; extern T; nt start = T;";

fn ns(duration: Duration) -> f64 { duration.as_secs_f64() * 1e9 }

fn summary(samples: &[f64]) -> Value {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let percentile = |p: usize| sorted[((p * sorted.len()).div_ceil(100)).saturating_sub(1)];
    json!({"samples": sorted.len(), "p50_ns": percentile(50),
        "p95_ns": percentile(95), "p99_ns": percentile(99),
        "min_ns": sorted[0], "max_ns": sorted[sorted.len() - 1]})
}

fn timed<T>(operation: impl FnOnce() -> T) -> (T, f64) {
    let start = Instant::now();
    let value = black_box(operation());
    (value, ns(start.elapsed()))
}

fn selection(name: &str, n: u32) -> Vec<RangeInclusive<u32>> {
    match name {
        "singleton" => vec![n - 1..=n - 1],
        "all" => vec![0..=n - 1],
        "all_except_middle" => vec![0..=n / 2 - 1, n / 2 + 1..=n - 1],
        "eight_disjoint_ranges" => (0..8).map(|i| {
            let start = i * n / 8 + n / 16;
            let end = (i + 1) * n / 8 - 1;
            start..=end
        }).collect(),
        _ => unreachable!(),
    }
}

fn run_case(n: u32, repetitions: usize, kind: &str, name: &str, mode: Optimization)
    -> Result<Value, Box<dyn std::error::Error>>
{
    let ranges = selection(name, n);
    // Fresh vocabulary for every case. Input-vector allocation and validation
    // index construction are INCLUDED, not shifted before this timer.
    let (vocab, vocab_ns) = timed(|| match kind {
        "exact_only" => Vocab::new_with_exact_token_ids(Vec::new(), 0..n),
        "byte_aliases" => Vocab::new((0..n).map(|id| (id, b"x".to_vec())).collect()),
        "empty_aliases" => Vocab::new((0..n).map(|id| (id, Vec::new())).collect()),
        _ => unreachable!(),
    });
    let (tokens, first_set_ns) = timed(|| vocab.token_ranges(ranges.clone()));
    let tokens = tokens?;
    let cardinality = tokens.len();
    let mut set_samples = Vec::with_capacity(200);
    for _ in 0..200 {
        let (result, elapsed) = timed(|| vocab.token_ranges(ranges.clone()));
        black_box(result?);
        set_samples.push(elapsed);
    }
    // The explicit-ID alternative includes iterating the IDs, allocation,
    // sorting/validation and compression. It is not a historical library build.
    let mut explicit_samples = Vec::with_capacity(5);
    for _ in 0..5 {
        let (result, elapsed) = timed(|| vocab.tokens(ranges.iter().cloned().flatten()));
        black_box(result?);
        explicit_samples.push(elapsed);
    }
    let options = BuildOptions::default().optimization(mode);
    let (bound, bind_ns) = timed(|| Grammar::from_glrm(SOURCE).bind("T", tokens.clone()));
    let bound = bound?;
    let (constraint, cold_compile_ns) = timed(|| bound.compile_with(&vocab, options.clone()));
    let constraint = constraint?;
    let mut warm_compiles = Vec::with_capacity(5);
    for _ in 0..5 {
        let (result, elapsed) = timed(|| bound.compile_with(&vocab, options.clone()));
        black_box(result?);
        warm_compiles.push(elapsed);
    }
    // Components are genuinely precompiled before the link timer. This runs
    // after source compilation, so shared vocabulary caches are already warm.
    let (open, precompile_ns) = timed(|| Grammar::from_glrm(SOURCE).compile_unlinked(&vocab));
    let open = open?;
    let (attached, compiled_bind_ns) = timed(|| open.bind("T", tokens.clone()));
    let attached = attached?;
    let (linked, first_link_ns) = timed(|| attached.link_with(options.clone()));
    let linked = linked?;
    let mut links = Vec::with_capacity(5);
    for _ in 0..5 {
        let (result, elapsed) = timed(|| attached.link_with(options.clone()));
        black_box(result?);
        links.push(elapsed);
    }
    let expected_mask = constraint.start().mask();
    assert_eq!(expected_mask.iter().map(|word| word.count_ones() as usize).sum::<usize>(), cardinality);
    assert_eq!(linked.start().mask(), expected_mask);
    let mut mask = vec![0u32; constraint.mask_len()];
    let mut start_samples = Vec::with_capacity(repetitions);
    let mut mask_samples = Vec::with_capacity(repetitions);
    let mut commit_samples = Vec::with_capacity(repetitions);
    let mut total_samples = Vec::with_capacity(repetitions);
    for _ in 0..repetitions {
        let total_started = Instant::now();
        let (mut state, start_ns) = timed(|| constraint.start());
        let (_, mask_ns) = timed(|| state.fill_mask(black_box(&mut mask)));
        let (result, commit_ns) = timed(|| state.commit_token(black_box(n - 1)));
        result?;
        let total_ns = ns(total_started.elapsed());
        assert!(state.is_accepting());
        black_box(&mask);
        start_samples.push(start_ns);
        mask_samples.push(mask_ns);
        commit_samples.push(commit_ns);
        total_samples.push(total_ns);
    }
    Ok(json!({"vocab_size": n, "vocab_kind": kind, "selection": name,
        "optimization": format!("{mode:?}"), "selected_tokens": cardinality,
        "normalized_ranges": tokens.ranges().len(), "mask_words": mask.len(),
        "vocab_prepare_ns": vocab_ns, "first_set_construct_ns": first_set_ns,
        "compact_set_construct": summary(&set_samples),
        "explicit_id_set_construct": summary(&explicit_samples),
        "source_bind_ns": bind_ns, "cold_source_compile_ns": cold_compile_ns,
        "warm_source_compile": summary(&warm_compiles),
        "precompile_open_component_warm_vocab_ns": precompile_ns,
        "compiled_binding_attach_ns": compiled_bind_ns,
        "first_precompiled_link_warm_vocab_ns": first_link_ns,
        "warm_precompiled_link": summary(&links),
        "start": summary(&start_samples), "fill_mask_reused_buffer": summary(&mask_samples),
        "commit": summary(&commit_samples), "start_mask_commit": summary(&total_samples)}))
}

// Same two-token language through ordinary bytes and through overlapping exact
// bindings plus a byte alternative. This exercises the DEFAULT-guard regression
// under production settings without presenting it as a historical baseline.
fn run_two_event_sanity(n: u32, repetitions: usize, mode: Optimization, mixed: bool)
    -> Result<Value, Box<dyn std::error::Error>>
{
    let (vocab, vocab_ns) = timed(|| Vocab::new(
        (0..n).map(|id| (id, b"x".to_vec())).collect(),
    ));
    let (grammar, prepare_ns) = timed(|| -> Result<Grammar<'static>, Box<dyn std::error::Error>> {
        if mixed {
            Ok(Grammar::from_glrm(
                "glrm 1; start start; extern A; extern B; nt start = (A | B) (A | \"x\");",
            ).bind("A", vocab.token_ranges([0..=n - 1])?)?
                .bind("B", vocab.token_ranges([0..=n - 2])?)?)
        } else {
            Ok(Grammar::from_glrm("glrm 1; start start; nt start = \"x\" \"x\";"))
        }
    });
    let grammar = grammar?;
    let (constraint, compile_ns) = timed(|| grammar.compile_with(
        &vocab, BuildOptions::default().optimization(mode),
    ));
    let constraint = constraint?;
    let mut check = constraint.start();
    for _ in 0..2 {
        assert_eq!(check.mask().iter().map(|word| word.count_ones() as u64).sum::<u64>(), u64::from(n));
        check.commit_token(n - 1)?;
    }
    assert!(check.is_accepting());
    assert!(check.mask().iter().all(|&word| word == 0));

    let mut mask = vec![0; constraint.mask_len()];
    let mut starts = Vec::with_capacity(repetitions);
    let mut first_masks = Vec::with_capacity(repetitions);
    let mut second_masks = Vec::with_capacity(repetitions);
    let mut first_commits = Vec::with_capacity(repetitions);
    let mut second_commits = Vec::with_capacity(repetitions);
    let mut totals = Vec::with_capacity(repetitions);
    for _ in 0..repetitions {
        let begin = Instant::now();
        let (mut state, elapsed) = timed(|| constraint.start());
        starts.push(elapsed);
        let (_, elapsed) = timed(|| state.fill_mask(black_box(&mut mask)));
        first_masks.push(elapsed);
        let (result, elapsed) = timed(|| state.commit_token(black_box(n - 1)));
        result?;
        first_commits.push(elapsed);
        let (_, elapsed) = timed(|| state.fill_mask(black_box(&mut mask)));
        second_masks.push(elapsed);
        let (result, elapsed) = timed(|| state.commit_token(black_box(n - 1)));
        result?;
        second_commits.push(elapsed);
        totals.push(ns(begin.elapsed()));
        assert!(state.is_accepting());
        black_box(&mask);
    }
    Ok(json!({"vocab_size": n, "optimization": format!("{mode:?}"),
        "grammar_kind": if mixed { "overlapping_exact_and_byte" } else { "ordinary_bytes" },
        "accepted_language": "exactly two tokens; every vocabulary token spells x",
        "vocab_prepare_ns": vocab_ns, "grammar_and_binding_prepare_ns": prepare_ns,
        "cold_source_compile_ns": compile_ns, "start": summary(&starts),
        "first_mask": summary(&first_masks), "second_mask": summary(&second_masks),
        "first_commit": summary(&first_commits), "second_commit": summary(&second_commits),
        "two_event_lifecycle": summary(&totals)}))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let n: u32 = args.next().as_deref().unwrap_or("131072").parse()?;
    let repetitions: usize = args.next().as_deref().unwrap_or("2000").parse()?;
    if n < 64 || n % 16 != 0 || repetitions < 10 || args.next().is_some() {
        return Err("usage: exact_token_range_bench [vocabulary-size-divisible-by-16 >=64] [repetitions >=10]".into());
    }
    let mut rows = Vec::new();
    for kind in ["exact_only", "byte_aliases", "empty_aliases"] {
        for mode in [Optimization::FastBuild, Optimization::Balanced, Optimization::FastRuntime, Optimization::Auto] {
            for name in ["singleton", "all", "all_except_middle", "eight_disjoint_ranges"] {
                rows.push(run_case(n, repetitions, kind, name, mode)?);
            }
        }
    }
    let mut sanity = Vec::new();
    for mode in [Optimization::FastBuild, Optimization::Balanced, Optimization::FastRuntime, Optimization::Auto] {
        for mixed in [false, true] {
            sanity.push(run_two_event_sanity(n, repetitions, mode, mixed)?);
        }
    }
    println!("{}", serde_json::to_string_pretty(&json!({
        "format": "glrmask-exact-token-range-lifecycle-v2",
        "profile": if cfg!(debug_assertions) { "development-not-performance-acceptance" } else { "release" },
        "notes": ["Fresh vocabulary for each case; vocabulary preparation includes input construction.",
            "Cold source compile is first compile with that vocabulary; warm builds share its reusable caches.",
            "Open-component compilation and binding are separate from final link; linking has warm vocabulary caches.",
            "Runtime samples use a fresh state; mask writes a preallocated buffer; start is timed separately and in lifecycle total.",
            "Explicit-ID construction is the current list API, not an old compiler baseline.",
            "Small build/link sample counts are descriptive, not tail-latency qualification.",
            "Two-event sanity compares equivalent ordinary-byte and mixed exact/byte languages in this version, not a historical baseline."],
        "rows": rows,
        "two_event_sanity": sanity
    }))?);
    Ok(())
}
