//! Matched public-API marker/no-marker latency benchmark.
//!
//! Build the SAME file with the parent revision and this revision, then run
//! each executable with `plain` to test unchanged workloads; run `marked` only
//! on the marker-aware revision. No private observer-disabling switch is used.
//! `cargo run --release --example marker_bench -- plain 51 16`
//! Each sample is one whole sequence, with start/compile outside commit timing.
//! fill_mask is timed after each actual commit on another state; legitimate
//! cross-commit cache reuse remains enabled in both revisions.
use std::{hint::black_box, time::Instant};
use glrmask::{BuildOptions, Constraint, Grammar, MaskStrategy, Vocab};
use serde_json::{json, Value};

fn stats(values: &[f64]) -> Value {
    let mut sorted = values.to_vec(); sorted.sort_by(f64::total_cmp);
    let at = |p: f64| sorted[((sorted.len()-1) as f64*p).round() as usize];
    json!({"samples":sorted.len(),"p50_ns":at(0.5),"p95_ns":at(0.95),"p99_ns":at(0.99),
        "min_ns":sorted[0],"max_ns":sorted[sorted.len()-1],"mean_ns":sorted.iter().sum::<f64>()/sorted.len() as f64})
}
fn marker(enabled: bool, id: usize) -> String {
    if enabled { format!(" @mark({id}) ") } else { " ".into() }
}
struct Workload { name: String, source: String, sequence: Vec<u32> }
fn workloads(marked: bool) -> Vec<Workload> {
    let m = |id| marker(marked,id);
    let mut recursive = vec![3;32]; recursive.push(5); recursive.extend(vec![4;32]);
    let mut result = vec![
        Workload { name:"alternating_single_tokens".into(), source:format!("start root\nnt root = (\"a\"{}\"b\"{})*\n",m(1),m(2)), sequence:[0,1].repeat(128) },
        Workload { name:"fused_eight_lexemes".into(), source:format!("start root\nnt root = (\"a\"{}\"b\"{})*\n",m(1),m(2)), sequence:vec![2;64] },
        Workload { name:"nested_parentheses".into(), source:format!("start root\nnt root = \"x\" | \"(\"{}root{}\")\"\n",m(1),m(2)), sequence:recursive },
        Workload { name:"long_fixed_lexeme_in_chunks".into(), source:format!("start root\nt WORD = \"{}\"\nnt root = {}WORD{}\n","a".repeat(256),m(1),m(2)), sequence:vec![0;256] },
        Workload { name:"dense_16_markers".into(), source:format!("start root\nnt root = (\"a\"{})*\n",(0..16).map(m).collect::<String>()),sequence:vec![0;256] },
    ];
    for count in [16usize,128,512] {
        let mut choices=vec!["\"a\"+".to_owned()];
        for id in 0..count { choices.push(format!("\"x{id}:\"{}\"b\"",m(id))); }
        result.push(Workload { name:format!("inactive_{count}_markers"),
            source:format!("start root\nnt root = {}\n",choices.join(" | ")),sequence:vec![0;256] });
    }
    result
}
fn measure(c: &Constraint, sequence: &[u32], samples: usize, rounds: usize) -> Result<Value,Box<dyn std::error::Error>> {
    // Validate once and warm both APIs. Timings never cover compilation/start.
    let mut mask=vec![0; c.mask_len()];
    for _ in 0..4 {
        let mut state=c.start();
        for &token in sequence { state.fill_mask(&mut mask);assert!(mask[token as usize/32]&(1<<(token%32))!=0);assert!(state.commit_token(token)); }
        assert!(state.is_accepting());
    }
    let mut commits=Vec::with_capacity(samples);let mut masks=Vec::with_capacity(samples);
    let mut totals=Vec::with_capacity(samples);
    for _ in 0..samples {
        let mut commit_ns=0u128; let mut mask_ns=0u128;let mut total_ns=0u128;
        for _ in 0..rounds {
            let mut state=c.start();
            let started=Instant::now();
            for &token in sequence { assert!(state.commit_token(black_box(token))); }
            commit_ns+=started.elapsed().as_nanos();black_box(&state);
            let mut state=c.start();
            // Every next mask follows a real commit, so this does not benchmark
            // repeatedly retrieving the same already-cached initial mask.
            let total_start=Instant::now();
            for &token in sequence {
                assert!(state.commit_token(token));
                let start=Instant::now();state.fill_mask(black_box(&mut mask));mask_ns+=start.elapsed().as_nanos();
                black_box(&mask);
            }
            total_ns+=total_start.elapsed().as_nanos();black_box(&state);
        }
        let n=(sequence.len()*rounds) as f64;
        commits.push(commit_ns as f64/n);masks.push(mask_ns as f64/n);totals.push(total_ns as f64/n);
    }
    Ok(json!({"commit":stats(&commits),"mask_after_commit":stats(&masks),"commit_plus_mask_with_timer_overhead":stats(&totals)}))
}
fn main() -> Result<(),Box<dyn std::error::Error>> {
    let args=std::env::args().skip(1).collect::<Vec<_>>();
    let marked=match args.first().map(String::as_str).unwrap_or("plain") {"plain"=>false,"marked"=>true,_=>return Err("use plain or marked".into())};
    let samples=args.get(1).map(|s|s.parse()).transpose()?.unwrap_or(51usize).max(3);
    let rounds=args.get(2).map(|s|s.parse()).transpose()?.unwrap_or(16usize).max(1);
    // Real byte trie with 2,065 distinct spellings; no model-specific tokenizer
    // dependency. All revisions use exactly the same generated vocabulary.
    let mut words=vec![b"a".to_vec(),b"b".to_vec(),b"abababab".to_vec(),b"(".to_vec(),b")".to_vec(),b"x".to_vec()];
    words.extend(b"0123456789:".iter().map(|&byte| vec![byte]));
    words.extend((0..2048).map(|i|format!("unused_{i}").into_bytes()));
    let vocab=Vocab::new(words.into_iter().enumerate().map(|(i,b)|(i as u32,b)).collect());
    let mut reports=Vec::new();
    for mode in [MaskStrategy::TokenTrie,MaskStrategy::TokenClassTrie,MaskStrategy::ParserDwa] {
        for case in workloads(marked) {
            let start=Instant::now();
            let c=Grammar::from_glrm(&case.source).compile_with_build_options(&vocab,BuildOptions::default().optimization(mode))?;
            let build_ms=start.elapsed().as_secs_f64()*1000.;
            let values=measure(&c,&case.sequence,samples,rounds)?;
            eprintln!("BENCH {} {:?} marked={} commit_p50_ns={:.1}",case.name,mode,marked,values["commit"]["p50_ns"].as_f64().unwrap());
            reports.push(json!({"case":case.name,"mode":format!("{mode:?}"),"marked":marked,"tokens_per_sequence":case.sequence.len(),"build_ms":build_ms,"timings":values}));
        }
    }
    println!("{}",serde_json::to_string_pretty(&json!({"toolchain":option_env!("MARKER_RUSTC").unwrap_or("record externally"),"samples":samples,"rounds":rounds,"vocab_size":2065,"marked":marked,"cases":reports}))?);
    Ok(())
}
