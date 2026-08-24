use std::{collections::BTreeMap, fs::File, io::{BufRead, BufReader, Write}, path::Path, time::Instant};
use glrmask::{Constraint, Grammar, Vocab};
use glrmask::__private::VocabExt;
use glrmask::__private::ConstraintExt;
use serde::Deserialize;

#[derive(Deserialize)]
struct Row { id: String, schema: String }

fn hex_to_bytes(hex: &str) -> Vec<u8> {
    (0..hex.len()).step_by(2).map(|i| u8::from_str_radix(&hex[i..i+2], 16).unwrap()).collect()
}
fn load_vocab(path: &Path) -> Vocab {
    let raw = std::fs::read_to_string(path).unwrap();
    let entries: BTreeMap<u32,String> = serde_json::from_str(&raw).unwrap();
    Vocab::new(entries.into_iter().map(|(id,h)|(id,hex_to_bytes(&h))).collect())
}
fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    let input = &args[1];
    let vocab_path = &args[2];
    let out_path = &args[3];
    let load_iters: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(7);
    let vocab = load_vocab(Path::new(vocab_path));
    vocab.prepare_for_compile();
    let reader = BufReader::new(File::open(input).unwrap());
    let mut out = File::create(out_path).unwrap();
    writeln!(out, "id\tcompile_ms\tsave_ms\tload_median_ms\tartifact_bytes").unwrap();
    let mut ok = 0usize;
    let mut unsupported = 0usize;
    for (index, line) in reader.lines().enumerate() {
        let row: Row = serde_json::from_str(&line.unwrap()).unwrap();
        let started = Instant::now();
        let compiled = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| Constraint::compile(Grammar::json_schema(&row.schema), &vocab)));
        let constraint = match compiled {
            Ok(Ok(c)) => c,
            Ok(Err(e)) => { unsupported += 1; eprintln!("ERROR\t{}\t{}", row.id, e); continue; },
            Err(_) => { unsupported += 1; eprintln!("PANIC\t{}", row.id); continue; },
        };
        let compile_ms = started.elapsed().as_secs_f64() * 1000.0;
        let profile_this_save = std::env::var("PROFILE_SAVE_ID")
            .ok()
            .is_some_and(|needle| row.id.contains(&needle));
        if profile_this_save {
            unsafe { std::env::set_var("GLRMASK_PROFILE_SERIALIZATION", "1") };
        }
        let started = Instant::now();
        let artifact = constraint.save();
        let save_ms = started.elapsed().as_secs_f64() * 1000.0;
        if profile_this_save {
            unsafe { std::env::remove_var("GLRMASK_PROFILE_SERIALIZATION") };
        }
        let mut loads = Vec::with_capacity(load_iters);
        for _ in 0..load_iters {
            let copy = artifact.clone();
            let started = Instant::now();
            let loaded = Constraint::load(copy).unwrap();
            loads.push(started.elapsed().as_secs_f64() * 1000.0);
            std::hint::black_box(loaded);
        }
        loads.sort_by(f64::total_cmp);
        let load_ms = loads[loads.len()/2];
        writeln!(out, "{}\t{compile_ms:.6}\t{save_ms:.6}\t{load_ms:.6}\t{}", row.id, artifact.len()).unwrap();
        ok += 1;
        if (index + 1) % 100 == 0 { eprintln!("progress {}/1000", index + 1); }
    }
    eprintln!("DONE ok={ok} unsupported={unsupported}");
}
