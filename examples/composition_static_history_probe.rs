//! Repeat a raw mask at an exact real-trace state, with full history.
//! Set PROBE_DYNAMIC=1 to explicitly force the dynamic engine (including its
//! bridge Vec allocation), rather than dispatching a static artifact to static.
//! Usage: VOCAB ARTIFACT TRACES TRACE STEP OUTPUT [REPEATS=501]
//! State cloning and correctness checks are outside the mask timer. Each clone
//! drops the state-local result cache; no cold observations are discarded.
//! Sample zero uses the original live state after all previous masks/commits.
//! macOS additionally reports thread CPU time so scheduler pauses remain
//! visible in wall time without being mistaken for added algorithmic work.
use glrmask::{Constraint, Vocab};
use glrmask::__private::ConstraintStateExt;
use serde::Deserialize;
use std::{fs, hint::black_box, io::{BufWriter, Write}, path::Path, time::Instant};

#[derive(Deserialize)]
struct Trace {
    token_ids: Vec<u32>,
}

#[cfg(target_os = "macos")]
fn thread_cpu_ns() -> u64 {
    // Darwin SDK _time.h defines CLOCK_THREAD_CPUTIME_ID as 16. This optional
    // benchmark-only clock observes this thread and changes no runtime state.
    unsafe extern "C" {
        fn clock_gettime_nsec_np(clock_id: i32) -> u64;
    }
    unsafe { clock_gettime_nsec_np(16) }
}

#[cfg(not(target_os = "macos"))]
fn thread_cpu_ns() -> u64 {
    0 // Thread CPU timing is unavailable; wall timing remains portable.
}

fn vocab(path: &Path) -> Result<Vocab, Box<dyn std::error::Error>> {
    let bytes = fs::read(path)?;
    let mut at = 0;
    let read = |at: &mut usize| {
        let value = u32::from_le_bytes(bytes[*at..*at + 4].try_into().unwrap());
        *at += 4;
        value
    };
    let count = read(&mut at);
    let mut entries = Vec::new();
    for _ in 0..count {
        let id = read(&mut at);
        let len = read(&mut at) as usize;
        entries.push((id, bytes[at..at + len].to_vec()));
        at += len;
    }
    assert_eq!(at, bytes.len());
    Ok(Vocab::new(entries))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let force_dynamic = std::env::var_os("PROBE_DYNAMIC").is_some();
    eprintln!("MASK_MODE {}", if force_dynamic { "forced-dynamic" } else { "artifact-dispatch" });
    let args: Vec<_> = std::env::args().collect();
    assert!(args.len() >= 7, "VOCAB ARTIFACT TRACES TRACE STEP OUTPUT [REPEATS]");
    let vocabulary = vocab(Path::new(&args[1]))?;
    let constraint = Constraint::load_with_vocab(fs::read(&args[2])?, &vocabulary)?;
    let traces: Vec<Trace> = serde_json::from_slice(&fs::read(&args[3])?)?;
    let selected = args[4].parse::<usize>()?;
    let step = args[5].parse::<usize>()?;
    let repeats = args.get(7).map(|v| v.parse::<usize>()).transpose()?.unwrap_or(501);
    assert!(selected < traces.len() && step <= traces[selected].token_ids.len() && repeats > 0);
    let mut mask = vec![0; constraint.mask_len()];
    for trace in traces.iter().take(selected) {
        let mut state = constraint.start();
        for &token in &trace.token_ids {
            if force_dynamic { mask = state.fill_mask_dynamic_vec(); } else { state.fill_mask(&mut mask); }
            state.commit_token(token)?;
        }
        if force_dynamic { mask = state.fill_mask_dynamic_vec(); } else { state.fill_mask(&mut mask); }
    }
    let mut template = constraint.start();
    for &token in &traces[selected].token_ids[..step] {
        if force_dynamic { mask = template.fill_mask_dynamic_vec(); } else { template.fill_mask(&mut mask); }
        template.commit_token(token)?;
    }
    eprintln!("STATE trace={selected} step={step} roots={} paths={}",
        template.parser_root_count(), template.parser_path_count(100_000));
    let mut out = BufWriter::new(fs::File::create(&args[6])?);
    writeln!(out, "sample,original_live,mask_ns,thread_ns,hash")?;
    let mut expected = None;
    for sample in 0..repeats {
        let clone = (sample != 0).then(|| template.clone());
        let state = clone.as_ref().unwrap_or(&template);
        let cpu_started = thread_cpu_ns();
        let started = Instant::now();
        if force_dynamic {
            mask = black_box(state).fill_mask_dynamic_vec();
        } else {
            state.fill_mask(black_box(&mut mask));
        }
        let elapsed = started.elapsed().as_nanos();
        let cpu_elapsed = thread_cpu_ns().saturating_sub(cpu_started);
        if let Some(expected) = &expected {
            assert_eq!(&mask, expected, "repeated-state mask changed");
        } else {
            expected = Some(mask.clone());
        }
        let hash = mask.iter().fold(0xcbf29ce484222325u64,
            |h, &word| (h ^ u64::from(word)).wrapping_mul(0x100000001b3));
        writeln!(out, "{sample},{},{elapsed},{cpu_elapsed},{hash:016x}", sample == 0)?;
    }
    out.flush()?;
    if !force_dynamic {
    let state = template.clone();
    let profile = state.fill_mask_profiled(&mut mask);
    eprintln!("PROFILE total={} cached={} single={} seed={} lookup={} apply={} accumulate={} queue_pop={} loop={} finalize={}",
        profile.total_ns, profile.cache_hit, profile.single_path_direct,
        profile.seed_decompose_ns, profile.transition_lookup_ns,
        profile.transition_apply_ns, profile.token_accumulation_ns,
        profile.queue_pop_ns, profile.loop_decompose_ns, profile.finalize_ns);
    }
    println!("EXACT_HISTORY completed trace={selected} step={step} samples={repeats}");
    Ok(())
}
