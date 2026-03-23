use std::fs;
use std::hint::black_box;
use std::path::PathBuf;
use std::thread;
use std::time::{Duration, Instant};

use glrmask::Constraint;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct ReplayPayload {
    prefix_token_ids: Vec<u32>,
    target_token_id: u32,
}

fn parse_args() -> (PathBuf, PathBuf, usize, u64, Option<PathBuf>) {
    let mut args = std::env::args().skip(1);
    let constraint_path = args
        .next()
        .map(PathBuf::from)
        .expect("usage: profile_commit_replay <constraint.bin> <payload.json> [batch_size] [pre_sleep_secs] [trigger_file]");
    let payload_path = args
        .next()
        .map(PathBuf::from)
        .expect("usage: profile_commit_replay <constraint.bin> <payload.json> [batch_size] [pre_sleep_secs] [trigger_file]");
    let batch_size = args
        .next()
        .as_deref()
        .map(str::parse)
        .transpose()
        .expect("batch_size must be an integer")
        .unwrap_or(200_000);
    let pre_sleep_secs = args
        .next()
        .as_deref()
        .map(str::parse)
        .transpose()
        .expect("pre_sleep_secs must be an integer")
        .unwrap_or(3);
    let trigger_path = args.next().map(PathBuf::from);
    (constraint_path, payload_path, batch_size, pre_sleep_secs, trigger_path)
}

fn main() {
    let (constraint_path, payload_path, batch_size, pre_sleep_secs, trigger_path) = parse_args();

    let constraint_bytes = fs::read(&constraint_path).expect("failed to read constraint bytes");
    let payload: ReplayPayload = serde_json::from_slice(
        &fs::read(&payload_path).expect("failed to read payload json"),
    )
    .expect("failed to parse payload json");

    let constraint = Constraint::load(&constraint_bytes).expect("failed to load constraint");
    let mut prefix_state = constraint.start();
    prefix_state
        .commit_tokens(&payload.prefix_token_ids)
        .expect("failed to commit prefix tokens");

    let mut batch = Vec::with_capacity(batch_size);
    let clone_start = Instant::now();
    for _ in 0..batch_size {
        batch.push(prefix_state.clone());
    }
    let clone_elapsed = clone_start.elapsed();

    println!(
        "READY pid={} batch_size={} clone_fill_ms={:.3} target_token_id={}",
        std::process::id(),
        batch_size,
        clone_elapsed.as_secs_f64() * 1_000.0,
        payload.target_token_id
    );
    if let Some(trigger_path) = trigger_path {
        println!("Waiting for trigger file {}", trigger_path.display());
        while !trigger_path.exists() {
            thread::sleep(Duration::from_millis(10));
        }
    } else {
        println!("Sleeping {}s before commit phase", pre_sleep_secs);
        thread::sleep(Duration::from_secs(pre_sleep_secs));
    }

    let commit_start = Instant::now();
    for state in &mut batch {
        state
            .commit_token(payload.target_token_id)
            .expect("target commit failed");
        black_box(state.is_finished());
    }
    let commit_elapsed = commit_start.elapsed();
    let per_commit_us = (commit_elapsed.as_secs_f64() * 1_000_000.0) / batch_size as f64;

    println!(
        "DONE commit_phase_ms={:.3} per_commit_us={:.3}",
        commit_elapsed.as_secs_f64() * 1_000.0,
        per_commit_us
    );
}