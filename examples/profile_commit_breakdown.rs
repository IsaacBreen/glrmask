use std::fs;
use std::path::PathBuf;

use glrmask::Constraint;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct ReplayPayload {
    prefix_token_ids: Vec<u32>,
    target_token_id: u32,
}

fn parse_args() -> (PathBuf, PathBuf, usize) {
    let mut args = std::env::args().skip(1);
    let constraint_path = args
        .next()
        .map(PathBuf::from)
        .expect("usage: profile_commit_breakdown <constraint.bin> <payload.json> [repeats]");
    let payload_path = args
        .next()
        .map(PathBuf::from)
        .expect("usage: profile_commit_breakdown <constraint.bin> <payload.json> [repeats]");
    let repeats = args
        .next()
        .as_deref()
        .map(str::parse)
        .transpose()
        .expect("repeats must be an integer")
        .unwrap_or(1000);
    (constraint_path, payload_path, repeats)
}

fn main() {
    let (constraint_path, payload_path, repeats) = parse_args();

    let constraint_bytes = fs::read(&constraint_path).expect("failed to read constraint bytes");
    let payload: ReplayPayload = serde_json::from_slice(
        &fs::read(&payload_path).expect("failed to read payload json"),
    )
    .expect("failed to parse payload json");

    let constraint = Constraint::load(&constraint_bytes).expect("failed to load constraint");
    let mut state = constraint.start();
    state
        .commit_tokens(&payload.prefix_token_ids)
        .expect("failed to commit prefix tokens");

    let breakdown = state
        .profile_commit_token_breakdown(payload.target_token_id, repeats)
        .expect("failed to benchmark commit breakdown");
    println!("{}", serde_json::to_string_pretty(&breakdown).unwrap());
}