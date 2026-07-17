use std::env;
use std::time::Instant;

use glrmask::{Constraint, Vocab};

fn byte_vocab() -> Vocab {
    let entries = (0..=255u32).map(|byte| (byte, vec![byte as u8])).collect();
    Vocab::new(entries)
}

fn main() {
    let max_length = env::args()
        .nth(1)
        .map(|value| value.parse::<usize>().expect("maxLength must be a positive integer"))
        .unwrap_or(65535);

    let schema = format!(
        r#"{{
        "$schema": "http://json-schema.org/draft-04/schema#",
        "type": "string",
        "maxLength": {max_length}
    }}"#
    );

    let started_at = Instant::now();
    let _constraint = Constraint::from_json_schema(&schema, &byte_vocab()).unwrap();
    println!(
        "compiled bounded string maxLength={} in {:.3}s",
        max_length,
        started_at.elapsed().as_secs_f64()
    );
}
