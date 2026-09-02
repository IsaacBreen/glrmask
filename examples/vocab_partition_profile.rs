use std::{collections::BTreeMap, fs, path::Path, time::Instant};

use glrmask::__private::VocabExt;
use glrmask::{Grammar, Vocab, VocabPartition};

fn decode_hex(value: &str) -> Vec<u8> {
    value
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap()
        })
        .collect()
}

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    let schema = fs::read_to_string(&args[1]).unwrap();
    let raw: BTreeMap<String, String> =
        serde_json::from_slice(&fs::read(Path::new(&args[2])).unwrap()).unwrap();
    let vocab = Vocab::new(
        raw.into_iter()
            .map(|(id, bytes)| (id.parse().unwrap(), decode_hex(&bytes)))
            .collect(),
    );
    vocab.prepare_for_compile();
    let started = Instant::now();
    let partition = VocabPartition::compile(Grammar::json_schema(&schema), &vocab).unwrap();
    eprintln!(
        "[vocab_partition_profile] total_ms={:.3} classes={}",
        started.elapsed().as_secs_f64() * 1000.0,
        partition.num_classes()
    );
}
