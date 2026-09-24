//! Prepare exactly one reusable component, without receiving a future caller.
//! Usage: composition_prepare_component <component> <vocab-dump> <output>
use glrmask::{Constraint, Vocab};
use glrmask::__private::ConstraintExt;
use std::{fs, path::Path, time::Instant};

fn read_vocab(path: &Path) -> Vocab {
    let input = fs::read(path).expect("read vocabulary");
    let mut offset = 0usize;
    let next = |offset: &mut usize| {
        let end = offset.checked_add(4).expect("vocabulary offset overflow");
        let bytes = input.get(*offset..end).expect("truncated vocabulary integer");
        *offset = end;
        u32::from_le_bytes(bytes.try_into().unwrap()) as usize
    };
    let count = next(&mut offset);
    assert!(count <= input.len() / 8, "invalid vocabulary entry count");
    let mut entries = Vec::with_capacity(count);
    for _ in 0..count {
        let id = next(&mut offset) as u32;
        let len = next(&mut offset);
        let end = offset.checked_add(len).expect("vocabulary length overflow");
        entries.push((id, input.get(offset..end).expect("truncated token").to_vec()));
        offset = end;
    }
    assert_eq!(offset, input.len(), "trailing vocabulary data");
    Vocab::new(entries)
}

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    assert_eq!(args.len(), 4, "usage: prepare-component <component> <vocab> <output>");
    let vocab = read_vocab(Path::new(&args[2]));
    let original = fs::read(&args[1]).expect("read component");
    let start = Instant::now();
    let mut component = Constraint::load_with_vocab(&original, &vocab).expect("load component");
    let load_ms = start.elapsed().as_secs_f64() * 1000.0;
    let start = Instant::now();
    component.prepare_for_composition(&vocab).expect("prepare independent component");
    let prepare_ms = start.elapsed().as_secs_f64() * 1000.0;
    let start = Instant::now();
    let bytes = component.save();
    let save_ms = start.elapsed().as_secs_f64() * 1000.0;
    let start = Instant::now();
    let reloaded = Constraint::load_with_vocab(&bytes, &vocab).expect("certify prepared component");
    let reload_ms = start.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(component.num_tokenizer_states(), reloaded.num_tokenizer_states());
    assert_eq!(reloaded.save(), bytes, "prepared component must roundtrip exactly");
    fs::write(&args[3], &bytes).expect("write prepared component");
    println!("COMPONENT_PREPARE independent=true load_ms={load_ms:.3} prepare_ms={prepare_ms:.3} save_ms={save_ms:.3} reload_ms={reload_ms:.3} original_bytes={} prepared_bytes={} output={}", original.len(), bytes.len(), args[3]);
}
