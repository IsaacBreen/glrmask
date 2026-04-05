use glrmask::{Constraint, Vocab};

fn byte_vocab() -> Vocab {
    let entries = (0..=255u32).map(|byte| (byte, vec![byte as u8])).collect();
    Vocab::new(entries, None)
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: profile_schema <schema.json>");
    let schema = std::fs::read_to_string(&path).expect("failed to read schema file");
    let vocab = byte_vocab();
    let _constraint = Constraint::from_json_schema(&schema, &vocab).unwrap();
    eprintln!("[profile_schema] compiled ok");
}
