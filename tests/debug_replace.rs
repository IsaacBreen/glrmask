/// Debug test: check which NTs get replace marks for bounded string.
/// Run with: cargo test --release --test debug_replace -- --nocapture

use glrmask::{Constraint, Vocab};

fn tiny_vocab() -> Vocab {
    let mut entries: Vec<(u32, Vec<u8>)> = Vec::new();
    for b in 0..=255u8 {
        entries.push((b as u32, vec![b]));
    }
    Vocab::new(entries, None)
}

#[test]
fn debug_replace_bounded_string() {
    let vocab = tiny_vocab();
    let constraint = Constraint::from_json_schema(
        r#"{"type": "string", "maxLength": 1025}"#,
        &vocab,
    ).unwrap();

    println!("\n{}", constraint.debug_table_stats());

    // Also test that the constraint works
    let mut state = constraint.start();
    // Commit opening quote
    state.commit_token(b'"' as u32).unwrap();
    // Commit 1025 'a' characters
    for _ in 0..1025 {
        state.commit_token(b'a' as u32).unwrap();
    }
    // Close quote should be allowed
    let mask = state.mask();
    let quote_word = (b'"' as u32) / 32;
    let quote_bit = (b'"' as u32) % 32;
    let quote_allowed = mask[quote_word as usize] & (1 << quote_bit) != 0;
    println!("After 1025 chars, quote allowed: {}", quote_allowed);

    // Commit closing quote
    state.commit_token(b'"' as u32).unwrap();
    println!("Successfully closed 1025-char string");

    // Now test 1026 chars
    let mut state2 = constraint.start();
    state2.commit_token(b'"' as u32).unwrap();
    for _ in 0..1026 {
        state2.commit_token(b'a' as u32).unwrap();
    }
    let mask2 = state2.mask();
    let quote_allowed2 = mask2[quote_word as usize] & (1 << quote_bit) != 0;
    println!("After 1026 chars, quote allowed: {}", quote_allowed2);
}
