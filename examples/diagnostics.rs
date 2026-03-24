use glrmask::{Constraint, Vocab};

fn main() {
    let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())], None);
    let (constraint, diagnostics) =
        Constraint::from_ebnf_with_diagnostics(r#"start ::= "a" "b""#, &vocab).unwrap();

    let state = constraint.start();
    let mask_metrics = state.mask_metrics();
    let commit_metrics = state.commit_token_metrics(0).unwrap();

    println!("glr states: {}", diagnostics.glr_table.num_states);
    println!("mask words: {}", mask_metrics.mask_words);
    println!("bytes committed in profile: {}", commit_metrics.bytes_len);
}
