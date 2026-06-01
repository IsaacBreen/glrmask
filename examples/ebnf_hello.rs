//! Minimal EBNF example for publication docs.

use glrmask::{Constraint, Vocab};

fn main() -> glrmask::Result<()> {
    let vocab = Vocab::new(
        vec![
            (0, b"hello".to_vec()),
            (1, b" ".to_vec()),
            (2, b"world".to_vec()),
        ],
        None,
    );

    let constraint = Constraint::from_ebnf("start = \"hello\" \" \" \"world\" ;", &vocab)?;
    let mut state = constraint.start();

    state.commit_token(0).unwrap();
    state.commit_token(1).unwrap();
    state.commit_token(2).unwrap();

    assert!(state.is_finished());
    Ok(())
}
