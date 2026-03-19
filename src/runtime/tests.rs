use crate::{Constraint, Vocab};

fn make_vocab(entries: &[&str]) -> Vocab {
    let entries: Vec<(u32, Vec<u8>)> = entries
        .iter()
        .enumerate()
        .map(|(i, s)| (i as u32, s.as_bytes().to_vec()))
        .collect();
    Vocab::new(entries, None)
}

fn token_allowed(mask: &[u32], id: usize) -> bool {
    let word = id / 32;
    if word >= mask.len() { return false; }
    (mask[word] >> (id % 32)) & 1 != 0
}

#[test]
fn test_trivial() {
    let vocab = make_vocab(&["a"]);
    let c = Constraint::from_ebnf("start ::= 'a' 'b'", &vocab).unwrap();

    let mut s = c.start();
    s.commit_bytes(b"a");
    let mask = s.mask();
    assert!(!token_allowed(&mask, 0), "token 'a' should NOT be allowed");
}