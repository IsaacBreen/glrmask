use glrmask::{Constraint, Vocab};

fn token_allowed(mask: &[u32], id: usize) -> bool {
    let word = id / 32;
    if word >= mask.len() {
        return false;
    }
    (mask[word] >> (id % 32)) & 1 != 0
}

fn make_vocab(entries: &[&[u8]]) -> Vocab {
    let entries: Vec<(u32, Vec<u8>)> = entries
        .iter()
        .enumerate()
        .map(|(i, entry)| (i as u32, entry.to_vec()))
        .collect();
    Vocab::new(entries, None)
}

fn o82710_schema() -> &'static str {
    r##"
    {
      "maxLength": 2400,
      "minLength": 0,
      "type": "string"
    }
    "##
}

fn o82710_step_580_prefix() -> Vec<u8> {
    let mut prefix = String::from(
        "\"",
    );
    prefix.push_str(&"a".repeat(0));
    prefix.into_bytes()
}

#[test]
fn test_o82710_step_580_allows_disputed_token_in_small_vocab() {
    let vocab = make_vocab(&[b"\\];?>\"", b" {:?},", b" Vimeo"]);
    let constraint = Constraint::from_json_schema(o82710_schema(), &vocab).unwrap();
    let mut state = constraint.start();
    state.commit_bytes(&o82710_step_580_prefix()).unwrap();

    let mask = state.mask();
    assert!(
        token_allowed(&mask, 0),
        "expected disputed token b'\\];?>\"' to be in mask"
    );
}

#[test]
fn test_o82710_step_580_allows_disputed_token_in_single_token_vocab() {
    let vocab = make_vocab(&[b"a"]);
    let constraint = Constraint::from_json_schema(o82710_schema(), &vocab).unwrap();
    let mut state = constraint.start();
    state.commit_bytes(&o82710_step_580_prefix()).unwrap();

    let mask = state.mask();
    assert!(
        token_allowed(&mask, 0),
        "expected disputed token b'\\];?>\"' to be in mask"
    );
}

#[test]
fn test_o82710_step_580_allows_disputed_token_in_single_token_vocab_temp() {
    let vocab = make_vocab(&[b"\\];?>\""]);
    let constraint = Constraint::from_json_schema(o82710_schema(), &vocab).unwrap();
    let mut state = constraint.start();
    state.commit_bytes(&o82710_step_580_prefix()).unwrap();
    println!("mask: {:?}", state.mask());
    state.commit_token(0).unwrap();
}
