use glrmask::{Constraint, Vocab};

fn vocab(entries: &[&[u8]]) -> Vocab {
    Vocab::new(
        entries
            .iter()
            .enumerate()
            .map(|(id, bytes)| (id as u32, bytes.to_vec()))
            .collect(),
        None,
    )
}

fn token_allowed(mask: &[u32], id: usize) -> bool {
    mask.get(id / 32)
        .map(|word| (word >> (id % 32)) & 1 != 0)
        .unwrap_or(false)
}

fn string_schema() -> &'static str {
    r#"{"type":"string","minLength":0,"maxLength":5000}"#
}

fn object_schema() -> &'static str {
    r#"{
        "type":"object",
        "properties":{
            "aside":{"type":"boolean"},
            "autoplay":{"type":"boolean"},
            "css_class":{"type":"string","pattern":"^[\\w\\s-]+$"},
            "description":{"type":"string","minLength":0,"maxLength":5000}
        },
        "required":[],
        "additionalProperties":true
    }"#
}

fn string_prefix() -> Vec<u8> {
    let mut prefix = String::from("\"");
    prefix.push_str(&"This is a Vimeo video block. ".repeat(79));
    prefix.push_str("This is a");
    prefix.into_bytes()
}

fn object_prefix() -> Vec<u8> {
    let mut prefix = String::from(
        "{\"aside\": true, \"autoplay\": false, \"css_class\": \"vimeo-video-block\", \"description\": \"",
    );
    prefix.push_str(&"This is a Vimeo video block. ".repeat(79));
    prefix.push_str("This is a");
    prefix.into_bytes()
}

fn assert_token_in_mask(schema: &str, prefix: &[u8], tokens: &[&[u8]], token_id: usize) {
    let vocab = vocab(tokens);
    let constraint = Constraint::from_json_schema(schema, &vocab).unwrap();
    let mut state = constraint.start();
    state.commit_bytes(prefix).unwrap();
    assert!(
        token_allowed(&state.mask(), token_id),
        "token {token_id} should be present in mask"
    );
}

#[test]
fn o82710_string_prefix_allows_disputed_token() {
    assert_token_in_mask(string_schema(), &string_prefix(), &[b"'];?>\"", b" Vimeo"], 0);
}

#[test]
fn o82710_string_prefix_allows_control_token() {
    assert_token_in_mask(string_schema(), &string_prefix(), &[b"');?>\"", b" Vimeo"], 1);
}

#[test]
fn o82710_string_prefix_commits_disputed_single_token() {
    let vocab = vocab(&[b"');?>\""]);
    let constraint = Constraint::from_json_schema(string_schema(), &vocab).unwrap();
    let mut state = constraint.start();
    state.commit_bytes(&string_prefix()).unwrap();
    assert!(token_allowed(&state.mask(), 0));
    state.commit_token(0).unwrap();
}

#[test]
fn o82710_object_prefix_allows_disputed_token() {
    assert_token_in_mask(object_schema(), &object_prefix(), &[b"');?>\"", b" Vimeo"], 0);
}
