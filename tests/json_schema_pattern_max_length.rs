use glrmask::{Constraint, Vocab};

const DISPUTED_TOKEN_ID: u32 = 28914;
const DISPUTED_TOKEN_BYTES: &[u8] = b"----------------------------";
const PREFIX_BYTES: &[u8] = b"\"1.0.0";
const JSON_SCHEMA_MRE: &str = r#"{
    "type": "string",
    "maxLength": 32,
    "pattern": "^(\\d+\\.\\d+\\.\\d+.*)$"
}"#;
const DIRECT_GLRM_MRE: &str = r#"
start start;
t D ::= "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9";
nt start ::= "\"" D+ "." D+ "." D+ "-"{0,27} "\"";
"#;

fn token_allowed(mask: &[u32], token_id: u32) -> bool {
    let word = token_id as usize / 32;
    let bit = token_id as usize % 32;
    word < mask.len() && ((mask[word] >> bit) & 1) != 0
}

fn byte_vocab_with_disputed_token() -> Vocab {
    let mut entries: Vec<(u32, Vec<u8>)> = (0u32..=255).map(|byte| (byte, vec![byte as u8])).collect();
    entries.push((DISPUTED_TOKEN_ID, DISPUTED_TOKEN_BYTES.to_vec()));
    Vocab::new(entries, None)
}

#[test]
fn json_schema_pattern_max_length_rejects_overlength_hyphen_token() {
    let vocab = byte_vocab_with_disputed_token();
    let constraint = Constraint::from_json_schema(JSON_SCHEMA_MRE, &vocab).unwrap();

    let mut base = constraint.start();
    base.commit_bytes(PREFIX_BYTES).unwrap();

    let mask_has_token = token_allowed(&base.mask(), DISPUTED_TOKEN_ID);
    let commit_token_ok = {
        let mut state = base.clone();
        state.commit_token(DISPUTED_TOKEN_ID).is_ok()
    };
    let commit_bytes_ok = {
        let mut state = base.clone();
        state.commit_bytes(DISPUTED_TOKEN_BYTES).is_ok()
    };

    assert!(
        !mask_has_token && !commit_token_ok && !commit_bytes_ok,
        "pattern+maxLength soundness hole: mask_has_token={mask_has_token} commit_token_ok={commit_token_ok} commit_bytes_ok={commit_bytes_ok}"
    );
}

#[test]
fn direct_glrm_remaining_length_budget_rejects_overlength_hyphen_token() {
    let vocab = byte_vocab_with_disputed_token();
    let constraint = Constraint::from_glrm_grammar(DIRECT_GLRM_MRE, &vocab).unwrap();

    let mut base = constraint.start();
    base.commit_bytes(PREFIX_BYTES).unwrap();

    assert!(!token_allowed(&base.mask(), DISPUTED_TOKEN_ID));

    let mut token_state = base.clone();
    assert!(token_state.commit_token(DISPUTED_TOKEN_ID).is_err());

    let mut bytes_state = base.clone();
    assert!(bytes_state.commit_bytes(DISPUTED_TOKEN_BYTES).is_err());
}
