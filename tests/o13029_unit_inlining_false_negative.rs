use glrmask::{Constraint, Vocab};

const DISPUTED_TOKEN_ID: u32 = 3605;
const DISPUTED_TOKEN_BYTES: &[u8] = b" \"/";
const MINIMIZED_SCHEMA: &str = r##"{"definitions":{"A":{"properties":{"h":{"type":"array","items":{"anyOf":[{"$ref":"#/definitions/A"},{"$ref":"#/definitions/B"},{"$ref":"#/definitions/C"}]}},"f":{"type":"array"},"m":{"properties":{"n":{"type":"string"}}},"x":{"type":"array"}}},"B":{"properties":{"h":{"type":"array","items":{"anyOf":[{"$ref":"#/definitions/A"},{"$ref":"#/definitions/B"},{"$ref":"#/definitions/C"}]}},"f":{"type":"array"},"m":{"properties":{"n":{"enum":["k"]}}},"n":{"enum":["r"]},"x":{"type":"array"}}},"C":{"properties":{"h":{"type":"array","items":{"anyOf":[{"$ref":"#/definitions/A"},{"$ref":"#/definitions/B"},{"$ref":"#/definitions/C"}]}},"f":{"type":"array"},"m":{"properties":{"n":{"enum":["k"]}}},"n":{"enum":["r"]},"x":{"type":"array"}}}},"properties":{"e":{"anyOf":[{"$ref":"#/definitions/A"},{"$ref":"#/definitions/B"},{"$ref":"#/definitions/C"}]}}}"##;
const PREFIX: &[u8] = br#"{"e": {"h": [], "f": [], "m": {"n":"#;

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
fn origin_dependent_unit_reduce_singleton_pushes_preserve_string_branch_context() {
    // Regression for o13029: origin-dependent unit reductions with singleton
    // pushed branch targets must remain reductions. Collapsing them into a
    // replace shift loses the branch-specific context needed for the later
    // string value and incorrectly rejects the disputed token.
    let constraint = Constraint::from_json_schema(MINIMIZED_SCHEMA, &byte_vocab_with_disputed_token()).unwrap();

    let mut base = constraint.start();
    base.commit_bytes(PREFIX).unwrap();

    let mask_has_token = token_allowed(&base.mask(), DISPUTED_TOKEN_ID);

    let mut commit_token_state = base.clone();
    let commit_token_ok = commit_token_state.commit_token(DISPUTED_TOKEN_ID).is_ok();

    let mut commit_bytes_state = base.clone();
    let commit_bytes_ok = commit_bytes_state.commit_bytes(DISPUTED_TOKEN_BYTES).is_ok();

    assert!(
        mask_has_token && commit_token_ok && commit_bytes_ok,
        "o13029 regression: mask_has_token={mask_has_token}, commit_token_ok={commit_token_ok}, commit_bytes_ok={commit_bytes_ok}, stacks={:?}",
        base.debug_parser_stacks(),
    );
}