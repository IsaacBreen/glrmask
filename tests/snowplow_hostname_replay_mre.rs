use glrmask::Constraint;

include!("snowplow_hostname_fixture.rsinc");

fn token_allowed(mask: &[u32], token_id: u32) -> bool {
    let word = token_id as usize / 32;
    let bit = token_id as usize % 32;
    word < mask.len() && ((mask[word] >> bit) & 1) != 0
}

fn has_disallowed_future(
    stacks: &[(u32, Vec<(Vec<u32>, Vec<(u32, Vec<u32>)>)>)],
    tokenizer_state: u32,
    terminal_id: u32,
) -> bool {
    stacks.iter().any(|(_, per_state)| {
        per_state.iter().any(|(_, disallowed)| {
            disallowed.iter().any(|(state, terminals)| {
                *state == tokenizer_state && terminals.contains(&terminal_id)
            })
        })
    })
}

#[test]
fn snowplow_hostname_replay_mask_includes_token_15() {
    let vocab = snowplow_vocab();
    let constraint = Constraint::from_json_schema(SNOWPLOW_SCHEMA, &vocab).unwrap();

    let mut bytes_state = constraint.start();
    bytes_state.commit_bytes(SNOWPLOW_PREFIX_BYTES).unwrap();
    let bytes_mask = bytes_state.mask();
    assert!(token_allowed(&bytes_mask, 15));
    let bytes_stacks = bytes_state.debug_parser_stacks();

    let mut replay_state = constraint.start();
    for &token_id in SNOWPLOW_REPLAY_IDS {
        replay_state.commit_token(token_id).unwrap();
    }
    let replay_mask = replay_state.mask();
    assert!(token_allowed(&replay_mask, 15));

    let replay_stacks = replay_state.debug_parser_stacks();
    assert_eq!(bytes_state.parser_root_count(), replay_state.parser_root_count());
    assert_eq!(
        bytes_state.parser_path_count(1_000_000),
        replay_state.parser_path_count(1_000_000)
    );
    assert_ne!(bytes_stacks, replay_stacks);
    assert!(!has_disallowed_future(&bytes_stacks, 1332, 14));
    assert!(has_disallowed_future(&replay_stacks, 1332, 14));

    let mut replay_commit_bytes = replay_state.clone();
    replay_commit_bytes.commit_bytes(b"0").unwrap();

    let mut replay_commit_token = replay_state.clone();
    replay_commit_token.commit_token(15).unwrap();
}