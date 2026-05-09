use glrmask::{Constraint, Vocab};

const CHUNK16_DIRECT_GLRM: &str = r#"
start start;
t JSON_STRING_CHAR ::= /[ ]/;
t JSON_STRING_CHAR_EXACT_16 ::= /[ ]{16}/;
t JSON_STRING_CHAR_UPTO_16 ::= /[ ]{0,16}/;
t JSON_STRING_CHAR_UPTO_CLOSE_16 ::= JSON_STRING_CHAR_UPTO_16 "\"";
t JSON_STRING_CHAR_UPTO_3 ::= /[ ]{0,3}/;
t JSON_STRING_CHAR_UPTO_CLOSE_3 ::= JSON_STRING_CHAR_UPTO_3 "\"";
nt ALT1 ::= JSON_STRING_CHAR_UPTO_CLOSE_16
  | JSON_STRING_CHAR_EXACT_16 JSON_STRING_CHAR_UPTO_CLOSE_16
  | JSON_STRING_CHAR_EXACT_16 JSON_STRING_CHAR_EXACT_16 JSON_STRING_CHAR_UPTO_CLOSE_16
  | JSON_STRING_CHAR_EXACT_16 JSON_STRING_CHAR_EXACT_16 JSON_STRING_CHAR_EXACT_16 JSON_STRING_CHAR_UPTO_CLOSE_16
  | JSON_STRING_CHAR_EXACT_16 JSON_STRING_CHAR_EXACT_16 JSON_STRING_CHAR_EXACT_16 JSON_STRING_CHAR_EXACT_16 JSON_STRING_CHAR_UPTO_CLOSE_16
  | JSON_STRING_CHAR_EXACT_16 JSON_STRING_CHAR_EXACT_16 JSON_STRING_CHAR_EXACT_16 JSON_STRING_CHAR_EXACT_16 JSON_STRING_CHAR_EXACT_16 JSON_STRING_CHAR_UPTO_CLOSE_16;
nt ALT2 ::= JSON_STRING_CHAR_EXACT_16 JSON_STRING_CHAR_EXACT_16 JSON_STRING_CHAR_EXACT_16 JSON_STRING_CHAR_EXACT_16 JSON_STRING_CHAR_EXACT_16 JSON_STRING_CHAR_EXACT_16 JSON_STRING_CHAR_UPTO_CLOSE_3;
nt start ::= "\"" JSON_STRING_CHAR (ALT1 | ALT2);
"#;

fn tiny_vocab() -> Vocab {
    Vocab::new(vec![(0, vec![b' ']), (1, vec![b'"'])], None)
}

fn boundary_vocab() -> Vocab {
    Vocab::new(
        vec![
            (0, vec![b' ']),
            (1, vec![b'"']),
            (2, vec![b' '; 21]),
            (3, vec![b' '; 22]),
            (4, vec![b' '; 23]),
            (5, vec![b' '; 24]),
        ],
        None,
    )
}

#[test]
fn direct_glrm_pipeline_has_no_exact16_shift_in_first_16_single_space_commits() {
    let constraint = Constraint::from_glrm_grammar(CHUNK16_DIRECT_GLRM, &tiny_vocab()).unwrap();
    let mut state = constraint.start();
    state.commit_token(1).unwrap();

    for step in 1..=16 {
        let (advances, _final_stacks, _profile) = state.commit_token_per_advance(0).unwrap();
        assert!(
            advances.iter().all(|entry| entry.terminal_id != 1),
            "exact16 terminal advanced too early at step {step}: {advances:?}",
        );
    }
}

#[test]
fn direct_glrm_pipeline_boundary_probe() {
    let constraint = Constraint::from_glrm_grammar(CHUNK16_DIRECT_GLRM, &boundary_vocab()).unwrap();

    for step_count in [1usize, 2, 15, 16] {
        let mut state = constraint.start();
        state.commit_token(1).unwrap();
        for _ in 0..step_count.saturating_sub(1) {
            state.commit_token(0).unwrap();
        }
        let (advances, final_stacks, profile) = state.commit_token_per_advance(0).unwrap();
        println!(
            "single-step={} advances={advances:?} final_stacks={final_stacks:?} profile_n_advances={}",
            step_count,
            profile.n_advances,
        );
    }

    for (token_id, len) in [(2u32, 21usize), (3u32, 22usize), (4u32, 23usize), (5u32, 24usize)] {
        let mut state = constraint.start();
        state.commit_token(1).unwrap();
        let result = state.commit_token_per_advance(token_id);
        println!("token-len={len} result={result:?}");
    }
}