use glrmask::{Constraint, ConstraintState, Vocab};
use glrmask::__private::ConstraintStateExt as _;

const FIXED_N5_ORDERED_SUFFIX_GLRM: &str = r#"
start start;

nt f_0 ::= "a";
nt f_1 ::= "b";
nt f_2 ::= "c";
nt f_3 ::= "d";
nt f_4 ::= "e";
nt f_5 ::= "f";
nt f_6 ::= "g";
nt f_7 ::= "h";
nt f_8 ::= "i";
nt f_9 ::= "j";
nt f_10 ::= "k";

nt v_0 ::= f_0 "," f_1 ("," f_2)? "," f_3 "," f_4 "," f_5 "," f_6 ("," f_7)? ("," f_8)? ("," f_9)? ("," f_10)?;
nt v_1 ::= f_0 "," f_1 ("," f_2)? "," f_3 "," f_4 "," f_5 ("," f_6)? "," f_7 ("," f_8)? ("," f_9)? ("," f_10)?;
nt v_2 ::= f_0 "," f_1 ("," f_2)? "," f_3 "," f_4 "," f_5 ("," f_6)? ("," f_7)? "," f_8 ("," f_9)? ("," f_10)?;
nt v_3 ::= f_0 "," f_1 ("," f_2)? "," f_3 "," f_4 "," f_5 ("," f_6)? ("," f_7)? ("," f_8)? "," f_9 ("," f_10)?;
nt v_4 ::= f_0 "," f_1 ("," f_2)? "," f_3 "," f_4 "," f_5 ("," f_6)? ("," f_7)? ("," f_8)? ("," f_9)? "," f_10;

nt start ::= v_0 | v_1 | v_2 | v_3 | v_4;
"#;

const FIXED_N5_INPUT: &str = "a,b,c,d,e,f,g,h,i,j";

fn byte_vocab() -> Vocab {
    Vocab::new((0u8..=127).map(|byte| (byte as u32, vec![byte])).collect(), None)
}

fn live_stack_count(state: &ConstraintState<'_>) -> usize {
    state
        .debug_parser_stacks()
        .iter()
        .map(|(_, stacks)| stacks.len())
        .sum()
}

fn measure_max_counts(constraint: &Constraint, text: &str) -> (usize, usize) {
    let mut state = constraint.start();
    let mut max_paths = state.parser_path_count(1_000_000);
    let mut max_stacks = live_stack_count(&state);

    for &byte in text.as_bytes() {
        state.commit_bytes(&[byte]).unwrap();
        max_paths = max_paths.max(state.parser_path_count(1_000_000));
        max_stacks = max_stacks.max(live_stack_count(&state));
    }

    (max_paths, max_stacks)
}

fn first_final_max_location(constraint: &Constraint, text: &str) -> Option<(usize, u8, usize, usize)> {
    let mut state = constraint.start();
    let mut max_paths = state.parser_path_count(1_000_000);
    let mut first_final_max = None;

    for (byte_index, &byte) in text.as_bytes().iter().enumerate() {
        state.commit_bytes(&[byte]).unwrap();
        let paths = state.parser_path_count(1_000_000);
        if paths > max_paths {
            max_paths = paths;
            first_final_max = Some((byte_index, byte, paths, live_stack_count(&state)));
        }
    }

    first_final_max
}

#[test]
fn direct_glrm_ordered_suffix_fixed_n5_regression() {
    let vocab = byte_vocab();
    let constraint = Constraint::from_glrm_grammar(FIXED_N5_ORDERED_SUFFIX_GLRM, &vocab).unwrap();
    let (max_paths, max_stacks) = measure_max_counts(&constraint, FIXED_N5_INPUT);

    println!(
        "input={} max_paths={max_paths} max_stacks={max_stacks}",
        FIXED_N5_INPUT,
    );

    let (byte_index, byte, first_max_paths, first_max_stacks) =
        first_final_max_location(&constraint, FIXED_N5_INPUT).unwrap();
    println!(
        "first_max byte_index={byte_index} char={} max_paths={first_max_paths} max_stacks={first_max_stacks}",
        byte as char,
    );

    assert_eq!(max_paths, 6);
    assert_eq!(max_stacks, 6);
    assert_eq!(byte_index, 15);
    assert_eq!(byte, b',');
    assert_eq!(first_max_paths, 6);
    assert_eq!(first_max_stacks, 6);
}
