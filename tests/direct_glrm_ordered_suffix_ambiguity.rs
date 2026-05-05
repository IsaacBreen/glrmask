use glrmask::{Constraint, Vocab};

fn byte_vocab() -> Vocab {
    Vocab::new((0u8..=127).map(|byte| (byte as u32, vec![byte])).collect(), None)
}

fn field_name(index: usize) -> char {
    (b'a' + index as u8) as char
}

fn direct_ordered_suffix_glrm(n_caps: usize) -> String {
    let prefix_required = [0usize, 1];
    let cap0 = 2usize;
    let middle_required = [3usize, 4, 5];
    let suffix_caps: Vec<usize> = (0..n_caps).map(|offset| 6 + offset).collect();

    let mut lines = Vec::new();
    lines.push("start start;".to_string());

    let total_fields = 6 + n_caps;
    for field in 0..total_fields {
        let ch = field_name(field);
        lines.push(format!("nt f_{field} ::= \"{ch}\";"));
    }

    let mut variants = Vec::new();
    for required_cap_index in 0..n_caps {
        let mut pieces = Vec::new();
        let mut present = vec![false; total_fields];

        for &field in &prefix_required {
            present[field] = true;
        }
        for &field in &middle_required {
            present[field] = true;
        }
        present[suffix_caps[required_cap_index]] = true;

        let mut emit_field = |field: usize, optional: bool| {
            if pieces.is_empty() {
                pieces.push(format!("f_{field}"));
            } else if optional {
                pieces.push(format!("(\",\" f_{field})?"));
            } else {
                pieces.push(format!("\",\" f_{field}"));
            }
        };

        for &field in &prefix_required {
            emit_field(field, false);
        }
        emit_field(cap0, !present[cap0]);
        for &field in &middle_required {
            emit_field(field, false);
        }
        for &field in &suffix_caps {
            emit_field(field, !present[field]);
        }

        let name = format!("v_{required_cap_index}");
        lines.push(format!("nt {name} ::= {};", pieces.join(" ")));
        variants.push(name);
    }

    lines.push(format!("nt start ::= {};", variants.join(" | ")));
    lines.join("\n") + "\n"
}

fn direct_ordered_suffix_example(n_caps: usize, mask: usize) -> String {
    let total_fields = 6 + n_caps;
    let mut present = vec![false; total_fields];
    present[0] = true;
    present[1] = true;
    present[3] = true;
    present[4] = true;
    present[5] = true;
    if mask & 1 != 0 {
        present[2] = true;
    }
    for offset in 0..n_caps {
        if mask & (1usize << offset) != 0 {
            present[6 + offset] = true;
        }
    }

    let mut out = String::new();
    for field in 0..total_fields {
        if !present[field] {
            continue;
        }
        if !out.is_empty() {
            out.push(',');
        }
        out.push(field_name(field));
    }
    out
}

fn live_stack_count(constraint_state: &glrmask::ConstraintState) -> usize {
    constraint_state
        .debug_parser_stacks()
        .iter()
        .map(|(_, stacks)| stacks.len())
        .sum()
}

fn measure_max_counts_for_text(constraint: &Constraint, text: &str) -> (usize, usize) {
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

fn first_final_max_location(
    constraint: &Constraint,
    text: &str,
) -> Option<(usize, u8, usize, usize)> {
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
fn direct_glrm_ordered_suffix_ambiguity_grows() {
    // This isolates duplicated ordered suffix variants directly in GLRM, without
    // the JSON Schema importer or lowering pipeline.
    let vocab = byte_vocab();
    let cases = [
        (2usize, 0b01usize, 1usize, 1usize),
        (3usize, 0b011usize, 3usize, 3usize),
        (5usize, 0b01111usize, 6usize, 6usize),
        (8usize, 0b00110011usize, 8usize, 8usize),
    ];

    for (n_caps, mask, expected_max_paths, expected_max_stacks) in cases {
        let grammar = direct_ordered_suffix_glrm(n_caps);
        let example = direct_ordered_suffix_example(n_caps, mask);
        let constraint = Constraint::from_glrm_grammar(&grammar, &vocab).unwrap();
        let (max_paths, max_stacks) = measure_max_counts_for_text(&constraint, &example);

        println!(
            "n_caps={n_caps} mask={mask:0width$b} max_paths={max_paths} max_stacks={max_stacks}",
            width = n_caps
        );

        assert_eq!(max_paths, expected_max_paths);
        assert_eq!(max_stacks, expected_max_stacks);

        if n_caps == 5 && mask == 0b01111 {
            let (byte_index, byte, first_max_paths, first_max_stacks) =
                first_final_max_location(&constraint, &example).unwrap();
            println!(
                "first_max byte_index={byte_index} char={} max_paths={first_max_paths} max_stacks={first_max_stacks}",
                byte as char
            );
            assert_eq!(byte_index, 17);
            assert_eq!(byte, b',');
            assert_eq!(first_max_paths, 6);
            assert_eq!(first_max_stacks, 6);
        }
    }
}