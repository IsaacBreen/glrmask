use glrmask::{Constraint, Vocab};
use serde_json::json;

fn bytes_vocab() -> Vocab {
    Vocab::new((0u8..=255).map(|b| (b as u32, vec![b])).collect(), None)
}

const BAD_FAMILY: [(&str, &str); 8] = [
    ("INSPIRE BAI", r"(\w+\.)+\d+"),
    ("ARXIV", r"\w+_(\w_)?\d+"),
    ("GOOGLESCHOLAR", r"(\w|-){12}"),
    ("WIKIPEDIA", r"\w+"),
    ("VIAF", r"\d{7,9}"),
    ("SCOPUS", r"\d{10,11}"),
    ("ORCID", r"\d{4}-\d{4}-\d{4}-\d{3}[0-9Xx]"),
    ("RESEARCHERID", r"[A-z]-\d{4}-\d{4}"),
];

fn object_branch(type_name: &str, pattern: &str) -> serde_json::Value {
    json!({
        "type": "object",
        "properties": {
            "type": {
                "enum": [type_name],
                "type": "string",
            },
            "value": {
                "pattern": pattern,
                "type": "string",
            },
        },
        "required": ["type", "value"],
    })
}

fn schema_for_indexes(indexes: &[usize]) -> String {
    assert!(indexes.len() >= 2);
    let any_of: Vec<_> = BAD_FAMILY
        .iter()
        .enumerate()
        .filter(|(index, _)| indexes.contains(index))
        .map(|(_, (type_name, pattern))| object_branch(type_name, pattern))
        .collect();

    json!({ "anyOf": any_of }).to_string()
}

fn schema_for_branches(branches: &[(&str, &str)]) -> String {
    assert!(branches.len() >= 2);
    schema_for_branches_with_combiner("anyOf", branches)
}

fn schema_for_branches_with_combiner(combiner: &str, branches: &[(&str, &str)]) -> String {
    assert!(branches.len() >= 2);
    json!({
        combiner: branches
            .iter()
            .map(|(type_name, pattern)| object_branch(type_name, pattern))
            .collect::<Vec<_>>()
    })
    .to_string()
}

fn standalone_schema_for_index(index: usize) -> String {
    let (type_name, pattern) = BAD_FAMILY[index];
    object_branch(type_name, pattern).to_string()
}

fn standalone_schema_for_branch(type_name: &str, pattern: &str) -> String {
    object_branch(type_name, pattern).to_string()
}

fn schema_for_n(n: usize) -> String {
    assert!((2..=BAD_FAMILY.len()).contains(&n));
    schema_for_indexes(&(0..n).collect::<Vec<_>>())
}

#[test]
fn o35155_bad_family_tokenizer_states_stay_under_current_budget() {
    let vocab = bytes_vocab();
    let mut tokenizer_states = Vec::new();
    let mut minimized_tokenizer_states = Vec::new();

    for n in 2..=BAD_FAMILY.len() {
        let constraint = Constraint::from_json_schema(&schema_for_n(n), &vocab).unwrap();
        tokenizer_states.push(constraint.num_tokenizer_states());
        minimized_tokenizer_states.push(constraint.num_forced_minimized_tokenizer_states());
    }

    eprintln!("o35155 tokenizer states by N=2..{}: {:?}", BAD_FAMILY.len(), tokenizer_states);
    eprintln!(
        "o35155 minimized tokenizer states by N=2..{}: {:?}",
        BAD_FAMILY.len(),
        minimized_tokenizer_states
    );

    // Adaptive JSON-pattern grouping keeps the known cross-pattern interference
    // below 10k states. This is intentionally loose enough to permit harmless
    // topology changes while preventing a return to the 20k+ shared-group DFA.
    let budget = 10_000usize;
    for (index, &state_count) in tokenizer_states.iter().enumerate() {
        let n = index + 2;
        assert!(
            state_count <= budget,
            "tokenizer states exceeded the current o35155 bad-family budget: N={n} states={state_count} budget={budget} counts={tokenizer_states:?}"
        );
    }
}

#[test]
fn o35155_bad_family_just_3_and_4_tokenizer_states() {
    let vocab = bytes_vocab();
    let constraint = Constraint::from_json_schema(&schema_for_indexes(&[2, 3]), &vocab).unwrap();

    eprintln!(
        "o35155 tokenizer states for just branches 3 and 4: current={} minimized={}",
        constraint.num_tokenizer_states(),
        constraint.num_forced_minimized_tokenizer_states()
    );
}

#[test]
fn o35155_bad_family_just_2_3_and_4_tokenizer_states() {
    let vocab = bytes_vocab();
    let constraint = Constraint::from_json_schema(&schema_for_indexes(&[1, 2, 3]), &vocab).unwrap();

    eprintln!(
        "o35155 tokenizer states for just branches 2, 3, and 4: current={} minimized={}",
        constraint.num_tokenizer_states(),
        constraint.num_forced_minimized_tokenizer_states()
    );
}

#[test]
fn o35155_bad_family_just_1_3_and_4_tokenizer_states() {
    let vocab = bytes_vocab();
    let constraint = Constraint::from_json_schema(&schema_for_indexes(&[0, 2, 3]), &vocab).unwrap();

    eprintln!(
        "o35155 tokenizer states for just branches 1, 3, and 4: current={} minimized={}",
        constraint.num_tokenizer_states(),
        constraint.num_forced_minimized_tokenizer_states()
    );
}

#[test]
fn o35155_bad_family_just_1_2_3_and_4_tokenizer_states() {
    let vocab = bytes_vocab();
    let constraint = Constraint::from_json_schema(&schema_for_indexes(&[0, 1, 2, 3]), &vocab).unwrap();

    eprintln!(
        "o35155 tokenizer states for just branches 1, 2, 3, and 4: current={} minimized={}",
        constraint.num_tokenizer_states(),
        constraint.num_forced_minimized_tokenizer_states()
    );
}

#[test]
fn o35155_bad_family_requested_prefix_window_tokenizer_states() {
    let vocab = bytes_vocab();
    let requested_subsets = [
        ("4, 5", vec![3, 4]),
        ("3-5", vec![2, 3, 4]),
        ("2-5", vec![1, 2, 3, 4]),
        ("1-5", vec![0, 1, 2, 3, 4]),
    ];

    for (label, indexes) in requested_subsets {
        let constraint = Constraint::from_json_schema(&schema_for_indexes(&indexes), &vocab).unwrap();
        eprintln!(
            "o35155 tokenizer states for just branches {label}: current={} minimized={}",
            constraint.num_tokenizer_states(),
            constraint.num_forced_minimized_tokenizer_states()
        );
    }
}

#[test]
fn o35155_branch_5_schema_is_now_lowerable() {
    let vocab = bytes_vocab();
    let constraint = Constraint::from_json_schema(&standalone_schema_for_index(4), &vocab)
        .expect("branch 5 schema should now lower successfully");

    eprintln!(
        "o35155 tokenizer states for just branch 5: current={} minimized={}",
        constraint.num_tokenizer_states(),
        constraint.num_forced_minimized_tokenizer_states()
    );
}

#[test]
fn o35155_bad_family_all_pairs_with_5_tokenizer_states() {
    let vocab = bytes_vocab();
    let requested_subsets = [
        ("1, 5", vec![0, 4]),
        ("2, 5", vec![1, 4]),
        ("3, 5", vec![2, 4]),
        ("4, 5", vec![3, 4]),
    ];

    for (label, indexes) in requested_subsets {
        let constraint = Constraint::from_json_schema(&schema_for_indexes(&indexes), &vocab).unwrap();
        eprintln!(
            "o35155 tokenizer states for just branches {label}: current={} minimized={}",
            constraint.num_tokenizer_states(),
            constraint.num_forced_minimized_tokenizer_states()
        );
    }
}

#[test]
fn o35155_bad_family_all_combinations_with_5_tokenizer_states() {
    let vocab = bytes_vocab();
    let requested_subsets = [
        ("1, 5", vec![0, 4]),
        ("2, 5", vec![1, 4]),
        ("3, 5", vec![2, 4]),
        ("4, 5", vec![3, 4]),
        ("1, 2, 5", vec![0, 1, 4]),
        ("1, 3, 5", vec![0, 2, 4]),
        ("1, 4, 5", vec![0, 3, 4]),
        ("2, 3, 5", vec![1, 2, 4]),
        ("2, 4, 5", vec![1, 3, 4]),
        ("3, 4, 5", vec![2, 3, 4]),
        ("1, 2, 3, 5", vec![0, 1, 2, 4]),
        ("1, 2, 4, 5", vec![0, 1, 3, 4]),
        ("1, 3, 4, 5", vec![0, 2, 3, 4]),
        ("2, 3, 4, 5", vec![1, 2, 3, 4]),
        ("1, 2, 3, 4, 5", vec![0, 1, 2, 3, 4]),
    ];

    for (label, indexes) in requested_subsets {
        let constraint = Constraint::from_json_schema(&schema_for_indexes(&indexes), &vocab).unwrap();
        eprintln!(
            "o35155 tokenizer states for just branches {label}: current={} minimized={}",
            constraint.num_tokenizer_states(),
            constraint.num_forced_minimized_tokenizer_states()
        );
    }
}

#[test]
fn o35155_bad_family_viaf_bound_variants_are_now_lowerable() {
    let vocab = bytes_vocab();

    let duplicate_singletons = [
        ("5 duplicated", vec![("VIAF", r"\d{7,9}"), ("VIAF", r"\d{7,9}")]),
        ("5(7,7) duplicated", vec![("VIAF", r"\d{7}"), ("VIAF", r"\d{7}")]),
        ("5(9,9) duplicated", vec![("VIAF", r"\d{9}"), ("VIAF", r"\d{9}")]),
    ];

    for (label, branches) in duplicate_singletons {
        let schema_variants = [
            ("anyOf", schema_for_branches_with_combiner("anyOf", &branches)),
            ("oneOf", schema_for_branches_with_combiner("oneOf", &branches)),
        ];

        let mut measured = false;
        for (combiner, schema) in schema_variants {
            if let Ok(constraint) = Constraint::from_json_schema(&schema, &vocab) {
                eprintln!(
                    "o35155 tokenizer states for just branch {label} via duplicated {combiner}: current={} minimized={}",
                    constraint.num_tokenizer_states(),
                    constraint.num_forced_minimized_tokenizer_states()
                );
                measured = true;
                break;
            }
        }

        if !measured {
            eprintln!(
                "o35155 tokenizer states for just branch {label}: no duplicated anyOf/oneOf encoding was lowerable"
            );
        }
    }

    let constraint = Constraint::from_json_schema(&standalone_schema_for_branch("VIAF", r"\d{7,9}"), &vocab)
        .expect("standalone VIAF singleton schema should now lower successfully");
    eprintln!(
        "o35155 standalone singleton branch 5 schema now lowers: current={} minimized={}",
        constraint.num_tokenizer_states(),
        constraint.num_forced_minimized_tokenizer_states()
    );

    let constraint = Constraint::from_json_schema(
        &schema_for_branches(&[("WIKIPEDIA", r"\w+"), ("VIAF", r"\d{7}")]),
        &vocab,
    )
    .unwrap();
    eprintln!(
        "o35155 tokenizer states for just branches 4 and modified 5(7,7): current={} minimized={}",
        constraint.num_tokenizer_states(),
        constraint.num_forced_minimized_tokenizer_states()
    );
}

#[test]
fn o35155_bad_family_viaf_7_7_selected_combinations() {
    let vocab = bytes_vocab();
    let requested_subsets = [
        (
            "2, 5(7,7)",
            vec![("ARXIV", r"\w+_(\w_)?\d+"), ("VIAF", r"\d{7}")],
        ),
        (
            "2, 3, 5(7,7)",
            vec![
                ("ARXIV", r"\w+_(\w_)?\d+"),
                ("GOOGLESCHOLAR", r"(\w|-){12}"),
                ("VIAF", r"\d{7}"),
            ],
        ),
    ];

    for (label, branches) in requested_subsets {
        let constraint = Constraint::from_json_schema(&schema_for_branches(&branches), &vocab).unwrap();
        eprintln!(
            "o35155 tokenizer states for just branches {label}: current={} minimized={}",
            constraint.num_tokenizer_states(),
            constraint.num_forced_minimized_tokenizer_states()
        );
    }
}

#[test]
fn o35155_bad_family_duplicate_branch_pairs() {
    let vocab = bytes_vocab();
    let requested_duplicates = [
        ("1, 1", vec![("INSPIRE BAI", r"(\w+\.)+\d+"), ("INSPIRE BAI", r"(\w+\.)+\d+")]),
        ("4, 4", vec![("WIKIPEDIA", r"\w+"), ("WIKIPEDIA", r"\w+")]),
        ("5, 5", vec![("VIAF", r"\d{7,9}"), ("VIAF", r"\d{7,9}")]),
        ("6, 6", vec![("SCOPUS", r"\d{10,11}"), ("SCOPUS", r"\d{10,11}")]),
    ];

    for (label, branches) in requested_duplicates {
        let schema_variants = [
            ("anyOf", schema_for_branches_with_combiner("anyOf", &branches)),
            ("oneOf", schema_for_branches_with_combiner("oneOf", &branches)),
        ];

        let mut measured = false;
        for (combiner, schema) in schema_variants {
            if let Ok(constraint) = Constraint::from_json_schema(&schema, &vocab) {
                eprintln!(
                    "o35155 tokenizer states for just branches {label} via duplicated {combiner}: current={} minimized={}",
                    constraint.num_tokenizer_states(),
                    constraint.num_forced_minimized_tokenizer_states()
                );
                measured = true;
                break;
            }
        }

        if !measured {
            eprintln!(
                "o35155 tokenizer states for just branches {label}: no duplicated anyOf/oneOf encoding was lowerable"
            );
        }
    }
}

#[test]
fn o35155_bad_family_with_2_and_2_3_selected_combinations() {
    let vocab = bytes_vocab();
    let requested_subsets = [
        ("2, 4", vec![1, 3]),
        ("2, 3, 4", vec![1, 2, 3]),
        ("2, 6", vec![1, 5]),
        ("2, 3, 6", vec![1, 2, 5]),
    ];

    for (label, indexes) in requested_subsets {
        let constraint = Constraint::from_json_schema(&schema_for_indexes(&indexes), &vocab).unwrap();
        eprintln!(
            "o35155 tokenizer states for just branches {label}: current={} minimized={}",
            constraint.num_tokenizer_states(),
            constraint.num_forced_minimized_tokenizer_states()
        );
    }
}

#[test]
fn o35155_bad_family_just_2_and_3_tokenizer_states() {
    let vocab = bytes_vocab();
    let constraint = Constraint::from_json_schema(&schema_for_indexes(&[1, 2]), &vocab).unwrap();

    eprintln!(
        "o35155 tokenizer states for just branches 2 and 3: current={} minimized={}",
        constraint.num_tokenizer_states(),
        constraint.num_forced_minimized_tokenizer_states()
    );
}

#[test]
fn o35155_bad_family_2_3_5_with_viaf_2_2_tokenizer_states() {
    let vocab = bytes_vocab();
    let constraint = Constraint::from_json_schema(
        &schema_for_branches(&[
            ("ARXIV", r"\w+_(\w_)?\d+"),
            ("GOOGLESCHOLAR", r"(\w|-){12}"),
            ("VIAF", r"\d{2}"),
        ]),
        &vocab,
    )
    .unwrap();

    eprintln!(
        "o35155 tokenizer states for just branches 2, 3, 5(2,2): current={} minimized={}",
        constraint.num_tokenizer_states(),
        constraint.num_forced_minimized_tokenizer_states()
    );
}

#[test]
fn o35155_bad_family_2_3rep2_5_variants() {
    let vocab = bytes_vocab();
    let requested_subsets = [
        (
            "2, 3(rep=2), 5",
            vec![
                ("ARXIV", r"\w+_(\w_)?\d+"),
                ("GOOGLESCHOLAR", r"(\w|-){2}"),
                ("VIAF", r"\d{7,9}"),
            ],
        ),
        (
            "2, 3(rep=2), 5(2,2)",
            vec![
                ("ARXIV", r"\w+_(\w_)?\d+"),
                ("GOOGLESCHOLAR", r"(\w|-){2}"),
                ("VIAF", r"\d{2}"),
            ],
        ),
    ];

    for (label, branches) in requested_subsets {
        let constraint = Constraint::from_json_schema(&schema_for_branches(&branches), &vocab).unwrap();
        eprintln!(
            "o35155 tokenizer states for just branches {label}: current={} minimized={}",
            constraint.num_tokenizer_states(),
            constraint.num_forced_minimized_tokenizer_states()
        );
    }
}

#[test]
fn o35155_bad_family_2_3rep2_and_2_5len2_variants() {
    let vocab = bytes_vocab();
    let requested_subsets = [
        (
            "2, 3(rep=2)",
            vec![
                ("ARXIV", r"\w+_(\w_)?\d+"),
                ("GOOGLESCHOLAR", r"(\w|-){2}"),
            ],
        ),
        (
            "2, 5(2,2)",
            vec![
                ("ARXIV", r"\w+_(\w_)?\d+"),
                ("VIAF", r"\d{2}"),
            ],
        ),
    ];

    for (label, branches) in requested_subsets {
        let constraint = Constraint::from_json_schema(&schema_for_branches(&branches), &vocab).unwrap();
        eprintln!(
            "o35155 tokenizer states for just branches {label}: current={} minimized={}",
            constraint.num_tokenizer_states(),
            constraint.num_forced_minimized_tokenizer_states()
        );
    }
}

#[test]
fn o35155_bad_family_singletons_for_2_3rep2_and_5len2() {
    let vocab = bytes_vocab();
    let singleton_branches = [
        ("2", ("ARXIV", r"\w+_(\w_)?\d+")),
        ("3(rep=2)", ("GOOGLESCHOLAR", r"(\w|-){2}")),
        ("5(2,2)", ("VIAF", r"\d{2}")),
    ];

    for (label, branch) in singleton_branches {
        let direct = Constraint::from_json_schema(
            &standalone_schema_for_branch(branch.0, branch.1),
            &vocab,
        );

        if direct.is_ok() {
            let constraint = direct.unwrap();
            eprintln!(
                "o35155 tokenizer states for just branch {label} via direct singleton: current={} minimized={}",
                constraint.num_tokenizer_states(),
                constraint.num_forced_minimized_tokenizer_states()
            );
            continue;
        }

        let duplicate_branches = vec![branch, branch];
        let schema_variants = [
            ("anyOf", schema_for_branches_with_combiner("anyOf", &duplicate_branches)),
            ("oneOf", schema_for_branches_with_combiner("oneOf", &duplicate_branches)),
        ];

        let mut measured = false;
        for (combiner, schema) in schema_variants {
            if let Ok(constraint) = Constraint::from_json_schema(&schema, &vocab) {
                eprintln!(
                    "o35155 tokenizer states for just branch {label} via duplicated {combiner}: current={} minimized={}",
                    constraint.num_tokenizer_states(),
                    constraint.num_forced_minimized_tokenizer_states()
                );
                measured = true;
                break;
            }
        }

        if !measured {
            eprintln!(
                "o35155 tokenizer states for just branch {label}: no direct or duplicated anyOf/oneOf encoding was lowerable"
            );
        }
    }
}

#[test]
fn o35155_bad_family_viaf_fixed_length_variants_with_2_and_2_3() {
    let vocab = bytes_vocab();
    let requested_subsets = [
        (
            "2, 5(6,6)",
            vec![("ARXIV", r"\w+_(\w_)?\d+"), ("VIAF", r"\d{6}")],
        ),
        (
            "2, 3, 5(6,6)",
            vec![
                ("ARXIV", r"\w+_(\w_)?\d+"),
                ("GOOGLESCHOLAR", r"(\w|-){12}"),
                ("VIAF", r"\d{6}"),
            ],
        ),
        (
            "2, 5(5,5)",
            vec![("ARXIV", r"\w+_(\w_)?\d+"), ("VIAF", r"\d{5}")],
        ),
        (
            "2, 3, 5(5,5)",
            vec![
                ("ARXIV", r"\w+_(\w_)?\d+"),
                ("GOOGLESCHOLAR", r"(\w|-){12}"),
                ("VIAF", r"\d{5}"),
            ],
        ),
        (
            "2, 5(4,4)",
            vec![("ARXIV", r"\w+_(\w_)?\d+"), ("VIAF", r"\d{4}")],
        ),
        (
            "2, 3, 5(4,4)",
            vec![
                ("ARXIV", r"\w+_(\w_)?\d+"),
                ("GOOGLESCHOLAR", r"(\w|-){12}"),
                ("VIAF", r"\d{4}"),
            ],
        ),
        (
            "2, 5(3,3)",
            vec![("ARXIV", r"\w+_(\w_)?\d+"), ("VIAF", r"\d{3}")],
        ),
        (
            "2, 3, 5(3,3)",
            vec![
                ("ARXIV", r"\w+_(\w_)?\d+"),
                ("GOOGLESCHOLAR", r"(\w|-){12}"),
                ("VIAF", r"\d{3}"),
            ],
        ),
    ];

    for (label, branches) in requested_subsets {
        let constraint = Constraint::from_json_schema(&schema_for_branches(&branches), &vocab).unwrap();
        eprintln!(
            "o35155 tokenizer states for just branches {label}: current={} minimized={}",
            constraint.num_tokenizer_states(),
            constraint.num_forced_minimized_tokenizer_states()
        );
    }
}
