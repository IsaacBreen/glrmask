// src/constraint_precompute3_eliminate_negative_pops.rs
use crate::datastructures::trie::{GodWrapper, Trie, Trie2Index};

pub fn eliminate_negative_pops<EK, EV, T, FGet, FMake, FMerge>(
    god: &GodWrapper<EK, EV, T>,
    roots: &[Trie2Index],
    mut get_pop: FGet,
    mut make_key: FMake,
    mut merge_ev: FMerge,
) where
    EK: Ord + Clone,
    EV: Clone,
    T: Clone,
    FGet: FnMut(&EK) -> isize,
    FMake: FnMut(&EK, isize) -> EK,
    FMerge: FnMut(&mut EV, EV),
{
    bubble_up_negative_pops(god, roots, &mut get_pop, &mut make_key, &mut merge_ev);
    neutralize_remaining_negative_pops(god, roots, &mut get_pop, &mut make_key);
}

pub fn bubble_up_negative_pops<EK, EV, T, FGet, FMake, FMerge>(
    _god: &GodWrapper<EK, EV, T>,
    _roots: &[Trie2Index],
    _get_pop: &mut FGet,
    _make_key: &mut FMake,
    _merge_ev: &mut FMerge,
) where
    EK: Ord + Clone,
    EV: Clone,
    T: Clone,
    FGet: FnMut(&EK) -> isize,
    FMake: FnMut(&EK, isize) -> EK,
    FMerge: FnMut(&mut EV, EV),
{
    // Will be implemented after finalizing expected behavior via tests.
    // This function will perform the graph version of "bubbling up" negative pops.
    todo!("bubble_up_negative_pops (graph version) not implemented yet");
}

pub fn neutralize_remaining_negative_pops<EK, EV, T, FGet, FMake>(
    _god: &GodWrapper<EK, EV, T>,
    _roots: &[Trie2Index],
    _get_pop: &mut FGet,
    _make_key: &mut FMake,
) where
    EK: Ord + Clone,
    EV: Clone,
    T: Clone,
    FGet: FnMut(&EK) -> isize,
    FMake: FnMut(&EK, isize) -> EK,
{
    // Will be implemented after finalizing expected behavior via tests.
    // This function will remove any trailing pop-only effects by converting
    // them to pop 0 unconditional (no check).
    todo!("neutralize_remaining_negative_pops (graph version) not implemented yet");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraint::PrecomputedNodeContents;
    use crate::datastructures::trie::{Trie, Trie2Index};
    use std::collections::BTreeSet;

    // --- Test Helpers ---

    // Test edge key: (pop_amount, check_value)
    type TestEK = (isize, usize);
    type TestEV = ();
    type TestT = PrecomputedNodeContents;
    type TestGod = GodWrapper<TestEK, TestEV, TestT>;

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    struct Instr {
        pop: isize,
        check: Option<usize>, // None means "unconditional pop-only"
    }

    impl Instr {
        fn with(pop: isize, check: Option<usize>) -> Self {
            Instr { pop, check }
        }
    }

    // Program canonicalization:
    // - Merge any pop-only step into the next step.
    // - Combine consecutive pop-only steps.
    fn compress_pop_only(mut prog: Vec<Instr>) -> Vec<Instr> {
        // Merge any pop-only prior to a checked step into that step.
        let mut i = 0;
        while i + 1 < prog.len() {
            if prog[i].check.is_none() {
                // Merge into next step
                prog[i + 1].pop += prog[i].pop;
                prog.remove(i);
                if i > 0 {
                    i -= 1;
                }
            } else if prog[i + 1].check.is_none() && i + 2 < prog.len() {
                // Merge next pop-only into the step after it
                let p = prog[i + 1].pop;
                prog[i + 2].pop += p;
                prog.remove(i + 1);
            } else {
                i += 1;
            }
        }

        // Combine trailing pop-only steps into a single one.
        let mut trailing_sum = 0isize;
        while let Some(last) = prog.last() {
            if last.check.is_none() {
                trailing_sum += last.pop;
                prog.pop();
            } else {
                break;
            }
        }
        if trailing_sum != 0 {
            prog.push(Instr::with(trailing_sum, None));
        }

        // Remove any zero-pop-only trailing no-op to keep consistent canonical form
        if let Some(last) = prog.last() {
            if last.check.is_none() && last.pop == 0 {
                // keep a single zero-pop-only if it exists; tests may expect it after neutralization
            }
        }
        prog
    }

    // Stack-level bubble-up-negative-pops (reference algorithm).
    // Implements the local two-instruction transformation repeatedly:
    //   (pop n, check x); (pop m, check y) with m < 0
    // =>
    //   (pop n+m, check y); (pop -m, check x); (pop m, no-check)
    // Then eagerly merges pop-only steps forward.
    fn bubble_up_negative_pops_stack(mut prog: Vec<Instr>) -> Vec<Instr> {
        prog = compress_pop_only(prog);

        let mut i = 0usize;
        while i + 1 < prog.len() {
            // Merge any pop-only at position i forward to simplify
            if prog[i].check.is_none() {
                if i + 1 < prog.len() {
                    prog[i + 1].pop += prog[i].pop;
                    prog.remove(i);
                    if i > 0 {
                        i -= 1;
                    }
                    continue;
                } else {
                    // trailing pop-only, nothing to do
                    break;
                }
            }

            // If next is pop-only, merge it ahead if possible
            if prog[i + 1].check.is_none() {
                if i + 2 < prog.len() {
                    let p = prog[i + 1].pop;
                    prog[i + 2].pop += p;
                    prog.remove(i + 1);
                    continue;
                } else {
                    // trailing pop-only
                    break;
                }
            }

            // Both i and i+1 have checks.
            if prog[i + 1].pop < 0 {
                // Apply the transformation
                let n = prog[i].pop;
                let x = prog[i].check.unwrap();
                let m = prog[i + 1].pop;
                let y = prog[i + 1].check.unwrap();

                let new1 = Instr::with(n + m, Some(y));
                let new2 = Instr::with(-m, Some(x));
                let new3 = Instr::with(m, None);

                // Replace [i, i+1] with [new1, new2, new3]
                prog.splice(i..=i + 1, vec![new1, new2, new3]);

                // Merge pop-only (new3) into the next step if present
                if i + 3 < prog.len() {
                    let pop_only = prog[i + 2].pop; // new3
                    prog[i + 3].pop += pop_only;
                    prog.remove(i + 2);
                }

                if i > 0 {
                    i -= 1;
                } else {
                    i = 0;
                }
                continue;
            } else {
                i += 1;
            }
        }

        compress_pop_only(prog)
    }

    // Stack-level neutralization: replace any trailing pop-only by pop 0, no check.
    fn neutralize_remaining_negative_pops_stack(mut prog: Vec<Instr>) -> Vec<Instr> {
        prog = compress_pop_only(prog);
        if let Some(last) = prog.last_mut() {
            if last.check.is_none() {
                last.pop = 0;
            }
        }
        prog
    }

    // Execute a program on a data stack and return whether all checks pass.
    // Pointer starts "after top" (at position stack.len()), pop n moves pointer left by n,
    // and check compares the element immediately left of pointer to 'check' value.
    fn run_on_stack(data_stack: &[usize], prog: &[Instr]) -> bool {
        let mut ptr: isize = data_stack.len() as isize; // pointer is "after index" (right side)
        for step in prog {
            ptr -= step.pop;
            if let Some(x) = step.check {
                // The value to check is immediately to the left of the pointer -> index (ptr - 1)
                let idx = ptr - 1;
                if idx < 0 || (idx as usize) >= data_stack.len() {
                    return false;
                }
                if data_stack[idx as usize] != x {
                    return false;
                }
            }
        }
        true
    }

    // --- Trie Construction Helpers ---

    fn new_node(god: &TestGod) -> Trie2Index {
        Trie2Index::new(god.insert(Trie::new(PrecomputedNodeContents::internal())))
    }

    fn add_edge(god: &TestGod, from: Trie2Index, key: TestEK, to: Trie2Index, val: TestEV) {
        let mut from_w = from.write(god).unwrap();
        from_w.children_mut().entry(key).or_default().insert(to, val);
    }

    // Build a linear path in a trie from root along the given program.
    fn build_linear_trie(god: &TestGod, root: Trie2Index, prog: &[Instr]) -> Trie2Index {
        let mut cur = root;
        for step in prog {
            let next = new_node(god);
            // In trie EK is (pop, check_value); for unconditional pop-only, we use check=None.
            // For test keys, we encode unconditional as check value 0 with special meaning?
            // However, in tests we only use edges with checks; pop-only edges arise only after
            // transformations (graph version not yet implemented). For building initial test tries,
            // we only add edges with checks.
            if let Some(cv) = step.check {
                add_edge(god, cur, (step.pop, cv), next, ());
            } else {
                // For sanity if ever needed in the future tests:
                add_edge(god, cur, (step.pop, 0usize), next, ());
            }
            cur = next;
        }
        cur
    }

    // Flatten a trie into a set of programs (each path is a sequence of Instr).
    fn flatten_trie_to_programs(god: &TestGod, roots: &[Trie2Index]) -> BTreeSet<Vec<Instr>> {
        let mut out = BTreeSet::new();
        let paths = Trie::<TestEK, TestEV, TestT>::get_all_paths(god, roots);
        for path in paths {
            // path is Vec<TestEK> = Vec<(pop, check_value)>
            let prog: Vec<Instr> = path
                .into_iter()
                .map(|(p, cv)| Instr::with(p, Some(cv)))
                .collect();
            out.insert(prog);
        }
        out
    }

    // --- Basic Stack-Only Unit Tests (reference behavior) ---

    #[test]
    fn test_stack_bubble_pair_transform_result() {
        // Example from the prompt:
        // (pop 3, check 0); (pop -2, check 2)
        // =>
        // (pop 1, check 2); (pop 2, check 0); (pop -2, no-check)
        let input = vec![Instr::with(3, Some(0)), Instr::with(-2, Some(2))];
        let got = bubble_up_negative_pops_stack(input);
        let expected = vec![
            Instr::with(1, Some(2)),
            Instr::with(2, Some(0)),
            Instr::with(-2, None),
        ];
        assert_eq!(got, expected);
    }

    #[test]
    fn test_stack_bubble_pair_transform_semantics() {
        // Using data stack [0,1,2,3], pointer starts after 3 (index 4).
        // Both original and transformed programs should pass.
        let data = vec![0usize, 1, 2, 3];
        let original = vec![Instr::with(3, Some(0)), Instr::with(-2, Some(2))];
        let transformed = vec![
            Instr::with(1, Some(2)),
            Instr::with(2, Some(0)),
            Instr::with(-2, None),
        ];
        assert!(run_on_stack(&data, &original));
        assert!(run_on_stack(&data, &transformed));
    }

    #[test]
    fn test_stack_bubble_multiple_negatives() {
        // Program: (2, a); (-3, b); (1, c); (-1, d)
        // We'll just verify it runs and that the transformation has no negative pops before checks.
        let input = vec![
            Instr::with(2, Some(10)),
            Instr::with(-3, Some(20)),
            Instr::with(1, Some(30)),
            Instr::with(-1, Some(40)),
        ];

        let transformed = bubble_up_negative_pops_stack(input.clone());
        // All steps except possibly the final should have non-negative pops for checked steps.
        for (i, st) in transformed.iter().enumerate() {
            if st.check.is_some() {
                assert!(
                    st.pop >= 0,
                    "Negative pop remained before a check at step {}, {:?}",
                    i,
                    st
                );
            }
        }

        // Semantics sanity on a larger data stack
        let data: Vec<usize> = (0..100).collect();
        assert!(run_on_stack(&data, &input));
        assert!(run_on_stack(&data, &transformed));
    }

    #[test]
    fn test_stack_neutralize_trailing_pop_only() {
        // Input with trailing pop-only
        let input = vec![
            Instr::with(1, Some(2)),
            Instr::with(2, Some(0)),
            Instr::with(-2, None),
        ];
        let got = neutralize_remaining_negative_pops_stack(input);
        let expected = vec![
            Instr::with(1, Some(2)),
            Instr::with(2, Some(0)),
            Instr::with(0, None), // trailing neutralized
        ];
        assert_eq!(got, expected);
    }

    #[test]
    fn test_stack_neutralize_semantics() {
        // Neutralization should not affect check outcomes.
        let data = vec![0usize, 1, 2, 3];
        let original = vec![
            Instr::with(1, Some(2)),
            Instr::with(2, Some(0)),
            Instr::with(-2, None),
        ];
        let neutralized = vec![
            Instr::with(1, Some(2)),
            Instr::with(2, Some(0)),
            Instr::with(0, None),
        ];
        assert!(run_on_stack(&data, &original));
        assert!(run_on_stack(&data, &neutralized));
    }

    // --- Graph vs Stack Tests ---
    //
    // These tests are marked #[ignore] until the graph versions of
    // bubble_up_negative_pops and neutralize_remaining_negative_pops are implemented.

    #[test]
    #[ignore = "graph bubble_up_negative_pops not implemented yet"]
    fn test_graph_vs_stack_bubble_up_single_path() {
        let god: TestGod = GodWrapper::new();
        let root = new_node(&god);

        // Build a single path in the trie:
        // (pop 3, check 0) -> (pop -2, check 2)
        let input_prog = vec![Instr::with(3, Some(0)), Instr::with(-2, Some(2))];
        let _last = build_linear_trie(&god, root, &input_prog);

        // Flatten trie to programs
        let before = flatten_trie_to_programs(&god, &[root]);

        // Stack reference transformation
        let stacks_after: BTreeSet<Vec<Instr>> = before
            .iter()
            .map(|p| bubble_up_negative_pops_stack(p.clone()))
            .collect();

        // Graph transformation
        let mut get_pop = |ek: &TestEK| ek.0;
        let mut make_key = |ek: &TestEK, new_pop: isize| (new_pop, ek.1);
        let mut merge_ev = |_a: &mut TestEV, _b: TestEV| {};
        bubble_up_negative_pops(&god, &[root], &mut get_pop, &mut make_key, &mut merge_ev);

        // Flatten again
        let after = flatten_trie_to_programs(&god, &[root]);

        // Compare with stack reference
        let after_set: BTreeSet<Vec<Instr>> = after
            .into_iter()
            .map(|path| compress_pop_only(path))
            .collect();

        assert_eq!(after_set, stacks_after);
    }

    #[test]
    #[ignore = "graph neutralize_remaining_negative_pops not implemented yet"]
    fn test_graph_vs_stack_neutralize_single_path() {
        let god: TestGod = GodWrapper::new();
        let root = new_node(&god);

        // Build path: (1,2); (2,0); (pop-only -2) [note: initial tries won't have pop-only edges; included to mirror post-bubble-up scenario]
        // For trie construction, we only add checked edges; emulate bubble-up scenario:
        let input_prog = vec![Instr::with(1, Some(2)), Instr::with(2, Some(0))];
        let _last = build_linear_trie(&god, root, &input_prog);

        // Flatten trie to programs (no pop-only yet)
        let before = flatten_trie_to_programs(&god, &[root]);

        // Stack reference: first bubble-up to get a trailing pop-only, then neutralize it.
        let mut stacks_after: BTreeSet<Vec<Instr>> = BTreeSet::new();
        for p in before {
            let bubbled = {
                // emulate that a prior bubble may have introduced a trailing pop-only; here we just append one manually to simulate
                let mut q = p.clone();
                q.push(Instr::with(-2, None));
                bubble_up_negative_pops_stack(q)
            };
            stacks_after.insert(neutralize_remaining_negative_pops_stack(bubbled));
        }

        // Graph neutralization
        let mut get_pop = |ek: &TestEK| ek.0;
        let mut make_key = |ek: &TestEK, new_pop: isize| (new_pop, ek.1);
        neutralize_remaining_negative_pops(&god, &[root], &mut get_pop, &mut make_key);

        // Flatten again
        let after = flatten_trie_to_programs(&god, &[root]);

        let after_set: BTreeSet<Vec<Instr>> =
            after.into_iter().map(|path| compress_pop_only(path)).collect();

        assert_eq!(after_set, stacks_after);
    }
}
