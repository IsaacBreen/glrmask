//! Independent literal stack-language fixtures for the active prepared
//! interpreter. No output cache, phase schedule or GSS sharing is used by
//! the reference evaluator.
use super::template_advance::advance_with_prepared_template;
use crate::runtime::{CommitTemplateDfas, FastCommitTemplateDfas};
use crate::automata::unweighted_u32::dfa::{DFA, DFAState};
use crate::compiler::glr::labels::{DEFAULT_LABEL, encode_negative_label, negative_to_positive_label};
use crate::compiler::glr::{accumulator::TerminalsDisallowed, parser::ParserGSS};
use std::collections::{BTreeMap, BTreeSet};

fn graph(n: usize) -> DFA {
    DFA { start_state: 0, states: (0..n).map(|_| DFAState { is_accepting: false, transitions: BTreeMap::new() }).collect() }
}
fn blank(n: usize) -> CommitTemplateDfas {
    CommitTemplateDfas { pop: graph(n), read: graph(n), push: graph(n), pop_to_read: Vec::new(), pop_to_push: Vec::new(), read_to_push: Vec::new() }
}
// Literal stack-language oracle: no fast rows, suffix cache, shared GSS,
// admission certificate, or production traversal helper is used here.
fn literal(t: &CommitTemplateDfas, input: &[u32]) -> BTreeSet<Vec<u32>> {
    let mut pending = vec![(0u8, t.pop.start_state, input.to_vec())];
    let mut output = BTreeSet::new();
    while let Some((phase, id, stack)) = pending.pop() {
        let g = match phase { 0 => &t.pop, 1 => &t.read, _ => &t.push };
        let row = &g.states[id as usize];
        if row.is_accepting { output.insert(stack.clone()); }
        if phase == 0 {
            if let Some(r) = t.pop_to_read.get(id as usize).copied().flatten() { pending.push((1, r, stack.clone())); }
            if let Some(p) = t.pop_to_push.get(id as usize).copied().flatten() { pending.push((2, p, stack.clone())); }
        } else if phase == 1
            && let Some(p) = t.read_to_push.get(id as usize).copied().flatten() { pending.push((2, p, stack.clone())); }
        match phase {
            0 => if let Some(&top) = stack.last()
                && let Some(&next) = row.transitions.get(&(top as i32)).or_else(||row.transitions.get(&DEFAULT_LABEL)) {
                    let mut s = stack; s.pop(); pending.push((0, next, s));
                },
            1 => if let Some(&top) = stack.last()
                && let Some(&next) = row.transitions.get(&(top as i32)) { pending.push((1, next, stack)); },
            _ => for (&label, &next) in &row.transitions {
                let mut s = stack.clone(); s.push(u32::try_from(negative_to_positive_label(label)).expect("valid PUSH label")); pending.push((2, next, s));
            },
        }
    }
    output
}
fn outputs(t: &CommitTemplateDfas, fast: &FastCommitTemplateDfas, source: &[u32]) -> BTreeSet<Vec<u32>> {
    let input = ParserGSS::from_single_stack(source.to_vec(), TerminalsDisallowed::new());
    advance_with_prepared_template(t, input, Some(fast)).to_stacks(4096)
        .expect("bounded literal fixture").into_iter().map(|(word, _)| word).collect()
}
fn next(x: &mut u64) -> u32 { *x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (*x >> 32) as u32 }

#[test]
fn generated_acyclic_prepared_relations_match_literal_oracle() {
    let mut seed = 608894u64;
    let mut words = vec![Vec::new()];
    for n in 1..=3 {
        for code in 0..3usize.pow(n) {
            let mut c = code; let mut w = Vec::new();
            for _ in 0..n { w.push((c % 3) as u32); c /= 3; }
            words.push(w);
        }
    }
    for _ in 0..512 {
        let mut t = blank(4);
        for (phase, g) in [&mut t.pop, &mut t.read, &mut t.push].into_iter().enumerate() {
            for i in 0..4 {
                g.states[i].is_accepting = next(&mut seed).is_multiple_of(3);
                if i == 3 { continue; }
                for label in 0..3 {
                    if next(&mut seed).is_multiple_of(2) {
                        let target = (i + 1 + next(&mut seed) as usize % (3-i)) as u32;
                        let l = if phase == 2 { encode_negative_label(label) } else { label as i32 };
                        g.states[i].transitions.insert(l, target);
                    }
                }
                if phase == 0 && next(&mut seed).is_multiple_of(2) {
                    g.states[i].transitions.insert(DEFAULT_LABEL, (i+1) as u32);
                }
            }
        }
        for links in [&mut t.pop_to_read, &mut t.pop_to_push, &mut t.read_to_push] {
            for _ in 0..(next(&mut seed)%5) {
                links.push(next(&mut seed).is_multiple_of(2).then(|| next(&mut seed)%4));
            }
        }
        let fast = FastCommitTemplateDfas::from_template(&t);
        for word in &words {
            let expected = literal(&t, word);
            assert_eq!(outputs(&t, &fast, word), expected, "word={word:?}");
        }
    }
}

#[test]
fn dead_default_shadow_and_read_epsilon_on_empty_are_exact() {
    let mut t = blank(3);
    t.pop.states[0].transitions.insert(DEFAULT_LABEL, 1);
    t.pop.states[0].transitions.insert(1, 2); // Explicit dead edge.
    t.pop.states[1].is_accepting = true;
    t.pop_to_read = vec![Some(0)];
    t.read.states[0].is_accepting = true;
    let fast = FastCommitTemplateDfas::from_template(&t);
    for source in [vec![], vec![0,1], vec![0,2]] {
        assert_eq!(outputs(&t, &fast, &source), literal(&t, &source));
    }
}

#[test]
fn large_cursor_and_push_languages_complete_after_acceleration_budgets() {
    let mut t = blank(300);
    t.pop_to_read = vec![Some(0)];
    for i in 0..299 {
        t.read.states[i].is_accepting = true;
        t.read.states[i].transitions.insert(0, (i+1) as u32);
    }
    let fast = FastCommitTemplateDfas::from_template(&t);
    assert_eq!(outputs(&t, &fast, &[0]), literal(&t, &[0]));

    let mut t = blank(2);
    t.pop_to_push = vec![Some(0)];
    for symbol in 0..80 {
        t.push.states[0].transitions.insert(encode_negative_label(symbol), 1);
    }
    t.push.states[1].is_accepting = true;
    let fast = FastCommitTemplateDfas::from_template(&t);
    assert_eq!(outputs(&t, &fast, &[0; 64]), literal(&t, &[0; 64]));
}
