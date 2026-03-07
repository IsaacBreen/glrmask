//! Template bundle construction.
//!
//! `sep1` does not build the parser automaton terminal-by-terminal. It first
//! groups equivalent terminal characterizations into reusable template bundles,
//! then composes those bundles into the final parser automaton. `glrmask`
//! still lacks the full tokenizer/template composition step, but it can and
//! should at least build the parser-side bundles explicitly instead of hiding
//! that structure inside `parser_dwa.rs`.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::automata::weighted::determinize::determinize;
use crate::automata::weighted::dwa::CompDwa;
use crate::automata::weighted::minimize::minimize_acyclic;
use crate::automata::weighted::nwa::{Nwa, NwaState};
use crate::automata::weighted::weight::Weight;
use crate::compiler::grammar_def::TerminalId;
use crate::compiler::labels::{DEFAULT_LABEL, encode_negative_label, encode_positive_label};
use crate::compiler::parser_dwa::TerminalCharacterization;
use crate::compiler::resolve_negatives::resolve_negative_codes_in_nwa;
use crate::compiler::terminal_dwa::TerminalDwa;

#[derive(Debug)]
struct TemplateFragment {
    start_states: Vec<u32>,
    final_states: Vec<(u32, Weight)>,
}

struct TemplateCompositionContext<'a> {
    bundles: &'a [TemplateBundle],
    terminal_dwa: &'a TerminalDwa,
    combined: Nwa,
    template_by_terminal: BTreeMap<TerminalId, usize>,
    blueprint_cache: HashMap<(usize, Weight), Nwa>,
    total_fragment_uses: usize,
    total_fragment_states: usize,
}

impl<'a> TemplateCompositionContext<'a> {
    fn new(
        bundles: &'a [TemplateBundle],
        terminal_dwa: &'a TerminalDwa,
        num_tsids: u32,
        max_token: u32,
    ) -> Self {
        let mut template_by_terminal = BTreeMap::new();
        for (bundle_idx, bundle) in bundles.iter().enumerate() {
            for &terminal in &bundle.terminals {
                template_by_terminal.insert(terminal, bundle_idx);
            }
        }

        Self {
            bundles,
            terminal_dwa,
            combined: Nwa::new(num_tsids, max_token),
            template_by_terminal,
            blueprint_cache: HashMap::new(),
            total_fragment_uses: 0,
            total_fragment_states: 0,
        }
    }

    fn fresh_fragment(
        &mut self,
        bundle_idx: usize,
        transition_weight: &Weight,
        num_tsids: u32,
        max_token: u32,
    ) -> TemplateFragment {
        // Cache the *blueprint* NWA per (bundle, weight) to avoid repeated
        // instantiate_template_dfa calls, but append a fresh copy to `combined`
        // for each use so that each transition gets independent NWA state IDs.
        //
        // This prevents cycles: sharing actual NWA state IDs between transitions
        // A→B and B→C creates cycle body_B → F.start →…→ F.final → body_B.
        // Fresh copies eliminate that back-edge opportunity while keeping build
        // cost proportional to (unique templates × uses) rather than (total transitions).
        let key = (bundle_idx, transition_weight.clone());
        if !self.blueprint_cache.contains_key(&key) {
            let blueprint = instantiate_template_dfa(
                &self.bundles[bundle_idx].template_dfa,
                transition_weight,
                num_tsids,
                max_token,
            );
            self.blueprint_cache.insert(key.clone(), blueprint);
        }
        let blueprint = &self.blueprint_cache[&key];
        self.total_fragment_uses += 1;
        self.total_fragment_states += blueprint.states.len();
        let offset = append_nwa(&mut self.combined, blueprint);

        // Diagnostic: verify append worked
        if std::env::var("GLRMASK_DUMP_DWA").unwrap_or_default() == "1" {
            eprintln!("  [fragment] bundle={}, offset={}, blueprint_states={}", bundle_idx, offset, blueprint.states.len());
            for (i, state) in blueprint.states.iter().enumerate() {
                let trans_count: usize = state.transitions.values().map(|v| v.len()).sum();
                let has_final = state.final_weight.is_some();
                eprintln!("    bp_state {}: {} trans, {} eps, final={}", i, trans_count, state.epsilons.len(), has_final);
            }
            for i in 0..blueprint.states.len() {
                let combined_idx = offset as usize + i;
                let cs = &self.combined.states[combined_idx];
                let ct: usize = cs.transitions.values().map(|v| v.len()).sum();
                let has_final = cs.final_weight.is_some();
                eprintln!("    combined_state {}: {} trans, {} eps, final={}", combined_idx, ct, cs.epsilons.len(), has_final);
            }
        }

        let start_states = blueprint
            .start_states
            .iter()
            .map(|state| offset + *state)
            .collect();
        let mut final_states = Vec::new();
        for (fragment_sid, fragment_state) in blueprint.states.iter().enumerate() {
            let Some(final_weight) = &fragment_state.final_weight else {
                continue;
            };
            let combined_state = offset + fragment_sid as u32;
            self.combined.states[combined_state as usize].final_weight = None;
            final_states.push((combined_state, final_weight.clone()));
        }
        TemplateFragment {
            start_states,
            final_states,
        }
    }

    /// Compose a terminal NWA state into the combined parser NWA.
    ///
    /// Returns the **continuation body** — a non-final state with template
    /// fragment entries for processing the next stack element.  A separate
    /// **final node** (if the terminal NWA state is accepting) receives the
    /// token-specific weight from fragment exits.  This separation ensures
    /// that the token constraint from one template step does not bleed into
    /// subsequent steps through the determinizer's epsilon-closure
    /// intersection.
    fn compose_terminal_state(
        &mut self,
        state_id: u32,
        body_cache: &mut BTreeMap<u32, (u32, Option<u32>)>,
        num_tsids: u32,
        max_token: u32,
    ) -> u32 {
        if let Some(&(cont, _)) = body_cache.get(&state_id) {
            return cont;
        }

        let cont_body = self.combined.add_state();

        // If the terminal NWA state is accepting, create a separate final
        // node.  Fragment exits will route finality here (with token weight)
        // and continuation to cont_body (with ALL weight).
        let final_node = {
            let terminal_state = &self.terminal_dwa.nwa.states[state_id as usize];
            if let Some(final_weight) = &terminal_state.final_weight {
                let node = self.combined.add_state();
                self.combined.set_final_weight(node, final_weight.clone());
                Some(node)
            } else {
                None
            }
        };

        body_cache.insert(state_id, (cont_body, final_node));

        let w_all = Weight::all(self.combined.max_position(), num_tsids);

        // Process transitions from the terminal NWA state (template entries).
        let transitions: Vec<_> = self.terminal_dwa.nwa.states[state_id as usize]
            .transitions
            .iter()
            .map(|(&l, t)| (l, t.clone()))
            .collect();

        for (label, targets) in &transitions {
            let Ok(terminal) = TerminalId::try_from(*label) else {
                continue;
            };
            let Some(&bundle_idx) = self.template_by_terminal.get(&terminal) else {
                continue;
            };

            for (dest, transition_weight) in targets {
                let fragment = self.fresh_fragment(bundle_idx, transition_weight, num_tsids, max_token);
                let dest_cont = self.compose_terminal_state(*dest, body_cache, num_tsids, max_token);
                let dest_final = body_cache.get(dest).and_then(|&(_, f)| f);
                for start in &fragment.start_states {
                    self.combined.add_epsilon(cont_body, *start, w_all.clone());
                }
                for (final_state, final_weight) in &fragment.final_states {
                    // Continuation: ALL weight → dest cont_body (no constraint)
                    self.combined.add_epsilon(*final_state, dest_cont, w_all.clone());
                    // Finality: token weight → dest final_node
                    if let Some(fn_id) = dest_final {
                        self.combined.add_epsilon(*final_state, fn_id, final_weight.clone());
                    }
                }
            }
        }

        // If the terminal NWA state is a leaf (final but no transitions),
        // add template fragment entries from ALL root states so the parser
        // NWA can continue processing subsequent stack entries.
        if transitions.is_empty() {
            for &root_id in &self.terminal_dwa.tsid_roots.clone() {
                let root_transitions: Vec<_> = self.terminal_dwa.nwa.states[root_id as usize]
                    .transitions
                    .iter()
                    .map(|(&l, t)| (l, t.clone()))
                    .collect();
                for (label, targets) in &root_transitions {
                    let Ok(terminal) = TerminalId::try_from(*label) else {
                        continue;
                    };
                    let Some(&bundle_idx) = self.template_by_terminal.get(&terminal) else {
                        continue;
                    };
                    for (dest, transition_weight) in targets {
                        let fragment = self.fresh_fragment(bundle_idx, transition_weight, num_tsids, max_token);
                        let dest_cont = self.compose_terminal_state(*dest, body_cache, num_tsids, max_token);
                        let dest_final = body_cache.get(dest).and_then(|&(_, f)| f);
                        for start in &fragment.start_states {
                            self.combined.add_epsilon(cont_body, *start, w_all.clone());
                        }
                        for (final_state, final_weight) in &fragment.final_states {
                            self.combined.add_epsilon(*final_state, dest_cont, w_all.clone());
                            if let Some(fn_id) = dest_final {
                                self.combined.add_epsilon(*final_state, fn_id, final_weight.clone());
                            }
                        }
                    }
                }
            }
        }

        cont_body
    }
}

/// A compiled structural template DFA, independent of lexical weights.
#[derive(Debug, Clone)]
pub struct TemplateDfa {
    pub dfa: CompDwa,
}

/// A parser-side template bundle shared by one or more terminals.
#[derive(Debug, Clone)]
pub struct TemplateBundle {
    /// Structural template DFA compiled from the shared characterization.
    pub template_dfa: TemplateDfa,
    /// All terminals that reuse the same parser-side template.
    pub terminals: Vec<TerminalId>,
}

/// Group terminals that share the same parser-side characterization.
///
/// This is the structural part of the `sep1` template-bundle design: parser
/// patterns are deduplicated before they are turned into automaton fragments.
/// The current `glrmask` rewrite still uses direct lexical weights instead of
/// true template/tokenizer composition, but bundling equivalent parser
/// templates keeps the architecture aligned with the intended direction.
pub(crate) fn build_template_bundles(
    characterizations: &BTreeMap<TerminalId, TerminalCharacterization>,
    used_terminals: &BTreeSet<TerminalId>,
) -> Vec<TemplateBundle> {
    let mut bundles: BTreeMap<TerminalCharacterization, TemplateBundle> = BTreeMap::new();

    for (&terminal, characterization) in characterizations {
        if !used_terminals.contains(&terminal) {
            continue;
        }

        let bundle = bundles
            .entry(characterization.clone())
            .or_insert_with(|| TemplateBundle {
                template_dfa: build_template_dfa(characterization),
                terminals: Vec::new(),
            });
        bundle.terminals.push(terminal);
    }

    bundles.into_values().collect()
}

fn ensure_nt_stack_state(
    nwa: &mut Nwa,
    nt_stacks: &mut BTreeMap<u32, Vec<u32>>,
    nt: u32,
    depth: usize,
    w_all: &Weight,
) -> u32 {
    let stack = nt_stacks.get_mut(&nt).expect("nt stack must exist");
    while stack.len() <= depth {
        let new_state = nwa.add_state();
        let prev_state = *stack.last().expect("nt stack must be non-empty");
        nwa.add_transition(new_state, DEFAULT_LABEL, prev_state, w_all.clone());
        stack.push(new_state);
    }
    stack[depth]
}

fn build_template_structure_nwa(characterization: &TerminalCharacterization) -> Nwa {
    let mut nwa = Nwa::new(1, 0);
    let w_all = Weight::all(0, 1);

    let start = nwa.add_state();
    let end = nwa.add_state();
    nwa.start_states.push(start);
    nwa.set_final_weight(end, w_all.clone());

    let mut nt_states: BTreeMap<u32, u32> = BTreeMap::new();
    let mut nt_stacks: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    for &nt in &characterization.all_nts {
        let state = nwa.add_state();
        nt_states.insert(nt, state);
        nt_stacks.insert(nt, vec![state]);
    }

    for &(state, shift_state) in &characterization.shifts {
        let s1 = nwa.add_state();
        let s2 = nwa.add_state();
        nwa.add_transition(start, encode_positive_label(state), s1, w_all.clone());
        nwa.add_transition(s1, encode_negative_label(state), s2, w_all.clone());
        nwa.add_transition(s2, encode_negative_label(shift_state), end, w_all.clone());
    }

    for &(state, len, nt) in &characterization.reduces {
        let target = ensure_nt_stack_state(&mut nwa, &mut nt_stacks, nt, len, &w_all);
        nwa.add_transition(start, encode_positive_label(state), target, w_all.clone());
    }

    for &(nt, revealed, goto_state, shift_state) in &characterization.nt_escapes {
        let nt_state = *nt_states.get(&nt).expect("nt state must exist");
        let s1 = nwa.add_state();
        let s2 = nwa.add_state();
        let s3 = nwa.add_state();
        nwa.add_transition(nt_state, encode_positive_label(revealed), s1, w_all.clone());
        nwa.add_transition(s1, encode_negative_label(revealed), s2, w_all.clone());
        nwa.add_transition(s2, encode_negative_label(goto_state), s3, w_all.clone());
        nwa.add_transition(s3, encode_negative_label(shift_state), end, w_all.clone());
    }

    for &(nt, revealed, remaining_len, target_nt) in &characterization.nt_rereduces {
        let nt_state = *nt_states.get(&nt).expect("nt state must exist");
        let target = ensure_nt_stack_state(
            &mut nwa,
            &mut nt_stacks,
            target_nt,
            remaining_len,
            &w_all,
        );
        nwa.add_transition(nt_state, encode_positive_label(revealed), target, w_all.clone());
    }

    nwa
}

fn is_acyclic(dwa: &CompDwa) -> bool {
    fn dfs(state_id: usize, states: &[crate::automata::weighted::dwa::CompDwaState], colors: &mut [u8]) -> bool {
        colors[state_id] = 1;
        for (target, _) in states[state_id].transitions.values() {
            let target = *target as usize;
            match colors[target] {
                1 => return false,
                0 => {
                    if !dfs(target, states, colors) {
                        return false;
                    }
                }
                _ => {}
            }
        }
        colors[state_id] = 2;
        true
    }

    let mut colors = vec![0u8; dwa.states.len()];
    for state_id in 0..dwa.states.len() {
        if colors[state_id] == 0 && !dfs(state_id, &dwa.states, &mut colors) {
            return false;
        }
    }
    true
}

fn build_template_dfa(characterization: &TerminalCharacterization) -> TemplateDfa {
    let mut nwa = build_template_structure_nwa(characterization);

    // Dump template NWA before resolve
    if std::env::var("GLRMASK_DUMP_DWA").unwrap_or_default() == "1" {
        eprintln!("\n  [template] NWA before resolve ({} states):", nwa.states.len());
        eprintln!("    starts: {:?}", nwa.start_states);
        for (i, state) in nwa.states.iter().enumerate() {
            let mut parts = Vec::new();
            for (&label, targets) in &state.transitions {
                for (dest, _) in targets {
                    let label_str = if label == crate::compiler::labels::DEFAULT_LABEL {
                        "DEFAULT".to_string()
                    } else if crate::compiler::labels::is_negative_label(label) {
                        format!("neg({})", crate::compiler::labels::negative_to_positive_label(label))
                    } else {
                        format!("{}", label)
                    };
                    parts.push(format!("--[{}]-->{}", label_str, dest));
                }
            }
            for (dest, _) in &state.epsilons {
                parts.push(format!("--[eps]-->{}", dest));
            }
            if state.final_weight.is_some() {
                parts.push("FINAL".to_string());
            }
            if !parts.is_empty() {
                eprintln!("    state {}: {}", i, parts.join(", "));
            }
        }
    }

    resolve_negative_codes_in_nwa(&mut nwa);

    // Dump template NWA after resolve
    if std::env::var("GLRMASK_DUMP_DWA").unwrap_or_default() == "1" {
        eprintln!("  [template] NWA after resolve ({} states):", nwa.states.len());
        for (i, state) in nwa.states.iter().enumerate() {
            let mut parts = Vec::new();
            for (&label, targets) in &state.transitions {
                for (dest, _) in targets {
                    parts.push(format!("--[{}]-->{}", label, dest));
                }
            }
            for (dest, _) in &state.epsilons {
                parts.push(format!("--[eps]-->{}", dest));
            }
            if state.final_weight.is_some() {
                parts.push("FINAL".to_string());
            }
            if !parts.is_empty() {
                eprintln!("    state {}: {}", i, parts.join(", "));
            }
        }
    }

    let dfa = determinize(&nwa);

    // Dump template DFA
    if std::env::var("GLRMASK_DUMP_DWA").unwrap_or_default() == "1" {
        eprintln!("  [template] DFA ({} states, start={}):", dfa.num_states(), dfa.start_state);
        for (i, state) in dfa.states.iter().enumerate() {
            let mut parts = Vec::new();
            for (&label, (target, _)) in &state.transitions {
                parts.push(format!("--[{}]-->{}", label, target));
            }
            if state.final_weight.is_some() {
                parts.push("FINAL".to_string());
            }
            if !parts.is_empty() {
                eprintln!("    state {}: {}", i, parts.join(", "));
            }
        }
    }

    assert!(
        is_acyclic(&dfa),
        "template DFA is cyclic after determinization"
    );
    let minimized = minimize_acyclic(&dfa);

    if std::env::var("GLRMASK_DUMP_DWA").unwrap_or_default() == "1" {
        eprintln!("  [template] Minimized DFA ({} states, start={}):", minimized.num_states(), minimized.start_state);
        for (i, state) in minimized.states.iter().enumerate() {
            let mut parts = Vec::new();
            for (&label, (target, _)) in &state.transitions {
                parts.push(format!("--[{}]-->{}", label, target));
            }
            if state.final_weight.is_some() {
                parts.push("FINAL".to_string());
            }
            if !parts.is_empty() {
                eprintln!("    state {}: {}", i, parts.join(", "));
            }
        }
    }

    TemplateDfa {
        dfa: minimized,
    }
}

fn instantiate_template_dfa(
    template_dfa: &TemplateDfa,
    terminal_weight: &Weight,
    num_tsids: u32,
    max_token: u32,
) -> Nwa {
    let mut nwa = Nwa::new(num_tsids, max_token);
    let w_all = Weight::all(nwa.max_position(), num_tsids);

    for _ in 0..template_dfa.dfa.states.len() {
        nwa.add_state();
    }
    nwa.start_states.push(template_dfa.dfa.start_state);

    for (sid, state) in template_dfa.dfa.states.iter().enumerate() {
        for (&label, (target, _)) in &state.transitions {
            nwa.add_transition(sid as u32, label, *target, w_all.clone());
        }
        if state.final_weight.is_some() {
            nwa.set_final_weight(sid as u32, terminal_weight.clone());
        }
    }

    nwa
}

fn append_nwa(target: &mut Nwa, fragment: &Nwa) -> u32 {
    debug_assert_eq!(target.num_tsids, fragment.num_tsids);
    debug_assert_eq!(target.max_token, fragment.max_token);

    let offset = target.states.len() as u32;
    target
        .states
        .extend((0..fragment.states.len()).map(|_| NwaState::default()));

    for (index, state) in fragment.states.iter().enumerate() {
        let new_state = &mut target.states[offset as usize + index];
        new_state.final_weight = state.final_weight.clone();
        new_state.transitions = state
            .transitions
            .iter()
            .map(|(&label, targets)| {
                (
                    label,
                    targets
                        .iter()
                        .map(|(dest, weight)| (dest + offset, weight.clone()))
                        .collect(),
                )
            })
            .collect();
        new_state.epsilons = state
            .epsilons
            .iter()
            .map(|(dest, weight)| (dest + offset, weight.clone()))
            .collect();
    }

    target
        .start_states
        .extend(fragment.start_states.iter().map(|state| state + offset));

    offset
}

/// Build the parser-side template NWA by unioning the per-bundle template NWAs.
pub(crate) fn build_template_nwa_from_bundles(
    bundles: &[TemplateBundle],
    terminal_dwa: &TerminalDwa,
    num_tsids: u32,
    max_token: u32,
) -> Nwa {
    let mut context = TemplateCompositionContext::new(bundles, terminal_dwa, num_tsids, max_token);
    let mut body_cache: BTreeMap<u32, (u32, Option<u32>)> = BTreeMap::new();
    context.combined.start_states = terminal_dwa
        .nwa
        .start_states
        .iter()
        .map(|state| context.compose_terminal_state(*state, &mut body_cache, num_tsids, max_token))
        .collect();
    // Compute blueprint size histogram
    let mut size_hist: BTreeMap<usize, usize> = BTreeMap::new();
    for bp in context.blueprint_cache.values() {
        *size_hist.entry(bp.states.len()).or_insert(0) += 1;
    }
    // Count unique bundle indices used
    let unique_bundles: std::collections::HashSet<usize> = context.blueprint_cache.keys().map(|(b,_)| *b).collect();
    eprintln!("[compose] blueprint_cache={} unique entries, terminal_states={}, body_cache={} entries, fragment_uses={}, fragment_states={}, unique_bundles={}, size_hist={:?}, total_nwa_states={}",
        context.blueprint_cache.len(), terminal_dwa.nwa.states.len(), body_cache.len(),
        context.total_fragment_uses, context.total_fragment_states,
        unique_bundles.len(), size_hist, context.combined.states.len());
    context.combined
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeMap, BTreeSet};

    #[test]
    fn test_build_template_bundles_groups_equivalent_terminals() {
        let shared = TerminalCharacterization {
            shifts: vec![(0, 1)],
            reduces: vec![(2, 1, 7)],
            nt_escapes: vec![(7, 0, 3, 4)],
            nt_rereduces: vec![(7, 1, 2, 9)],
            all_nts: BTreeSet::from([7, 9]),
        };

        let characterizations = BTreeMap::from([(0, shared.clone()), (1, shared)]);
        let used_terminals = BTreeSet::from([0, 1]);
        let bundles = build_template_bundles(&characterizations, &used_terminals);

        assert_eq!(bundles.len(), 1);
        assert_eq!(bundles[0].terminals, vec![0, 1]);
        assert!(bundles[0].template_dfa.dfa.num_states() > 0);
    }
}
