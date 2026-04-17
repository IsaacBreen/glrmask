//! Template-DFA compilation from terminal characterizations.
//!
//! Builds each template as a lightweight NFA (fresh intermediate states per
//! path, epsilon-connected to NT nodes) and then determinizes + minimizes to
//! produce an acyclic unweighted DFA.

use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use crate::automata::weighted::nwa::{NWA, NWAState};
use crate::automata::unweighted_u32::dfa::DFA as UnweightedDfa;
use crate::automata::unweighted_u32::determinize::determinize;
use crate::automata::unweighted_u32::minimize_acyclic::minimize_acyclic as minimize_dfa;
use crate::automata::unweighted_u32::nfa::NFA;
use crate::compiler::glr::labels::{encode_negative_label, encode_positive_label, DEFAULT_LABEL};
use crate::grammar::flat::TerminalID;
use crate::compiler::stages::templates::characterize::TerminalCharacterization;
use crate::ds::weight::Weight;

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct TemplateCompileProfile {
    pub(crate) build_nfa_ms: f64,
    pub(crate) determinize_ms: f64,
    pub(crate) minimize_ms: f64,
    pub(crate) total_ms: f64,
    pub(crate) num_terminals: usize,
    pub(crate) unique_characterizations: usize,
    pub(crate) max_characterization_multiplicity: usize,
    pub(crate) total_nfa_states: usize,
    pub(crate) max_nfa_states: usize,
    pub(crate) total_nfa_transitions: usize,
    pub(crate) max_nfa_transitions: usize,
    pub(crate) total_dfa_states: usize,
    pub(crate) max_dfa_states: usize,
    pub(crate) total_dfa_transitions: usize,
    pub(crate) max_dfa_transitions: usize,
}

impl TemplateCompileProfile {
    pub(crate) fn avg_nfa_states(&self) -> f64 {
        average(self.total_nfa_states, self.num_terminals)
    }

    pub(crate) fn avg_nfa_transitions(&self) -> f64 {
        average(self.total_nfa_transitions, self.num_terminals)
    }

    pub(crate) fn avg_dfa_states(&self) -> f64 {
        average(self.total_dfa_states, self.num_terminals)
    }

    pub(crate) fn avg_dfa_transitions(&self) -> f64 {
        average(self.total_dfa_transitions, self.num_terminals)
    }

    fn observe_compilation(&mut self, sample: &TemplateCompilationSample) {
        self.build_nfa_ms += sample.build_nfa_ms;
        self.determinize_ms += sample.determinize_ms;
        self.minimize_ms += sample.minimize_ms;
        self.total_ms += sample.total_ms();
        self.num_terminals += 1;
        self.total_nfa_states += sample.nfa_states;
        self.max_nfa_states = self.max_nfa_states.max(sample.nfa_states);
        self.total_nfa_transitions += sample.nfa_transitions;
        self.max_nfa_transitions = self.max_nfa_transitions.max(sample.nfa_transitions);
        self.total_dfa_states += sample.dfa_states;
        self.max_dfa_states = self.max_dfa_states.max(sample.dfa_states);
        self.total_dfa_transitions += sample.dfa_transitions;
        self.max_dfa_transitions = self.max_dfa_transitions.max(sample.dfa_transitions);
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct TemplateCompilationSample {
    build_nfa_ms: f64,
    determinize_ms: f64,
    minimize_ms: f64,
    nfa_states: usize,
    nfa_transitions: usize,
    dfa_states: usize,
    dfa_transitions: usize,
}

impl TemplateCompilationSample {
    fn total_ms(&self) -> f64 {
        self.build_nfa_ms + self.determinize_ms + self.minimize_ms
    }
}

fn elapsed_ms(started_at: Instant) -> f64 {
    started_at.elapsed().as_secs_f64() * 1000.0
}

fn average(total: usize, count: usize) -> f64 {
    if count == 0 {
        0.0
    } else {
        total as f64 / count as f64
    }
}

fn nfa_size(nfa: &NFA) -> (usize, usize) {
    let transitions = nfa
        .states
        .iter()
        .map(|state| {
            state
                .transitions
                .values()
                .map(Vec::len)
                .sum::<usize>()
                + state.epsilons.len()
        })
        .sum();
    (nfa.states.len(), transitions)
}

fn dfa_size(dfa: &UnweightedDfa) -> (usize, usize) {
    let transitions = dfa
        .states
        .iter()
        .map(|state| state.transitions.len())
        .sum();
    (dfa.states.len(), transitions)
}

fn dfa_to_nwa_skeleton(dfa: &UnweightedDfa) -> NWA {
    let states = dfa
        .states
        .iter()
        .map(|state| NWAState {
            final_weight: state.is_accepting.then(Weight::empty),
            transitions: state
                .transitions
                .iter()
                .map(|(&label, &target)| (label, vec![(target, Weight::empty())]))
                .collect(),
            epsilons: Vec::new(),
        })
        .collect();

    NWA {
        states,
        start_states: vec![dfa.start_state],
    }
}

fn compile_template_with_profile(
    characterization: &TerminalCharacterization,
) -> (UnweightedDfa, NWA, TemplateCompilationSample) {
    let build_nfa_started_at = Instant::now();
    let nfa = build_template_nfa(characterization);
    let build_nfa_ms = elapsed_ms(build_nfa_started_at);
    let (nfa_states, nfa_transitions) = nfa_size(&nfa);

    let determinize_started_at = Instant::now();
    let determinized = determinize(&nfa);
    let determinize_ms = elapsed_ms(determinize_started_at);

    let minimize_started_at = Instant::now();
    let dfa = minimize_dfa(&determinized);
    let minimize_ms = elapsed_ms(minimize_started_at);
    let (dfa_states, dfa_transitions) = dfa_size(&dfa);

    let skeleton = dfa_to_nwa_skeleton(&dfa);

    (
        dfa,
        skeleton,
        TemplateCompilationSample {
            build_nfa_ms,
            determinize_ms,
            minimize_ms,
            nfa_states,
            nfa_transitions,
            dfa_states,
            dfa_transitions,
        },
    )
}

pub(crate) fn emit_template_profile_summary(
    characterize_ms: f64,
    profile: &TemplateCompileProfile,
) {
    eprintln!(
        "[glrmask/profile][templates] characterize_ms={:.3} compile_ms={:.3} build_nfa_ms={:.3} determinize_ms={:.3} minimize_ms={:.3} num_terminals={} unique_characterizations={} max_characterization_multiplicity={} avg_nfa_states={:.1} max_nfa_states={} avg_nfa_transitions={:.1} max_nfa_transitions={} avg_dfa_states={:.1} max_dfa_states={} avg_dfa_transitions={:.1} max_dfa_transitions={} total_ms={:.3}",
        characterize_ms,
        profile.total_ms,
        profile.build_nfa_ms,
        profile.determinize_ms,
        profile.minimize_ms,
        profile.num_terminals,
        profile.unique_characterizations,
        profile.max_characterization_multiplicity,
        profile.avg_nfa_states(),
        profile.max_nfa_states,
        profile.avg_nfa_transitions(),
        profile.max_nfa_transitions,
        profile.avg_dfa_states(),
        profile.max_dfa_states,
        profile.avg_dfa_transitions(),
        profile.max_dfa_transitions,
        characterize_ms + profile.total_ms,
    );
}

#[derive(Debug, Clone, Default)]
pub struct Templates {
    pub by_terminal: BTreeMap<TerminalID, UnweightedDfa>,
    pub by_terminal_nwa: BTreeMap<TerminalID, NWA>,
}

impl Templates {
    pub(crate) fn from_characterizations(
        characterizations: &BTreeMap<TerminalID, TerminalCharacterization>,
    ) -> Self {
        use rayon::prelude::*;

        let compiled: Vec<(TerminalID, UnweightedDfa, NWA)> = characterizations
            .par_iter()
            .map(|(&terminal, characterization)| {
                let nfa = build_template_nfa(characterization);
                let dfa = minimize_dfa(&determinize(&nfa));
                let skeleton = dfa_to_nwa_skeleton(&dfa);
                (terminal, dfa, skeleton)
            })
            .collect();

        let mut by_terminal = BTreeMap::new();
        let mut by_terminal_nwa = BTreeMap::new();
        for (terminal, dfa, skeleton) in compiled {
            by_terminal.insert(terminal, dfa);
            by_terminal_nwa.insert(terminal, skeleton);
        }

        Self {
            by_terminal,
            by_terminal_nwa,
        }
    }

    pub(crate) fn from_characterizations_profiled(
        characterizations: &BTreeMap<TerminalID, TerminalCharacterization>,
    ) -> (Self, TemplateCompileProfile) {
        use rayon::prelude::*;

        let mut multiplicities = BTreeMap::<&TerminalCharacterization, usize>::new();
        for characterization in characterizations.values() {
            *multiplicities.entry(characterization).or_default() += 1;
        }

        let compiled: Vec<(TerminalID, UnweightedDfa, NWA, TemplateCompilationSample)> =
            characterizations
                .par_iter()
                .map(|(&terminal, characterization)| {
                    let (dfa, skeleton, sample) = compile_template_with_profile(characterization);
                    (terminal, dfa, skeleton, sample)
                })
                .collect();

        let mut profile = TemplateCompileProfile {
            unique_characterizations: multiplicities.len(),
            max_characterization_multiplicity: multiplicities.values().copied().max().unwrap_or(0),
            ..TemplateCompileProfile::default()
        };

        let mut by_terminal = BTreeMap::new();
        let mut by_terminal_nwa = BTreeMap::new();
        for (terminal, dfa, skeleton, sample) in compiled {
            if std::env::var("GLRMASK_DEBUG_PROFILE").is_ok() && sample.nfa_states > 1000 {
                eprintln!(
                    "[glrmask/debug][template_large] terminal={} nfa_states={} nfa_transitions={} dfa_states={} dfa_transitions={} build_nfa_ms={:.1} det_ms={:.1} min_ms={:.1}",
                    terminal, sample.nfa_states, sample.nfa_transitions, sample.dfa_states, sample.dfa_transitions,
                    sample.build_nfa_ms, sample.determinize_ms, sample.minimize_ms
                );
            }
            profile.observe_compilation(&sample);
            by_terminal.insert(terminal, dfa);
            by_terminal_nwa.insert(terminal, skeleton);
        }

        (
            Self {
                by_terminal,
                by_terminal_nwa,
            },
            profile,
        )
    }
}

fn build_nonterminal_nodes(
    nfa: &mut NFA,
    characterization: &TerminalCharacterization,
) -> BTreeMap<u32, u32> {
    let mut nonterminal_nodes = BTreeMap::new();
    for &nonterminal in &characterization.all_nts {
        let state = nfa.add_state();
        nonterminal_nodes.insert(nonterminal, state);
    }
    nonterminal_nodes
}

fn append_default_pop_chain(nfa: &mut NFA, mut from: u32, pop_count: usize, target: u32) {
    for pop_index in 0..pop_count {
        let to = if pop_index == pop_count - 1 {
            target
        } else {
            nfa.add_state()
        };
        nfa.add_transition(from, DEFAULT_LABEL, to);
        from = to;
    }
}

fn add_positive_transition_chain(
    nfa: &mut NFA,
    from: u32,
    revealed_state: u32,
    pop_count: usize,
    target: u32,
) {
    if pop_count == 0 {
        // Zero-length reduce: no stack pop — epsilon to the nonterminal node.
        nfa.add_epsilon(from, target);
        return;
    }
    if pop_count == 1 {
        nfa.add_transition(from, encode_positive_label(revealed_state), target);
        return;
    }
    let first_target = nfa.add_state();
    nfa.add_transition(from, encode_positive_label(revealed_state), first_target);
    append_default_pop_chain(nfa, first_target, pop_count - 1, target);
}

/// A shared DEFAULT-labeled pop chain ending at `target`.
///
/// `chain[i]` is an NFA state such that there is a sequence of `i+1`
/// consecutive DEFAULT transitions from `chain[i]` to `target`. That is:
/// - `chain[0]` has a DEFAULT transition to `target` (one pop).
/// - `chain[i]` has a DEFAULT transition to `chain[i - 1]` (i+1 pops).
///
/// A caller wanting `k` pops leading to `target` (`k >= 1`) directs its
/// positive transition to `chain[k - 1]`, reusing all DEFAULT-pop states
/// shared by other reduces targeting the same nonterminal. This keeps
/// the template NFA size at O(num_nonterminals × max_pop_count) instead
/// of O(total_reduces × avg_pop_count).
struct PopChain {
    states: Vec<u32>,
}

struct PopChainPool {
    chains: BTreeMap<u32, PopChain>,
}

impl PopChainPool {
    fn new() -> Self {
        Self {
            chains: BTreeMap::new(),
        }
    }

    /// Return the NFA state that has a chain of `pop_count` DEFAULT transitions
    /// terminating at the nonterminal node `target_state`, extending the shared
    /// chain for `target_nt` as needed. Requires `pop_count >= 1`.
    fn entry_state(
        &mut self,
        nfa: &mut NFA,
        target_nt: u32,
        target_state: u32,
        pop_count: usize,
    ) -> u32 {
        debug_assert!(pop_count >= 1);
        let chain = self.chains.entry(target_nt).or_insert_with(|| PopChain {
            states: Vec::new(),
        });
        while chain.states.len() < pop_count {
            let idx = chain.states.len();
            let predecessor = if idx == 0 {
                target_state
            } else {
                chain.states[idx - 1]
            };
            let new_state = nfa.add_state();
            nfa.add_transition(new_state, DEFAULT_LABEL, predecessor);
            chain.states.push(new_state);
        }
        chain.states[pop_count - 1]
    }
}

fn add_positive_transition_chain_shared(
    nfa: &mut NFA,
    pool: &mut PopChainPool,
    from: u32,
    revealed_state: u32,
    pop_count: usize,
    target_nt: u32,
    target_state: u32,
) {
    if pop_count == 0 {
        nfa.add_epsilon(from, target_state);
        return;
    }
    if pop_count == 1 {
        nfa.add_transition(from, encode_positive_label(revealed_state), target_state);
        return;
    }
    let entry = pool.entry_state(nfa, target_nt, target_state, pop_count - 1);
    nfa.add_transition(from, encode_positive_label(revealed_state), entry);
}

/// Build an unweighted NFA from a terminal characterization.
///
/// Each shift/reduce/escape/re-reduce path gets its own fresh intermediate
/// states, connected to the shared start state (via epsilon) and to shared
/// NT-node states.
fn build_template_nfa(characterization: &TerminalCharacterization) -> NFA {
    let mut nfa = NFA::new();
    let start = 0u32; // NFA::new() creates state 0 as start

    let nonterminal_nodes = build_nonterminal_nodes(&mut nfa, characterization);
    let mut pool = PopChainPool::new();

    // Shared escape-chain tail.
    //
    // An "escape chain" is the sequence
    //     positive(revealed_state) → negative(pushes[0]) → … → negative(pushes[n]) → accepting
    // emitted for every `(escape)` and `(nt_escape)` entry in the
    // characterization. Rather than materialise a distinct entry node per
    // signature and splice the source via an epsilon, each source adds its
    // positive transition directly to a shared "pos-target" state that
    // represents the state reached just after firing `positive(revealed)`.
    // The pos-target state is cached per `pushes` (the `revealed` component
    // differs per caller but never affects the negative-chain tail).
    //
    // A source dedup set eliminates duplicate positive transitions when the
    // characterization repeats `(source, revealed, pushes)` tuples.

    // Suffix trie over *reversed* push sequences, all rooted at a single
    // shared accepting state. If two signatures share a common `pushes`
    // suffix, they share the corresponding NFA states and negative
    // transitions. For `(pushes = [p0, p1, …, pn])`, the trie walk starts at
    // the shared `accept_root` and consumes `pn, pn-1, …, p0` in reverse;
    // the state reached after consuming all pushes is the pos-target that
    // the caller's positive transition points at.
    //
    // Key: `(child_state, push_label)` → `parent_state` such that
    // `parent_state` has a `negative(push_label)` transition to `child_state`.
    let accept_root = nfa.add_state();
    nfa.set_accepting(accept_root);
    let mut suffix_trie: BTreeMap<(u32, u32), u32> = BTreeMap::new();

    // Cache of pos-target states keyed by `pushes`.
    let mut pos_target_cache: BTreeMap<Vec<u32>, u32> = BTreeMap::new();

    // Dedup set for emitted `(source, revealed, pushes)` positive transitions.
    // Keying includes `pushes` rather than `pos_target` because two distinct
    // `pushes` sequences may resolve (under suffix sharing) to the same
    // `pos_target`, yet still represent logically distinct escapes; we dedupe
    // purely to avoid inserting the same transition twice when the
    // characterization contains exact duplicates.
    let mut emitted_escapes: BTreeSet<(u32, u32, Vec<u32>)> = BTreeSet::new();

    // Resolve (or build) the pos-target for a `pushes` suffix by walking
    // it in reverse through the suffix trie rooted at `accept_root`.
    let resolve_pos_target =
        |nfa: &mut NFA,
         pos_target_cache: &mut BTreeMap<Vec<u32>, u32>,
         suffix_trie: &mut BTreeMap<(u32, u32), u32>,
         pushes: &[u32]|
         -> u32 {
            if let Some(&cached) = pos_target_cache.get(pushes) {
                return cached;
            }
            let mut cur = accept_root;
            for &push_state in pushes.iter().rev() {
                let key = (cur, push_state);
                cur = if let Some(&existing) = suffix_trie.get(&key) {
                    existing
                } else {
                    let s = nfa.add_state();
                    nfa.add_transition(s, encode_negative_label(push_state), cur);
                    suffix_trie.insert(key, s);
                    s
                };
            }
            pos_target_cache.insert(pushes.to_vec(), cur);
            cur
        };

    // Initial escapes: start → positive(initial_state) → [shared suffix tail] → accept_root
    for &(initial_state, ref pushes) in &characterization.escapes {
        if !emitted_escapes.insert((start, initial_state, pushes.to_vec())) {
            continue;
        }
        let pos_target = resolve_pos_target(
            &mut nfa,
            &mut pos_target_cache,
            &mut suffix_trie,
            pushes,
        );
        nfa.add_transition(start, encode_positive_label(initial_state), pos_target);
    }

    for &(initial_state, pop_count, nonterminal) in &characterization.reduces {
        let Some(&target_nonterminal_state) = nonterminal_nodes.get(&nonterminal) else {
            continue;
        };

        // Emit the reduce chain directly from `start`; no staging state is
        // needed because `start` is already the common source for every
        // `reduces` entry, and the chain itself handles the pop/reveal split.
        add_positive_transition_chain_shared(
            &mut nfa,
            &mut pool,
            start,
            initial_state,
            pop_count,
            nonterminal,
            target_nonterminal_state,
        );
    }

    // NT escapes: source_nt_node → positive(revealed) → [shared suffix tail] → accept_root.
    // The suffix tail is shared across every `(source, revealed, pushes)` that
    // agrees on the `pushes` tail; the positive transition is added directly
    // from the source, with dedup against exact `(source, revealed, pushes)`
    // duplicates.
    for &(source_nonterminal, revealed_state, ref pushes) in &characterization.nt_escapes {
        let Some(&source_state) = nonterminal_nodes.get(&source_nonterminal) else {
            continue;
        };
        if !emitted_escapes.insert((source_state, revealed_state, pushes.to_vec())) {
            continue;
        }
        let pos_target = resolve_pos_target(
            &mut nfa,
            &mut pos_target_cache,
            &mut suffix_trie,
            pushes,
        );
        nfa.add_transition(source_state, encode_positive_label(revealed_state), pos_target);
    }

    for &(source_nonterminal, revealed_state, pop_count, target_nonterminal) in &characterization.nt_rereduces {
        let (Some(&source_state), Some(&target_state)) =
            (nonterminal_nodes.get(&source_nonterminal), nonterminal_nodes.get(&target_nonterminal))
        else {
            continue;
        };

        // Emit the rereduce chain directly from `source_state`; per-entry
        // staging states are redundant (the chain's own states disambiguate
        // distinct entries).
        add_positive_transition_chain_shared(
            &mut nfa,
            &mut pool,
            source_state,
            revealed_state,
            pop_count,
            target_nonterminal,
            target_state,
        );
    }

    nfa
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::automata::unweighted_u32::dfa::DFA as UnweightedDfa;
    use crate::automata::unweighted_u32::nfa::NFAState;
    use crate::automata::weighted::dwa::DWA;
    use crate::automata::weighted::nwa::NWA as WeightedNwa;
    use crate::compiler::glr::labels::{DEFAULT_LABEL, is_negative_label, negative_to_positive_label};
    use crate::compiler::glr::analysis::AnalyzedGrammar;
    use crate::compiler::glr::table::GLRTable;
    use crate::compiler::grammar::transforms::prepare_grammar_transforms_only;
    use crate::compiler::pipeline::build_tokenizer;
    use crate::compiler::stages::equiv_types::InternalIdMap;
    use crate::compiler::stages::parser_dwa::build_parser_dwa_from_terminal_dwa_with_precomputed_templates;
    use crate::compiler::stages::templates::characterize::characterize_terminals;
    use crate::compiler::stages::terminal_dwa_compat::build_terminal_dwa_for_existing_id_map;
    use crate::grammar::flat::{GrammarDef, Symbol, Terminal};
    use crate::import::lark::parse_lark;
    use crate::Constraint;
    use crate::Vocab;

    fn minimal_repro_grammar() -> &'static str {
        r#"
        start: item+ "d"
        item: "d" node
        node: leaf
        leaf: "d"
        "#
    }

    fn minimal_boundary_pass_grammar() -> &'static str {
        r#"
        start: item+ "d"
        item: "d" leaf
        leaf: "d"
        "#
    }

    fn minimal_repro_vocab() -> Vocab {
        Vocab::new(vec![(0, b"d".to_vec())], None)
    }

    fn mask_allows(mask: &[u32], token: u32) -> bool {
        let word = (token / 32) as usize;
        let bit = 1u32 << (token % 32);
        mask.get(word).is_some_and(|value| (value & bit) != 0)
    }

    fn describe_label(label: i32) -> String {
        if label == DEFAULT_LABEL {
            "DEFAULT".to_string()
        } else if is_negative_label(label) {
            format!("push({})", negative_to_positive_label(label))
        } else {
            format!("pop({})", label)
        }
    }

    fn print_unweighted_dfa(label: &str, dfa: &UnweightedDfa) {
        eprintln!("{} start_state={}", label, dfa.start_state);
        for (state_id, state) in dfa.states.iter().enumerate() {
            eprintln!("  state {} accepting={}", state_id, state.is_accepting);
            let mut transitions = state.transitions.iter().collect::<Vec<_>>();
            transitions.sort_by_key(|(edge_label, _)| **edge_label);
            for (edge_label, target) in transitions {
                eprintln!("    {} -> {}", describe_label(*edge_label), target);
            }
        }
    }

    fn print_nfa(label: &str, nfa: &NFA) {
        eprintln!("{} start_states={:?}", label, nfa.start_states);
        for (state_id, state) in nfa.states.iter().enumerate() {
            print_nfa_state(state_id, state);
        }
    }

    fn print_nfa_state(state_id: usize, state: &NFAState) {
        eprintln!("  state {} accepting={} epsilons={:?}", state_id, state.is_accepting, state.epsilons);
        let mut transitions = state.transitions.iter().collect::<Vec<_>>();
        transitions.sort_by_key(|(edge_label, _)| **edge_label);
        for (edge_label, targets) in transitions {
            eprintln!("    {} -> {:?}", describe_label(*edge_label), targets);
        }
    }

    fn print_weighted_nwa(label: &str, nwa: &WeightedNwa) {
        eprintln!("{} start_states={:?}", label, nwa.start_states);
        for (state_id, state) in nwa.states.iter().enumerate() {
            eprintln!("  state {} final_weight={:?} epsilons={:?}", state_id, state.final_weight, state.epsilons);
            let mut transitions = state.transitions.iter().collect::<Vec<_>>();
            transitions.sort_by_key(|(edge_label, _)| **edge_label);
            for (edge_label, targets) in transitions {
                eprintln!("    {} -> {:?}", describe_label(*edge_label), targets);
            }
        }
    }

    fn print_weighted_dwa(label: &str, dwa: &DWA) {
        eprintln!("{} start_state={}", label, dwa.start_state);
        for (state_id, state) in dwa.states.iter().enumerate() {
            eprintln!("  state {} final_weight={:?}", state_id, state.final_weight);
            let mut transitions = state.transitions.iter().collect::<Vec<_>>();
            transitions.sort_by_key(|(edge_label, _)| **edge_label);
            for (edge_label, (target, weight)) in transitions {
                eprintln!("    {} -> {} weight={:?}", describe_label(*edge_label), target, weight);
            }
        }
    }

    fn print_grammar(grammar: &GrammarDef) {
        eprintln!("=== Prepared Grammar ===");
        eprintln!("start_nt={} ignore_terminal={:?}", grammar.start, grammar.ignore_terminal);
        eprintln!("nonterminal_names={:?}", grammar.nonterminal_names);
        for terminal in &grammar.terminals {
            match terminal {
                Terminal::Literal { id, bytes } => {
                    eprintln!("terminal {} literal {:?}", id, String::from_utf8_lossy(bytes));
                }
                Terminal::Pattern { id, pattern, utf8 } => {
                    eprintln!("terminal {} pattern {:?} utf8={}", id, pattern, utf8);
                }
                Terminal::Expr { id, expr } => {
                    eprintln!("terminal {} expr {:?}", id, expr);
                }
            }
        }
        for (index, rule) in grammar.rules.iter().enumerate() {
            let rhs = rule
                .rhs
                .iter()
                .map(|symbol| match symbol {
                    Symbol::Terminal(terminal) => format!("T{}", terminal),
                    Symbol::Nonterminal(nonterminal) => format!("N{}", nonterminal),
                })
                .collect::<Vec<_>>()
                .join(" ");
            eprintln!("rule {}: N{} -> {}", index, rule.lhs, rhs);
        }
    }

    fn print_analyzed_grammar(grammar: &AnalyzedGrammar) {
        eprintln!("=== Analyzed Grammar ===");
        eprintln!(
            "num_terminals={} num_nonterminals={} nullable={:?}",
            grammar.num_terminals,
            grammar.num_nonterminals,
            grammar.nullable
        );
        for (index, rule) in grammar.rules.iter().enumerate() {
            let rhs = rule
                .rhs
                .iter()
                .map(|symbol| match symbol {
                    Symbol::Terminal(terminal) => format!("T{}", terminal),
                    Symbol::Nonterminal(nonterminal) => format!("N{}", nonterminal),
                })
                .collect::<Vec<_>>()
                .join(" ");
            eprintln!("analyzed rule {}: N{} -> {}", index, rule.lhs, rhs);
        }
        for nonterminal in 0..grammar.num_nonterminals {
            eprintln!(
                "FIRST(N{})={:?} FOLLOW(N{})={:?}",
                nonterminal,
                grammar.first[nonterminal as usize],
                nonterminal,
                grammar.follow[nonterminal as usize]
            );
        }
    }

    fn print_glr_table(table: &GLRTable) {
        eprintln!("=== GLR Table ===");
        eprintln!(
            "num_states={} num_terminals={} num_rules={}",
            table.num_states,
            table.num_terminals,
            table.num_rules
        );
        for (index, rule) in table.rules.iter().enumerate() {
            eprintln!("table rule {}: lhs={} rhs={:?}", index, rule.lhs, rule.rhs);
        }
        for state in 0..table.num_states as usize {
            eprintln!("state {}", state);
            let mut actions = table.action[state].iter().collect::<Vec<_>>();
            actions.sort_by_key(|(terminal, _)| **terminal);
            for (terminal, action) in actions {
                eprintln!("  action T{} -> {:?}", terminal, action);
            }
            let mut gotos = table.goto[state].iter().collect::<Vec<_>>();
            gotos.sort_by_key(|(nonterminal, _)| **nonterminal);
            for (nonterminal, (target, replace)) in gotos {
                eprintln!("  goto N{} -> state {} replace={}", nonterminal, target, replace);
            }
        }
    }

    fn dump_case(label: &str, grammar_text: &str) {
        let vocab = minimal_repro_vocab();
        let grammar = parse_lark(grammar_text).unwrap();
        let prepared = prepare_grammar_transforms_only(grammar);
        eprintln!("\n================ {} ================", label);
        print_grammar(&prepared);

        let mut tokenizer = build_tokenizer(&prepared);
        tokenizer.isolate_start_state_and_drain_nullable_terminals();

        let analyzed = AnalyzedGrammar::from_grammar_def(&prepared);
        print_analyzed_grammar(&analyzed);

        let table = GLRTable::build(&analyzed);
        print_glr_table(&table);

        let characterizations = characterize_terminals(&table, &analyzed);
        eprintln!("=== Terminal Characterizations ===");
        for (terminal, characterization) in &characterizations {
            eprintln!("terminal {} characterization {:#?}", terminal, characterization);
            let template_nfa = build_template_nfa(characterization);
            print_nfa(&format!("terminal {} template_nfa", terminal), &template_nfa);
        }

        let templates = Templates::from_characterizations(&characterizations);
        eprintln!("=== Template DFAs / Skeleton NWAs ===");
        for (terminal, dfa) in &templates.by_terminal {
            print_unweighted_dfa(&format!("terminal {} template_dfa", terminal), dfa);
        }
        for (terminal, nwa) in &templates.by_terminal_nwa {
            print_weighted_nwa(&format!("terminal {} template_nwa_skeleton", terminal), nwa);
        }

        let id_map = InternalIdMap::build_identity(&tokenizer, &vocab);
        eprintln!("=== Internal ID Map ===");
        eprintln!("tokenizer_states {:#?}", id_map.tokenizer_states);
        eprintln!("vocab_tokens {:#?}", id_map.vocab_tokens);

        let terminal_dwa = build_terminal_dwa_for_existing_id_map(
            &analyzed,
            &tokenizer,
            &vocab,
            &id_map,
            prepared.ignore_terminal,
        );
        eprintln!("=== Terminal DWA ===");
        print_weighted_dwa("terminal_dwa", &terminal_dwa);

        let parser_dwa = build_parser_dwa_from_terminal_dwa_with_precomputed_templates(
            &table,
            &analyzed,
            &terminal_dwa,
            templates,
        );
        eprintln!("=== Parser DWA ===");
        print_weighted_dwa("parser_dwa", &parser_dwa);

        let constraint = Constraint::from_lark(grammar_text, &vocab).unwrap();
        let mut state = constraint.start();
        state.commit_bytes(b"dd").unwrap();
        let mask = state.mask();
        eprintln!("=== Runtime Probe ===");
        eprintln!("mask_after_dd={:?}", mask);
        eprintln!("closing_token_allowed={}", mask_allows(&mask, 0));
        let mut accepting = constraint.start();
        eprintln!("commit_ddd_result={:?}", accepting.commit_bytes(b"ddd"));
    }

    #[test]
    #[ignore = "debug-only artifact dump for the minimized goto repro"]
    fn dump_minimal_goto_repro_artifacts() {
        dump_case("FAILING_MINIMAL", minimal_repro_grammar());
        dump_case("PASSING_BOUNDARY", minimal_boundary_pass_grammar());
    }
}
