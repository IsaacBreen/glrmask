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
use crate::compiler::glr::labels::{DEFAULT_LABEL, encode_negative_label, encode_positive_label};
use crate::grammar::flat::TerminalID;
use crate::compiler::stages::templates::characterize::{StackMatcher, TerminalCharacterization};
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
    pub(crate) total_premin_dfa_states: usize,
    pub(crate) max_premin_dfa_states: usize,
    pub(crate) total_premin_dfa_transitions: usize,
    pub(crate) max_premin_dfa_transitions: usize,
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

    pub(crate) fn avg_premin_dfa_states(&self) -> f64 {
        average(self.total_premin_dfa_states, self.num_terminals)
    }

    pub(crate) fn avg_premin_dfa_transitions(&self) -> f64 {
        average(self.total_premin_dfa_transitions, self.num_terminals)
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
        self.total_premin_dfa_states += sample.premin_dfa_states;
        self.max_premin_dfa_states = self.max_premin_dfa_states.max(sample.premin_dfa_states);
        self.total_premin_dfa_transitions += sample.premin_dfa_transitions;
        self.max_premin_dfa_transitions =
            self.max_premin_dfa_transitions.max(sample.premin_dfa_transitions);
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
    premin_dfa_states: usize,
    premin_dfa_transitions: usize,
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

    NWA::from_parts(
        states,
        vec![dfa.start_state],
    )
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
    let (premin_dfa_states, premin_dfa_transitions) = dfa_size(&determinized);

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
            premin_dfa_states,
            premin_dfa_transitions,
        },
    )
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
        Self::from_characterizations_profiled(characterizations).0
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

fn add_matcher_transition(nfa: &mut NFA, from: u32, matcher: &StackMatcher, to: u32) {
    match matcher {
        StackMatcher::Any => {
            nfa.add_transition(from, DEFAULT_LABEL, to);
        }
        StackMatcher::State(state) => {
            nfa.add_transition(from, encode_positive_label(*state), to);
        }
        StackMatcher::States(states) => {
            for &state in states {
                nfa.add_transition(from, encode_positive_label(state), to);
            }
        }
    }
}

fn add_pop_pattern_path(nfa: &mut NFA, from: u32, pop: &[StackMatcher], to: u32) {
    if pop.is_empty() {
        nfa.add_epsilon(from, to);
        return;
    }

    let mut current = from;
    for (index, matcher) in pop.iter().enumerate() {
        let next = if index + 1 == pop.len() {
            to
        } else {
            nfa.add_state()
        };
        add_matcher_transition(nfa, current, matcher, next);
        current = next;
    }
}

fn simple_exact_then_any(pop: &[StackMatcher]) -> Option<(u32, usize)> {
    let (first, rest) = pop.split_first()?;
    let StackMatcher::State(first_state) = first else {
        return None;
    };

    if rest.iter().all(|matcher| matches!(matcher, StackMatcher::Any)) {
        Some((*first_state, pop.len()))
    } else {
        None
    }
}

fn add_reduce_pattern_path(
    nfa: &mut NFA,
    pool: &mut PopChainPool,
    from: u32,
    pop: &[StackMatcher],
    target_nt: u32,
    target_state: u32,
) {
    if let Some((first_state, pop_count)) = simple_exact_then_any(pop) {
        add_positive_transition_chain_shared(
            nfa,
            pool,
            from,
            first_state,
            pop_count,
            target_nt,
            target_state,
        );
    } else {
        add_pop_pattern_path(nfa, from, pop, target_state);
    }
}

fn resolve_pos_target(
    nfa: &mut NFA,
    pos_target_cache: &mut BTreeMap<Vec<u32>, u32>,
    suffix_trie: &mut BTreeMap<(u32, u32), u32>,
    accept_root: u32,
    pushes: &[u32],
) -> u32 {
    if let Some(&cached) = pos_target_cache.get(pushes) {
        return cached;
    }
    let mut cur = accept_root;
    for &push_state in pushes.iter().rev() {
        let key = (cur, push_state);
        cur = if let Some(&existing) = suffix_trie.get(&key) {
            existing
        } else {
            let state = nfa.add_state();
            nfa.add_transition(state, encode_negative_label(push_state), cur);
            suffix_trie.insert(key, state);
            state
        };
    }
    pos_target_cache.insert(pushes.to_vec(), cur);
    cur
}

fn add_escape_pattern_path(
    nfa: &mut NFA,
    pos_target_cache: &mut BTreeMap<Vec<u32>, u32>,
    suffix_trie: &mut BTreeMap<(u32, u32), u32>,
    emitted_escapes: &mut BTreeSet<(u32, Vec<StackMatcher>, Vec<u32>)>,
    accept_root: u32,
    from: u32,
    pop: &[StackMatcher],
    pushes: &[u32],
) {
    if !emitted_escapes.insert((from, pop.to_vec(), pushes.to_vec())) {
        return;
    }
    let pos_target = resolve_pos_target(nfa, pos_target_cache, suffix_trie, accept_root, pushes);
    add_pop_pattern_path(nfa, from, pop, pos_target);
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
    let mut emitted_escapes: BTreeSet<(u32, Vec<StackMatcher>, Vec<u32>)> = BTreeSet::new();

    // Initial escapes: start → positive(initial_state) → [extra DEFAULT pops] → [shared suffix tail] → accept_root
    for escape in &characterization.escapes {
        add_escape_pattern_path(
            &mut nfa,
            &mut pos_target_cache,
            &mut suffix_trie,
            &mut emitted_escapes,
            accept_root,
            start,
            &escape.pop,
            &escape.pushes,
        );
    }

    for reduce in &characterization.reduces {
        let Some(&target_nonterminal_state) = nonterminal_nodes.get(&reduce.nonterminal) else {
            continue;
        };

        add_reduce_pattern_path(
            &mut nfa,
            &mut pool,
            start,
            &reduce.pop,
            reduce.nonterminal,
            target_nonterminal_state,
        );
    }

    // NT escapes: source_nt_node → positive(revealed) → [shared suffix tail] → accept_root.
    // The suffix tail is shared across every `(source, revealed, pushes)` that
    // agrees on the `pushes` tail; the positive transition is added directly
    // from the source, with dedup against exact `(source, revealed, pushes)`
    // duplicates.
    for nt_escape in &characterization.nt_escapes {
        let Some(&source_state) = nonterminal_nodes.get(&nt_escape.source_nonterminal) else {
            continue;
        };
        add_escape_pattern_path(
            &mut nfa,
            &mut pos_target_cache,
            &mut suffix_trie,
            &mut emitted_escapes,
            accept_root,
            source_state,
            &nt_escape.pop,
            &nt_escape.pushes,
        );
    }

    for nt_rereduce in &characterization.nt_rereduces {
        let (Some(&source_state), Some(&target_state)) =
            (
                nonterminal_nodes.get(&nt_rereduce.source_nonterminal),
                nonterminal_nodes.get(&nt_rereduce.target_nonterminal),
            )
        else {
            continue;
        };

        add_reduce_pattern_path(
            &mut nfa,
            &mut pool,
            source_state,
            &nt_rereduce.pop,
            nt_rereduce.target_nonterminal,
            target_state,
        );
    }

    nfa
}

