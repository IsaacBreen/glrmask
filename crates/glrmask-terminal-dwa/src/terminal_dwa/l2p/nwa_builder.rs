//! NWA builder: trie-walk construction of the terminal NWA.
//!
//! Contains the core `TerminalNwaBuilder` that walks the vocab prefix trie
//! and constructs NWA transitions for each (byte, tokenizer-state) pair.

use crate::automata::lexer::Lexer;
use crate::automata::lexer::observation::{ConfigActionCache, LexicalStateAction};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use range_set_blaze::RangeSetBlaze;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::Vocab;
use crate::automata::lexer::tokenizer::{Tokenizer, TokenizerMatch};
use crate::automata::weighted::nwa::NWA;
use crate::grammar::flat::TerminalID;
use crate::compiler::possible_matches::PossibleMatchesComputer;
use crate::compiler::stages::equiv_types::InternalIdMap;
use crate::ds::u8set::U8Set;
use crate::ds::vocab_prefix_tree::VocabPrefixTreeNode;
use crate::ds::weight::Weight;

use crate::compiler::stages::id_map_and_terminal_dwa::types::{
    ColorId, TerminalColoring, TerminalDwaBuildProfile,
};

/// NWA state identifier (index into `NWA.states`).
type NwaState = u32;
/// Tokenizer state identifier.
type TokenizerState = u32;
type LeafTokenIds = SmallVec<[u32; 8]>;
type DeferredFutureStateGroups = SmallVec<[(TokenizerState, LeafTokenIds); 4]>;
type DeferredUncoloredFutureLeafBuffer = Vec<DeferredFutureStateGroups>;
/// One active future-terminal relation per logical scanner state. The color
/// projection is derived once, only on colored builders. Its first-seen color
/// order is retained separately from the sorted color-to-terminals lookup.
struct FutureObservation {
    terminals: Arc<[TerminalID]>,
    ignore_present: bool,
    colored: Option<Box<ColoredFuture>>,
}

struct ColoredFuture {
    groups: SmallVec<[(ColorId, SmallVec<[TerminalID; 4]>); 8]>,
    colors: SmallVec<[ColorId; 8]>,
}

impl FutureObservation {
    fn terminals_for_color(&self, color: ColorId) -> &[TerminalID] {
        let Some(colored) = &self.colored else { return &[]; };
        match colored.groups.binary_search_by_key(&color, |(key, _)| *key) {
            Ok(index) => &colored.groups[index].1,
            Err(_) => &[],
        }
    }
}

/// Scalar access is an acceleration of the same epsilon-closed byte action.
/// A borrowed induced view must additionally prove that every observed raw
/// state belongs to its coordinate; failure prevents artifact publication.
trait CursorSource {
    fn has_epsilon(&self, state: u32) -> bool;
    fn step(&self, state: u32, byte: u8) -> Option<u32>;
    fn visit_finalizers(&self, state: u32, visit: impl FnMut(TerminalID)) -> bool;
    fn check_config(&self, _states: &[u32]) -> bool { true }
}

struct RawCursorSource<'a> {
    tokenizer: &'a Tokenizer,
    flat: Option<&'a [u32]>,
}

impl CursorSource for RawCursorSource<'_> {
    #[inline]
    fn has_epsilon(&self, state: u32) -> bool {
        self.tokenizer.state_has_epsilon_transitions(state)
    }

    #[inline]
    fn step(&self, state: u32, byte: u8) -> Option<u32> {
        match self.flat {
            Some(table) => {
                let target = table[state as usize * 256 + byte as usize];
                (target != u32::MAX).then_some(target)
            }
            None => self.tokenizer.step(state, byte),
        }
    }

    #[inline]
    fn visit_finalizers(&self, state: u32, visit: impl FnMut(TerminalID)) -> bool {
        self.tokenizer.matched_terminals_iter(state).for_each(visit);
        true
    }
}

/// Ordered longest-match projection over the same source-pinned action used
/// by possible-match pruning. The active-terminal observation cache belongs
/// to this projection, not to the unrestricted epsilon action. Raw TSIDs
/// remain the NWA coordinate; configuration IDs are local memo keys.
struct NfaTrieScanCache<'tok> {
    tokenizer: &'tok Tokenizer,
    active_terminals: Option<Vec<bool>>,
    observations: Vec<Option<Box<[(TerminalID, u32)]>>>,
    match_seen_stamp: Vec<u32>,
    match_width: Vec<usize>,
    match_end_states: Vec<SmallVec<[u32; 1]>>,
    touched_terminals: Vec<TerminalID>,
    match_stamp: u32,
    strict_reference: bool,
}

impl<'tok> NfaTrieScanCache<'tok> {
    fn new(tokenizer: &'tok Tokenizer, active_terminals: Option<Vec<bool>>) -> Self {
        let terminal_count = tokenizer.num_terminals() as usize;
        Self {
            tokenizer,
            active_terminals,
            observations: Vec::new(),
            match_seen_stamp: vec![0; terminal_count],
            match_width: vec![0; terminal_count],
            match_end_states: (0..terminal_count).map(|_| SmallVec::new()).collect(),
            touched_terminals: Vec::new(),
            match_stamp: 0,
            strict_reference: std::env::var_os(
                "GLRMASK_L2P_NWA_NFA_SCAN_STRICT_REFERENCE",
            )
            .is_some(),
        }
    }

    fn ensure_observations(&mut self, action: &LexicalStateAction<'_>, config: u32) {
        let config_index = config as usize;
        if config_index >= self.observations.len() {
            self.observations.resize_with(config_index + 1, || None);
        }
        if self.observations[config_index].is_some() {
            return;
        }
        let mut observations = Vec::<(TerminalID, u32)>::new();
        for &state in action.states(config) {
            observations.extend(
                self.tokenizer
                    .matched_terminals_iter(state)
                    .filter(|&terminal| {
                        self.active_terminals.as_ref().is_none_or(|active| {
                            active
                                .get(terminal as usize)
                                .copied()
                                .unwrap_or(false)
                        })
                    })
                    .map(|terminal| (terminal, state)),
            );
        }
        self.observations[config_index] = Some(observations.into_boxed_slice());
    }

    #[inline]
    fn record_match(&mut self, terminal: TerminalID, state: u32, width: usize, stamp: u32) {
        let t = terminal as usize;
        if self.match_seen_stamp[t] != stamp {
            self.match_seen_stamp[t] = stamp;
            self.match_width[t] = width;
            self.match_end_states[t].clear();
            self.match_end_states[t].push(state);
            self.touched_terminals.push(terminal);
        } else if width > self.match_width[t] {
            self.match_width[t] = width;
            self.match_end_states[t].clear();
            self.match_end_states[t].push(state);
        } else if width == self.match_width[t] && !self.match_end_states[t].contains(&state) {
            self.match_end_states[t].push(state);
        }
    }

    /// One ordered longest-match observation recurrence. Scalar mode merely
    /// represents a singleton epsilon-free configuration without interning it.
    fn execute_cursor<const SCALAR: bool, S: CursorSource>(
        &mut self, action: &mut LexicalStateAction<'_>, source: &S,
        input: &[u8], start: u32, matches: &mut Vec<TokenizerMatch>,
    ) -> SmallVec<[u32; 1]> {
        self.match_stamp = self.match_stamp.wrapping_add(1);
        if self.match_stamp == 0 {
            self.match_seen_stamp.fill(0);
            self.match_stamp = 1;
        }
        let stamp = self.match_stamp;
        self.touched_terminals.clear();
        matches.clear();

        let mut scalar = (SCALAR && !source.has_epsilon(start)).then_some(start);
        let mut config = if scalar.is_none() {
            action.start(start)
        } else {
            ConfigActionCache::DEAD
        };
        if scalar.is_none() && !source.check_config(action.states(config)) {
            return SmallVec::new();
        }

        let mut alive = true;
        for (index, &byte) in input.iter().enumerate() {
            if let Some(state) = scalar {
                let Some(target) = source.step(state, byte) else {
                    alive = false;
                    break;
                };
                if source.has_epsilon(target) {
                    scalar = None;
                    config = action.start(target);
                } else {
                    scalar = Some(target);
                }
            } else {
                config = action.step(config, byte);
                if config == ConfigActionCache::DEAD {
                    alive = false;
                    break;
                }
                let states = action.states(config);
                if !source.check_config(states) {
                    alive = false;
                    break;
                }
                if SCALAR && states.len() == 1 && !source.has_epsilon(states[0]) {
                    scalar = Some(states[0]);
                }
            }
            let width = index + 1;
            if let Some(state) = scalar {
                if !source.visit_finalizers(state, |terminal| {
                    if self.active_terminals.as_ref().is_none_or(|active|
                        active.get(terminal as usize).copied().unwrap_or(false))
                    {
                        self.record_match(terminal, state, width, stamp);
                    }
                }) {
                    alive = false;
                    break;
                }
            } else {
                if config == ConfigActionCache::DEAD || !source.check_config(action.states(config)) {
                    alive = false;
                    break;
                }
                self.ensure_observations(action, config);
                let observations = self.observations[config as usize]
                    .as_ref().expect("configuration observations initialized").len();
                for i in 0..observations {
                    let (terminal, state) = self.observations[config as usize].as_ref().unwrap()[i];
                    self.record_match(terminal, state, width, stamp);
                }
            }
        }

        for &terminal in &self.touched_terminals {
            let index = terminal as usize;
            let width = self.match_width[index];
            matches.extend(self.match_end_states[index].iter().copied().map(|end_state|
                TokenizerMatch { id: terminal, width, end_state }
            ));
        }
        if !alive {
            SmallVec::new()
        } else if let Some(state) = scalar {
            SmallVec::from_buf([state])
        } else {
            SmallVec::from_slice(action.states(config))
        }
    }

    fn verify_reference(
        &self, input: &[u8], start: u32, matches: &[TokenizerMatch], ends: &[u32],
    ) {
        let mut reference = self.tokenizer.execute_from_state(input, start);
        reference.matches.retain(|matched| {
            self.active_terminals.as_ref().is_none_or(|active|
                active.get(matched.id as usize).copied().unwrap_or(false))
        });
        let mut actual = matches.to_vec();
        actual.sort_unstable_by_key(|m| (m.id, m.width, m.end_state));
        reference.matches.sort_unstable_by_key(|m| (m.id, m.width, m.end_state));
        assert_eq!(ends, reference.end_state.as_slice(), "cached NFA scanner end-state mismatch");
        assert_eq!(actual, reference.matches, "cached NFA scanner longest-match mismatch");
    }

    fn verify_scalar_reference(
        &mut self, action: &mut LexicalStateAction<'_>, input: &[u8], start: u32, matches: &[TokenizerMatch], ends: &[u32],
    ) {
        let mut reference = Vec::new();
        let reference_ends = self.execute_into(action, input, start, &mut reference);
        assert_eq!(ends, reference_ends.as_slice(), "scalar cursor changed ordered end states");
        assert_eq!(matches, reference.as_slice(), "scalar cursor changed ordered longest matches");
        if !self.strict_reference {
            self.verify_reference(input, start, matches, ends);
        }
    }

    fn execute_into(
        &mut self, action: &mut LexicalStateAction<'_>, input: &[u8], start: u32, matches: &mut Vec<TokenizerMatch>,
    ) -> SmallVec<[u32; 1]> {
        let source = RawCursorSource { tokenizer: self.tokenizer, flat: None };
        let ends = self.execute_cursor::<false, _>(action, &source, input, start, matches);
        if self.strict_reference {
            self.verify_reference(input, start, matches, &ends);
        }
        ends
    }
}

fn all_token_weight(internal_tsid: u32, max_token_id: u32) -> Weight {
    Weight::from_token_set_for_tsid(
        internal_tsid,
        RangeSetBlaze::from_iter(std::iter::once(0..=max_token_id)),
    )
}

#[derive(Clone)]
pub struct NodesByTokenizerState {
    pub entries: FxHashMap<TokenizerState, Vec<NwaState>>,
}

impl NodesByTokenizerState {
    fn new() -> Self {
        Self {
            entries: FxHashMap::default(),
        }
    }

    fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn merge(&mut self, state: TokenizerState, nodes: &[NwaState]) {
        self.entries.entry(state).or_default().extend_from_slice(nodes);
    }

    fn first(&self, state: TokenizerState) -> Option<NwaState> {
        self.entries.get(&state).and_then(|nodes| nodes.first().copied())
    }

    fn first_any(&self) -> Option<NwaState> {
        self.entries
            .values()
            .find_map(|nodes| nodes.first().copied())
    }

    fn push_one(&mut self, state: TokenizerState, node: NwaState) {
        self.entries.entry(state).or_default().push(node);
    }

    fn iter(&self) -> impl Iterator<Item = (TokenizerState, &[NwaState])> {
        self.entries
            .iter()
            .map(|(&state, nodes)| (state, nodes.as_slice()))
    }
}

impl IntoIterator for NodesByTokenizerState {
    type Item = (TokenizerState, Vec<NwaState>);
    type IntoIter = <FxHashMap<TokenizerState, Vec<NwaState>> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.entries.into_iter()
    }
}

/// Every trie event has a uniform TSID domain; only seed edges have a
/// nonuniform selector. Choose coefficient storage and graph publication
/// independently of the lexical recurrence. Native bounds remain enforced
/// by its sink and compact-buffer admission checks.
trait EventCoefficient: Clone + PartialEq + std::fmt::Debug {
    fn empty() -> Self;
    fn is_empty(&self) -> bool;
    fn union(&self, other: &Self) -> Self;
    fn from_uniform(rows: std::ops::RangeInclusive<u32>, tokens: RangeSetBlaze<u32>) -> Self;
    fn from_reachable(num_tsids: u32, tokens: &RangeSetBlaze<usize>) -> Self;
    fn from_leaf_tokens(num_tsids: u32, tokens: &[u32]) -> Self;
    fn without_token(&self, num_tsids: u32, token: u32) -> Option<Self>;
}

trait LeafIdsBuffer: Default {
    fn slot(&mut self, source: u32, terminal: TerminalID) -> &mut LeafTokenIds;
    fn len(&self) -> usize;
    fn into_entries(self) -> impl Iterator<Item = (u32, TerminalID, LeafTokenIds)>;
}

impl LeafIdsBuffer for Vec<Vec<LeafTokenIds>> {
    fn slot(&mut self, source: u32, terminal: TerminalID) -> &mut LeafTokenIds {
        let source = source as usize;
        let terminal = terminal as usize;
        if source >= self.len() {
            self.resize_with(source + 1, Vec::new);
        }
        let labels = &mut self[source];
        if terminal >= labels.len() {
            labels.resize_with(terminal + 1, SmallVec::new);
        }
        &mut labels[terminal]
    }

    fn len(&self) -> usize { Vec::len(self) }

    fn into_entries(self) -> impl Iterator<Item = (u32, TerminalID, LeafTokenIds)> {
        self.into_iter().enumerate().flat_map(|(source, labels)| {
            labels.into_iter().enumerate().map(move |(label, tokens)| {
                (source as u32, label as TerminalID, tokens)
            })
        })
    }
}

impl LeafIdsBuffer for FxHashMap<(u32, TerminalID), LeafTokenIds> {
    fn slot(&mut self, source: u32, terminal: TerminalID) -> &mut LeafTokenIds {
        self.entry((source, terminal)).or_default()
    }

    fn len(&self) -> usize { std::collections::HashMap::len(self) }

    fn into_entries(self) -> impl Iterator<Item = (u32, TerminalID, LeafTokenIds)> {
        self.into_iter().map(|((source, label), tokens)| (source, label, tokens))
    }
}

trait EventSink {
    type Coefficient: EventCoefficient;
    type LeafIds: LeafIdsBuffer;
    const NATIVE: bool;
    fn states_len(&self) -> usize;
    fn add_state(&mut self) -> u32;
    fn emit_epsilon(&mut self, from: u32, to: u32, weight: Self::Coefficient);
    fn emit_transition(&mut self, from: u32, label: i32, to: u32, weight: Self::Coefficient);
    fn emit_leaf_transitions(&mut self, from: u32, leaf: u32, entries: Vec<(i32, Self::Coefficient)>) {
        for (label, weight) in entries {
            self.emit_transition(from, label, leaf, weight);
        }
    }
}

impl EventCoefficient for Weight {
    fn empty() -> Self { Weight::empty() }
    fn is_empty(&self) -> bool { Weight::is_empty(self) }
    fn union(&self, other: &Self) -> Self { Weight::union(self, other) }

    fn from_uniform(rows: std::ops::RangeInclusive<u32>, tokens: RangeSetBlaze<u32>) -> Self {
        Weight::from_uniform(rows, tokens)
    }

    fn from_reachable(num_tsids: u32, tokens: &RangeSetBlaze<usize>) -> Self {
        if num_tsids == 0 || tokens.is_empty() {
            return Weight::empty();
        }
        Weight::from_uniform(0..=num_tsids - 1, tokens.ranges()
            .map(|range| (*range.start() as u32)..=(*range.end() as u32)).collect())
    }

    fn from_leaf_tokens(num_tsids: u32, tokens: &[u32]) -> Self {
        if num_tsids == 0 { return Weight::empty(); }
        Weight::from_uniform(0..=num_tsids - 1, RangeSetBlaze::from_iter(
            tokens.iter().copied().map(|id| id..=id)))
    }

    fn without_token(&self, num_tsids: u32, token: u32) -> Option<Self> {
        Some(self.difference(&Self::from_leaf_tokens(num_tsids, &[token])))
    }
}

impl EventSink for NWA {
    type Coefficient = Weight;
    type LeafIds = Vec<Vec<LeafTokenIds>>;
    const NATIVE: bool = false;

    fn states_len(&self) -> usize { self.states().len() }
    fn add_state(&mut self) -> u32 { NWA::add_state(self) }

    fn emit_epsilon(&mut self, from: u32, to: u32, weight: Weight) {
        self.states_mut().get_mut(from as usize)
            .expect("buffered epsilon source state must exist").epsilons.push((to, weight));
    }

    fn emit_transition(&mut self, from: u32, label: i32, to: u32, weight: Weight) {
        self.states_mut().get_mut(from as usize)
            .expect("buffered transition source state must exist")
            .transitions.entry(label).or_default().push((to, weight));
    }
    fn emit_leaf_transitions(&mut self, from: u32, leaf: u32, entries: Vec<(i32, Weight)>) {
        let state = self.states_mut().get_mut(from as usize)
            .expect("buffered leaf transition source state must exist");
        for (label, weight) in entries {
            state.transitions.entry(label).or_default().push((leaf, weight));
        }
    }
}

struct TerminalNwaBuilder<'tok, 'pm, 'nwa, S: EventSink = NWA> {
    tokenizer: &'tok Tokenizer,
    borrowed_map:Option<BorrowedObservation<'tok>>,
    borrowed_cursor:Option<native_borrowed_cursor::BorrowedScalarCache<'tok>>,
    mapping_failed:bool,
    terminal_coloring: TerminalColoring,
    future_observations: FxHashMap<TokenizerState, FutureObservation>,
    possible_matches: &'pm mut PossibleMatchesComputer<'tok>,
    nwa: &'nwa mut S,
    num_tsids: u32,
    leaf_state: u32,
    ignore_terminal: Option<TerminalID>,
    /// States reached from the NWA start using only epsilon transitions.
    ///
    /// Ignore is transparent only after a terminal boundary. An ignore match
    /// from one of these sources is the token's first terminal and must remain
    /// labelled so the parser DWA can account for it with the ignore template.
    initial_source_states: Vec<bool>,
    use_terminal_coloring: bool,
    active_terminals: Option<Vec<bool>>,
    self_loop_bytes: FxHashMap<TokenizerState, U8Set>,
    leaf_token_ids_buffer: S::LeafIds,
    future_leaf_buffer: FxHashMap<(u32, TokenizerState, ColorId), BufferedLeafTransition<S::Coefficient>>,
    deferred_uncolored_future_leaf_buffer: DeferredUncoloredFutureLeafBuffer,
    defer_uncolored_future_leaves: bool,
    reachable_weight_cache: FxHashMap<usize, S::Coefficient>,
    pruned_weight_cache: FxHashMap<(usize, usize, u32, TerminalID), S::Coefficient>,
    leaf_weight_cache: FxHashMap<LeafTokenIds, S::Coefficient>,
    transition_buffer: FxHashMap<(u32, i32, u32), S::Coefficient>,
    compact_buffer: Option<native_compact_buffer::CompactBuffer<S::Coefficient>>,
    epsilon_buffer: FxHashMap<(u32, u32), S::Coefficient>,
    pub profile: TerminalDwaBuildProfile,
    flat_transitions: Vec<Option<Box<[u32; 256]>>>,
    nfa_scan_cache: Option<NfaTrieScanCache<'tok>>,
    shared_flat_transitions: Option<&'tok [u32]>,
    self_loop_subtree_skip_enabled: bool,
    profile_timing: bool,
    has_epsilon_transitions: bool,
    scalar_deterministic_dispatch: bool,
    scalar_cursor:bool,
    lean_frontiers:bool,
    prove_future_absence:bool,
    validate_future_absence:bool,
    future_absence_hits:usize,
    cached_reset_roots:Vec<u32>,
    batch_leaf_flush:bool,
    leaf_flush_failed:bool,
    validate_scalar_cursor:bool,
    dfa_scan_strict_reference: bool,
}

struct BufferedLeafTransition<C = Weight> {
    token_ids: LeafTokenIds,
    weight: Option<C>,
}

impl<C> Default for BufferedLeafTransition<C> {
    fn default() -> Self {
        Self { token_ids: LeafTokenIds::new(), weight: None }
    }
}

fn deferred_future_token_ids_for(
    buffer: &mut DeferredUncoloredFutureLeafBuffer,
    source: u32,
    tokenizer_state: TokenizerState,
) -> &mut LeafTokenIds {
    let source = source as usize;
    if source >= buffer.len() {
        buffer.resize_with(source + 1, SmallVec::new);
    }
    let groups = &mut buffer[source];
    if let Some(index) = groups
        .iter()
        .position(|(state, _)| *state == tokenizer_state)
    {
        return &mut groups[index].1;
    }
    groups.push((tokenizer_state, LeafTokenIds::new()));
    &mut groups.last_mut().expect("just pushed deferred future group").1
}

impl<'tok, 'pm, 'nwa, S: EventSink> TerminalNwaBuilder<'tok, 'pm, 'nwa, S> {
    pub fn new(
        tokenizer: &'tok Tokenizer,
        terminal_coloring: TerminalColoring,
        possible_matches: &'pm mut PossibleMatchesComputer<'tok>,
        nwa: &'nwa mut S,
        num_tsids: u32,
        leaf_state: u32,
        ignore_terminal: Option<TerminalID>,
        initial_source_states: Vec<bool>,
        use_terminal_coloring: bool,
        active_terminals: Option<Vec<bool>>,
        num_tokenizer_states: usize,
        shared_flat_transitions: Option<&'tok [u32]>,
    ) -> Self {
        let has_epsilon_transitions = tokenizer.has_epsilon_transitions();
        let scalar_deterministic_dispatch = tokenizer.has_scalar_deterministic_dispatch();
        let nfa_scan_cache = has_epsilon_transitions
            .then(|| NfaTrieScanCache::new(tokenizer, active_terminals.clone()));
        Self {
            tokenizer,
            borrowed_map:None,
            borrowed_cursor:None,
            mapping_failed:false,
            terminal_coloring,
            future_observations: FxHashMap::default(),
            possible_matches,
            nwa,
            num_tsids,
            leaf_state,
            ignore_terminal,
            initial_source_states,
            use_terminal_coloring,
            active_terminals,
            self_loop_bytes: FxHashMap::default(),
            leaf_token_ids_buffer: S::LeafIds::default(),
            future_leaf_buffer: FxHashMap::default(),
            deferred_uncolored_future_leaf_buffer: Vec::new(),
            defer_uncolored_future_leaves: std::env::var_os(
                "GLRMASK_DISABLE_L2P_DEFER_UNCOLORED_FUTURE_LEAVES",
            )
            .is_none(),
            reachable_weight_cache: FxHashMap::default(),
            pruned_weight_cache: FxHashMap::default(),
            leaf_weight_cache: FxHashMap::default(),
            transition_buffer: FxHashMap::default(),
            compact_buffer:(S::NATIVE && crate::terminal_dwa::env_policy::enabled("GLRMASK_BOUNDARY_NATIVE_COMPACT_EVENT_BUFFER")).then(native_compact_buffer::CompactBuffer::default),
            epsilon_buffer: FxHashMap::default(),
            profile: TerminalDwaBuildProfile::default(),
            flat_transitions: vec![None; num_tokenizer_states],
            nfa_scan_cache,
            shared_flat_transitions: shared_flat_transitions
                .filter(|transitions| transitions.len() == num_tokenizer_states * 256),
            self_loop_subtree_skip_enabled: std::env::var_os(
                "GLRMASK_ENABLE_L2P_SELF_LOOP_SUBTREE_SKIP",
            )
            .is_some(),
            profile_timing: std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some(),
            has_epsilon_transitions,
            scalar_deterministic_dispatch,
            scalar_cursor:S::NATIVE && crate::terminal_dwa::env_policy::enabled("GLRMASK_BOUNDARY_NATIVE_SCALAR_CURSOR"),
            lean_frontiers:S::NATIVE && crate::terminal_dwa::env_policy::enabled("GLRMASK_BOUNDARY_NATIVE_FRONTIER_LIFETIME"),
            prove_future_absence:S::NATIVE && crate::terminal_dwa::env_policy::enabled("GLRMASK_BOUNDARY_NATIVE_FUTURE_ABSENCE")
                && !tokenizer.has_virtual_residual_runtime(),
            validate_future_absence:S::NATIVE && std::env::var_os("GLRMASK_VALIDATE_NATIVE_FUTURE_ABSENCE").is_some(),
            future_absence_hits:0,
            batch_leaf_flush:S::NATIVE && crate::terminal_dwa::env_policy::enabled("GLRMASK_BOUNDARY_NATIVE_FACTORED_LEAF_FLUSH"),
            leaf_flush_failed:false,
            cached_reset_roots:if S::NATIVE && crate::terminal_dwa::env_policy::enabled("GLRMASK_BOUNDARY_NATIVE_FRONTIER_LIFETIME"){tokenizer.deterministic_reset_states().into_vec()}else{Vec::new()},
            validate_scalar_cursor:S::NATIVE && std::env::var_os("GLRMASK_VALIDATE_NATIVE_SCALAR_CURSOR").is_some(),
            dfa_scan_strict_reference: std::env::var_os(
                "GLRMASK_L2P_NWA_DFA_SCAN_STRICT_REFERENCE",
            )
            .is_some(),
        }
    }

    #[inline]
    fn original_state(&self, state: u32) -> u32 {
        if S::NATIVE { self.borrowed_map.map_or(state, |map| map.original(state)) } else { state }
    }

    fn reset_roots(&self) -> SmallVec<[u32; 1]> {
        let roots = self.tokenizer.deterministic_reset_states();
        if let Some(map) = self.borrowed_map {
            roots.into_iter().filter_map(|state| map.logical(state)).collect()
        } else {
            roots
        }
    }

    /// O(1) DFA step using lazily-built flat transition table.
    #[inline]
    fn fast_step(&mut self, state: u32, byte: u8) -> Option<u32> {
        let state_idx = state as usize;
        if self.flat_transitions[state_idx].is_none() {
            let mut row = self.tokenizer.transition_row(self.original_state(state));
            if let Some(map) = self.borrowed_map {
                for target in row.iter_mut() {
                    if *target != u32::MAX {
                        *target = map.logical(*target).unwrap_or(u32::MAX);
                    }
                }
            }
            self.flat_transitions[state_idx] = Some(row);
        }
        let next = self.flat_transitions[state_idx].as_ref().unwrap()[byte as usize];
        (next != u32::MAX).then_some(next)
    }

    fn leaf_token_ids_for(&mut self,source:u32,label:TerminalID)->&mut LeafTokenIds {
        self.leaf_token_ids_buffer.slot(source, label)
    }

    fn buffer_leaf_token_id(&mut self, source: u32, label: TerminalID, internal_token_id: u32) {
        self.leaf_token_ids_for(source, label).push(internal_token_id);
    }

    fn terminal_is_active(&self, terminal: TerminalID) -> bool {
        self.active_terminals.as_ref().is_none_or(|active| {
            active.get(terminal as usize).copied().unwrap_or(false)
        })
    }

    fn future_observation(&mut self, state: TokenizerState) -> &FutureObservation {
        let original = self.original_state(state);
        let tokenizer = self.tokenizer;
        let active = self.active_terminals.as_deref();
        let ignore = self.ignore_terminal;
        let coloring = self.use_terminal_coloring.then_some(&self.terminal_coloring);
        self.future_observations.entry(state).or_insert_with(|| {
            let terminals: Arc<[TerminalID]> = tokenizer.possible_future_terminals_iter(original)
                .filter(|&terminal| active.is_none_or(|mask| {
                    mask.get(terminal as usize).copied().unwrap_or(false)
                }))
                .collect::<Vec<_>>().into();
            let ignore_present = ignore.is_some_and(|terminal| terminals.contains(&terminal));
            let colored = coloring.map(|coloring| {
                let mut groups = BTreeMap::<ColorId, SmallVec<[TerminalID; 4]>>::new();
                let mut colors = SmallVec::new();
                for &terminal in terminals.iter() {
                    if Some(terminal) == ignore {
                        continue;
                    }
                    let color = coloring.color_for(terminal);
                    let entry = groups.entry(color).or_default();
                    if entry.is_empty() {
                        colors.push(color);
                    }
                    entry.push(terminal);
                }
                Box::new(ColoredFuture { groups: groups.into_iter().collect(), colors })
            });
            FutureObservation { terminals, ignore_present, colored }
        })
    }

    fn possible_future_terminals_for_state(&mut self, state: TokenizerState) -> Arc<[TerminalID]> {
        Arc::clone(&self.future_observation(state).terminals)
    }

    fn ignore_terminal_possible_for_state(&mut self, state: TokenizerState) -> bool {
        self.ignore_terminal.is_some() && self.future_observation(state).ignore_present
    }

    fn future_terminal_colors_for_state(&mut self, state: TokenizerState) -> SmallVec<[ColorId; 8]> {
        self.future_observation(state).colored.as_ref()
            .map(|colored| colored.colors.clone())
            .unwrap_or_default()
    }

    fn buffer_future_leaf_token_id(
        &mut self,
        source: u32,
        internal_tsid: TokenizerState,
        color: ColorId,
        internal_token_id: u32,
    ) {
        self.profile.future_terminal_additions += 1;
        self.future_leaf_buffer
            .entry((source, internal_tsid, color))
            .or_default()
            .token_ids
            .push(internal_token_id);
    }

    fn add_future_leaf_token_from_sources(
        &mut self,
        sources: &[u32],
        tokenizer_state: TokenizerState,
        internal_token_id: u32,
    ) {
        if !self.use_terminal_coloring {
            if self.defer_uncolored_future_leaves {
                // Ignore is the one future label whose first-vs-later source
                // semantics cannot be deferred into an ordinary leaf edge.
                // Preserve that label eagerly and defer every other label as
                // one token-set contribution per (source, scanner state).
                if let Some(ignore_terminal) = self.ignore_terminal
                    && self.ignore_terminal_possible_for_state(tokenizer_state)
                {
                    self.profile.future_terminal_additions += sources.len() as u64;
                    self.add_leaf_token_from_sources(
                        sources,
                        ignore_terminal,
                        internal_token_id,
                    );
                }
                for &source in sources {
                    deferred_future_token_ids_for(
                        &mut self.deferred_uncolored_future_leaf_buffer,
                        source,
                        tokenizer_state,
                    )
                    .push(internal_token_id);
                }
                return;
            }

            let future_terminals = self.possible_future_terminals_for_state(tokenizer_state);
            self.profile.future_terminal_additions +=
                (sources.len() * future_terminals.len()) as u64;
            for &terminal_id in future_terminals.iter() {
                self.add_leaf_token_from_sources(sources, terminal_id, internal_token_id);
            }
            return;
        }

        if let Some(ignore_terminal) = self.ignore_terminal
            && self.ignore_terminal_possible_for_state(tokenizer_state) {
                self.profile.future_terminal_additions += sources.len() as u64;
                self.add_leaf_token_from_sources(sources, ignore_terminal, internal_token_id);
            }

        let colors = self.future_terminal_colors_for_state(tokenizer_state);
        for color in colors {
            for &source in sources {
                self.buffer_future_leaf_token_id(source, tokenizer_state, color, internal_token_id);
            }
        }
    }

    fn add_future_weighted_match_from_sources(
        &mut self,
        sources: &[u32],
        tokenizer_state: TokenizerState,
        weight: &S::Coefficient,
    ) {
        if !self.use_terminal_coloring {
            let future_terminals = self.possible_future_terminals_for_state(tokenizer_state);
            self.profile.future_terminal_additions +=
                (sources.len() * future_terminals.len()) as u64;
            for &terminal_id in future_terminals.iter() {
                self.add_match_from_sources(sources, terminal_id, self.leaf_state, weight);
            }
            return;
        }

        if let Some(ignore_terminal) = self.ignore_terminal
            && self.ignore_terminal_possible_for_state(tokenizer_state) {
                self.profile.future_terminal_additions += sources.len() as u64;
                self.add_match_from_sources(sources, ignore_terminal, self.leaf_state, weight);
            }

        let colors = self.future_terminal_colors_for_state(tokenizer_state);
        for color in colors {
            if weight.is_empty() {
                continue;
            }
            for &source in sources {
                self.profile.future_terminal_additions += 1;
                let entry = self.future_leaf_buffer
                    .entry((source, tokenizer_state, color))
                    .or_default();
                if let Some(existing) = &mut entry.weight {
                    *existing = existing.union(weight);
                } else {
                    entry.weight = Some(weight.clone());
                }
            }
        }
    }

    fn cached_reachable_weight(&mut self, token_ids: &RangeSetBlaze<usize>) -> S::Coefficient {
        let cache_key = token_ids as *const RangeSetBlaze<usize> as usize;
        if let Some(weight) = self.reachable_weight_cache.get(&cache_key) {
            return weight.clone();
        }

        let weight = self.token_set_weight_fast(token_ids);
        self.reachable_weight_cache.insert(cache_key, weight.clone());
        weight
    }

    /// Build a weight covering all tsids for the given set of internal token IDs.
    fn token_set_weight_fast(&self, tokens: &RangeSetBlaze<usize>) -> S::Coefficient {
        S::Coefficient::from_reachable(self.num_tsids, tokens)
    }

    fn cached_leaf_weight(&mut self, mut tokens: LeafTokenIds) -> S::Coefficient {
        if S::NATIVE {
            return S::Coefficient::from_leaf_tokens(self.num_tsids, &tokens);
        }
        tokens.sort_unstable();
        tokens.dedup();
        if let Some(weight) = self.leaf_weight_cache.get(&tokens) {
            return weight.clone();
        }
        let weight = S::Coefficient::from_leaf_tokens(self.num_tsids, &tokens);
        self.leaf_weight_cache.insert(tokens, weight.clone());
        weight
    }

    fn continuation_weight_for_match(
        &mut self,
        child_node: VocabPrefixTreeNode<'_>,
        leaf_token_id: u32,
        terminal_id: TerminalID,
        end_state: Option<u32>,
        remaining_segment: &[u8],
    ) -> Option<S::Coefficient> {
        if end_state.is_none() && !(remaining_segment.is_empty() && child_node.has_token()) {
            return Some(self.cached_reachable_weight(child_node.reachable_token_ids()));
        }

        let remove_leaf = remaining_segment.is_empty() && child_node.has_token();
        // Strict future excludes zero-byte acceptance. At an epsilon-free
        // matching state, absence of this terminal proves that no positive
        // continuation can extend it. The current trie-node token is still
        // removed independently, exactly as in the ordinary scanner.
        if self.prove_future_absence && end_state.is_some_and(|q|
            !self.tokenizer.state_has_epsilon_transitions(self.original_state(q))
            && !self.tokenizer.possible_future_terminals(self.original_state(q)).get(terminal_id as usize))
        {
            self.future_absence_hits += 1;
            let mut weight=self.cached_reachable_weight(child_node.reachable_token_ids());
            if remove_leaf {
                weight = weight.without_token(self.num_tsids, leaf_token_id)?;
            }
            if self.validate_future_absence {
                let original = self.original_state(end_state.expect("checked matching state"));
                let ordinary = self.possible_matches.possible_matches_for_suffix_and_node(
                    remaining_segment, child_node, original);
                let mut expected=child_node.reachable_token_ids().clone();
                if remove_leaf{expected.remove(leaf_token_id as usize);}
                if let Some(tokens)=ordinary.get(&terminal_id){subtract_possible_matches(&mut expected,tokens);}
                let expected=self.token_set_weight_fast(&expected);
                assert_eq!(weight, expected,
                    "strict-future absence changed exact continuation coefficient");
            }
            return (!weight.is_empty()).then_some(weight);
        }
        let original_end=end_state.map(|q|self.original_state(q));
        let possible_matches = original_end.map(|end_state| {
            self.possible_matches
                .possible_matches_for_suffix_and_node(
                    remaining_segment,
                    child_node,
                    end_state,
                )
        });
        let matches_for_terminal = possible_matches
            .as_ref()
            .and_then(|matches| matches.get(&terminal_id));

        // Most continuations do not actually exclude anything. In that case the
        // exact continuation domain is simply the child's reachable-token set,
        // whose uniform weight is already cached by node identity. Avoid cloning
        // the token set and rebuilding/reinterning the same weight per match.
        if !remove_leaf && matches_for_terminal.is_none() {
            return Some(self.cached_reachable_weight(child_node.reachable_token_ids()));
        }

        // The context-keyed cache is useful only for the rare genuinely pruned
        // continuations. Checking it after the no-pruning fast path avoids a hash
        // lookup on the overwhelmingly common case.
        let cache_key = (
            child_node.cache_key(),
            remaining_segment.len(),
            end_state.unwrap_or(u32::MAX),
            terminal_id,
        );
        if let Some(weight) = self.pruned_weight_cache.get(&cache_key) {
            return Some(weight.clone());
        }

        let mut remaining = child_node.reachable_token_ids().clone();
        if remove_leaf {
            remaining.remove(leaf_token_id as usize);
        }
        if let Some(matches_for_terminal) = matches_for_terminal {
            subtract_possible_matches(&mut remaining, matches_for_terminal);
        }

        if remaining.is_empty() {
            return None;
        }

        let weight = self.token_set_weight_fast(&remaining);
        self.pruned_weight_cache.insert(cache_key, weight.clone());
        Some(weight)
    }

    fn add_leaf_token_from_sources(
        &mut self,
        sources: &[u32],
        label: TerminalID,
        internal_token_id: u32,
    ) {
        if self.ignore_terminal == Some(label) {
            let weight = if self.num_tsids == 0 {
                S::Coefficient::empty()
            } else {
                S::Coefficient::from_uniform(
                    0..=self.num_tsids - 1,
                    RangeSetBlaze::from_iter(std::iter::once(internal_token_id..=internal_token_id)),
                )
            };
            self.add_match_from_sources(sources, label, self.leaf_state, &weight);
            return;
        }

        for &source in sources {
            self.buffer_leaf_token_id(source, label, internal_token_id);
        }
    }

    fn can_skip_self_loop_subtree(
        &mut self,
        node: VocabPrefixTreeNode<'_>,
        tokenizer_state: TokenizerState,
    ) -> bool {
        if self.has_epsilon_transitions
            && !self.scalar_deterministic_dispatch
        {
            return false;
        }
        self.profile.trie_self_loop_checks += 1;
        if let Some(flat_transitions) = self.shared_flat_transitions {
            let raw=self.original_state(tokenizer_state);
            let base = raw as usize * 256;
            let can_skip = U8Set::from_words(*node.subtree_bytes())
                .iter()
                .all(|byte| flat_transitions[base + byte as usize] == raw);
            if can_skip {
                self.profile.trie_self_loop_skips += 1;
            }
            return can_skip;
        }
        if !self.self_loop_bytes.contains_key(&tokenizer_state) {
            self.profile.trie_self_loop_cache_misses += 1;
        }
        let self_loop_bytes = self
            .self_loop_bytes
            .entry(tokenizer_state)
            .or_insert_with(|| self.tokenizer.self_loop_bytes(self.borrowed_map.map_or(tokenizer_state,|m|m.original(tokenizer_state))));
        let can_skip = U8Set::from_words(*node.subtree_bytes()).is_subset(self_loop_bytes);
        if can_skip {
            self.profile.trie_self_loop_skips += 1;
        }
        can_skip
    }

    fn emit_self_loop_leaf_only_subtree(
        &mut self,
        node: VocabPrefixTreeNode<'_>,
        assoc_by_state: &NodesByTokenizerState,
    ) {
        let mut accessible = node.reachable_token_ids().clone();
        if node.has_token() {
            accessible.remove(node.token_id());
        }
        if accessible.is_empty() {
            return;
        }
        let accessible_weight = self.token_set_weight_fast(&accessible);
        for (internal_tsid, source_nodes) in assoc_by_state.iter() {
            self.add_future_weighted_match_from_sources(
                source_nodes,
                internal_tsid,
                &accessible_weight,
            );
        }
    }

    fn add_match_from_sources(
        &mut self,
        sources: &[u32],
        label: TerminalID,
        target: u32,
        weight: &S::Coefficient,
    ) {
        let lower_ignore_to_epsilon = self.ignore_terminal == Some(label);
        for &source in sources {
            if lower_ignore_to_epsilon
                && !self
                    .initial_source_states
                    .get(source as usize)
                    .copied()
                    .unwrap_or(false)
            {
                self.epsilon_buffer
                    .entry((source, target))
                    .and_modify(|existing| *existing = existing.union(weight))
                    .or_insert_with(|| weight.clone());
            } else {
                if let Some(buffer)=&mut self.compact_buffer{buffer.add((source,label as i32,target),weight);continue}
                self.transition_buffer
                    .entry((source, label as i32, target))
                    .and_modify(|existing| *existing = existing.union(weight))
                    .or_insert_with(|| weight.clone());
            }
        }
    }

    pub fn flush_transition_buffer(&mut self) {
        if self.batch_leaf_flush && !self.use_terminal_coloring && self.future_leaf_buffer.is_empty(){
            self.flush_factored_leaf_outputs();return;
        }
        let flush_start = std::time::Instant::now();
        let mut leaf_transition_buckets: Vec<FxHashMap<i32, BufferedLeafTransition<S::Coefficient>>> =
            (0..self.nwa.states_len()).map(|_| FxHashMap::default()).collect();

        for (from, label, token_ids) in std::mem::take(&mut self.leaf_token_ids_buffer).into_entries() {
            if !token_ids.is_empty(){leaf_transition_buckets[from as usize].entry(label as i32).or_default().token_ids.extend(token_ids);}
        }
        let flush_leaf_ms = flush_start.elapsed().as_secs_f64() * 1000.0;

        let deferred = std::mem::take(&mut self.deferred_uncolored_future_leaf_buffer);
        let ignore = self.ignore_terminal;
        let mut future_additions = 0;
        for (source, groups) in deferred.into_iter().enumerate() {
            for (state, token_ids) in groups {
                if token_ids.is_empty() {
                    continue;
                }
                for &terminal in self.future_observation(state).terminals.iter() {
                    if Some(terminal) == ignore {
                        continue;
                    }
                    future_additions += token_ids.len() as u64;
                    leaf_transition_buckets[source].entry(terminal as i32)
                        .or_default().token_ids.extend_from_slice(&token_ids);
                }
            }
        }
        self.profile.future_terminal_additions += future_additions;

        let flush_future_start = std::time::Instant::now();
        let buffer = std::mem::take(&mut self.future_leaf_buffer);

        for ((source, tokenizer_state, color), buffered) in buffer {
            if buffered.token_ids.is_empty() && buffered.weight.as_ref().is_none_or(|w| w.is_empty()) {
                continue;
            }
            let terminals = self.future_observation(tokenizer_state).terminals_for_color(color);
            for &terminal_id in terminals {
                let entry = leaf_transition_buckets[source as usize]
                    .entry(terminal_id as i32)
                    .or_default();
                if !buffered.token_ids.is_empty() {
                    entry.token_ids.extend_from_slice(&buffered.token_ids);
                }
                if let Some(w) = &buffered.weight {
                    if let Some(existing) = &mut entry.weight {
                        *existing = existing.union(w);
                    } else {
                        entry.weight = Some(w.clone());
                    }
                }
            }
        }
        let flush_future_ms = flush_future_start.elapsed().as_secs_f64() * 1000.0;

        let flush_weight_start = std::time::Instant::now();
        self.flush_non_leaf_events();

        for (from, bucket) in leaf_transition_buckets.into_iter().enumerate() {
            if bucket.is_empty() {
                continue;
            }

            let mut entries: Vec<(i32, BufferedLeafTransition<S::Coefficient>)> = bucket.into_iter().collect();
            entries.sort_unstable_by_key(|(label, _)| *label);

            let mut finalized_entries = Vec::with_capacity(entries.len());
            for (label, mut entry) in entries {
                let mut weight = entry.weight.take().unwrap_or_else(S::Coefficient::empty);
                if !entry.token_ids.is_empty() {
                    let token_weight = self.cached_leaf_weight(entry.token_ids);
                    weight = if weight.is_empty() {
                        token_weight
                    } else {
                        weight.union(&token_weight)
                    };
                }
                if !weight.is_empty() {
                    finalized_entries.push((label, weight));
                }
            }

            self.nwa.emit_leaf_transitions(from as u32, self.leaf_state, finalized_entries);
        }
        let flush_weight_ms = flush_weight_start.elapsed().as_secs_f64() * 1000.0;
        self.profile.flush_leaf_ms = flush_leaf_ms;
        self.profile.flush_future_ms = flush_future_ms;
        self.profile.flush_weight_ms = flush_weight_ms;
    }

    fn flush_non_leaf_events(&mut self) {
        let mut epsilon_entries: Vec<_> = std::mem::take(&mut self.epsilon_buffer).into_iter().collect();
        epsilon_entries.sort_unstable_by_key(|((from, target), _)| (*from, *target));
        for ((from, target), weight) in epsilon_entries {
            self.nwa.emit_epsilon(from, target, weight);
        }

        if let Some(buffer)=&mut self.compact_buffer{buffer.flush(self.nwa);}
        let mut transition_entries: Vec<_> = std::mem::take(&mut self.transition_buffer).into_iter().collect();
        transition_entries.sort_unstable_by_key(|((from, label, target), _)| (*from, *label, *target));
        for ((from, label, target), weight) in transition_entries {
            self.nwa.emit_transition(from, label, target, weight);
        }
    }

    /// Compile sparse leaf observations in two factored arrays. A deferred
    /// (source, lexer-state, token-set) contributes one coefficient to every
    /// possible terminal; do not clone that token vector into a hash map for
    /// every source/terminal pair. Only the (source,label,coefficient-ID) keys
    /// are sorted. Exact union and final edge order are unchanged.
    fn flush_factored_leaf_outputs(&mut self) {
        let started=std::time::Instant::now();
        let mut coefficients=Vec::<S::Coefficient>::new();
        let mut entries=Vec::<(u32,u32,u32)>::new();
        if self.leaf_token_ids_buffer.len()>100_000{self.leaf_flush_failed=true;return}
        for (source, label, tokens) in std::mem::take(&mut self.leaf_token_ids_buffer).into_entries() {
            if tokens.is_empty(){continue}
            let weight=self.cached_leaf_weight(tokens);
            let id=coefficients.len()as u32;coefficients.push(weight);entries.push((source,label,id));
        }
        let deferred=std::mem::take(&mut self.deferred_uncolored_future_leaf_buffer);
        let ignore = self.ignore_terminal;
        let mut future_additions = 0u64;
        for (source,groups)in deferred.into_iter().enumerate(){
            for (state,tokens)in groups {
                if tokens.is_empty(){continue}
                let count=tokens.len()as u64;
                if coefficients.len()>=100_000{self.leaf_flush_failed=true;return}
                let weight=self.cached_leaf_weight(tokens);
                let id=coefficients.len()as u32;coefficients.push(weight);
                let terminals = &self.future_observation(state).terminals;
                for &terminal in terminals.iter(){
                    if ignore == Some(terminal) { continue; }
                    if entries.len() >= 100_000 { self.leaf_flush_failed = true; return; }
                    future_additions += count;
                    entries.push((source as u32,terminal,id));
                }
                if entries.len()>100_000||coefficients.len()>100_000{self.leaf_flush_failed=true;return}
            }
        }
        if entries.len()>100_000||coefficients.len()>100_000{self.leaf_flush_failed=true;return}
        self.profile.future_terminal_additions += future_additions;
        // Keep all non-leaf transitions exactly as before, before appending
        // any leaf branch. Duplicate targets must not be coalesced here.
        self.flush_non_leaf_events();
        entries.sort_unstable();
        let mut i=0;
        while i<entries.len(){
            let(from,label,_)=entries[i];let mut weight=S::Coefficient::empty();
            while i<entries.len()&&entries[i].0==from&&entries[i].1==label {
                weight=weight.union(&coefficients[entries[i].2 as usize]);i+=1;
            }
            if !weight.is_empty(){self.nwa.emit_transition(from, label as i32, self.leaf_state, weight);}
        }
        self.profile.flush_weight_ms=started.elapsed().as_secs_f64()*1000.;
        if std::env::var_os("GLRMASK_PROFILE_NATIVE_BUILDER").is_some(){
            eprintln!("[glrmask/profile][native_factored_leaf] coefficients={} contributions={} exact_source_order=true",coefficients.len(),entries.len());
        }
    }

    pub fn build_from_trie(
        &mut self,
        node: VocabPrefixTreeNode<'_>,
        assoc_by_state: &NodesByTokenizerState,
    ) {
        if !self.self_loop_subtree_skip_enabled {
            for (segment_bytes, child_node) in node.iter_children() {
                let next_level_nodes =
                    self.process_child_segment(segment_bytes, child_node, assoc_by_state);
                if !next_level_nodes.is_empty() {
                    self.build_from_trie(child_node, &next_level_nodes);
                }
            }
            return;
        }

        let self_loop_started_at = self.profile_timing.then(std::time::Instant::now);
        let mut recursive_nodes = NodesByTokenizerState::new();
        let mut self_loop_only_nodes = NodesByTokenizerState::new();
        for (tokenizer_state, source_nodes) in assoc_by_state.iter() {
            self.profile.trie_self_loop_source_nodes += source_nodes.len() as u64;
            if self.can_skip_self_loop_subtree(node, tokenizer_state) {
                self.profile.trie_self_loop_skipped_source_nodes += source_nodes.len() as u64;
                self_loop_only_nodes.merge(tokenizer_state, source_nodes);
            } else {
                recursive_nodes.merge(tokenizer_state, source_nodes);
            }
        }
        if let Some(started_at) = self_loop_started_at {
            self.profile.trie_self_loop_ms += started_at.elapsed().as_secs_f64() * 1000.0;
        }

        if !self_loop_only_nodes.is_empty() {
            self.emit_self_loop_leaf_only_subtree(node, &self_loop_only_nodes);
        }

        if recursive_nodes.is_empty() {
            return;
        }

        for (segment_bytes, child_node) in node.iter_children() {
            let next_level_nodes = self.process_child_segment(
                segment_bytes,
                child_node,
                &recursive_nodes,
            );
            if !next_level_nodes.is_empty() {
                self.build_from_trie(child_node, &next_level_nodes);
            }
        }
    }

    fn process_child_segment(
        &mut self,
        segment_bytes: &[u8],
        child_node: VocabPrefixTreeNode<'_>,
        initial_nodes: &NodesByTokenizerState,
    ) -> NodesByTokenizerState {
        // Token IDs in the trie are already internal (equivalence class) IDs.
        let leaf_token_id = child_node.token_id() as u32;
        let mut next_level_nodes = NodesByTokenizerState::new();
        let mut pending_by_offset = BTreeMap::<usize, NodesByTokenizerState>::new();
        // The initial frontier is immutable for this entire segment. Borrow it
        // instead of allocating/copying one source Vec per raw lexer state.
        let mut initial = Some(if self.lean_frontiers {
            std::borrow::Cow::Borrowed(initial_nodes)
        } else {
            std::borrow::Cow::Owned(initial_nodes.clone())
        });
        // No recursive caller observes the continuation frontier of a trie
        // leaf. Token-end futures and longest-match events are still emitted.
        let keep_next_frontier = !self.lean_frontiers || !child_node.children().is_empty();

        // Reusable DFA longest-match buffer. The pre-epsilon lexer path used
        // this scalar scanner and is dramatically faster than routing every
        // compressed trie segment through the general state-set executor.
        let mut match_map_buf = FxHashMap::<TerminalID, (usize, u32)>::default();
        let mut matches_buf: Vec<TokenizerMatch> = Vec::new();

        loop {
            let batch = initial.take().map(|nodes| (0, nodes)).or_else(||
                pending_by_offset.pop_first().map(|(offset,nodes)| (offset,std::borrow::Cow::Owned(nodes))));
            let Some((offset,nodes_at_offset)) = batch else {break};
            if offset == segment_bytes.len() {
                if keep_next_frontier {
                    for (tokenizer_state, nwa_states) in nodes_at_offset.iter() {
                        next_level_nodes.merge(tokenizer_state, nwa_states);
                    }
                }
                continue;
            }

            // FxHashMap::iter and its owning iterator visit the same buckets;
            // borrowing changes storage lifetime, not discovery/event order.
            for (tokenizer_state, source_nodes) in nodes_at_offset.iter() {
                let remaining = &segment_bytes[offset..];
                let execute_started_at = self.profile_timing.then(std::time::Instant::now);
                let end_states = if self.has_epsilon_transitions {
                    let raw=self.original_state(tokenizer_state);
                    let cache=self.nfa_scan_cache.as_mut().expect("epsilon tokenizer must initialize NFA trie scan cache");
                    let action = self.possible_matches.state_action(self.tokenizer);
                    let mut ends=if let Some(prepared)=self.borrowed_cursor.as_ref(){
                        cache.execute_native_borrowed_into(action,remaining,raw,&mut matches_buf,prepared,self.validate_scalar_cursor)
                    }else if self.scalar_cursor {
                        cache.execute_native_scalar_into(action,remaining,raw,&mut matches_buf,
                            self.shared_flat_transitions,self.validate_scalar_cursor)
                    }else{cache.execute_into(action,remaining,raw,&mut matches_buf)};
                    if let Some(map)=self.borrowed_map {
                        let mut failed=false;
                        matches_buf.retain_mut(|matched|{if let Some(q)=map.logical(matched.end_state){matched.end_state=q;true}else{failed=true;false}});
                        ends.retain_mut(|q|{if let Some(logical)=map.logical(*q){*q=logical;true}else{failed=true;false}});
                        self.mapping_failed|=failed;
                    }
                    ends
                } else {
                    match_map_buf.clear();
                    let mut scan_state = tokenizer_state;
                    let mut scan_alive = true;
                    for (index, &byte) in segment_bytes[offset..].iter().enumerate() {
                        if let Some(next) = self.fast_step(scan_state, byte) {
                            scan_state = next;
                            for terminal in self.tokenizer.matched_terminals_iter(self.original_state(scan_state)) {
                                if !self.terminal_is_active(terminal) {
                                    continue;
                                }
                                match_map_buf.insert(terminal, (index + 1, scan_state));
                            }
                        } else {
                            scan_alive = false;
                            break;
                        }
                    }

                    matches_buf.clear();
                    matches_buf.extend(match_map_buf.iter().map(
                        |(&id, &(width, end_state))| TokenizerMatch {
                            id,
                            width,
                            end_state,
                        },
                    ));

                    let end_states = if scan_alive {
                        SmallVec::from_buf([scan_state])
                    } else {
                        SmallVec::new()
                    };

                    if self.dfa_scan_strict_reference {
                        let mut reference = self
                            .tokenizer
                            .execute_from_state(remaining, self.original_state(tokenizer_state));
                        reference
                            .matches
                            .retain(|matched| self.terminal_is_active(matched.id));
                        if let Some(map)=self.borrowed_map {
                            reference.matches.retain_mut(|m|{if let Some(q)=map.logical(m.end_state){m.end_state=q;true}else{self.mapping_failed=true;false}});
                            reference.end_state.retain_mut(|q|{if let Some(v)=map.logical(*q){*q=v;true}else{self.mapping_failed=true;false}});
                        }
                        let mut actual_matches = matches_buf.clone();
                        actual_matches.sort_unstable_by_key(|matched| {
                            (matched.id, matched.width, matched.end_state)
                        });
                        reference.matches.sort_unstable_by_key(|matched| {
                            (matched.id, matched.width, matched.end_state)
                        });
                        assert_eq!(
                            end_states, reference.end_state,
                            "historical DFA trie scanner end-state mismatch"
                        );
                        assert_eq!(
                            actual_matches, reference.matches,
                            "historical DFA trie scanner longest-match mismatch"
                        );
                    }

                    end_states
                };
                if let Some(started_at) = execute_started_at {
                    self.profile.trie_execute_ms +=
                        started_at.elapsed().as_secs_f64() * 1000.0;
                    self.profile.trie_execute_calls += 1;
                    self.profile.trie_execute_input_bytes += remaining.len() as u64;
                    self.profile.trie_matches += matches_buf.len() as u64;
                }

                let end_state_started_at = self.profile_timing.then(std::time::Instant::now);
                for &end_state in &end_states {
                    if child_node.has_token() {
                        self.add_future_leaf_token_from_sources(
                            source_nodes,
                            end_state,
                            leaf_token_id,
                        );
                    }

                    if keep_next_frontier {
                        next_level_nodes.merge(end_state, source_nodes);
                    }
                }
                if let Some(started_at) = end_state_started_at {
                    self.profile.trie_end_state_ms +=
                        started_at.elapsed().as_secs_f64() * 1000.0;
                    self.profile.trie_end_states += end_states.len() as u64;
                }

                let match_process_started_at = self.profile_timing.then(std::time::Instant::now);
                for matched in &matches_buf {
                    let next_offset = offset + matched.width;

                    if next_offset == segment_bytes.len()
                        && child_node.has_token()
                        && ((self.has_epsilon_transitions
                            && !self.scalar_deterministic_dispatch)
                            || !end_states.iter().copied().any(|s| {
                                self.future_observation(s).terminals.contains(&matched.id)
                            }))
                    {
                        self.profile.match_transition_additions += source_nodes.len() as u64;
                        self.add_leaf_token_from_sources(
                            source_nodes,
                            matched.id,
                            leaf_token_id,
                        );
                    }

                    let continuation_weight_started_at =
                        self.profile_timing.then(std::time::Instant::now);
                    let continuation_weight = self.continuation_weight_for_match(
                        child_node,
                        leaf_token_id,
                        matched.id,
                        Some(matched.end_state),
                        &segment_bytes[next_offset..],
                    );
                    if let Some(started_at) = continuation_weight_started_at {
                        self.profile.trie_continuation_weight_ms +=
                            started_at.elapsed().as_secs_f64() * 1000.0;
                    }
                    let Some(continuation_weight) = continuation_weight else {
                        continue;
                    };
                    if continuation_weight.is_empty() {
                        continue;
                    }

                    let continuation_nodes = pending_by_offset
                        .entry(next_offset)
                        .or_insert_with(NodesByTokenizerState::new);
                    let uncached_reset_states;
                    let reset_states = if self.lean_frontiers {
                        self.cached_reset_roots.as_slice()
                    } else {
                        uncached_reset_states = self.reset_roots();
                        uncached_reset_states.as_slice()
                    };
                    let destination = ensure_continuation_state(
                        continuation_nodes,
                        reset_states,
                        self.nwa,
                    );

                    self.profile.match_transition_additions += source_nodes.len() as u64;
                    self.add_match_from_sources(
                        source_nodes,
                        matched.id,
                        destination,
                        &continuation_weight,
                    );
                }
                if let Some(started_at) = match_process_started_at {
                    self.profile.trie_match_process_ms +=
                        started_at.elapsed().as_secs_f64() * 1000.0;
                }
            }
        }

        next_level_nodes
    }
}
fn subtract_possible_matches(
    continuation_tokens: &mut RangeSetBlaze<usize>,
    possible_matches: &RangeSetBlaze<u32>,
) {
    for token_id in possible_matches.iter() {
        continuation_tokens.remove(token_id as usize);
    }
}

fn ensure_continuation_state<S: EventSink>(
    pending: &mut NodesByTokenizerState,
    tokenizer_states: &[TokenizerState],
    nwa: &mut S,
) -> NwaState {
    if let Some(existing) = pending.first_any() {
        return existing;
    }

    let state = nwa.add_state();
    for &tokenizer_state in tokenizer_states {
        pending.push_one(tokenizer_state, state);
    }
    state
}

pub fn internal_vocab_entries(vocab: &Vocab, id_map: &InternalIdMap) -> Vec<(u32, Vec<u8>)> {
    id_map
        .vocab_tokens
        .iter_representative_ids()
        .enumerate()
        .filter_map(|(internal_token_id, representative)| {
            vocab
                .entries_map()
                .get(&representative)
                .map(|bytes| (internal_token_id as u32, bytes.to_vec()))
        })
        .collect()
}

pub fn seed_root_nodes(
    tokenizer: &Tokenizer,
    nwa: &mut NWA,
    start_state: u32,
    id_map: &InternalIdMap,
) -> NodesByTokenizerState {
    seed_root_nodes_inner(tokenizer, nwa, start_state, id_map, None)
}

/// Seed walk roots only for internal-TSID classes that intersect a raw-state
/// allow-list (`keep_raw_state[raw_state]`, length = `tokenizer.num_states()`).
/// This is exact for walks that must cover exactly the allowed start states:
/// every allowed raw state belongs to a kept class whose representative root
/// covers the class behavior, and dropped classes cover no allowed state.
/// `None` (via [`seed_root_nodes`]) preserves the historical seed-all behavior.
pub fn seed_root_nodes_filtered(
    tokenizer: &Tokenizer,
    nwa: &mut NWA,
    start_state: u32,
    id_map: &InternalIdMap,
    keep_raw_state: &[bool],
) -> NodesByTokenizerState {
    assert_eq!(
        keep_raw_state.len(),
        tokenizer.num_states() as usize,
        "seed filter must cover the raw tokenizer-state domain",
    );
    seed_root_nodes_inner(tokenizer, nwa, start_state, id_map, Some(keep_raw_state))
}

fn seed_root_nodes_inner(
    tokenizer: &Tokenizer,
    nwa: &mut NWA,
    start_state: u32,
    id_map: &InternalIdMap,
    keep_raw_state: Option<&[bool]>,
) -> NodesByTokenizerState {
    let mut roots_by_tokenizer_state = NodesByTokenizerState::new();

    for (internal_tsid, representative_state) in id_map
        .tokenizer_states
        .iter_representative_ids()
        .enumerate()
    {
        let seed_state = if let Some(keep) = keep_raw_state {
            let Some(&member) = id_map.tokenizer_states.internal_to_originals[internal_tsid]
                .iter()
                .find(|&&raw| keep.get(raw as usize).copied().unwrap_or(false))
            else {
                continue;
            };
            member
        } else {
            representative_state
        };
        let root = nwa.add_state();
        let start_weight = all_token_weight(internal_tsid as u32, id_map.max_internal_token_id());
        nwa.add_epsilon(start_state, root, start_weight);
        if seed_state == tokenizer.initial_state_id()
            && let Some(dispatch_roots) = tokenizer.deterministic_dispatch_roots()
        {
            for &dispatch_root in dispatch_roots {
                roots_by_tokenizer_state.merge(dispatch_root, &[root]);
            }
            continue;
        }
        debug_assert!(keep_raw_state.is_none_or(|keep| {
            keep.get(seed_state as usize).copied().unwrap_or(false)
        }));
        roots_by_tokenizer_state.merge(seed_state, &[root]);
    }

    roots_by_tokenizer_state
}


#[derive(Default)]
struct RootFragmentBuffers {
    leaf_token_ids_buffer: Vec<Vec<LeafTokenIds>>,
    future_leaf_buffer: FxHashMap<(u32, TokenizerState, ColorId), BufferedLeafTransition>,
    deferred_uncolored_future_leaf_buffer: DeferredUncoloredFutureLeafBuffer,
    transition_buffer: FxHashMap<(u32, i32, u32), Weight>,
    epsilon_buffer: FxHashMap<(u32, u32), Weight>,
}

fn merge_terminal_dwa_build_profile(
    dst: &mut TerminalDwaBuildProfile,
    src: TerminalDwaBuildProfile,
) {
    dst.future_terminal_additions += src.future_terminal_additions;
    dst.match_transition_additions += src.match_transition_additions;
    dst.trie_walk_ms += src.trie_walk_ms;
    dst.flush_ms += src.flush_ms;
    dst.flush_leaf_ms += src.flush_leaf_ms;
    dst.flush_future_ms += src.flush_future_ms;
    dst.flush_weight_ms += src.flush_weight_ms;
    dst.trie_self_loop_ms += src.trie_self_loop_ms;
    dst.trie_execute_ms += src.trie_execute_ms;
    dst.trie_match_filter_ms += src.trie_match_filter_ms;
    dst.trie_end_state_ms += src.trie_end_state_ms;
    dst.trie_match_process_ms += src.trie_match_process_ms;
    dst.trie_continuation_weight_ms += src.trie_continuation_weight_ms;
    dst.trie_execute_calls += src.trie_execute_calls;
    dst.trie_execute_input_bytes += src.trie_execute_input_bytes;
    dst.trie_matches += src.trie_matches;
    dst.trie_end_states += src.trie_end_states;
    dst.trie_self_loop_checks += src.trie_self_loop_checks;
    dst.trie_self_loop_skips += src.trie_self_loop_skips;
    dst.trie_self_loop_source_nodes += src.trie_self_loop_source_nodes;
    dst.trie_self_loop_skipped_source_nodes += src.trie_self_loop_skipped_source_nodes;
    dst.trie_self_loop_cache_misses += src.trie_self_loop_cache_misses;
}

fn remap_fragment_target(target: u32, base_states: usize, appended_base: u32) -> u32 {
    if (target as usize) < base_states {
        target
    } else {
        appended_base + target - base_states as u32
    }
}

/// Merge one independently-built root-subtree fragment into `dst`, identifying
/// the shared seeded states `[0, base_states)` and keeping every fragment-local
/// state disjoint. NWA nondeterminism makes appended transitions from a shared
/// root exact; later canonicalization/determinization can re-coalesce them.
fn merge_root_fragment(dst: &mut NWA, base: &NWA, fragment: NWA) -> u32 {
    let base_states = base.states().len();
    let (mut fragment_states, _) = fragment.into_parts();
    debug_assert!(fragment_states.len() >= base_states);
    let appended_base = dst.states().len() as u32;

    {
        let dst_states = dst.states_mut();
        for state_id in 0..base_states {
            let base_state = &base.states()[state_id];
            let fragment_state = &fragment_states[state_id];
            let dst_state = &mut dst_states[state_id];

            for (target, weight) in fragment_state
                .epsilons
                .iter()
                .skip(base_state.epsilons.len())
            {
                dst_state.epsilons.push((
                    remap_fragment_target(*target, base_states, appended_base),
                    weight.clone(),
                ));
            }
            for (&label, targets) in &fragment_state.transitions {
                let base_count = base_state
                    .transitions
                    .get(&label)
                    .map_or(0, Vec::len);
                let dst_targets = dst_state.transitions.entry(label).or_default();
                dst_targets.extend(targets.iter().skip(base_count).map(|(target, weight)| {
                    (
                        remap_fragment_target(*target, base_states, appended_base),
                        weight.clone(),
                    )
                }));
            }
        }

        for mut state in fragment_states.drain(base_states..) {
            for (target, _) in &mut state.epsilons {
                *target = remap_fragment_target(*target, base_states, appended_base);
            }
            for targets in state.transitions.values_mut() {
                for (target, _) in targets {
                    *target = remap_fragment_target(*target, base_states, appended_base);
                }
            }
            dst_states.push(state);
        }
    }
    appended_base
}

fn remap_fragment_state_id(state: u32, base_states: usize, appended_base: u32) -> u32 {
    remap_fragment_target(state, base_states, appended_base)
}

fn merge_root_fragment_buffers(
    merged: &mut RootFragmentBuffers,
    buffers: RootFragmentBuffers,
    base_states: usize,
    appended_base: u32,
) {
    let remap = |state: u32| remap_fragment_state_id(state, base_states, appended_base);
    for (local_source, labels) in buffers.leaf_token_ids_buffer.into_iter().enumerate() {
        if labels.is_empty() { continue; }
        let source = remap(local_source as u32) as usize;
        if source >= merged.leaf_token_ids_buffer.len() {
            merged.leaf_token_ids_buffer.resize_with(source + 1, Vec::new);
        }
        let dst_labels = &mut merged.leaf_token_ids_buffer[source];
        if dst_labels.len() < labels.len() {
            dst_labels.resize_with(labels.len(), SmallVec::new);
        }
        for (label, token_ids) in labels.into_iter().enumerate() {
            dst_labels[label].extend(token_ids);
        }
    }
    for ((source, tokenizer_state, color), buffered) in buffers.future_leaf_buffer {
        let dst = merged.future_leaf_buffer.entry((remap(source), tokenizer_state, color)).or_default();
        dst.token_ids.extend(buffered.token_ids);
        if let Some(weight) = buffered.weight {
            if let Some(existing) = &mut dst.weight { *existing = existing.union(&weight); }
            else { dst.weight = Some(weight); }
        }
    }
    for (local_source, groups) in buffers
        .deferred_uncolored_future_leaf_buffer
        .into_iter()
        .enumerate()
    {
        if groups.is_empty() {
            continue;
        }
        let source = remap(local_source as u32);
        for (tokenizer_state, token_ids) in groups {
            deferred_future_token_ids_for(
                &mut merged.deferred_uncolored_future_leaf_buffer,
                source,
                tokenizer_state,
            )
            .extend(token_ids);
        }
    }
    for ((source, label, target), weight) in buffers.transition_buffer {
        merged.transition_buffer.entry((remap(source), label, remap(target)))
            .and_modify(|existing| *existing = existing.union(&weight)).or_insert(weight);
    }
    for ((source, target), weight) in buffers.epsilon_buffer {
        merged.epsilon_buffer.entry((remap(source), remap(target)))
            .and_modify(|existing| *existing = existing.union(&weight)).or_insert(weight);
    }
}

pub fn build_nwa_via_trie_walk<'a>(
    tokenizer: &'a Tokenizer,
    terminal_coloring: &TerminalColoring,
    use_terminal_coloring: bool,
    ignore_terminal: Option<TerminalID>,
    nwa: &mut NWA,
    leaf_state: u32,
    num_tsids: u32,
    vocab_tree_root: VocabPrefixTreeNode<'_>,
    roots: &NodesByTokenizerState,
    possible_matches: &mut PossibleMatchesComputer<'a>,
    shared_flat_transitions: Option<&'a [u32]>,
    active_terminals: &[bool],
) -> TerminalDwaBuildProfile {
    let num_tokenizer_states = tokenizer.num_states() as usize;
    let mut initial_source_states = vec![false; nwa.states().len()];
    for (_, source_nodes) in roots.iter() {
        for &source in source_nodes {
            initial_source_states[source as usize] = true;
        }
    }
    let parallel_root_trie_forced =
        std::env::var_os("GLRMASK_L2P_PARALLEL_ROOT_TRIE").is_some();
    let parallel_root_trie_disabled =
        std::env::var_os("GLRMASK_DISABLE_L2P_PARALLEL_ROOT_TRIE").is_some();
    let parallel_root_trie = !parallel_root_trie_disabled
        && std::env::var_os("GLRMASK_ENABLE_L2P_SELF_LOOP_SUBTREE_SKIP").is_none()
        && vocab_tree_root.children().len() >= 2
        && rayon::current_num_threads() > 1
        && (parallel_root_trie_forced || vocab_tree_root.reachable_token_ids().len() >= 512);

    if parallel_root_trie {
        let base_nwa = nwa.clone();
        let base_state_count = base_nwa.states().len();
        debug_assert_eq!(base_state_count, initial_source_states.len());
        let mut children = vocab_tree_root.iter_children().collect::<Vec<_>>();
        children.sort_unstable_by_key(|(_, child)| {
            std::cmp::Reverse(child.reachable_token_ids().len())
        });
        let task_count = std::env::var("GLRMASK_L2P_ROOT_TRIE_TASKS")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .filter(|&value| value > 0)
            .unwrap_or_else(rayon::current_num_threads)
            .min(children.len())
            .max(1);
        let mut bins = (0..task_count).map(|_| Vec::new()).collect::<Vec<_>>();
        let mut bin_loads = vec![0usize; task_count];
        for child in children {
            let index = bin_loads
                .iter()
                .enumerate()
                .min_by_key(|(_, load)| **load)
                .map(|(index, _)| index)
                .unwrap_or(0);
            bin_loads[index] += child.1.reachable_token_ids().len() as usize;
            bins[index].push(child);
        }

        let parallel_started_at = std::time::Instant::now();
        let fragments = bins
            .into_par_iter()
            .map(|children| {
                let mut local_nwa = base_nwa.clone();
                let mut local_possible_matches = PossibleMatchesComputer::new(tokenizer);
                let mut local_builder = TerminalNwaBuilder::new(
                    tokenizer,
                    terminal_coloring.clone(),
                    &mut local_possible_matches,
                    &mut local_nwa,
                    num_tsids,
                    leaf_state,
                    ignore_terminal,
                    initial_source_states.clone(),
                    use_terminal_coloring,
                    Some(active_terminals.to_vec()),
                    num_tokenizer_states,
                    shared_flat_transitions,
                );
                let trie_started_at = std::time::Instant::now();
                for (segment_bytes, child_node) in children {
                    let next_level_nodes =
                        local_builder.process_child_segment(segment_bytes, child_node, roots);
                    if !next_level_nodes.is_empty() {
                        local_builder.build_from_trie(child_node, &next_level_nodes);
                    }
                }
                local_builder.profile.trie_walk_ms =
                    trie_started_at.elapsed().as_secs_f64() * 1000.0;
                let buffers = RootFragmentBuffers {
                    leaf_token_ids_buffer: std::mem::take(&mut local_builder.leaf_token_ids_buffer),
                    future_leaf_buffer: std::mem::take(&mut local_builder.future_leaf_buffer),
                    deferred_uncolored_future_leaf_buffer: std::mem::take(
                        &mut local_builder.deferred_uncolored_future_leaf_buffer,
                    ),
                    transition_buffer: std::mem::take(&mut local_builder.transition_buffer),
                    epsilon_buffer: std::mem::take(&mut local_builder.epsilon_buffer),
                };
                let profile = local_builder.profile;
                drop(local_builder);
                (local_nwa, profile, buffers)
            })
            .collect::<Vec<_>>();
        let parallel_wall_ms = parallel_started_at.elapsed().as_secs_f64() * 1000.0;

        let mut profile = TerminalDwaBuildProfile::default();
        let mut merged_buffers = RootFragmentBuffers::default();
        for (fragment, fragment_profile, buffers) in fragments {
            merge_terminal_dwa_build_profile(&mut profile, fragment_profile);
            let appended_base = merge_root_fragment(nwa, &base_nwa, fragment);
            merge_root_fragment_buffers(
                &mut merged_buffers,
                buffers,
                base_state_count,
                appended_base,
            );
        }

        let mut flush_initial_sources = vec![false; nwa.states().len()];
        for (_, source_nodes) in roots.iter() {
            for &source in source_nodes {
                flush_initial_sources[source as usize] = true;
            }
        }
        let mut flush_builder = TerminalNwaBuilder::new(
            tokenizer,
            terminal_coloring.clone(),
            possible_matches,
            nwa,
            num_tsids,
            leaf_state,
            ignore_terminal,
            flush_initial_sources,
            use_terminal_coloring,
            Some(active_terminals.to_vec()),
            num_tokenizer_states,
            shared_flat_transitions,
        );
        flush_builder.leaf_token_ids_buffer = merged_buffers.leaf_token_ids_buffer;
        flush_builder.future_leaf_buffer = merged_buffers.future_leaf_buffer;
        flush_builder.deferred_uncolored_future_leaf_buffer =
            merged_buffers.deferred_uncolored_future_leaf_buffer;
        flush_builder.transition_buffer = merged_buffers.transition_buffer;
        flush_builder.epsilon_buffer = merged_buffers.epsilon_buffer;
        let flush_started_at = std::time::Instant::now();
        flush_builder.flush_transition_buffer();
        profile.future_terminal_additions += flush_builder.profile.future_terminal_additions;
        profile.flush_ms = flush_started_at.elapsed().as_secs_f64() * 1000.0;
        profile.flush_leaf_ms = flush_builder.profile.flush_leaf_ms;
        profile.flush_future_ms = flush_builder.profile.flush_future_ms;
        profile.flush_weight_ms = flush_builder.profile.flush_weight_ms;
        drop(flush_builder);

        // Per-fragment trie times are CPU-like sums; the phase field is intended
        // to report the observed critical-path wall time.
        profile.trie_walk_ms = parallel_wall_ms;
        profile
    } else {
        let mut builder = TerminalNwaBuilder::new(
            tokenizer,
            terminal_coloring.clone(),
            possible_matches,
            nwa,
            num_tsids,
            leaf_state,
            ignore_terminal,
            initial_source_states,
            use_terminal_coloring,
            Some(active_terminals.to_vec()),
            num_tokenizer_states,
            shared_flat_transitions,
        );
        let trie_start = std::time::Instant::now();
        builder.build_from_trie(vocab_tree_root, roots);
        let trie_ms = trie_start.elapsed().as_secs_f64() * 1000.0;
        builder.profile.trie_walk_ms = trie_ms;

        let flush_start = std::time::Instant::now();
        builder.flush_transition_buffer();
        let flush_ms = flush_start.elapsed().as_secs_f64() * 1000.0;
        builder.profile.flush_ms = flush_ms;

        let profile = builder.profile;
        drop(builder);
        profile
    }
}


#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::automata::lexer::ast::Expr;
    use crate::automata::lexer::compile::build_regex_monolithic as build_regex;
    use crate::automata::weighted::determinize::determinize;
    use crate::automata::weighted::minimize::minimize_owned;
    use crate::compiler::stages::equiv_types::ManyToOneIdMap;
    use crate::compiler::stages::id_map_and_terminal_dwa::l2p::terminal_interchangeability::{
        binary_transport_modes, discover_one_round, visible_output_raw_labels,
    };
    use crate::compiler::stages::id_map_and_terminal_dwa::types::{
        LocalIdMapTerminalDwa, TerminalDwaPhaseProfile,
    };
    use crate::compiler::stages::id_map_and_terminal_dwa::l2p::terminal_dwa_equivalence::compare as compare_terminal_dwa;
    use crate::ds::vocab_prefix_tree::VocabPrefixTree;

    fn one_terminal_tokenizer() -> Tokenizer {
        let expressions = vec![Expr::U8Seq(b" ".to_vec())];
        build_regex(&expressions).into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        )
    }

    #[test]
    fn ignore_matches_are_labelled_initially_and_epsilon_later() {
        let tokenizer = one_terminal_tokenizer();
        let mut possible_matches = PossibleMatchesComputer::new(&tokenizer);
        let mut nwa = NWA::new(1, 0);
        let initial_source = nwa.add_state();
        let later_source = nwa.add_state();
        let target = nwa.add_state();

        let mut builder = TerminalNwaBuilder::new(
            &tokenizer,
            TerminalColoring::identity(1),
            &mut possible_matches,
            &mut nwa,
            1,
            target,
            Some(0),
            vec![true, false, false],
            false,
            None,
            tokenizer.num_states() as usize,
            None,
        );
        builder.add_match_from_sources(
            &[initial_source, later_source],
            0,
            target,
            &Weight::all(),
        );
        builder.flush_transition_buffer();
        drop(builder);

        assert!(
            nwa.states()[initial_source as usize]
                .transitions
                .get(&0)
                .is_some_and(|edges| edges.iter().any(|(dst, _)| *dst == target)),
            "an initial ignore match must remain a labelled terminal edge"
        );
        assert!(
            nwa.states()[later_source as usize]
                .epsilons
                .iter()
                .any(|(dst, _)| *dst == target),
            "a non-initial ignore match must remain transparent"
        );
        assert!(
            !nwa.states()[initial_source as usize]
                .epsilons
                .iter()
                .any(|(dst, _)| *dst == target),
            "the initial ignore match must not also be lowered to epsilon"
        );
        assert!(
            !nwa.states()[later_source as usize]
                .transitions
                .contains_key(&0),
            "the non-initial ignore match must not become a terminal edge"
        );
    }

    #[test]
    fn member_reconstruction_emits_only_its_target_member() {
        let expressions = vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"b".to_vec()),
            Expr::U8Seq(b"c".to_vec()),
        ];
        let tokenizer = build_regex(&expressions).into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );
        let tree = VocabPrefixTree::build(&[(0, b"a".to_vec()), (1, b"c".to_vec())]);
        let state_ids = (0..tokenizer.num_states()).collect::<Vec<_>>();
        let id_map = InternalIdMap {
            tokenizer_states:
                ManyToOneIdMap::from_singleton_original_to_internal_with_representatives(
                    state_ids.clone(), state_ids,
                ),
            vocab_tokens: ManyToOneIdMap::from_singleton_original_to_internal_with_representatives(
                vec![0, 1], vec![0, 1],
            ),
        };
        let mut possible_matches = PossibleMatchesComputer::new(&tokenizer);
        let mut nwa = NWA::new(id_map.num_tsids(), id_map.max_internal_token_id());
        let leaf = nwa.add_state();
        nwa.set_final_weight(leaf, Weight::all());
        let start = nwa.add_state();
        nwa.start_states_mut().push(start);
        let identity = TransportScannerStateMap::from_explicit(
            (0..tokenizer.num_states()).collect::<Vec<_>>(),
        );
        let modes = [
            TerminalNwaTransportMode::ordinary(tokenizer.num_states() as usize),
            TerminalNwaTransportMode::member(identity, 0, 1),
        ];

        build_transport_nwa_via_trie_walk(
            &tokenizer,
            None,
            &mut nwa,
            start,
            leaf,
            &id_map,
            tree.root(),
            &mut possible_matches,
            &[true, false, true],
            &modes,
        );

        let root = nwa.states()[start as usize].epsilons[0].0;
        let transitions = &nwa.states()[root as usize].transitions;
        assert!(transitions.contains_key(&0), "ordinary representative scan must emit rep");
        assert!(transitions.contains_key(&1), "member reconstruction must emit its target member");
        assert!(transitions.contains_key(&2), "ordinary representative scan must retain unrelated rep");
        let member_weight = transitions
            .get(&1)
            .and_then(|edges| {
                edges
                    .iter()
                    .find_map(|(destination, weight)| (*destination == leaf).then_some(weight))
            })
            .expect("member reconstruction must create a leaf edge");
        let member_tokens = member_weight.tokens_for_tsid(0);
        assert!(member_tokens.contains(0), "member must accept the representative token");
        assert!(
            !member_tokens.contains(1),
            "member reconstruction must not emit its member for unrelated raw outputs",
        );
    }

    fn singleton_id_map(num_tokenizer_states: u32, num_tokens: usize) -> InternalIdMap {
        let tokenizer_states = (0..num_tokenizer_states).collect::<Vec<_>>();
        let vocab_tokens = (0..num_tokens as u32).collect::<Vec<_>>();
        InternalIdMap {
            tokenizer_states:
                ManyToOneIdMap::from_singleton_original_to_internal_with_representatives(
                    tokenizer_states.clone(),
                    tokenizer_states,
                ),
            vocab_tokens: ManyToOneIdMap::from_singleton_original_to_internal_with_representatives(
                vocab_tokens.clone(),
                vocab_tokens,
            ),
        }
    }

    fn finish_test_artifact(nwa: &NWA, id_map: InternalIdMap) -> LocalIdMapTerminalDwa {
        LocalIdMapTerminalDwa {
            id_map,
            dwa: minimize_owned(determinize(nwa).expect("test NWA must determinize")),
            profile: TerminalDwaPhaseProfile::default(),
        }
    }

    fn build_baseline_test_artifact(
        tokenizer: &Tokenizer,
        tree: &VocabPrefixTree,
        id_map: &InternalIdMap,
    ) -> LocalIdMapTerminalDwa {
        build_baseline_test_artifact_with_self_loop_skip(tokenizer, tree, id_map, false)
    }

    fn build_baseline_test_artifact_with_self_loop_skip(
        tokenizer: &Tokenizer,
        tree: &VocabPrefixTree,
        id_map: &InternalIdMap,
        self_loop_subtree_skip_enabled: bool,
    ) -> LocalIdMapTerminalDwa {
        build_baseline_test_artifact_with_options(
            tokenizer,
            tree,
            id_map,
            self_loop_subtree_skip_enabled,
            false,
        )
    }

    fn build_baseline_test_artifact_with_options(
        tokenizer: &Tokenizer,
        tree: &VocabPrefixTree,
        id_map: &InternalIdMap,
        self_loop_subtree_skip_enabled: bool,
        defer_uncolored_future_leaves: bool,
    ) -> LocalIdMapTerminalDwa {
        let mut possible_matches = PossibleMatchesComputer::new(tokenizer);
        let mut nwa = NWA::new(id_map.num_tsids(), id_map.max_internal_token_id());
        let leaf = nwa.add_state();
        nwa.set_final_weight(leaf, Weight::all());
        let start = nwa.add_state();
        nwa.start_states_mut().push(start);
        let roots = seed_root_nodes(tokenizer, &mut nwa, start, id_map);
        let mut initial_source_states = vec![false; nwa.states().len()];
        for (_, source_nodes) in roots.iter() {
            for &source in source_nodes {
                initial_source_states[source as usize] = true;
            }
        }
        let mut builder = TerminalNwaBuilder::new(
            tokenizer,
            TerminalColoring::identity(tokenizer.num_terminals() as usize),
            &mut possible_matches,
            &mut nwa,
            id_map.num_tsids(),
            leaf,
            None,
            initial_source_states,
            false,
            None,
            tokenizer.num_states() as usize,
            None,
        );
        builder.self_loop_subtree_skip_enabled = self_loop_subtree_skip_enabled;
        builder.defer_uncolored_future_leaves = defer_uncolored_future_leaves;
        builder.build_from_trie(tree.root(), &roots);
        builder.flush_transition_buffer();
        drop(builder);
        finish_test_artifact(&nwa, id_map.clone())
    }

    #[test]
    fn deferred_uncolored_future_leaves_match_eager_expansion() {
        let expressions = vec![
            Expr::Repeat {
                expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"ab"))),
                min: 1,
                max: None,
            },
            Expr::U8Seq(b"ac".to_vec()),
            Expr::U8Seq(b"ba".to_vec()),
        ];
        let tokenizer = build_regex(&expressions).into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );
        let tree = VocabPrefixTree::build(&[
            (0, b"a".to_vec()),
            (1, b"aa".to_vec()),
            (2, b"ab".to_vec()),
            (3, b"aba".to_vec()),
            (4, b"abb".to_vec()),
            (5, b"ac".to_vec()),
            (6, b"b".to_vec()),
            (7, b"ba".to_vec()),
        ]);
        let id_map = singleton_id_map(tokenizer.num_states(), 8);

        let eager = build_baseline_test_artifact_with_options(
            &tokenizer,
            &tree,
            &id_map,
            false,
            false,
        );
        let deferred = build_baseline_test_artifact_with_options(
            &tokenizer,
            &tree,
            &id_map,
            false,
            true,
        );

        compare_terminal_dwa(&eager, &deferred)
            .expect("deferred future-leaf expansion must preserve the exact terminal language");
    }

    #[test]
    fn deferred_uncolored_future_leaves_survive_root_fragment_buffer_merge() {
        let mut merged = RootFragmentBuffers::default();
        let mut fragment = RootFragmentBuffers::default();
        deferred_future_token_ids_for(
            &mut fragment.deferred_uncolored_future_leaf_buffer,
            0,
            7,
        )
        .extend([11, 12]);
        deferred_future_token_ids_for(
            &mut fragment.deferred_uncolored_future_leaf_buffer,
            3,
            9,
        )
        .push(21);

        // Fragment states below base_states are shared. State 3 is the second
        // fragment-local state, so it is appended at 10 + (3 - 2) = 11.
        merge_root_fragment_buffers(&mut merged, fragment, 2, 10);

        assert_eq!(
            merged
                .deferred_uncolored_future_leaf_buffer.first()
                .and_then(|groups| groups.iter().find(|(state, _)| *state == 7))
                .map(|(_, ids)| ids.as_slice()),
            Some(&[11, 12][..]),
        );
        assert_eq!(
            merged
                .deferred_uncolored_future_leaf_buffer
                .get(11)
                .and_then(|groups| groups.iter().find(|(state, _)| *state == 9))
                .map(|(_, ids)| ids.as_slice()),
            Some(&[21][..]),
        );
    }

    #[test]
    fn parallel_root_trie_preserves_deferred_uncolored_future_leaves() {
        let expressions = vec![
            Expr::Repeat {
                expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"ab"))),
                min: 1,
                max: None,
            },
            Expr::U8Seq(b"aba".to_vec()),
            Expr::U8Seq(b"bab".to_vec()),
        ];
        let tokenizer = build_regex(&expressions).into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );

        // 512 distinct binary strings make the production auto-parallel root
        // gate fire without mutating process-global environment variables.
        let vocab = (0_usize..512)
            .map(|token_id| {
                let bytes = (0..9)
                    .rev()
                    .map(|bit| if token_id & (1 << bit) == 0 { b'a' } else { b'b' })
                    .collect::<Vec<_>>();
                (token_id, bytes)
            })
            .collect::<Vec<_>>();
        let tree = VocabPrefixTree::build(&vocab);
        let id_map = singleton_id_map(tokenizer.num_states(), vocab.len());

        let eager = build_baseline_test_artifact_with_options(
            &tokenizer,
            &tree,
            &id_map,
            false,
            false,
        );

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .expect("test Rayon pool must build");
        let parallel_deferred = pool.install(|| {
            let mut possible_matches = PossibleMatchesComputer::new(&tokenizer);
            let mut nwa = NWA::new(id_map.num_tsids(), id_map.max_internal_token_id());
            let leaf = nwa.add_state();
            nwa.set_final_weight(leaf, Weight::all());
            let start = nwa.add_state();
            nwa.start_states_mut().push(start);
            let roots = seed_root_nodes(&tokenizer, &mut nwa, start, &id_map);
            let active_terminals = vec![true; tokenizer.num_terminals() as usize];

            build_nwa_via_trie_walk(
                &tokenizer,
                &TerminalColoring::identity(tokenizer.num_terminals() as usize),
                false,
                None,
                &mut nwa,
                leaf,
                id_map.num_tsids(),
                tree.root(),
                &roots,
                &mut possible_matches,
                None,
                &active_terminals,
            );
            finish_test_artifact(&nwa, id_map.clone())
        });

        compare_terminal_dwa(&eager, &parallel_deferred).expect(
            "parallel root fragments must preserve deferred uncolored future-leaf semantics",
        );
    }

    #[test]
    fn self_loop_subtree_skip_matches_exact_recursive_walk() {
        let expressions = vec![
            Expr::Repeat {
                expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"ab"))),
                min: 1,
                max: None,
            },
            Expr::U8Seq(b"ac".to_vec()),
        ];
        let tokenizer = build_regex(&expressions).into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );
        let tree = VocabPrefixTree::build(&[
            (0, b"aa".to_vec()),
            (1, b"ab".to_vec()),
            (2, b"aba".to_vec()),
            (3, b"abb".to_vec()),
            (4, b"ac".to_vec()),
        ]);
        let id_map = singleton_id_map(tokenizer.num_states(), 5);

        let exact = build_baseline_test_artifact_with_self_loop_skip(
            &tokenizer,
            &tree,
            &id_map,
            false,
        );
        let skipped = build_baseline_test_artifact_with_self_loop_skip(
            &tokenizer,
            &tree,
            &id_map,
            true,
        );

        compare_terminal_dwa(&exact, &skipped)
            .expect("self-loop subtree skip must preserve the exact terminal artifact language");
    }

    fn build_transport_test_artifact(
        tokenizer: &Tokenizer,
        tree: &VocabPrefixTree,
        id_map: &InternalIdMap,
        visible_output_raw_labels: &[bool],
        modes: &[TerminalNwaTransportMode],
    ) -> LocalIdMapTerminalDwa {
        let mut possible_matches = PossibleMatchesComputer::new(tokenizer);
        let mut nwa = NWA::new(id_map.num_tsids(), id_map.max_internal_token_id());
        let leaf = nwa.add_state();
        nwa.set_final_weight(leaf, Weight::all());
        let start = nwa.add_state();
        nwa.start_states_mut().push(start);
        build_transport_nwa_via_trie_walk(
            tokenizer,
            None,
            &mut nwa,
            start,
            leaf,
            id_map,
            tree.root(),
            &mut possible_matches,
            visible_output_raw_labels,
            modes,
        );
        finish_test_artifact(&nwa, id_map.clone())
    }

    fn artifact_accepts(
        artifact: &LocalIdMapTerminalDwa,
        original_state: u32,
        original_token: u32,
        labels: &[i32],
    ) -> bool {
        let internal_state = artifact.id_map.tokenizer_states.original_to_internal
            [original_state as usize];
        let internal_token = artifact.id_map.vocab_tokens.original_to_internal
            [original_token as usize];
        let mut state = artifact.dwa.start_state();
        for &label in labels {
            let Some((target, weight)) = artifact.dwa.states()[state as usize]
                .transitions
                .get(&label)
            else {
                return false;
            };
            if !weight.is_full()
                && !weight
                    .tokens_for_tsid(internal_state)
                    .contains(internal_token)
            {
                return false;
            }
            state = *target;
        }
        artifact.dwa.states()[state as usize]
            .final_weight
            .as_ref()
            .is_some_and(|weight| {
                weight.is_full()
                    || weight
                        .tokens_for_tsid(internal_state)
                        .contains(internal_token)
            })
    }

    #[test]
    fn residual_prefix_can_complete_terminal_at_vocab_leaf() {
        let expressions = vec![
            Expr::Choice(vec![
                Expr::U8Seq(b"a".to_vec()),
                Expr::U8Seq(b"ab".to_vec()),
            ]),
            Expr::Choice(vec![
                Expr::U8Seq(b"ab".to_vec()),
                Expr::U8Seq(b"b".to_vec()),
                Expr::U8Seq(b"ba".to_vec()),
            ]),
            Expr::Choice(vec![
                Expr::U8Seq(b"abc".to_vec()),
                Expr::U8Seq(b"bc".to_vec()),
                Expr::U8Seq(b"c".to_vec()),
            ]),
        ];
        let tokenizer = build_regex(&expressions).into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );
        let tokens = [
            b"a".as_slice(),
            b"b".as_slice(),
            b"c".as_slice(),
            b"aa".as_slice(),
            b"bb".as_slice(),
            b"cc".as_slice(),
            b"ab".as_slice(),
            b"ac".as_slice(),
            b"ba".as_slice(),
            b"bc".as_slice(),
            b"abc".as_slice(),
            b"aab".as_slice(),
            b"abb".as_slice(),
            b"acc".as_slice(),
        ];
        let tree = VocabPrefixTree::build(
            &tokens
                .iter()
                .enumerate()
                .map(|(id, bytes)| (id, bytes.to_vec()))
                .collect::<Vec<_>>(),
        );
        let id_map = singleton_id_map(tokenizer.num_states(), tokens.len());
        let artifact = build_baseline_test_artifact(&tokenizer, &tree, &id_map);
        let residual_a = tokenizer
            .step(tokenizer.initial_state_id(), b'a')
            .expect("a must enter a live lexer state");

        assert!(
            artifact_accepts(&artifact, residual_a, 1, &[1]),
            "token b from residual prefix a must complete B=ab"
        );
        assert!(
            artifact_accepts(&artifact, residual_a, 9, &[2]),
            "token bc from residual prefix a must complete C=abc"
        );
    }

    #[test]
    fn residual_prefix_can_complete_and_restart_within_one_vocab_token() {
        let expressions = vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::Choice(vec![
                Expr::U8Seq(b"b".to_vec()),
                Expr::U8Seq(b"ba".to_vec()),
            ]),
        ];
        let tokenizer = build_regex(&expressions).into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );
        let tokens = [
            b"a".as_slice(),
            b"b".as_slice(),
            b"c".as_slice(),
            b"ab".as_slice(),
            b"ba".as_slice(),
            b"bc".as_slice(),
            b"abc".as_slice(),
        ];
        let tree = VocabPrefixTree::build(
            &tokens
                .iter()
                .enumerate()
                .map(|(id, bytes)| (id, bytes.to_vec()))
                .collect::<Vec<_>>(),
        );
        let id_map = singleton_id_map(tokenizer.num_states(), tokens.len());
        let artifact = build_baseline_test_artifact(&tokenizer, &tree, &id_map);
        let residual_b = tokenizer
            .step(tokenizer.initial_state_id(), b'b')
            .expect("b must enter a live lexer state");

        assert!(
            artifact_accepts(&artifact, residual_b, 3, &[1, 1]),
            "token ab from residual prefix b must emit B=ba then B=b"
        );
    }

    #[test]
    fn same_terminal_rematch_inside_compressed_segment_prunes_shorter_boundary() {
        let expressions = vec![
            Expr::Repeat {
                expr: Box::new(Expr::U8Seq(b"_".to_vec())),
                min: 1,
                max: None,
            },
            Expr::U8Seq(b".".to_vec()),
        ];
        let tokenizer = build_regex(&expressions).into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );
        // A singleton vocabulary keeps the whole token on one compressed trie
        // edge. The first possible ID boundary therefore occurs inside that
        // edge rather than at its destination node.
        let tree = VocabPrefixTree::build(&[(0, b"__.__".to_vec())]);
        let id_map = singleton_id_map(tokenizer.num_states(), 1);
        let artifact = build_baseline_test_artifact(&tokenizer, &tree, &id_map);
        let residual_identifier = tokenizer
            .step(tokenizer.initial_state_id(), b'_')
            .expect("underscore must enter the identifier state");

        assert!(
            artifact_accepts(&artifact, residual_identifier, 0, &[0, 1, 0]),
            "the longest identifier match must allow ID DOT ID",
        );
        assert!(
            !artifact_accepts(&artifact, residual_identifier, 0, &[0, 0, 1, 0]),
            "a shorter ID boundary must be pruned when the same identifier can rematch later inside the compressed edge",
        );
    }

    #[test]
    fn transport_shares_one_nwa_root_per_actual_tsid() {
        // A large class makes the old reference builder allocate one root per
        // (actual TSID, mode). The quotient-union builder must retain the
        // exact language while allocating roots only for actual TSIDs.
        let expressions = (0..16)
            .map(|_| Expr::U8Seq(b"a".to_vec()))
            .collect::<Vec<_>>();
        let tokenizer = build_regex(&expressions).into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );
        let partition = discover_one_round(
            &tokenizer,
            &vec![true; tokenizer.num_terminals() as usize],
            &[true; 256],
            None,
        );
        let modes = binary_transport_modes(
            &tokenizer,
            &vec![true; tokenizer.num_terminals() as usize],
            &partition,
            &[true; 256],
            None,
        );
        assert!(modes.len() >= 16, "the test requires many transport modes");

        let tree = VocabPrefixTree::build(&[
            (0, b"a".to_vec()),
            (1, b"aa".to_vec()),
            (2, b"aaa".to_vec()),
            (3, b"aaaa".to_vec()),
        ]);
        let id_map = singleton_id_map(tokenizer.num_states(), 4);
        let mut possible_matches = PossibleMatchesComputer::new(&tokenizer);
        let mut nwa = NWA::new(id_map.num_tsids(), id_map.max_internal_token_id());
        let leaf = nwa.add_state();
        nwa.set_final_weight(leaf, Weight::all());
        let start = nwa.add_state();
        nwa.start_states_mut().push(start);

        build_transport_nwa_via_trie_walk(
            &tokenizer,
            None,
            &mut nwa,
            start,
            leaf,
            &id_map,
            tree.root(),
            &mut possible_matches,
            &visible_output_raw_labels(
                &partition,
                &vec![true; tokenizer.num_terminals() as usize],
            ),
            &modes,
        );

        assert_eq!(
            nwa.states()[start as usize].epsilons.len(),
            id_map.num_tsids() as usize,
            "transport modes must share each actual tokenizer-state root"
        );
    }

    #[test]
    fn compact_transport_output_filter_preserves_the_baseline_language() {
        // These duplicate literals are rooted-interchangeable. Raw
        // nonrepresentatives remain present in the lexer metadata, but their
        // output edges are redundant: the transport mode using its
        // representative edge supplies the corresponding member label.
        let expressions = vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"a".to_vec()),
        ];
        let tokenizer = build_regex(&expressions).into_tokenizer(
            expressions.len() as u32,
            Some(Arc::from(expressions.into_boxed_slice())),
        );
        let partition = discover_one_round(&tokenizer, &[true, true], &[true; 256], None);
        let modes = binary_transport_modes(
            &tokenizer,
            &[true, true],
            &partition,
            &[true; 256],
            None,
        );
        let tree = VocabPrefixTree::build(&[
            (0, b"a".to_vec()),
            (1, b"aaa".to_vec()),
            (2, b"aaaaa".to_vec()),
            (3, b"aaaaaaa".to_vec()),
        ]);
        let id_map = singleton_id_map(tokenizer.num_states(), 4);
        let baseline = build_baseline_test_artifact(&tokenizer, &tree, &id_map);
        let full = build_transport_test_artifact(
            &tokenizer,
            &tree,
            &id_map,
            &[true, true],
            &modes,
        );
        super::super::terminal_dwa_equivalence::compare(&baseline, &full)
            .expect("full raw-output transport must reproduce the ordinary NWA");

        let compact = build_transport_test_artifact(
            &tokenizer,
            &tree,
            &id_map,
            &visible_output_raw_labels(
                &partition,
                &vec![true; tokenizer.num_terminals() as usize],
            ),
            &modes,
        );
        super::super::terminal_dwa_equivalence::compare(&baseline, &compact)
            .expect("raw-edge filtering must preserve the completed terminal language");
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TransportScannerStateMap {
    Quotient {
        class_for_original: Arc<[u32]>,
        representative_for_class: Arc<[TokenizerState]>,
        source_class_for_target_deviations: Box<[(u32, u32)]>,
    },
    /// Temporary composition of round-local TI scanner coordinates. Keeping
    /// this lazy lets iterative TI replay share every accepted round map until
    /// the transport NWA actually queries a scanner state.
    Composed {
        outer: Arc<TransportScannerStateMap>,
        inner: Arc<TransportScannerStateMap>,
    },
}

impl TransportScannerStateMap {
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            Self::Quotient { class_for_original, .. } => class_for_original.len(),
            Self::Composed { outer, inner } => {
                debug_assert_eq!(outer.len(), inner.len());
                outer.len()
            }
        }
    }

    #[inline]
    pub fn scanner_state(&self, original_state: TokenizerState) -> TokenizerState {
        match self {
            Self::Quotient {
                class_for_original,
                representative_for_class,
                source_class_for_target_deviations,
                ..
            } => {
                let target_class = class_for_original[original_state as usize];
                let source_class = source_class_for_target_deviations
                    .binary_search_by_key(&target_class, |&(target, _)| target)
                    .map(|index| source_class_for_target_deviations[index].1)
                    .unwrap_or(target_class);
                representative_for_class[source_class as usize]
            }
            Self::Composed { outer, inner } => {
                outer.scanner_state(inner.scanner_state(original_state))
            }
        }
    }

    /// Return the scanner coordinate for applying `inner` first and `outer`
    /// second. TI replay composes every accepted round-local witness this way.
    pub fn compose(
        outer: Arc<TransportScannerStateMap>,
        inner: Arc<TransportScannerStateMap>,
    ) -> Arc<TransportScannerStateMap> {
        assert_eq!(
            outer.len(),
            inner.len(),
            "TI scanner-coordinate composition requires equal state domains",
        );
        Arc::new(Self::Composed { outer, inner })
    }

    /// Literal test oracle input. A discrete source partition represents an
    /// arbitrary raw function without introducing a different execution path.
    #[cfg(test)]
    pub fn from_explicit(states: Vec<TokenizerState>) -> Self {
        Self::Quotient {
            class_for_original: (0..states.len() as u32).collect::<Vec<_>>().into(),
            representative_for_class: states.into(),
            source_class_for_target_deviations: Box::from([]),
        }
    }

    #[cfg(test)]
    pub fn materialized(&self) -> Arc<[TokenizerState]> {
        (0..self.len()).map(|state| self.scanner_state(state as TokenizerState))
            .collect::<Vec<_>>().into()
    }

    /// Append this temporary transport program in application order.  A
    /// composed transform applies its inner map first, then its outer map.
    pub fn append_atomic_transforms<'a>(
        &'a self,
        output: &mut Vec<&'a TransportScannerStateMap>,
    ) {
        match self {
            Self::Composed { outer, inner } => {
                inner.append_atomic_transforms(output);
                outer.append_atomic_transforms(output);
            }
            Self::Quotient { .. } => {
                output.push(self)
            }
        }
    }

}

/// One temporary binary-witness scanner mode for strict terminal
/// interchangeability. The ordinary mode emits only final representatives. A
/// member mode scans from one certified binary transport state map and emits
/// exactly one label: its member whenever the transported scan emits that
/// member's representative. It never relabels or emits unrelated terminals.
#[derive(Clone, Debug)]
pub enum TerminalNwaTransportMode {
    Ordinary { state_count: usize },
    Member {
        scanner: TransportScannerStateMap,
        representative: TerminalID,
        member: TerminalID,
    },
}

impl TerminalNwaTransportMode {
    pub fn ordinary(state_count: usize) -> Self {
        Self::Ordinary { state_count }
    }

    pub fn member(scanner: TransportScannerStateMap, representative: TerminalID, member: TerminalID) -> Self {
        Self::Member { scanner, representative, member }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Ordinary { state_count } => *state_count,
            Self::Member { scanner, .. } => scanner.len(),
        }
    }

    pub fn scanner_state(&self, raw: TokenizerState) -> TokenizerState {
        match self {
            Self::Ordinary { state_count } => {
                assert!((raw as usize) < *state_count);
                raw
            }
            Self::Member { scanner, .. } => scanner.scanner_state(raw),
        }
    }

    pub fn scanner_map(&self) -> Option<&TransportScannerStateMap> {
        match self {
            Self::Ordinary { .. } => None,
            Self::Member { scanner, .. } => Some(scanner),
        }
    }

    pub fn member_reconstruction(&self) -> Option<(TerminalID, TerminalID)> {
        match self {
            Self::Ordinary { .. } => None,
            Self::Member { representative, member, .. } => Some((*representative, *member)),
        }
    }

    #[inline]
    fn emitted_terminal(
        &self,
        raw_terminal: TerminalID,
        visible_representatives: &[bool],
    ) -> Option<TerminalID> {
        match self.member_reconstruction() {
            None => visible_representatives
                .get(raw_terminal as usize)
                .copied()
                .unwrap_or(false)
                .then_some(raw_terminal),
            Some((representative, member)) => {
                (raw_terminal == representative).then_some(member)
            }
        }
    }
}

/// A scanner context is a raw lexer state together with the set of terminal
/// permutations that reached it. The parser/NWA state is deliberately *not*
/// duplicated for every permutation: all modes have the same parser-side
/// weight, so the exact union is represented by one NWA source with one context
/// per distinct raw scanner state.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct TransportContext {
    scanner_state: TokenizerState,
    mode_set: usize,
}

/// Exact mode subset representation. The common scanner context contains all
/// transport modes except a sparse deviation list, avoiding a full mode vector
/// at every root.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
enum TransportModeSet {
    Explicit(Arc<[usize]>),
    AllExcept(Arc<[usize]>),
}

#[derive(Clone, Default)]
struct NodesByTransportContext {
    entries: FxHashMap<TransportContext, Vec<NwaState>>,
}

impl NodesByTransportContext {
    fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn merge(&mut self, context: TransportContext, nodes: &[NwaState]) {
        self.entries.entry(context).or_default().extend_from_slice(nodes);
    }

    fn first_any(&self) -> Option<NwaState> {
        self.entries.values().find_map(|nodes| nodes.first().copied())
    }

    fn push_one(&mut self, context: TransportContext, node: NwaState) {
        self.entries.entry(context).or_default().push(node);
    }
}

#[derive(Default)]
struct TransportNwaTiming {
    enabled: bool,
    root_context_ms: f64,
    reset_context_ms: f64,
    mode_set_intern_ms: f64,
    mode_set_intern_hits: u64,
    mode_set_intern_new: u64,
    mapped_label_cache_hit_ms: f64,
    mapped_label_compute_ms: f64,
    mapped_label_cache_hits: u64,
    mapped_label_cache_misses: u64,
    trie_scan_ms: f64,
    trie_scan_calls: u64,
    trie_scan_bytes: u64,
    pending_context_ms: f64,
    pending_context_ops: u64,
    context_merge_ms: f64,
    future_edge_ms: f64,
    endpoint_leaf_edge_ms: f64,
    match_edge_ms: f64,
    edge_flush_ms: f64,
    edge_buffer_insertions: u64,
    total_ms: f64,
}

impl TransportNwaTiming {
    fn new() -> Self {
        Self {
            enabled: std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some(),
            ..Self::default()
        }
    }

    #[inline]
    fn start(&self) -> Option<Instant> {
        self.enabled.then(Instant::now)
    }

    #[inline]
    fn elapsed_ms(&self, started: Option<Instant>) -> f64 {
        started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0)
    }

    fn emit(&self, modes: usize, roots: usize, mode_sets: usize) {
        if !self.enabled {
            return;
        }
        let edge_insertion_ms = self.future_edge_ms
            + self.endpoint_leaf_edge_ms
            + self.match_edge_ms
            + self.edge_flush_ms;
        eprintln!(
            "[glrmask/profile][transport_nwa] modes={} roots={} mode_sets={} root_context_ms={:.3} reset_context_ms={:.3} mode_set_intern_ms={:.3} mode_set_intern_hits={} mode_set_intern_new={} mapped_label_cache_hit_ms={:.3} mapped_label_compute_ms={:.3} mapped_label_cache_hits={} mapped_label_cache_misses={} trie_scan_ms={:.3} trie_scan_calls={} trie_scan_bytes={} pending_context_ms={:.3} pending_context_ops={} context_merge_ms={:.3} future_edge_ms={:.3} endpoint_leaf_edge_ms={:.3} match_edge_ms={:.3} edge_flush_ms={:.3} edge_insertion_ms={:.3} edge_buffer_insertions={} total_ms={:.3}",
            modes,
            roots,
            mode_sets,
            self.root_context_ms,
            self.reset_context_ms,
            self.mode_set_intern_ms,
            self.mode_set_intern_hits,
            self.mode_set_intern_new,
            self.mapped_label_cache_hit_ms,
            self.mapped_label_compute_ms,
            self.mapped_label_cache_hits,
            self.mapped_label_cache_misses,
            self.trie_scan_ms,
            self.trie_scan_calls,
            self.trie_scan_bytes,
            self.pending_context_ms,
            self.pending_context_ops,
            self.context_merge_ms,
            self.future_edge_ms,
            self.endpoint_leaf_edge_ms,
            self.match_edge_ms,
            self.edge_flush_ms,
            edge_insertion_ms,
            self.edge_buffer_insertions,
            self.total_ms,
        );
    }
}

impl IntoIterator for NodesByTransportContext {
    type Item = (TransportContext, Vec<NwaState>);
    type IntoIter = <FxHashMap<TransportContext, Vec<NwaState>> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.entries.into_iter()
    }
}

/// Interns subsets of transport modes and gives their exact raw-output label
/// expansion. Modes that enter the same raw scanner state are grouped because
/// a DFA continuation from that point is identical; only their output labels
/// differ.
struct TransportModePlanner<'m> {
    modes: &'m [TerminalNwaTransportMode],
    visible_representatives: &'m [bool],
    mode_set_ids: FxHashMap<TransportModeSet, usize>,
    mode_sets: Vec<TransportModeSet>,
    mapped_labels: Vec<FxHashMap<TerminalID, SmallVec<[TerminalID; 4]>>>,
    timing: TransportNwaTiming,
}

impl<'m> TransportModePlanner<'m> {
    fn new(
        modes: &'m [TerminalNwaTransportMode],
        visible_representatives: &'m [bool],
    ) -> Self {
        assert!(!modes.is_empty());
        Self {
            modes,
            visible_representatives,
            mode_set_ids: FxHashMap::default(),
            mode_sets: Vec::new(),
            mapped_labels: Vec::new(),
            timing: TransportNwaTiming::new(),
        }
    }

    fn intern_mode_set(&mut self, mode_set: TransportModeSet) -> usize {
        let timer = self.timing.start();
        debug_assert!(match &mode_set {
            TransportModeSet::Explicit(modes) | TransportModeSet::AllExcept(modes) => {
                modes.windows(2).all(|pair| pair[0] < pair[1])
            }
        });
        if let Some(&id) = self.mode_set_ids.get(&mode_set) {
            self.timing.mode_set_intern_hits += 1;
            self.timing.mode_set_intern_ms += self.timing.elapsed_ms(timer);
            return id;
        }
        let id = self.mode_sets.len();
        self.mode_set_ids.insert(mode_set.clone(), id);
        self.mode_sets.push(mode_set);
        self.mapped_labels.push(FxHashMap::default());
        self.timing.mode_set_intern_new += 1;
        self.timing.mode_set_intern_ms += self.timing.elapsed_ms(timer);
        id
    }

    /// The original direct grouping is cheaper when there are few transport
    /// modes, because a two-pass batch setup cannot amortize its own work.
    fn contexts_for_original_state_small(
        &mut self,
        original_state: TokenizerState,
    ) -> SmallVec<[TransportContext; 4]> {
        let mut modes_by_scanner_state = BTreeMap::<TokenizerState, Vec<usize>>::new();
        for (mode, transport) in self.modes.iter().enumerate() {
            let scanner_state = transport.scanner_state(original_state);
            modes_by_scanner_state
                .entry(scanner_state)
                .or_default()
                .push(mode);
        }
        modes_by_scanner_state
            .into_iter()
            .map(|(scanner_state, modes)| TransportContext {
                scanner_state,
                mode_set: self.intern_mode_set(TransportModeSet::Explicit(modes.into())),
            })
            .collect()
    }

    /// Build all root/reset contexts at once. The modal scanner state for a
    /// raw state is represented as all modes except sparse deviations, which
    /// avoids per-root ordered maps and full mode-vector interning.
    fn contexts_for_original_states(
        &mut self,
        original_states: &[TokenizerState],
    ) -> Vec<SmallVec<[TransportContext; 4]>> {
        const BATCH_MODE_THRESHOLD: usize = 128;
        if self.modes.len() < BATCH_MODE_THRESHOLD {
            return original_states
                .iter()
                .map(|&original_state| self.contexts_for_original_state_small(original_state))
                .collect();
        }

        type ScannerCounts = SmallVec<[(TokenizerState, usize); 4]>;
        type ExplicitGroups = SmallVec<[(TokenizerState, SmallVec<[usize; 4]>); 4]>;

        let state_count = original_states.len();
        if state_count == 0 {
            return Vec::new();
        }

        let mut counts = vec![ScannerCounts::new(); state_count];
        for transport in self.modes {
            for (state_index, &original_state) in original_states.iter().enumerate() {
                let scanner_state = transport.scanner_state(original_state);
                if let Some((_, count)) = counts[state_index]
                    .iter_mut()
                    .find(|(state, _)| *state == scanner_state)
                {
                    *count += 1;
                } else {
                    counts[state_index].push((scanner_state, 1));
                }
            }
        }

        let modal_scanner_states: Vec<_> = counts
            .iter()
            .map(|scanner_counts| {
                let mut modal = scanner_counts[0];
                for &(scanner_state, count) in scanner_counts.iter().skip(1) {
                    if count > modal.1 || (count == modal.1 && scanner_state < modal.0) {
                        modal = (scanner_state, count);
                    }
                }
                modal.0
            })
            .collect();

        let mut exclusions = vec![SmallVec::<[usize; 4]>::new(); state_count];
        let mut explicit_groups = vec![ExplicitGroups::new(); state_count];
        for (mode, transport) in self.modes.iter().enumerate() {
            for (state_index, &original_state) in original_states.iter().enumerate() {
                let scanner_state = transport.scanner_state(original_state);
                if scanner_state == modal_scanner_states[state_index] {
                    continue;
                }
                exclusions[state_index].push(mode);
                if let Some((_, mode_indices)) = explicit_groups[state_index]
                    .iter_mut()
                    .find(|(state, _)| *state == scanner_state)
                {
                    mode_indices.push(mode);
                } else {
                    explicit_groups[state_index]
                        .push((scanner_state, SmallVec::from_vec(vec![mode])));
                }
            }
        }

        let mut contexts_by_state = Vec::with_capacity(state_count);
        for state_index in 0..state_count {
            let excluded: Vec<_> = std::mem::take(&mut exclusions[state_index]).into_vec();
            let mut contexts = SmallVec::<[TransportContext; 4]>::new();
            contexts.push(TransportContext {
                scanner_state: modal_scanner_states[state_index],
                mode_set: self.intern_mode_set(TransportModeSet::AllExcept(excluded.into())),
            });
            for (scanner_state, mode_indices) in std::mem::take(&mut explicit_groups[state_index]) {
                let mode_indices: Vec<_> = mode_indices.into_vec();
                contexts.push(TransportContext {
                    scanner_state,
                    mode_set: self.intern_mode_set(TransportModeSet::Explicit(mode_indices.into())),
                });
            }
            contexts.sort_unstable_by_key(|context| context.scanner_state);
            contexts_by_state.push(contexts);
        }
        contexts_by_state
    }

    fn mapped_labels_for(
        &mut self,
        mode_set: usize,
        terminal: TerminalID,
    ) -> SmallVec<[TerminalID; 4]> {
        let timer = self.timing.start();
        if let Some(labels) = self.mapped_labels[mode_set].get(&terminal) {
            self.timing.mapped_label_cache_hits += 1;
            self.timing.mapped_label_cache_hit_ms += self.timing.elapsed_ms(timer);
            return labels.clone();
        }
        self.timing.mapped_label_cache_misses += 1;
        let mut labels = SmallVec::<[TerminalID; 4]>::new();
        match &self.mode_sets[mode_set] {
            TransportModeSet::Explicit(modes) => {
                for &mode in modes.iter() {
                    if let Some(label) = self.modes[mode]
                        .emitted_terminal(terminal, self.visible_representatives)
                    {
                        labels.push(label);
                    }
                }
            }
            TransportModeSet::AllExcept(excluded) => {
                let mut excluded_index = 0;
                for mode in 0..self.modes.len() {
                    if excluded.get(excluded_index).copied() == Some(mode) {
                        excluded_index += 1;
                    } else if let Some(label) = self.modes[mode]
                        .emitted_terminal(terminal, self.visible_representatives)
                    {
                        labels.push(label);
                    }
                }
            }
        }
        labels.sort_unstable();
        labels.dedup();
        self.mapped_labels[mode_set].insert(terminal, labels.clone());
        self.timing.mapped_label_compute_ms += self.timing.elapsed_ms(timer);
        labels
    }
}

struct TransportNwaBuilder<'tok, 'pm, 'nwa, 'm> {
    base: TerminalNwaBuilder<'tok, 'pm, 'nwa>,
    transport: TransportModePlanner<'m>,
    /// After every terminal boundary every temporary witness is available again.
    /// This vector partitions those modes by their reset scanner state.
    reset_contexts: SmallVec<[TransportContext; 4]>,
}

impl<'tok, 'pm, 'nwa, 'm> TransportNwaBuilder<'tok, 'pm, 'nwa, 'm> {
    fn mapped_labels(&mut self, context: TransportContext, terminal: TerminalID) -> SmallVec<[TerminalID; 4]> {
        self.transport.mapped_labels_for(context.mode_set, terminal)
    }

    fn add_future_leaf_token_from_sources(
        &mut self,
        sources: &[NwaState],
        context: TransportContext,
        internal_token_id: u32,
    ) {
        let future_edge_timer = self.transport.timing.start();
        let future = self
            .base
            .possible_future_terminals_for_state(context.scanner_state);
        for &terminal in future.iter() {
            let mapped_labels = self.mapped_labels(context, terminal);
            if mapped_labels.is_empty() {
                continue;
            }
            self.base.profile.future_terminal_additions +=
                (sources.len() * mapped_labels.len()) as u64;
            self.transport.timing.edge_buffer_insertions +=
                (sources.len() * mapped_labels.len()) as u64;
            for mapped_label in mapped_labels {
                self.base
                    .add_leaf_token_from_sources(sources, mapped_label, internal_token_id);
            }
        }
        self.transport.timing.future_edge_ms +=
            self.transport.timing.elapsed_ms(future_edge_timer);
    }

    fn build_from_trie(
        &mut self,
        node: VocabPrefixTreeNode<'_>,
        contexts: &NodesByTransportContext,
    ) {
        if contexts.is_empty() {
            return;
        }
        for (segment_bytes, child) in node.iter_children() {
            let next = self.process_child_segment(segment_bytes, child, contexts);
            if !next.is_empty() {
                self.build_from_trie(child, &next);
            }
        }
    }

    fn process_child_segment(
        &mut self,
        segment_bytes: &[u8],
        child_node: VocabPrefixTreeNode<'_>,
        initial_contexts: &NodesByTransportContext,
    ) -> NodesByTransportContext {
        let leaf_token_id = child_node.token_id() as u32;
        let mut next_level = NodesByTransportContext::default();
        let mut pending = BTreeMap::<usize, NodesByTransportContext>::new();
        let pending_timer = self.transport.timing.start();
        pending.insert(0, initial_contexts.clone());
        self.transport.timing.pending_context_ms +=
            self.transport.timing.elapsed_ms(pending_timer);
        self.transport.timing.pending_context_ops += 1;
        let mut match_map = FxHashMap::<TerminalID, (usize, TokenizerState)>::default();
        let mut matches = Vec::<TokenizerMatch>::new();

        while let Some((offset, contexts_at_offset)) = pending.pop_first() {
            if offset == segment_bytes.len() {
                let merge_timer = self.transport.timing.start();
                for (context, nodes) in contexts_at_offset {
                    next_level.merge(context, &nodes);
                }
                self.transport.timing.context_merge_ms +=
                    self.transport.timing.elapsed_ms(merge_timer);
                continue;
            }

            for (context, source_nodes) in contexts_at_offset {
                match_map.clear();
                let mut scan_state = context.scanner_state;
                let mut alive = true;
                let scan_timer = self.transport.timing.start();
                let mut scanned_bytes = 0_u64;
                for (index, &byte) in segment_bytes[offset..].iter().enumerate() {
                    scanned_bytes += 1;
                    let Some(next) = self.base.fast_step(scan_state, byte) else {
                        alive = false;
                        break;
                    };
                    scan_state = next;
                    for terminal in self.base.tokenizer.matched_terminals_iter(scan_state) {
                        if !self.mapped_labels(context, terminal).is_empty() {
                            match_map.insert(terminal, (index + 1, scan_state));
                        }
                    }
                }
                self.transport.timing.trie_scan_ms +=
                    self.transport.timing.elapsed_ms(scan_timer);
                self.transport.timing.trie_scan_calls += 1;
                self.transport.timing.trie_scan_bytes += scanned_bytes;
                let end_state = alive.then_some(scan_state);

                matches.clear();
                matches.extend(match_map.iter().map(|(&id, &(width, end_state))| TokenizerMatch {
                    id,
                    width,
                    end_state,
                }));

                if let Some(end_state) = end_state {
                    if child_node.has_token() {
                        self.add_future_leaf_token_from_sources(
                            &source_nodes,
                            TransportContext { scanner_state: end_state, ..context },
                            leaf_token_id,
                        );
                    }
                    let merge_timer = self.transport.timing.start();
                    next_level.merge(
                        TransportContext { scanner_state: end_state, ..context },
                        &source_nodes,
                    );
                    self.transport.timing.context_merge_ms +=
                        self.transport.timing.elapsed_ms(merge_timer);
                }

                for matched in &matches {
                    let next_offset = offset + matched.width;
                    let mapped_labels = self.mapped_labels(context, matched.id);
                    if next_offset == segment_bytes.len()
                        && child_node.has_token()
                        && !end_state.is_some_and(|state| {
                            self.base
                                .future_observation(state).terminals.contains(&matched.id)
                        })
                    {
                        self.base.profile.match_transition_additions +=
                            (source_nodes.len() * mapped_labels.len()) as u64;
                        let endpoint_edge_timer = self.transport.timing.start();
                        for mapped_label in &mapped_labels {
                            self.base.add_leaf_token_from_sources(
                                &source_nodes,
                                *mapped_label,
                                leaf_token_id,
                            );
                        }
                        self.transport.timing.endpoint_leaf_edge_ms +=
                            self.transport.timing.elapsed_ms(endpoint_edge_timer);
                        self.transport.timing.edge_buffer_insertions +=
                            (source_nodes.len() * mapped_labels.len()) as u64;
                    }

                    let Some(weight) = self.base.continuation_weight_for_match(
                        child_node,
                        leaf_token_id,
                        matched.id,
                        end_state,
                        &segment_bytes[next_offset..],
                    ) else {
                        continue;
                    };
                    if weight.is_empty() {
                        continue;
                    }

                    // A terminal boundary resets the actual lexer. Every
                    // transport mode is available for the next terminal, so one
                    // NWA continuation state carries their exact union; it is
                    // merely attached to each distinct reset scanner context.
                    let pending_timer = self.transport.timing.start();
                    let continuation_contexts = pending.entry(next_offset).or_default();
                    let destination = ensure_transport_continuation_state(
                        continuation_contexts,
                        self.base.nwa,
                    );
                    for &next_context in &self.reset_contexts {
                        continuation_contexts.push_one(next_context, destination);
                    }
                    self.transport.timing.pending_context_ms +=
                        self.transport.timing.elapsed_ms(pending_timer);
                    self.transport.timing.pending_context_ops += 1;
                    self.base.profile.match_transition_additions +=
                        (source_nodes.len() * mapped_labels.len()) as u64;
                    let match_edge_timer = self.transport.timing.start();
                    let mapped_label_count = mapped_labels.len();
                    for mapped_label in mapped_labels {
                        self.base.add_match_from_sources(
                            &source_nodes,
                            mapped_label,
                            destination,
                            &weight,
                        );
                    }
                    self.transport.timing.match_edge_ms +=
                        self.transport.timing.elapsed_ms(match_edge_timer);
                    self.transport.timing.edge_buffer_insertions +=
                        (source_nodes.len() * mapped_label_count) as u64;
                }
            }
        }

        next_level
    }
}

fn ensure_transport_continuation_state(
    pending: &mut NodesByTransportContext,
    nwa: &mut NWA,
) -> NwaState {
    if let Some(existing) = pending.first_any() {
        return existing;
    }
    let state = nwa.add_state();
    // The caller immediately attaches this state to the complete reset-mode
    // partition. Keeping the helper context-free prevents one identical NFA
    // union state from being cloned per terminal interchangeability mode.
    state
}

/// Slow reference trie walk for strict terminal interchangeability. It is used
/// only while the feature is validation-gated. Normal L2P construction remains
/// on `build_nwa_via_trie_walk` above.
pub fn build_transport_nwa_via_trie_walk<'a>(
    tokenizer: &'a Tokenizer,
    ignore_terminal: Option<TerminalID>,
    nwa: &mut NWA,
    start_state: u32,
    leaf_state: u32,
    id_map: &InternalIdMap,
    vocab_tree_root: VocabPrefixTreeNode<'_>,
    possible_matches: &mut PossibleMatchesComputer<'a>,
    visible_output_raw_labels: &[bool],
    modes: &[TerminalNwaTransportMode],
) -> TerminalDwaBuildProfile {
    assert!(!modes.is_empty());
    assert!(
        !tokenizer.has_epsilon_transitions(),
        "TI transport construction requires a deterministic tokenizer",
    );
    let mut roots = NodesByTransportContext::default();
    let mut initial_source_states = Vec::<bool>::new();
    let mut transport = TransportModePlanner::new(modes, visible_output_raw_labels);
    let build_timer = transport.timing.start();
    let representative_states: Vec<_> = id_map
        .tokenizer_states
        .iter_representative_ids()
        .collect();
    let root_context_timer = transport.timing.start();
    let root_contexts = transport.contexts_for_original_states(&representative_states);

    for (internal_tsid, contexts) in root_contexts.into_iter().enumerate() {
        // All transport modes start from the same parser-side condition for
        // this actual tokenizer state. A single NWA root therefore represents
        // their exact union; contexts retain only the distinct raw scanner
        // continuations needed to discover the next terminal boundary.
        let root = nwa.add_state();
        let weight = all_token_weight(internal_tsid as u32, id_map.max_internal_token_id());
        nwa.add_epsilon(start_state, root, weight);
        if root as usize >= initial_source_states.len() {
            initial_source_states.resize(root as usize + 1, false);
        }
        initial_source_states[root as usize] = true;
        for context in contexts {
            roots.merge(context, &[root]);
        }
    }
    transport.timing.root_context_ms += transport.timing.elapsed_ms(root_context_timer);
    let reset_context_timer = transport.timing.start();
    let reset_contexts = transport
        .contexts_for_original_states(&[tokenizer.initial_state_id()])
        .into_iter()
        .next()
        .expect("one reset state must produce one context set");
    transport.timing.reset_context_ms += transport.timing.elapsed_ms(reset_context_timer);

    let base = TerminalNwaBuilder::new(
        tokenizer,
        TerminalColoring::identity(tokenizer.num_terminals() as usize),
        possible_matches,
        nwa,
        id_map.num_tsids(),
        leaf_state,
        ignore_terminal,
        initial_source_states,
        false,
        None,
        tokenizer.num_states() as usize,
        None,
    );
    let mut builder = TransportNwaBuilder {
        base,
        transport,
        reset_contexts,
    };
    builder.build_from_trie(vocab_tree_root, &roots);
    let flush_timer = builder.transport.timing.start();
    builder.base.flush_transition_buffer();
    builder.transport.timing.edge_flush_ms +=
        builder.transport.timing.elapsed_ms(flush_timer);
    builder.transport.timing.total_ms = builder.transport.timing.elapsed_ms(build_timer);
    builder.transport.timing.emit(
        modes.len(),
        id_map.num_tsids() as usize,
        builder.transport.mode_sets.len(),
    );
    builder.base.profile
}

#[path="nwa_builder/native_builder.rs"] pub mod native_builder;
use native_builder::BorrowedObservation;
#[path="nwa_builder/native_scalar_cursor.rs"] mod native_scalar_cursor;
#[path="nwa_builder/native_borrowed_cursor.rs"] mod native_borrowed_cursor;
#[path="nwa_builder/native_compact_buffer.rs"] mod native_compact_buffer;
