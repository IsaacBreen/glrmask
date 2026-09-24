//! Checked boundary-analysis scope.
//!
//! Ordinary terminal-DWA construction observes the whole raw tokenizer state
//! domain. Composition boundary shards may start only in states owned by one
//! immediate component, while reset after a terminal commit still follows the
//! merged tokenizer. Keep those domains explicit so a quotient cannot seed a
//! representative owned by another component.

use std::sync::Arc;

use crate::automata::lexer::{Lexer, tokenizer::Tokenizer};
use crate::compiler::stages::equiv_types::ManyToOneIdMap;
use crate::ds::bitset::BitSet;
use crate::Vocab;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ImmediateComponentId(pub u32);

/// Checked ownership of concrete terminal labels by the immediate components
/// of the link currently being compiled. A nested component may own several
/// concrete leaves; callers provide the leaf-to-immediate projection explicitly
/// so leaf identity is never confused with immediate-component identity.
#[derive(Debug, Clone)]
pub struct BoundaryOwnership {
    terminal_owner: Arc<[ImmediateComponentId]>,
    num_immediate_components: u32,
}

impl BoundaryOwnership {
    pub fn from_leaf_layout(
        terminal_offsets: &[u32],
        num_terminals: u32,
        leaf_to_immediate: &[ImmediateComponentId],
        num_immediate_components: u32,
    ) -> Result<Self, String> {
        if terminal_offsets.is_empty() || terminal_offsets[0] != 0 {
            return Err("boundary ownership terminal offsets must start at zero".to_owned());
        }
        if terminal_offsets.len() != leaf_to_immediate.len() {
            return Err(format!(
                "boundary ownership has {} terminal leaf offsets but {} leaf owners",
                terminal_offsets.len(),
                leaf_to_immediate.len(),
            ));
        }
        if num_immediate_components == 0 {
            return Err("boundary ownership has no immediate components".to_owned());
        }
        if terminal_offsets.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err("boundary ownership terminal offsets are not strictly increasing".to_owned());
        }
        if terminal_offsets.last().copied().unwrap_or(0) >= num_terminals && num_terminals != 0 {
            return Err("boundary ownership final leaf starts outside terminal domain".to_owned());
        }
        if leaf_to_immediate
            .iter()
            .any(|owner| owner.0 >= num_immediate_components)
        {
            return Err("boundary ownership references an out-of-range immediate component".to_owned());
        }
        let mut terminal_owner = Vec::with_capacity(num_terminals as usize);
        for terminal in 0..num_terminals {
            let leaf = terminal_offsets
                .partition_point(|&offset| offset <= terminal)
                .saturating_sub(1);
            let owner = *leaf_to_immediate
                .get(leaf)
                .ok_or_else(|| format!("terminal {terminal} has no owning leaf"))?;
            terminal_owner.push(owner);
        }
        Ok(Self {
            terminal_owner: Arc::from(terminal_owner.into_boxed_slice()),
            num_immediate_components,
        })
    }

    pub fn flat(terminal_offsets: &[u32], num_terminals: u32) -> Result<Self, String> {
        let owners = (0..terminal_offsets.len())
            .map(|index| ImmediateComponentId(index as u32))
            .collect::<Vec<_>>();
        Self::from_leaf_layout(
            terminal_offsets,
            num_terminals,
            &owners,
            terminal_offsets.len() as u32,
        )
    }

    #[inline]
    pub fn owner_of_terminal(&self, terminal: u32) -> Option<ImmediateComponentId> {
        self.terminal_owner.get(terminal as usize).copied()
    }

    #[inline]
    pub fn num_terminals(&self) -> usize {
        self.terminal_owner.len()
    }

    #[inline]
    pub fn num_immediate_components(&self) -> u32 {
        self.num_immediate_components
    }
}

#[derive(Debug, Clone)]
pub struct InitialStateDomain {
    keep_raw: Arc<[bool]>,
    exact_singleton_map: ManyToOneIdMap,
}

impl InitialStateDomain {
    pub fn from_mask(num_raw_states: usize, keep_raw: Vec<bool>) -> Result<Self, String> {
        if keep_raw.len() != num_raw_states {
            return Err(format!(
                "boundary initial-state mask has length {}, expected {num_raw_states}",
                keep_raw.len(),
            ));
        }
        if !keep_raw.iter().any(|&keep| keep) {
            return Err("boundary initial-state domain is empty".to_owned());
        }

        let mut original_to_internal = vec![u32::MAX; num_raw_states];
        let mut representative_original_ids = Vec::new();
        for (raw, &keep) in keep_raw.iter().enumerate() {
            if !keep {
                continue;
            }
            let internal = representative_original_ids.len() as u32;
            original_to_internal[raw] = internal;
            representative_original_ids.push(raw as u32);
        }
        let exact_singleton_map =
            ManyToOneIdMap::from_singleton_original_to_internal_with_representatives(
                original_to_internal,
                representative_original_ids,
            );
        Ok(Self {
            keep_raw: Arc::from(keep_raw.into_boxed_slice()),
            exact_singleton_map,
        })
    }

    #[inline]
    pub fn keep_raw(&self) -> &[bool] {
        &self.keep_raw
    }

    #[inline]
    pub fn exact_singleton_map(&self) -> &ManyToOneIdMap {
        &self.exact_singleton_map
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.exact_singleton_map.representative_original_ids.len()
    }

    #[inline]
    pub fn contains(&self, raw: u32) -> bool {
        self.keep_raw.get(raw as usize).copied().unwrap_or(false)
    }
}

/// Preserve an already-proved scoped quotient for represented token-start
/// states while giving every excluded raw state its own continuation-only
/// TSID. Appending singletons keeps all existing TSID names stable, so weights
/// built over the scoped domain need no remap. Root seeding remains separately
/// controlled by [`InitialStateDomain`].
pub fn complete_with_continuation_singletons(map: &mut ManyToOneIdMap) {
    for raw in 0..map.original_to_internal.len() {
        if map.original_to_internal[raw] != u32::MAX {
            continue;
        }
        let internal = map.internal_to_originals.len() as u32;
        map.original_to_internal[raw] = internal;
        map.internal_to_originals.push(vec![raw as u32]);
        map.representative_original_ids.push(raw as u32);
    }
}

/// Restrict a certified observation quotient to a token-start domain.
///
/// Each output class is an intersection of one proved class with the allowed
/// raw domain, so every retained member still has the same observation as its
/// representative. This is an INITIAL observation map, not a total transition
/// congruence: callers must keep the original total quotient separately when
/// constructing a continuation topology.
pub fn restrict_quotient_to_initial_domain(
    quotient: &ManyToOneIdMap,
    domain: &ManyToOneIdMap,
) -> Option<ManyToOneIdMap> {
    let n = domain.original_to_internal.len();
    if quotient.original_to_internal.len() != n { return None; }
    let mut remap = vec![u32::MAX; quotient.num_internal_ids() as usize];
    let mut original_to_internal = vec![u32::MAX; n];
    let mut groups = Vec::<Vec<u32>>::new();
    let mut representatives = Vec::new();
    for (raw, &allowed) in domain.original_to_internal.iter().enumerate() {
        if allowed == u32::MAX { continue; }
        let class = quotient.original_to_internal[raw] as usize;
        let slot = remap.get_mut(class)?;
        if *slot == u32::MAX {
            *slot = groups.len() as u32;
            groups.push(Vec::new());
            representatives.push(raw as u32);
        }
        original_to_internal[raw] = *slot;
        groups[*slot as usize].push(raw as u32);
    }
    Some(ManyToOneIdMap {
        original_to_internal,
        internal_to_originals: groups,
        representative_original_ids: representatives,
    })
}

/// Necessary initial-state support for a crossing-only lexical walk.
///
/// A crossing terminal word either completes a terminal on a token prefix,
/// or its first observed terminal is foreign and only partially consumed.
/// The event predicate therefore includes *any* terminal completion and any
/// foreign future observation. A state with no such event on any vocabulary
/// prefix cannot begin a crossing word. Local parser/follow restrictions are
/// deliberately ignored, so this is a safe superset, not an admission oracle.
///
/// The ordinary byte scanners report matches only AFTER consuming a byte.
/// A local finalizer already present at token entry is therefore not an event
/// by itself. Foreign finalizer/future observations at entry are retained for
/// empty-token and epsilon cases. Final-byte matches remain conservatively
/// included; no assumption about positive-byte CALL/RETURN is made.
/// Only token-start seeds are filtered. All reset/continuation raw coordinates
/// remain available to the ordinary compiler and its exact lifting maps.
pub fn crossing_prefix_seed_support(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    flat_trans: &[u32],
    ownership: &BoundaryOwnership,
    start: ImmediateComponentId,
) -> Option<(Vec<bool>, usize)> {
    use super::l2p::equivalence_analysis::state_equivalence::nfa::build_token_bounded_analysis_trie_sorted;

    if vocab.is_empty() || tokenizer.has_virtual_residual_runtime() {
        return None;
    }
    if vocab.entries_map().values().try_fold(0usize, |n, v| n.checked_add(v.len()))? > 262_144 {
        return None;
    }
    let n = tokenizer.num_states() as usize;
    let mut events = Vec::with_capacity(n);
    let mut initial_events = Vec::with_capacity(n);
    let mut closures = Vec::new();
    let mut closure_volume = 0usize;
    for q in 0..n {
        if tokenizer.state_is_virtual_runtime(q as u32) { return None; }
        let foreign_future = tokenizer.possible_future_terminals_iter(q as u32)
            .any(|t| ownership.owner_of_terminal(t) != Some(start));
        events.push(tokenizer.matched_terminals_iter(q as u32).next().is_some() || foreign_future);
        initial_events.push(foreign_future || tokenizer.matched_terminals_iter(q as u32)
            .any(|t| ownership.owner_of_terminal(t) != Some(start)));
        if tokenizer.state_has_epsilon_transitions(q as u32) {
            let closure = tokenizer.singleton_epsilon_closure(q as u32);
            closure_volume = closure_volume.checked_add(closure.len())?;
            if closure_volume > 262_144 { return None; }
            closures.push((q, closure));
        }
    }
    let mut words = vocab.entries_map().values().map(Vec::as_slice).collect::<Vec<_>>();
    words.sort_unstable(); words.dedup();
    let trie = build_token_bounded_analysis_trie_sorted(&words);
    trie.prefix_event_sources(flat_trans, &events, &closures, Some(&initial_events))
}

#[derive(Debug, Clone)]
pub struct BoundaryAnalysisScope {
    // Ensures differential tests can construct a genuinely unfiltered oracle
    // even when the shared family builder automatically narrows partitions.
    #[cfg(test)]
    automatic_prefix_support: bool,
    initial_states: InitialStateDomain,
    reset_states: Arc<[u32]>,
    ownership: Arc<BoundaryOwnership>,
    start_component: ImmediateComponentId,
    require_crossing: bool,
    follow_transparent: Option<BitSet>,
}

impl BoundaryAnalysisScope {
    pub fn automatic_prefix_support_enabled(&self) -> bool {
        #[cfg(test)]
        { return self.automatic_prefix_support; }
        #[cfg(not(test))]
        { true }
    }
    /// Relocate only raw lexer coordinates for an independently certified
    /// query-observation view. Terminal ownership and all crossing/follow
    /// observations remain identical. Every requested initial/reset state
    /// must have a defined image; no seed is silently dropped.
    pub fn relocate_query_view(&self, map:&[u32], state_count:usize, resets:Vec<u32>) -> Result<Self,String>{
        if map.len()!=self.initial_states.keep_raw.len(){return Err("query map source size".into());}
        let mut keep=vec![false;state_count];
        for (q,&yes) in self.initial_states.keep_raw.iter().enumerate(){if yes{
            let mapped=map[q] as usize;if mapped>=state_count{return Err("query view omitted initial state".into());}
            keep[mapped]=true;
        }}
        let mut expected=self.reset_states.iter().map(|&q|map[q as usize]).collect::<Vec<_>>();
        expected.sort_unstable();expected.dedup();let mut actual=resets.clone();actual.sort_unstable();actual.dedup();
        if expected!=actual || expected.iter().any(|&q|q as usize>=state_count){return Err("query view reset relation changed".into());}
        let result=Self::new(InitialStateDomain::from_mask(state_count,keep)?,resets,
            Arc::clone(&self.ownership),self.start_component,self.require_crossing,self.follow_transparent.clone())?;
        #[cfg(test)] { let mut result=result; result.automatic_prefix_support=self.automatic_prefix_support; return Ok(result); }
        #[cfg(not(test))] { Ok(result) }
    }

    /// Intersect the query's token-start domain without changing continuation
    /// topology, ownership, or the zero-width/follow contracts.
    pub fn intersect_initial_support(&self, support: &[bool]) -> Option<Self> {
        assert_eq!(support.len(), self.initial_states.keep_raw.len());
        let keep = self.initial_states.keep_raw.iter().zip(support)
            .map(|(&initial, &live)| initial && live).collect();
        let initial_states = InitialStateDomain::from_mask(support.len(), keep).ok()?;
        Some(Self {
            #[cfg(test)]
            automatic_prefix_support: self.automatic_prefix_support,
            initial_states,
            reset_states: Arc::clone(&self.reset_states),
            ownership: Arc::clone(&self.ownership),
            start_component: self.start_component,
            require_crossing: self.require_crossing,
            follow_transparent: self.follow_transparent.clone(),
        })
    }

    pub fn new(
        initial_states: InitialStateDomain,
        reset_states: Vec<u32>,
        ownership: Arc<BoundaryOwnership>,
        start_component: ImmediateComponentId,
        require_crossing: bool,
        follow_transparent: Option<BitSet>,
    ) -> Result<Self, String> {
        if start_component.0 >= ownership.num_immediate_components() {
            return Err(format!(
                "boundary start component {} is outside {} ownership blocks",
                start_component.0,
                ownership.num_immediate_components(),
            ));
        }
        let mut reset_states = reset_states;
        reset_states.sort_unstable();
        reset_states.dedup();
        if reset_states.is_empty() {
            return Err("boundary reset-state domain is empty".to_owned());
        }
        if reset_states
            .iter()
            .any(|&raw| raw as usize >= initial_states.keep_raw().len())
        {
            return Err("boundary reset state lies outside raw tokenizer domain".to_owned());
        }
        Ok(Self {
            #[cfg(test)]
            automatic_prefix_support: true,
            initial_states,
            reset_states: Arc::from(reset_states.into_boxed_slice()),
            ownership,
            start_component,
            require_crossing,
            follow_transparent,
        })
    }

    #[inline]
    pub fn initial_states(&self) -> &InitialStateDomain {
        &self.initial_states
    }

    #[inline]
    pub fn reset_states(&self) -> &[u32] {
        &self.reset_states
    }

    #[inline]
    pub fn ownership(&self) -> &BoundaryOwnership {
        &self.ownership
    }

    #[inline]
    pub fn start_component(&self) -> ImmediateComponentId {
        self.start_component
    }

    #[inline]
    pub fn require_crossing(&self) -> bool {
        self.require_crossing
    }

    #[inline]
    pub fn follow_transparent(&self) -> Option<&BitSet> {
        self.follow_transparent.as_ref()
    }

    #[inline]
    pub fn owner_of_terminal(&self, terminal: u32) -> Option<ImmediateComponentId> {
        self.ownership.owner_of_terminal(terminal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn restricted_quotient_keeps_seed_members_but_not_foreign_representatives() {
        let quotient = ManyToOneIdMap::from_original_to_internal_with_representatives(
            vec![0, 1, 0, 1, 2, 2], 3, vec![0, 1, 4],
        );
        let domain = InitialStateDomain::from_mask(6, vec![false, false, true, true, false, true]).unwrap();
        let restricted = restrict_quotient_to_initial_domain(&quotient, domain.exact_singleton_map()).unwrap();
        assert_eq!(restricted.original_to_internal, vec![u32::MAX, u32::MAX, 0, 1, u32::MAX, 2]);
        assert_eq!(restricted.representative_original_ids, vec![2, 3, 5]);
        // The separate total certificate remains unchanged and has all raw
        // successors, including the representatives excluded from seed scope.
        assert_eq!(quotient.representative_original_ids, vec![0, 1, 4]);
        let domain = InitialStateDomain::from_mask(6, vec![true, false, true, false, false, false]).unwrap();
        let restricted = restrict_quotient_to_initial_domain(&quotient, domain.exact_singleton_map()).unwrap();
        assert_eq!(restricted.internal_to_originals, vec![vec![0, 2]]);
    }

    #[test]
    fn prefix_support_preserves_complete_boundary_weighted_language() {
        use crate::automata::lexer::ast::{Expr, bytes, choice, plus};
        use crate::automata::lexer::compile::{build_regex_monolithic, build_regex_partitioned};
        use crate::automata::weighted::terminal_automaton::TerminalAutomaton;
        use crate::compiler::glr::analysis::AnalyzedGrammar;
        use crate::grammar::flat::{Rule, Symbol};
        use super::super::{build_scoped_boundary_id_map_and_terminal_dwa, l1, l2p, types};
        use std::collections::BTreeMap;

        let expressions = vec![
            bytes(b"abcdefghijklmn"),
            choice(vec![bytes(b"xyz"), plus(bytes(b"q")), Expr::Epsilon]),
            bytes(b"!"),
        ];
        for tokenizer in [
            build_regex_monolithic(&expressions).into_tokenizer(
                3, Some(Arc::from(expressions.clone().into_boxed_slice())),
            ),
            build_regex_partitioned(&expressions, &[0, 1, 2]).into_tokenizer(
                3, Some(Arc::from(expressions.clone().into_boxed_slice())),
            ),
        ] {
        let n = tokenizer.num_states() as usize;
        let vocab = Vocab::new(vec![
            (0, vec![]), (3, b"mn!".to_vec()), (11, b"mn!".to_vec()),
            (24, b"!ab".to_vec()), (81, b"z!".to_vec()), (100, b"q!".to_vec()),
            (103, b"!q!".to_vec()), (109, b"!".to_vec()), (200, vec![0xc2]),
        ]);
        let rules = vec![
            Rule { lhs: 1, rhs: vec![Symbol::Nonterminal(0)] },
            Rule { lhs: 0, rhs: vec![Symbol::Terminal(0), Symbol::Terminal(2)] },
            Rule { lhs: 0, rhs: vec![Symbol::Terminal(1), Symbol::Terminal(2)] },
            Rule { lhs: 0, rhs: vec![Symbol::Terminal(2), Symbol::Terminal(0)] },
            Rule { lhs: 0, rhs: vec![Symbol::Terminal(2), Symbol::Terminal(1), Symbol::Terminal(2)] },
        ];
        let grammar = AnalyzedGrammar::from_composed_rules(
            rules, 3, vec!["long".into(), "pattern".into(), "foreign".into()],
            vec!["doc".into(), "augmented".into()], 1,
        );
        let ownership = Arc::new(BoundaryOwnership::flat(&[0, 2], 3).unwrap());
        let flat: Arc<[u32]> = Arc::from(l1::build_flat_transition_table(&tokenizer));
        let mut total_removed = 0;
        for start in [ImmediateComponentId(0), ImmediateComponentId(1)] {
            let (support, _) = crossing_prefix_seed_support(&tokenizer, &vocab, &flat, &ownership, start).unwrap();
            total_removed += support.iter().filter(|&&keep| !keep).count();
            let build = |mask: Vec<bool>, automatic_prefix_support| {
                let mut scope = BoundaryAnalysisScope::new(
                    InitialStateDomain::from_mask(n, mask).unwrap(),
                    tokenizer.deterministic_reset_states().into_iter().collect(),
                    Arc::clone(&ownership), start, true, None,
                ).unwrap();
                scope.automatic_prefix_support = automatic_prefix_support;
                let (mapped, profile) = build_scoped_boundary_id_map_and_terminal_dwa(
                    &tokenizer, &vocab, &types::TerminalColoring::identity(3),
                    None, &grammar, &BTreeMap::new(), Arc::clone(&flat), &scope,
                );
                let (automaton, id_map) = mapped.into_parts();
                let TerminalAutomaton::Dwa(dwa) = automaton else { panic!("boundary must publish DWA") };
                types::LocalIdMapTerminalDwa { dwa, id_map, profile }
            };
            // The reference cannot run partition-local support implicitly:
            // otherwise both sides could share the same pruning bug.
            l2p::terminal_dwa_equivalence::compare(
                &build(vec![true; n], false), &build(support, true),
            ).unwrap();
        }
        assert!(total_removed > 0, "test must exercise actual seed removal");
        }
    }

    #[test]
    fn prefix_support_keeps_epsilon_and_foreign_partial_observations() {
        let tokenizer = crate::automata::lexer::tokenizer::arbitrary_epsilon_l1_test_tokenizer();
        let flat = super::super::l1::build_flat_transition_table(&tokenizer);
        let ownership = BoundaryOwnership::flat(&[0, 1], 2).unwrap();
        let vocab = Vocab::new(vec![(8, b"a!".to_vec()), (20, b"b!".to_vec())]);
        let (support, _) = crossing_prefix_seed_support(&tokenizer, &vocab, &flat, &ownership, ImmediateComponentId(0)).unwrap();
        assert!(support[0] && support[1] && support[2] && support[4]);
        // Local acceptance at entry is not a completion of this token: the
        // ordinary scanners record matches only after a consumed byte.
        assert!(!support[3] && !support[6]);
        // Fixture state5 completes foreign terminal1. State6, like state3,
        // completes LOCAL terminal0 and has no outgoing byte transitions.
        assert!(support[5]);
    }

    #[test]
    fn initial_domain_builds_only_in_scope_singleton_classes() {
        let domain = InitialStateDomain::from_mask(
            6,
            vec![false, true, false, true, true, false],
        )
        .unwrap();
        assert_eq!(domain.len(), 3);
        assert_eq!(domain.exact_singleton_map().representative_original_ids, vec![1, 3, 4]);
        assert_eq!(
            domain.exact_singleton_map().original_to_internal,
            vec![u32::MAX, 0, u32::MAX, 1, 2, u32::MAX],
        );
    }

    #[test]
    fn nested_leaf_layout_maps_to_immediate_owner_blocks() {
        let ownership = BoundaryOwnership::from_leaf_layout(
            &[0, 2, 5, 7],
            9,
            &[
                ImmediateComponentId(0),
                ImmediateComponentId(0),
                ImmediateComponentId(1),
                ImmediateComponentId(1),
            ],
            2,
        )
        .unwrap();
        assert_eq!(ownership.owner_of_terminal(0), Some(ImmediateComponentId(0)));
        assert_eq!(ownership.owner_of_terminal(6), Some(ImmediateComponentId(1)));
        assert_eq!(ownership.owner_of_terminal(8), Some(ImmediateComponentId(1)));
    }

    #[test]
    fn continuation_completion_preserves_scoped_classes() {
        let domain = InitialStateDomain::from_mask(
            5,
            vec![false, true, true, false, false],
        )
        .unwrap();
        let mut map = domain.exact_singleton_map().clone();
        complete_with_continuation_singletons(&mut map);
        assert_eq!(map.original_to_internal, vec![2, 0, 1, 3, 4]);
        assert_eq!(map.internal_to_originals, vec![vec![1], vec![2], vec![0], vec![3], vec![4]]);
    }
}
