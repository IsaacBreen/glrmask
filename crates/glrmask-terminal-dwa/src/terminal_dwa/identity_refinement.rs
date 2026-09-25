//! Exact over-refinement for small scoped vocabulary queries.
//!
//! This entry point constructs its own proof-trivial map: every physical lexer
//! state remains distinct, and model tokens merge only on identical byte words.
//! It accepts no externally asserted equivalence certificate. The expensive
//! equivalence/TI machinery is optional compression, not required for exactness.
//! The ordinary L2P, follow, crossing and final ignore passes remain unchanged.
use super::*;

/// A permutation, never an equivalence quotient. Every raw state has exactly
/// one distinct internal coordinate; selected token-entry coordinates form
/// a dense prefix so weighted sets can use shorter interval representations.
fn singleton_state_map(selected: &[bool], packed: bool) -> ManyToOneIdMap {
    let n = selected.len() as u32;
    let representatives = if packed {
        (0..n).filter(|&q| selected[q as usize])
            .chain((0..n).filter(|&q| !selected[q as usize])).collect::<Vec<_>>()
    } else { (0..n).collect() };
    let mut original_to_internal = vec![u32::MAX; n as usize];
    for (internal, &raw) in representatives.iter().enumerate() {
        original_to_internal[raw as usize] = internal as u32;
    }
    ManyToOneIdMap::from_singleton_original_to_internal_with_representatives(
        original_to_internal, representatives,
    )
}

/// A native minimum whose graph has not undergone ignore erasure or another
/// transformation. The private constructor prevents generic callers from
/// treating an arbitrary equivalent TDWA as a minimization fixed point.
/// This certificate belongs to the accompanying output and is consumed before
/// any change to that graph. It is never persisted or exposed to mask runtime.
pub struct NativeMinimizationFixedPoint { _private: () }

impl NativeMinimizationFixedPoint {
    // Reproduce the ordinary second pass's *small-direct* eligibility. Larger
    // outputs can be graph-isomorphic fixed points yet have a different raw
    // row order after generic hash-class enumeration. They must keep the old
    // second pass; a language or isomorphism proof alone is not sufficient.
    fn from_native(dwa:&crate::automata::weighted_u32::dwa::DWA,ignore:Option<TerminalID>)->Option<Self>{
        let n=dwa.states().len();
        if ignore.is_some() || n==0 || n>192 { return None; }
        let mut heights=vec![0usize;n];
        let mut buckets=vec![0usize;n];
        for (source,state) in dwa.states().iter().enumerate(){
            let mut height=0;
            for &(target,ref weight) in state.transitions.values(){
                // The native output is constructed bottom-up with live edges.
                // Refuse any future backend representation that changes this.
                if target as usize>=source || weight.is_empty(){return None;}
                height=height.max(heights[target as usize]+1);
            }
            heights[source]=height;buckets[height]+=1;
            if height!=0 && buckets[height]>64{return None;}
        }
        Some(Self{_private:()})
    }
}

pub fn build_scoped_boundary_identity_refinement(
    tokenizer:&Tokenizer,vocab:&Vocab,ignore_terminal:Option<TerminalID>,
    grammar:&AnalyzedGrammar,disallowed_follows:&BTreeMap<u32,BitSet>,
    flat_trans:Arc<[u32]>,scope:&scope::BoundaryAnalysisScope,
)->Option<(MappedArtifact<TerminalAutomaton>,TerminalDwaPhaseProfile)>{
    build_scoped_boundary_identity_with_certificate(tokenizer,vocab,ignore_terminal,
        grammar,disallowed_follows,flat_trans,scope).map(|(mapped,profile,_)|(mapped,profile))
}

/// Same public construction contract, with an additional narrowly scoped
/// certificate for a completed, unmodified native minimization.
pub fn build_scoped_boundary_identity_with_certificate(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    ignore_terminal: Option<TerminalID>,
    grammar: &AnalyzedGrammar,
    disallowed_follows: &BTreeMap<u32, BitSet>,
    flat_trans: Arc<[u32]>,
    scope: &scope::BoundaryAnalysisScope,
) -> Option<(MappedArtifact<TerminalAutomaton>, TerminalDwaPhaseProfile, Option<NativeMinimizationFixedPoint>)> {
    let started = Instant::now();
    let n = tokenizer.num_states();
    let max_token = *vocab.entries_map().keys().max()?;
    if n == 0 || n > 200_000 || max_token >= 2_000_000 || vocab.len() > 4096
        || tokenizer.has_virtual_residual_runtime()
        || vocab.entries_map().values().any(Vec::is_empty)
        || scope.initial_states().keep_raw().len() != n as usize
        || grammar.num_terminals != tokenizer.num_terminals()
        || flat_trans.len() != (n as usize).checked_mul(256)?
    {
        return None;
    }
    let mut roots = tokenizer.deterministic_reset_states().into_vec();
    roots.sort_unstable(); roots.dedup();
    if roots != scope.reset_states() { return None; }
    let packed = std::env::var_os("GLRMASK_BOUNDARY_PACK_INITIAL_TSIDS").is_some();
    let states = singleton_state_map(scope.initial_states().keep_raw(), packed);
    let mut words = BTreeMap::<Vec<u8>, Vec<u32>>::new();
    for (&id, word) in vocab.entries_map() {
        words.entry(word.clone()).or_default().push(id);
    }
    let groups = words.into_values().collect::<Vec<_>>();
    let mut original_to_internal = vec![u32::MAX; max_token as usize + 1];
    for (class, originals) in groups.iter().enumerate() {
        for &id in originals { original_to_internal[id as usize] = class as u32; }
    }
    let tokens = ManyToOneIdMap {
        original_to_internal,
        representative_original_ids: groups.iter().map(|g| g[0]).collect(),
        internal_to_originals: groups,
    };
    let map_ms = started.elapsed().as_secs_f64() * 1000.0;
    let length_count = |bound| vocab.entries_map().values().filter(|w| w.len() > bound).count();
    // The private common-builder transport accepts a map regardless of whether
    // it was obtained by optional compression or by exact identity refinement.
    // Keep that distinction at this typed public entry point, not in callers.
    let shared = l2p::SharedL2pEquivalence {
        id_map: InternalIdMap { tokenizer_states: states, vocab_tokens: tokens,
            deferred_vocab_singleton_original_ids: None },
        id_map_ms: map_ms,
        profile: l2p::equivalence_analysis::combined::CombinedEquivalenceProfile {
            initial_states_considered: scope.initial_states().len(), max_length_skipped: true,
            max_token_len: vocab.max_token_byte_len(), token_len_gt_4: length_count(4),
            token_len_gt_8: length_count(8), token_len_gt_16: length_count(16),
            token_len_gt_32: length_count(32), token_len_gt_64: length_count(64),
            raw_analysis_base_init_ms: 0.0, analysis_view_build_ms: 0.0,
            active_mask_filter_ms: 0.0, effective_follows_normalize_ms: 0.0,
            prepare_inputs_ms: 0.0, byte_class_setup_ms: 0.0,
            vocab_analysis_dfa_build_ms: 0.0, token_dedup_ms: map_ms,
            restricted_observation_state_equiv_ms: 0.0, max_length_state_equiv_ms: 0.0,
            vocab_equiv_ms: 0.0, exact_state_equiv_ms: 0.0, id_map_finalize_ms: 0.0,
            restricted_observation_reps: n as usize, max_length_reps: n as usize,
            exact_reps: n as usize, exact_rep_confirmation_used: false,
        },
    };
    let options = l2p::L2pShardBuildOptions {
        shared_equivalence: Some(&shared), skip_ti_discovery: true,
        ti_candidate_groups: None,
        crossing_filter: scope.require_crossing().then_some(l2p::L2pCrossingFilter {
            ownership: scope.ownership(), start_component: scope.start_component(),
        }),
        skip_core_compact: true, follow_transparent: scope.follow_transparent(),
        initial_state_domain_is_exact: true,
    };
    let active = vec![true; tokenizer.num_terminals() as usize];
    let ordinary = || {
        let always = compute_always_allowed_follows(grammar);
        l2p::build_l2p_id_map_and_terminal_dwa_mode(
        "boundary_identity_refinement", tokenizer, vocab,
        &TerminalColoring::identity(active.len()), false, ignore_terminal, grammar,
        &always, &active, disallowed_follows,
        None, None, None, None, None, None, None, Some(&flat_trans), None,
        Some(&shared.id_map.tokenizer_states), false,
        Some(scope.initial_states().keep_raw()), Some(&options),
        )
    };
    let direct=(std::env::var_os("GLRMASK_BOUNDARY_DIRECT_IDENTITY_ENTRY").is_some()
        &&std::env::var_os("GLRMASK_BOUNDARY_NATIVE_EVENT_PIPELINE").is_some()
        &&std::env::var_os("GLRMASK_BOUNDARY_NATIVE_TERMINAL_ALGEBRA").is_some())
        .then(||l2p::native_identity::try_build(tokenizer,vocab,ignore_terminal,grammar,
            disallowed_follows,Some(&flat_trans),scope,&shared.id_map)).flatten();
    let (mut output, fixed_point)=match direct {
        Some(candidate)=>{
            if std::env::var_os("GLRMASK_VALIDATE_BOUNDARY_DIRECT_IDENTITY").is_some(){
                let reference=ordinary()?;
                l2p::native_identity::validate(&candidate,&reference);
                eprintln!("[glrmask/validate][boundary_direct_identity] exact_graph=true exact_maps=true tokens={}",vocab.len());
            }
            // Native greedy groups are pairwise incompatible on their live
            // domains or label targets. Taking compatible unions cannot remove
            // these witnesses. Weights are already backward-pushed; by height
            // induction a second identical-policy minimization changes neither
            // weights nor groups. Ignore erasure could invalidate that proof.
            let proof=NativeMinimizationFixedPoint::from_native(&candidate.dwa,ignore_terminal);
            (candidate,proof)
        }
        None=>(ordinary()?,None),
    };
    let finish = Instant::now();
    let dwa = match erase_ignore_after_ti(TerminalAutomaton::Dwa(output.dwa), ignore_terminal) {
        TerminalAutomaton::Dwa(dwa) => dwa,
        TerminalAutomaton::TokenDeterministicNwa(nwa) | TerminalAutomaton::EpsilonNwa(nwa) => {
            crate::automata::weighted::determinize::determinize(&nwa)
                .expect("ordinary ignore-erased boundary NWA must be acyclic")
        }
    };
    output.profile.id_map_ms += map_ms;
    output.profile.terminal_dwa_ms += finish.elapsed().as_secs_f64() * 1000.0;
    if compile_profile_enabled() {
        eprintln!("[glrmask/profile][boundary_identity_refinement] tokens={} states={n} initial={} packed={packed} map_ms={map_ms:.3} total_ms={:.3}",
            vocab.len(), scope.initial_states().len(), started.elapsed().as_secs_f64() * 1000.0);
    }
    Some((MappedArtifact::new(TerminalAutomaton::Dwa(dwa), output.id_map), output.profile, fixed_point))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automata::lexer::ast::{bytes, choice, plus};
    use crate::automata::lexer::compile::{build_regex_monolithic, build_regex_partitioned};

    #[test]
    fn native_fixedpoint_certificate_requires_the_unchanged_small_direct_order() {
        use crate::automata::weighted_u32::dwa::{DWA,DWAState};
        use crate::ds::weight::Weight;
        let w=Weight::from_token_set_for_tsid(0,[0].into_iter().collect());
        let mut rows=vec![DWAState::default();3];rows[0].final_weight=Some(w.clone());
        rows[1].transitions.insert(0,(0,w.clone()));rows[2].transitions.insert(1,(1,w.clone()));
        let good=DWA::from_parts(rows.clone(),2);
        assert!(NativeMinimizationFixedPoint::from_native(&good,None).is_some());
        assert!(NativeMinimizationFixedPoint::from_native(&good,Some(2)).is_none());
        rows[1].transitions.insert(2,(2,w.clone()));
        assert!(NativeMinimizationFixedPoint::from_native(&DWA::from_parts(rows,2),None).is_none());
        assert!(NativeMinimizationFixedPoint::from_native(&DWA::from_parts(vec![DWAState::default();193],0),None).is_none());
        let mut wide=vec![DWAState::default();66];wide[0].final_weight=Some(w.clone());
        for row in &mut wide[1..]{row.transitions.insert(0,(0,w.clone()));}
        assert!(NativeMinimizationFixedPoint::from_native(&DWA::from_parts(wide,65),None).is_none());
    }

    #[test]
    fn initial_coordinate_packing_is_a_total_bijection() {
        for n in [1usize, 2, 7, 65, 513] {
            for stride in 1..6 {
                let selected = (0..n).map(|q| q % stride == stride - 1).collect::<Vec<_>>();
                let map = singleton_state_map(&selected, true);
                let count = selected.iter().filter(|&&x| x).count();
                let mut seen = vec![false; n];
                for (raw, &id) in map.original_to_internal.iter().enumerate() {
                    assert!(!seen[id as usize]); seen[id as usize] = true;
                    assert_eq!(map.representative_original_ids[id as usize], raw as u32);
                    assert_eq!((id as usize) < count, selected[raw]);
                    assert_eq!(map.internal_to_originals[id as usize], vec![raw as u32]);
                }
                assert!(seen.into_iter().all(|x| x));
            }
        }
    }

    #[test]
    fn identity_refinement_matches_ordinary_scoped_languages_with_ignores() {
        let expressions = vec![choice(vec![bytes(b"a"), bytes(b"ab"), bytes(b"ba")]),
            plus(choice(vec![bytes(b"b"), bytes(b"!")])), plus(bytes(b" ")), bytes(b"!a")];
        let rules = vec![Rule { lhs: 0, rhs: vec![Symbol::Nonterminal(1)] }].into_iter()
            .chain((0..4).flat_map(|t| [Rule { lhs: 1, rhs: vec![Symbol::Terminal(t)] },
                Rule { lhs: 1, rhs: vec![Symbol::Terminal(t), Symbol::Nonterminal(1)] }])).collect();
        let grammar = AnalyzedGrammar::from_composed_rules(rules, 4,
            (0..4).map(|t| format!("t{t}")).collect(), vec!["start".into(), "body".into()], 0);
        let mut seed = 244134u64;
        let mut random = || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1); (seed >> 32) as usize };
        let mut direct_cases=0;
        let mut fixed_points=0;
        for partitioned in [false, true] {
            let tokenizer = if partitioned { build_regex_partitioned(&expressions, &[0, 1, 2, 3]) }
                else { build_regex_monolithic(&expressions) }.into_tokenizer(4, Some(expressions.clone().into()));
            let flat: Arc<[u32]> = Arc::from(l1::build_flat_transition_table(&tokenizer));
            for case in 0..32 {
                let mut words = vec![b"ab!".to_vec(), b"ba".to_vec(), b"a b".to_vec(), b"!a".to_vec(), b"ab!".to_vec()];
                for _ in 0..8 { let len = 1 + random() % 5; words.push((0..len).map(|_| b"ab! "[random() % 4]).collect()); }
                let vocab = Vocab::new(words.into_iter().enumerate().map(|(i,w)| ((i * 7) as u32,w)).collect());
                let mut seeds = (0..tokenizer.num_states()).map(|_| random() % 3 != 0).collect::<Vec<_>>(); seeds[0] = true;
                let ignore = (case % 4 == 1).then_some(2);
                let transparent = (case % 4 == 2).then(|| { let mut v = BitSet::new(4); v.set(2); v });
                let scope = scope::BoundaryAnalysisScope::new(
                    scope::InitialStateDomain::from_mask(seeds.len(), seeds).unwrap(),
                    tokenizer.deterministic_reset_states().into_vec(),
                    Arc::new(scope::BoundaryOwnership::flat(&[0,3],4).unwrap()),
                    scope::ImmediateComponentId((case % 2) as u32), case % 3 != 0, transparent).unwrap();
                let mut follows = BTreeMap::new();
                if case % 4 == 3 { let mut v = BitSet::new(4); v.set(3); follows.insert(0, v); }
                let (candidate, _) = build_scoped_boundary_identity_refinement(&tokenizer, &vocab,
                    ignore, &grammar, &follows, Arc::clone(&flat), &scope).unwrap();
                let (reference, _) = build_scoped_boundary_id_map_and_terminal_dwa(&tokenizer,
                    &vocab, &TerminalColoring::identity(4), ignore, &grammar, &follows,
                    Arc::clone(&flat), &scope);
                let parts = |mapped: MappedArtifact<TerminalAutomaton>| {
                    let (automaton, id_map) = mapped.into_parts();
                    let TerminalAutomaton::Dwa(dwa) = automaton else { panic!("DWA expected") };
                    LocalIdMapTerminalDwa { dwa, id_map, profile: Default::default() }
                };
                let candidate=parts(candidate);
                if let Some(mut direct)=l2p::native_identity::try_build(&tokenizer,&vocab,ignore,
                    &grammar,&follows,Some(&flat),&scope,&candidate.id_map){
                    if ignore.is_none(){
                        let twice=crate::automata::weighted::minimize::minimize_owned(direct.dwa.clone());
                        assert_eq!(direct.dwa.start_state(),twice.start_state());
                        assert_eq!(direct.dwa.states(),twice.states(),"native output must be an exact fixed point");
                        fixed_points+=1;
                    }
                    direct.dwa=match erase_ignore_after_ti(TerminalAutomaton::Dwa(direct.dwa),ignore){
                        TerminalAutomaton::Dwa(dwa)=>dwa,
                        TerminalAutomaton::TokenDeterministicNwa(nwa)|TerminalAutomaton::EpsilonNwa(nwa)=>{
                            crate::automata::weighted::determinize::determinize(&nwa).unwrap()
                        }
                    };
                    l2p::native_identity::validate(&direct,&candidate);
                    direct_cases+=1;
                }
                l2p::terminal_dwa_equivalence::compare(&parts(reference), &candidate)
                    .unwrap_or_else(|e| panic!("partitioned={partitioned} case={case}: {e}"));
            }
        }
        assert!(direct_cases>=8,"direct native path not exercised: {direct_cases}");
        assert!(fixed_points>=8,"native fixedpoint not exercised: {fixed_points}");
    }
}
