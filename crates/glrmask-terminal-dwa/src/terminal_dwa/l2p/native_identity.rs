//! Typed entry to the established native pipeline for proof-trivial identity
//! maps. This does not infer a quotient or weaken the normal L2P contract.
//! The caller has already checked the total singleton lexer-state permutation
//! and byte-identical token groups. Every unsupported case uses ordinary L2P.
use super::*;
use crate::terminal_dwa::scope::BoundaryAnalysisScope;

#[allow(clippy::too_many_arguments)]
pub(in crate::terminal_dwa) fn try_build(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    ignore: Option<TerminalID>,
    grammar: &AnalyzedGrammar,
    follows: &BTreeMap<u32, BitSet>,
    flat: Option<&std::sync::Arc<[u32]>>,
    scope: &BoundaryAnalysisScope,
    id_map: &InternalIdMap,
) -> Option<LocalIdMapTerminalDwa> {
    if !scope.require_crossing()
        || vocab.max_token_byte_len() <= 2
        || std::env::var_os("GLRMASK_SKIP_L2P_MINIMIZE").is_some()
        || tokenizer.num_states() >= 16_384
    {
        return None;
    }
    let started = Instant::now();
    let words = nwa_builder::internal_vocab_entries(vocab, id_map);
    let tree = VocabPrefixTree::build_owned(
        words.into_iter().map(|(id, bytes)| (id as usize, bytes)).collect(),
    );
    let owners = (0..grammar.num_terminals)
        .map(|t| scope.ownership().owner_of_terminal(t).map(|owner| owner.0))
        .collect::<Option<Vec<_>>>()?;
    let mut seed = NWA::new(id_map.num_tsids(), id_map.max_internal_token_id());
    let leaf = seed.add_state();
    seed.set_final_weight(leaf, Weight::all());
    let start = seed.add_state();
    seed.start_states_mut().push(start);
    let roots = nwa_builder::seed_root_nodes_filtered(
        tokenizer, &mut seed, start, id_map, scope.initial_states().keep_raw(),
    );
    let active = vec![true; tokenizer.num_terminals() as usize];
    let coloring = TerminalColoring::identity(active.len());
    let native = nwa_builder::native_builder::build(
        tokenizer, &coloring, ignore, &seed, leaf, id_map.num_tsids(),
        &tree.root, &roots, flat.map(AsRef::as_ref), &active,
    )?;
    let build_ms = native.build_ms;
    let (dwa, post, core) = native.sink.finish(
        follows, grammar.num_terminals as usize, ignore,
        scope.follow_transparent(), &owners, scope.start_component().0,
    )?;
    let elapsed = started.elapsed().as_secs_f64() * 1000.0;
    if l2p_timing_profile_enabled() {
        eprintln!("[glrmask/profile][boundary_direct_identity] selected=true tokens={} states={} total_ms={elapsed:.3} event_ms={build_ms:.3} post={post:?} core={core:?}",
            vocab.len(), tokenizer.num_states());
    }
    Some(LocalIdMapTerminalDwa {
        dwa,
        id_map: id_map.clone(),
        profile: TerminalDwaPhaseProfile {
            terminal_dwa_ms: elapsed,
            determinize_ms: core.import_ms + core.compute_ms,
            minimize_ms: core.min_ms + core.export_ms,
            ..Default::default()
        },
    })
}
#[allow(clippy::too_many_arguments)]
pub(in crate::terminal_dwa) fn try_build_borrowed(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    ignore: Option<TerminalID>,
    grammar: &AnalyzedGrammar,
    follows: &BTreeMap<u32, BitSet>,
    flat: Option<&std::sync::Arc<[u32]>>,
    scope: &BoundaryAnalysisScope,
    id_map: &InternalIdMap,
    borrowed:nwa_builder::native_builder::BorrowedObservation<'_>,
) -> Option<LocalIdMapTerminalDwa> {
    if !scope.require_crossing()
        || vocab.max_token_byte_len() <= 2
        || std::env::var_os("GLRMASK_SKIP_L2P_MINIMIZE").is_some()
        || borrowed.len() >= 16_384
    {
        return None;
    }
    let started = Instant::now();
    let words = nwa_builder::internal_vocab_entries(vocab, id_map);
    let tree = VocabPrefixTree::build_owned(
        words.into_iter().map(|(id, bytes)| (id as usize, bytes)).collect(),
    );
    let owners = (0..grammar.num_terminals)
        .map(|t| scope.ownership().owner_of_terminal(t).map(|owner| owner.0))
        .collect::<Option<Vec<_>>>()?;
    let direct_seed=std::env::var_os("GLRMASK_BOUNDARY_BORROWED_NATIVE_SEED").is_some();
    let mut seed = NWA::new(id_map.num_tsids(), id_map.max_internal_token_id());
    let leaf = seed.add_state();
    seed.set_final_weight(leaf, Weight::all());
    let start = seed.add_state();
    seed.start_states_mut().push(start);
    let mut groups=BTreeMap::<u32,Vec<u32>>::new();
    for (raw,&keep)in scope.initial_states().keep_raw().iter().enumerate(){if keep{
        let class=*id_map.tokenizer_states.original_to_internal.get(raw)?;
        if class!=u32::MAX{groups.entry(class).or_default().push(raw as u32);}
    }}
    let mut roots=nwa_builder::NodesByTokenizerState{entries:Default::default()};
    let mut classes=Vec::with_capacity(groups.len());
    for (class,raws)in groups{
        let node=if direct_seed{classes.len()as u32+2}else{seed.add_state()};classes.push(class);
        if !direct_seed{seed.add_epsilon(start,node,Weight::from_token_set_for_tsid(class,[0..=id_map.max_internal_token_id()].into_iter().collect()));}
        for raw in raws{roots.entries.entry(raw).or_default().push(node);}
    }
    let active = vec![true; tokenizer.num_terminals() as usize];
    let coloring = TerminalColoring::identity(active.len());
    let native = if direct_seed{
        nwa_builder::native_builder::build_borrowed_direct_seed(tokenizer,&coloring,ignore,&classes,id_map.max_internal_token_id(),id_map.num_tsids(),&tree.root,&roots,flat.map(AsRef::as_ref),&active,borrowed)
    }else{nwa_builder::native_builder::build_borrowed(
        tokenizer, &coloring, ignore, &seed, leaf, id_map.num_tsids(),
        &tree.root, &roots, flat.map(AsRef::as_ref), &active,borrowed,
    )}?;
    let build_ms = native.build_ms;
    let (dwa, post, core) = native.sink.finish(
        follows, grammar.num_terminals as usize, ignore,
        scope.follow_transparent(), &owners, scope.start_component().0,
    )?;
    let elapsed = started.elapsed().as_secs_f64() * 1000.0;
    if l2p_timing_profile_enabled() {
        eprintln!("[glrmask/profile][boundary_direct_identity] selected=true tokens={} states={} total_ms={elapsed:.3} event_ms={build_ms:.3} post={post:?} core={core:?}",
            vocab.len(), tokenizer.num_states());
    }
    Some(LocalIdMapTerminalDwa {
        dwa,
        id_map: id_map.clone(),
        profile: TerminalDwaPhaseProfile {
            terminal_dwa_ms: elapsed,
            determinize_ms: core.import_ms + core.compute_ms,
            minimize_ms: core.min_ms + core.export_ms,
            ..Default::default()
        },
    })
}

pub(in crate::terminal_dwa) fn validate(
    candidate: &LocalIdMapTerminalDwa, reference: &LocalIdMapTerminalDwa,
) {
    for (a, b) in [
        (&candidate.id_map.tokenizer_states, &reference.id_map.tokenizer_states),
        (&candidate.id_map.vocab_tokens, &reference.id_map.vocab_tokens),
    ] {
        assert_eq!(a.original_to_internal, b.original_to_internal);
        assert_eq!(a.internal_to_originals, b.internal_to_originals);
        assert_eq!(a.representative_original_ids, b.representative_original_ids);
    }
    assert_eq!(candidate.id_map.deferred_vocab_singleton_original_ids,
        reference.id_map.deferred_vocab_singleton_original_ids);
    native_pipeline::graph_isomorphism(&candidate.dwa, &reference.dwa)
        .expect("direct identity changed the ordinary terminal graph");
}
