    use std::cell::RefCell;
    use std::collections::{HashMap, HashSet, VecDeque};
    use std::sync::Arc;

    use rustc_hash::FxHashMap;

    use crate::constraint::{
        GrammarConstraint, GrammarConstraintState, LLMTokenBV, PrecomputeNode1Index,
    };
    use crate::datastructures::gss_leveled_adapter::{
        allow_only_llm_tokens_and_prune_arc, GSSNode,
    };
    use crate::datastructures::hybrid_bitset::HybridBitset;
    use crate::glr::parser::{GLRParser, GLRParserState, ParseStateEdgeContent};
    use crate::glr::table::{NonTerminalID, StateID};
    use crate::types::TerminalID;

    // Types for special precomputation
    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    pub enum SpecialPrecomputeDest {
        Reduce { pop: usize, dest_nt: NonTerminalID },
        Escape { push_states: Vec<StateID> },
    }

    // (Option<NonTerminalID>, StateID, TerminalID, SpecialPrecomputeDest)
    pub type SpecialPrecomputeNormalEdge =
        (Option<NonTerminalID>, StateID, TerminalID, SpecialPrecomputeDest);

    // (Option<NonTerminalID>, TerminalID, (usize, NonTerminalID), LLMTokenBV, PrecomputeNode1Index, PrecomputeNode1Index)
    pub type SpecialPrecomputeSuperEdge = (
        Option<NonTerminalID>,
        TerminalID,
        (usize, NonTerminalID),
        LLMTokenBV,
        PrecomputeNode1Index,
        PrecomputeNode1Index,
    );

    #[derive(Debug, Clone, Default, PartialEq, Eq)]
    pub struct SpecialPrecomputation {
        pub normal_edges: HashSet<SpecialPrecomputeNormalEdge>,
        pub super_edges: HashSet<SpecialPrecomputeSuperEdge>,
    }

    pub fn precompute_special(_gc: &GrammarConstraint) -> SpecialPrecomputation {
        todo!()
    }

    pub fn get_mask4(gcs: &GrammarConstraintState) -> LLMTokenBV {
        // This function implements the mask calculation by traversing the precompute1 trie
        // and simulating the GLR parser using the `special_precomputation` graph.
        // This avoids the need for the large `trie3` and the expensive `process_token` calls.

        let final_mask_internal = RefCell::new(HybridBitset::zeros());
        if gcs.state.is_empty() {
            return gcs
                .parent
                .internal_bv_to_original_precompute1(&final_mask_internal.into_inner());
        }

        // The special precomputation should ideally be cached on the GrammarConstraint.
        // For now, we compute it if it's not present.
        let sp = gcs.parent.precompute_special();

        // Group edges for efficient lookup during traversal.
        let mut edges_by_src_term: FxHashMap<
            (Option<NonTerminalID>, TerminalID),
            Vec<&SpecialPrecomputeNormalEdge>,
        > = FxHashMap::default();
        for edge in &sp.normal_edges {
            edges_by_src_term
                .entry((edge.0, edge.2))
                .or_default()
                .push(edge);
        }

        // Queue for the main traversal over the precompute1 trie.
        // State is (pci1_node_index, glr_parser_state).
        let mut pci1_queue: VecDeque<(PrecomputeNode1Index, GLRParserState)> = VecDeque::new();
        let mut pci1_visited: FxHashMap<(PrecomputeNode1Index, Arc<GSSNode>), ()> =
            FxHashMap::default();

        for (&tokenizer_state_id, glr_state) in &gcs.state {
            if glr_state.active_state.stack.is_empty() {
                continue;
            }
            if let Some(pci1_root) = gcs.parent.precomputed1.get(&tokenizer_state_id) {
                let key = (pci1_root.clone(), glr_state.active_state.stack.clone());
                if pci1_visited.insert(key, ()).is_none() {
                    pci1_queue.push_back((pci1_root.clone(), glr_state.clone()));
                }
            }
        }

        // Helper function to find all GSS nodes that could be the top of an "initial stack"
        // for a special precomputation step.
        fn find_initial_stack_tops<'a>(
            gss_head: &Arc<GSSNode>,
            src_nt: Option<NonTerminalID>,
            parser: &'a GLRParser,
        ) -> Vec<(Arc<GSSNode>, StateID)> {
            let mut results = Vec::new();
            let mut q = vec![gss_head.clone()];
            let mut visited = HashSet::new();

            while let Some(node) = q.pop() {
                if node.is_root() || !visited.insert(node.clone()) {
                    continue;
                }

                if let Some(nt_id) = src_nt {
                    if let Some(preds) = node.get_predecessors() {
                        for pred in preds {
                            if pred.is_root() {
                                continue;
                            }
                            let start_state_id = pred.edge_value().state_id;
                            if let Some(goto_dests) =
                                parser.table.get(&start_state_id).and_then(|s| s.gotos.get(&nt_id))
                            {
                                if goto_dests.contains(&node.edge_value().state_id) {
                                    results.push((node.clone(), start_state_id));
                                }
                            }
                        }
                    }
                } else {
                    // src_nt is None
                    results.push((node.clone(), node.edge_value().state_id));
                }

                if let Some(preds) = node.get_predecessors() {
                    q.extend(preds.iter().cloned());
                }
            }
            results
        }

        while let Some((pci1_idx, glr_state)) = pci1_queue.pop_front() {
            let pci1_node_guard = pci1_idx.read(&gcs.parent.trie1_god).unwrap();

            if pci1_node_guard.value.end {
                *final_mask_internal.borrow_mut() |=
                    &glr_state.active_state.stack.allowed_llm_tokens();
            }

            for (grammar_token_opt, dest_map) in pci1_node_guard.children() {
                for (next_pci1_idx, edge_llm_bv) in dest_map {
                    let mut base_glr_state = glr_state.clone();
                    allow_only_llm_tokens_and_prune_arc(
                        &mut base_glr_state.active_state.stack,
                        edge_llm_bv,
                        &mut HashMap::new(),
                    );
                    if !base_glr_state.is_ok() {
                        continue;
                    }

                    let new_gss_nodes = if let Some(terminal_id) = grammar_token_opt {
                        let mut final_gss_for_terminal: Vec<Arc<GSSNode>> = Vec::new();
                        let mut sp_queue: VecDeque<(Option<NonTerminalID>, Arc<GSSNode>)> =
                            VecDeque::new();
                        let mut sp_visited: FxHashMap<(Option<NonTerminalID>, Arc<GSSNode>), ()> =
                            FxHashMap::default();

                        let initial_key = (None, base_glr_state.active_state.stack.clone());
                        if sp_visited.insert(initial_key.clone(), ()).is_none() {
                            sp_queue.push_back(initial_key);
                        }

                        while let Some((src_nt, gss_head)) = sp_queue.pop_front() {
                            let mut tops_by_start_state: FxHashMap<StateID, Vec<Arc<GSSNode>>> =
                                FxHashMap::default();
                            for (top_node, start_state_id) in
                                find_initial_stack_tops(&gss_head, src_nt, &gcs.parent.parser)
                            {
                                tops_by_start_state
                                    .entry(start_state_id)
                                    .or_default()
                                    .push(top_node);
                            }

                            if let Some(edges) = edges_by_src_term.get(&(src_nt, *terminal_id)) {
                                for edge in edges {
                                    let start_state_id = edge.1;
                                    if let Some(top_nodes) = tops_by_start_state.get(&start_state_id) {
                                        for top_node in top_nodes {
                                            let dest = &edge.3;
                                            let depth_to_pop = if src_nt.is_some() { 2 } else { 1 };
                                            match dest {
                                                SpecialPrecomputeDest::Reduce { pop, dest_nt } => {
                                                    for base_item in top_node.popn(depth_to_pop) {
                                                        for item in base_item.node.popn(*pop) {
                                                            let key = (Some(*dest_nt), item.node.clone());
                                                            if sp_visited.insert(key.clone(), ()).is_none()
                                                            {
                                                                sp_queue.push_back(key);
                                                            }
                                                        }
                                                    }
                                                }
                                                SpecialPrecomputeDest::Escape { push_states } => {
                                                    let mut new_gss_head = top_node.clone();
                                                    for state_id in push_states {
                                                        new_gss_head = GSSNode::with_single_pred(
                                                            new_gss_head,
                                                            ParseStateEdgeContent {
                                                                state_id: *state_id,
                                                                ..Default::default()
                                                            },
                                                        );
                                                    }
                                                    final_gss_for_terminal.push(new_gss_head);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        final_gss_for_terminal
                    } else {
                        vec![base_glr_state.active_state.stack]
                    };

                    if !new_gss_nodes.is_empty() {
                        let merged_gss = GSSNode::merge_many(new_gss_nodes);
                        let mut new_glr_state = base_glr_state.clone();
                        new_glr_state.active_state.stack = merged_gss;
                        if new_glr_state.is_ok() {
                            let key = (*next_pci1_idx, new_glr_state.active_state.stack.clone());
                            if pci1_visited.insert(key, ()).is_none() {
                                pci1_queue.push_back((*next_pci1_idx, new_glr_state));
                            }
                        }
                    }
                }
            }
        }
        gcs.parent
            .internal_bv_to_original_precompute1(&final_mask_internal.into_inner())
    }