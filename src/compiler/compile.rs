#[cfg(test)]
use crate::compiler::grammar::transforms::prepare_grammar_for_compile;
#[cfg(test)]
use crate::Vocab;
#[cfg(test)]
use crate::grammar::flat::GrammarDef;
#[cfg(test)]
use crate::runtime::Constraint;
pub(crate) use super::pipeline::{
    build_tokenizer,
    compile_owned,
    compile_owned_profiled,
    compile_profile_enabled,
    compute_disallowed_follows,
    emit_compile_profile_summary,
};

#[cfg(test)]
pub(crate) use super::pipeline::build_tokenizer_from_exprs;

#[cfg(test)]
pub(crate) fn compile(grammar: &GrammarDef, vocab: &Vocab) -> Constraint {
    let (prepared_grammar, _tokenizer) = prepare_grammar_for_compile(grammar);
    super::pipeline::compile_prepared(prepared_grammar, vocab)
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::automata::regex::Expr;
    use crate::compiler::glr::accumulator::TerminalsDisallowed;
    use crate::compiler::glr::labels::{DEFAULT_LABEL, encode_positive_label};
    use crate::compiler::glr::analysis::AnalyzedGrammar;
    use crate::compiler::glr::parser::{ParserGSS, advance_stacks, stack_may_advance_on};
    use crate::compiler::glr::table::{Action, GLRTable};
    use crate::grammar::flat::tests::*;
    use crate::grammar::flat::{NonterminalID, Rule, Symbol, Terminal};
    use crate::compiler::grammar::transforms::{
        compact_unused_terminals,
        expand_nullable_terminals,
        inline_single_use_nonterminals,
        prepare_grammar_for_compile,
        prepare_owned_grammar_for_compile,
    };
    use crate::compiler::stages::id_map_and_terminal_dwa::l2p::equivalence_analysis::combined::analyze_equivalences;
    use crate::compiler::stages::templates::compile_dfa::Templates;
    use crate::compiler::stages::templates::characterize::{StackMatcher, characterize_terminals};
    use crate::import::json_schema::json_schema_to_grammar;
    use std::collections::BTreeMap;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    fn elapsed_ms(started_at: Instant) -> f64 {
        started_at.elapsed().as_secs_f64() * 1000.0
    }

    #[derive(Debug, Clone, Eq, Hash, PartialEq)]
    struct PmObservableOutputSignature {
        matched_terminals: Vec<u32>,
        is_end: bool,
    }

    #[derive(Debug, Clone, Eq, Hash, PartialEq)]
    struct PmObservableStateSignature {
        output: PmObservableOutputSignature,
        transition_classes: Vec<u32>,
    }

    #[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
    struct SegmentWalkOutcome {
        terminals_id: u32,
        end_state: Option<u32>,
    }

    type DensePossibleMatchMap = BTreeMap<u32, Box<[u64]>>;

    #[derive(Default)]
    struct TerminalSetInterner {
        ids: std::collections::HashMap<Vec<u32>, u32>,
        sets: Vec<Vec<u32>>,
    }

    impl TerminalSetInterner {
        fn intern_slice(&mut self, terminals: &[u32]) -> u32 {
            if let Some(&id) = self.ids.get(terminals) {
                return id;
            }

            let id = self.sets.len() as u32;
            let owned = terminals.to_vec();
            self.ids.insert(owned.clone(), id);
            self.sets.push(owned);
            id
        }

        fn intern_vec(&mut self, terminals: Vec<u32>) -> u32 {
            if let Some(&id) = self.ids.get(&terminals) {
                return id;
            }

            let id = self.sets.len() as u32;
            self.ids.insert(terminals.clone(), id);
            self.sets.push(terminals);
            id
        }

        fn get(&self, id: u32) -> &[u32] {
            &self.sets[id as usize]
        }
    }

    struct BuiltNodeClasses {
        classes: Vec<u32>,
        class_maps: Vec<std::rc::Rc<DensePossibleMatchMap>>,
    }

    fn merge_pm_bitmaps(into: &mut [u64], other: &[u64]) {
        for (lhs, rhs) in into.iter_mut().zip(other.iter()) {
            *lhs |= *rhs;
        }
    }

    fn merge_pm_maps(into: &mut DensePossibleMatchMap, other: &DensePossibleMatchMap, num_words: usize) {
        for (&terminal, bitmap) in other {
            let entry = into
                .entry(terminal)
                .or_insert_with(|| vec![0u64; num_words].into_boxed_slice());
            merge_pm_bitmaps(entry, bitmap);
        }
    }

    fn reachable_bitmap_for_test(
        node: &crate::ds::vocab_prefix_tree::VocabPrefixTreeNode,
        num_words: usize,
    ) -> Box<[u64]> {
        let mut words = vec![0u64; num_words];
        for range in node.reachable_token_ids().ranges() {
            let lo = *range.start() as u32;
            let hi = *range.end() as u32;
            for token_id in lo..=hi {
                words[token_id as usize / 64] |= 1u64 << (token_id % 64);
            }
        }
        words.into_boxed_slice()
    }

    fn compute_pm_observable_tokenizer_classes(tokenizer: &crate::automata::lexer::tokenizer::Tokenizer) -> Vec<u32> {
        let num_states = tokenizer.num_states() as usize;

        let outputs: Vec<PmObservableOutputSignature> = (0..tokenizer.num_states())
            .map(|state| PmObservableOutputSignature {
                matched_terminals: tokenizer.matched_terminals_iter(state).collect(),
                is_end: tokenizer.is_end(state),
            })
            .collect();

        let mut output_classes = std::collections::HashMap::new();
        let mut classes = vec![0u32; num_states];
        for (state, output) in outputs.iter().enumerate() {
            let next_id = output_classes.len() as u32;
            let class_id = *output_classes.entry(output.clone()).or_insert(next_id);
            classes[state] = class_id;
        }

        loop {
            let mut signature_classes = std::collections::HashMap::new();
            let mut next_classes = vec![0u32; num_states];

            for state in 0..tokenizer.num_states() {
                let mut transition_classes = vec![u32::MAX; 256];
                let dfa_state = &tokenizer.dfa.states()[state as usize];
                for (byte, &target) in dfa_state.transitions.iter() {
                    transition_classes[byte as usize] = classes[target as usize];
                }

                let signature = PmObservableStateSignature {
                    output: outputs[state as usize].clone(),
                    transition_classes,
                };

                let next_id = signature_classes.len() as u32;
                let class_id = *signature_classes.entry(signature).or_insert(next_id);
                next_classes[state as usize] = class_id;
            }

            if next_classes == classes {
                return classes;
            }

            classes = next_classes;
        }
    }

    fn compute_trie_pm_root_classes_with_depth(
        tokenizer: &crate::automata::lexer::tokenizer::Tokenizer,
        root: &crate::ds::vocab_prefix_tree::VocabPrefixTreeNode,
        max_depth: Option<usize>,
    ) -> Vec<u32> {
        let matched_terminals: Vec<Vec<u32>> = (0..tokenizer.num_states())
            .map(|state| tokenizer.matched_terminals_iter(state).collect())
            .collect();
        let is_end: Vec<bool> = (0..tokenizer.num_states())
            .map(|state| tokenizer.is_end(state))
            .collect();
        let flat_transitions: Vec<[u32; 256]> = (0..tokenizer.num_states() as usize)
            .map(|state_idx| {
                let dfa_state = &tokenizer.dfa.states()[state_idx];
                let mut flat = [u32::MAX; 256];
                for (byte, &target) in dfa_state.transitions.iter() {
                    flat[byte as usize] = target;
                }
                flat
            })
            .collect();
        let self_loop_bytes: Vec<crate::ds::u8set::U8Set> = (0..tokenizer.num_states() as usize)
            .map(|state_idx| {
                let dfa_state = &tokenizer.dfa.states()[state_idx];
                let mut bytes = crate::ds::u8set::U8Set::empty();
                for (byte, &target) in dfa_state.transitions.iter() {
                    if target == state_idx as u32 {
                        bytes.insert(byte);
                    }
                }
                bytes
            })
            .collect();
        let mut terminal_set_ids = std::collections::HashMap::<Vec<u32>, u32>::new();
        let mut intern_terminal_set = |terminals: &[u32]| {
            let next_id = terminal_set_ids.len() as u32;
            *terminal_set_ids
                .entry(terminals.to_vec())
                .or_insert(next_id)
        };
        let empty_terminals_id = intern_terminal_set(&[]);
        let node_terminal_ids: Vec<u32> = matched_terminals
            .iter()
            .map(|terminals| intern_terminal_set(terminals))
            .collect();
        let mut segment_cache = std::collections::HashMap::<Vec<u8>, usize>::new();
        let mut segment_outcome_tables = Vec::<Vec<SegmentWalkOutcome>>::new();

        fn compute_node_classes(
            node: &crate::ds::vocab_prefix_tree::VocabPrefixTreeNode,
            tokenizer: &crate::automata::lexer::tokenizer::Tokenizer,
            active_states: &[u32],
            remaining_depth: Option<usize>,
            matched_terminals: &[Vec<u32>],
            node_terminal_ids: &[u32],
            empty_terminals_id: u32,
            is_end: &[bool],
            flat_transitions: &[[u32; 256]],
            self_loop_bytes: &[crate::ds::u8set::U8Set],
            terminal_set_ids: &mut std::collections::HashMap<Vec<u32>, u32>,
            segment_cache: &mut std::collections::HashMap<Vec<u8>, usize>,
            segment_outcome_tables: &mut Vec<Vec<SegmentWalkOutcome>>,
        ) -> Vec<u32> {
            let mut child_data = Vec::new();
            for (segment, child) in node.iter_children() {
                let segment_table_idx = if let Some(&table_idx) = segment_cache.get(segment) {
                    table_idx
                } else {
                    let mut outcomes = vec![
                        SegmentWalkOutcome {
                            terminals_id: empty_terminals_id,
                            end_state: None,
                        };
                        tokenizer.num_states() as usize
                    ];

                    for start_state in 0..tokenizer.num_states() {
                        let mut current_state = start_state;
                        let mut blocked = false;
                        let mut encountered_terminals = Vec::new();

                        for &byte in segment {
                            let next_state = flat_transitions[current_state as usize][byte as usize];
                            if next_state == u32::MAX {
                                blocked = true;
                                break;
                            }
                            current_state = next_state;
                            encountered_terminals
                                .extend_from_slice(&matched_terminals[current_state as usize]);
                        }

                        if encountered_terminals.len() > 1 {
                            encountered_terminals.sort_unstable();
                            encountered_terminals.dedup();
                        }

                        let next_id = terminal_set_ids.len() as u32;
                        let terminals_id = *terminal_set_ids
                            .entry(encountered_terminals)
                            .or_insert(next_id);
                        outcomes[start_state as usize] = SegmentWalkOutcome {
                            terminals_id,
                            end_state: (!blocked).then_some(current_state),
                        };
                    }

                    let table_idx = segment_outcome_tables.len();
                    segment_outcome_tables.push(outcomes);
                    segment_cache.insert(segment.to_vec(), table_idx);
                    table_idx
                };

                let child_subtree_bytes = crate::ds::u8set::U8Set::from_words(*child.subtree_bytes());
                let mut child_active_states = Vec::new();
                for &state in active_states {
                    let segment_outcome = segment_outcome_tables[segment_table_idx][state as usize];
                    if let Some(end_state) = segment_outcome.end_state {
                        if !is_end[end_state as usize]
                            && !child_subtree_bytes.is_subset(&self_loop_bytes[end_state as usize])
                        {
                            child_active_states.push(end_state);
                        }
                    }
                }
                child_active_states.sort_unstable();
                child_active_states.dedup();

                let child_classes = if matches!(remaining_depth, Some(0)) {
                    vec![u32::MAX; tokenizer.num_states() as usize]
                } else {
                    compute_node_classes(
                        child,
                        tokenizer,
                        &child_active_states,
                        remaining_depth.map(|depth| depth - 1),
                        matched_terminals,
                        node_terminal_ids,
                        empty_terminals_id,
                        is_end,
                        flat_transitions,
                        self_loop_bytes,
                        terminal_set_ids,
                        segment_cache,
                        segment_outcome_tables,
                    )
                };

                child_data.push((segment_table_idx, child_subtree_bytes, child_classes));
            }

            let mut signature_ids = std::collections::HashMap::new();
            let mut classes = vec![u32::MAX; tokenizer.num_states() as usize];

            for &state in active_states {
                let node_terminals_id = if node.has_token() {
                    node_terminal_ids[state as usize]
                } else {
                    empty_terminals_id
                };

                let mut child_signature_words = Vec::with_capacity(child_data.len() * 2 + 1);
                child_signature_words.push(node_terminals_id);
                for (segment_table_idx, child_subtree_bytes, child_class_ids) in child_data.iter() {
                    let segment_outcome = segment_outcome_tables[*segment_table_idx][state as usize];
                    let child_class_id = if let Some(end_state) = segment_outcome.end_state {
                        if is_end[end_state as usize] || child_subtree_bytes.is_subset(&self_loop_bytes[end_state as usize]) {
                            None
                        } else if matches!(remaining_depth, Some(0)) {
                            Some(0)
                        } else {
                            Some(child_class_ids[end_state as usize])
                        }
                    } else {
                        None
                    };

                    child_signature_words.push(segment_outcome.terminals_id);
                    child_signature_words.push(child_class_id.unwrap_or(u32::MAX));
                }

                let next_id = signature_ids.len() as u32;
                let class_id = *signature_ids.entry(child_signature_words).or_insert(next_id);
                classes[state as usize] = class_id;
            }

            classes
        }

        compute_node_classes(
            root,
            tokenizer,
            &(0..tokenizer.num_states()).collect::<Vec<_>>(),
            max_depth,
            &matched_terminals,
            &node_terminal_ids,
            empty_terminals_id,
            &is_end,
            &flat_transitions,
            &self_loop_bytes,
            &mut terminal_set_ids,
            &mut segment_cache,
            &mut segment_outcome_tables,
        )
    }

    fn compute_trie_pm_root_classes(
        tokenizer: &crate::automata::lexer::tokenizer::Tokenizer,
        root: &crate::ds::vocab_prefix_tree::VocabPrefixTreeNode,
    ) -> Vec<u32> {
        compute_trie_pm_root_classes_with_depth(tokenizer, root, None)
    }

    fn build_trie_pm_root_maps_by_class(
        tokenizer: &crate::automata::lexer::tokenizer::Tokenizer,
        root: &crate::ds::vocab_prefix_tree::VocabPrefixTreeNode,
        num_internal_tokens: u32,
    ) -> BuiltNodeClasses {
        let num_words = (num_internal_tokens as usize + 63) / 64;
        let matched_terminals: Vec<Vec<u32>> = (0..tokenizer.num_states())
            .map(|state| tokenizer.matched_terminals_iter(state).collect())
            .collect();
        let is_end: Vec<bool> = (0..tokenizer.num_states())
            .map(|state| tokenizer.is_end(state))
            .collect();
        let flat_transitions: Vec<[u32; 256]> = (0..tokenizer.num_states() as usize)
            .map(|state_idx| {
                let dfa_state = &tokenizer.dfa.states()[state_idx];
                let mut flat = [u32::MAX; 256];
                for (byte, &target) in dfa_state.transitions.iter() {
                    flat[byte as usize] = target;
                }
                flat
            })
            .collect();
        let self_loop_bytes: Vec<crate::ds::u8set::U8Set> = (0..tokenizer.num_states() as usize)
            .map(|state_idx| {
                let dfa_state = &tokenizer.dfa.states()[state_idx];
                let mut bytes = crate::ds::u8set::U8Set::empty();
                for (byte, &target) in dfa_state.transitions.iter() {
                    if target == state_idx as u32 {
                        bytes.insert(byte);
                    }
                }
                bytes
            })
            .collect();
        let mut terminal_sets = TerminalSetInterner::default();
        let empty_terminals_id = terminal_sets.intern_slice(&[]);
        let node_terminal_ids: Vec<u32> = matched_terminals
            .iter()
            .map(|terminals| terminal_sets.intern_slice(terminals))
            .collect();
        let mut segment_cache = std::collections::HashMap::<Vec<u8>, usize>::new();
        let mut segment_outcome_tables = Vec::<Vec<SegmentWalkOutcome>>::new();

        fn build_node(
            node: &crate::ds::vocab_prefix_tree::VocabPrefixTreeNode,
            tokenizer: &crate::automata::lexer::tokenizer::Tokenizer,
            active_states: &[u32],
            matched_terminals: &[Vec<u32>],
            node_terminal_ids: &[u32],
            empty_terminals_id: u32,
            is_end: &[bool],
            flat_transitions: &[[u32; 256]],
            self_loop_bytes: &[crate::ds::u8set::U8Set],
            terminal_sets: &mut TerminalSetInterner,
            segment_cache: &mut std::collections::HashMap<Vec<u8>, usize>,
            segment_outcome_tables: &mut Vec<Vec<SegmentWalkOutcome>>,
            num_words: usize,
        ) -> BuiltNodeClasses {
            struct ChildBuildData {
                segment_table_idx: usize,
                subtree_bytes: crate::ds::u8set::U8Set,
                reachable: Box<[u64]>,
                result: BuiltNodeClasses,
            }

            let mut child_data = Vec::new();
            for (segment, child) in node.iter_children() {
                let segment_table_idx = if let Some(&table_idx) = segment_cache.get(segment) {
                    table_idx
                } else {
                    let mut outcomes = vec![
                        SegmentWalkOutcome {
                            terminals_id: empty_terminals_id,
                            end_state: None,
                        };
                        tokenizer.num_states() as usize
                    ];

                    for start_state in 0..tokenizer.num_states() {
                        let mut current_state = start_state;
                        let mut blocked = false;
                        let mut encountered_terminals = Vec::new();

                        for &byte in segment {
                            let next_state = flat_transitions[current_state as usize][byte as usize];
                            if next_state == u32::MAX {
                                blocked = true;
                                break;
                            }
                            current_state = next_state;
                            encountered_terminals
                                .extend_from_slice(&matched_terminals[current_state as usize]);
                        }

                        if encountered_terminals.len() > 1 {
                            encountered_terminals.sort_unstable();
                            encountered_terminals.dedup();
                        }

                        let terminals_id = terminal_sets.intern_vec(encountered_terminals);
                        outcomes[start_state as usize] = SegmentWalkOutcome {
                            terminals_id,
                            end_state: (!blocked).then_some(current_state),
                        };
                    }

                    let table_idx = segment_outcome_tables.len();
                    segment_outcome_tables.push(outcomes);
                    segment_cache.insert(segment.to_vec(), table_idx);
                    table_idx
                };

                let subtree_bytes = crate::ds::u8set::U8Set::from_words(*child.subtree_bytes());
                let mut child_active_states = Vec::new();
                for &state in active_states {
                    let segment_outcome = segment_outcome_tables[segment_table_idx][state as usize];
                    if let Some(end_state) = segment_outcome.end_state {
                        if !is_end[end_state as usize]
                            && !subtree_bytes.is_subset(&self_loop_bytes[end_state as usize])
                        {
                            child_active_states.push(end_state);
                        }
                    }
                }
                child_active_states.sort_unstable();
                child_active_states.dedup();

                let result = build_node(
                    child,
                    tokenizer,
                    &child_active_states,
                    matched_terminals,
                    node_terminal_ids,
                    empty_terminals_id,
                    is_end,
                    flat_transitions,
                    self_loop_bytes,
                    terminal_sets,
                    segment_cache,
                    segment_outcome_tables,
                    num_words,
                );

                child_data.push(ChildBuildData {
                    segment_table_idx,
                    subtree_bytes,
                    reachable: reachable_bitmap_for_test(child, num_words),
                    result,
                });
            }

            let mut signature_ids = std::collections::HashMap::<Vec<u32>, u32>::new();
            let mut representative_states = Vec::new();
            let mut classes = vec![u32::MAX; tokenizer.num_states() as usize];

            for &state in active_states {
                let node_terminals_id = if node.has_token() {
                    node_terminal_ids[state as usize]
                } else {
                    empty_terminals_id
                };

                let mut signature_words = Vec::with_capacity(child_data.len() * 2 + 1);
                signature_words.push(node_terminals_id);
                for child in child_data.iter() {
                    let segment_outcome = segment_outcome_tables[child.segment_table_idx][state as usize];
                    let child_class_id = if let Some(end_state) = segment_outcome.end_state {
                        if is_end[end_state as usize]
                            || child.subtree_bytes.is_subset(&self_loop_bytes[end_state as usize])
                        {
                            u32::MAX
                        } else {
                            child.result.classes[end_state as usize]
                        }
                    } else {
                        u32::MAX
                    };
                    signature_words.push(segment_outcome.terminals_id);
                    signature_words.push(child_class_id);
                }

                let next_id = signature_ids.len() as u32;
                let class_id = *signature_ids.entry(signature_words).or_insert_with(|| {
                    representative_states.push(state);
                    next_id
                });
                classes[state as usize] = class_id;
            }

            let mut class_maps = Vec::with_capacity(representative_states.len());
            for &state in representative_states.iter() {
                let mut result = DensePossibleMatchMap::default();

                if node.has_token() {
                    let token_id = node.token_id() as u32;
                    for &terminal in terminal_sets.get(node_terminal_ids[state as usize]) {
                        let entry = result
                            .entry(terminal)
                            .or_insert_with(|| vec![0u64; num_words].into_boxed_slice());
                        entry[token_id as usize / 64] |= 1u64 << (token_id % 64);
                    }
                }

                for child in child_data.iter() {
                    let segment_outcome = segment_outcome_tables[child.segment_table_idx][state as usize];
                    for &terminal in terminal_sets.get(segment_outcome.terminals_id) {
                        let entry = result
                            .entry(terminal)
                            .or_insert_with(|| vec![0u64; num_words].into_boxed_slice());
                        merge_pm_bitmaps(entry, &child.reachable);
                    }

                    if let Some(end_state) = segment_outcome.end_state {
                        if is_end[end_state as usize]
                            || child.subtree_bytes.is_subset(&self_loop_bytes[end_state as usize])
                        {
                            continue;
                        }

                        let child_class_id = child.result.classes[end_state as usize];
                        if child_class_id != u32::MAX {
                            merge_pm_maps(
                                &mut result,
                                child.result.class_maps[child_class_id as usize].as_ref(),
                                num_words,
                            );
                        }
                    }
                }

                class_maps.push(std::rc::Rc::new(result));
            }

            BuiltNodeClasses { classes, class_maps }
        }

        let active_states = (0..tokenizer.num_states()).collect::<Vec<_>>();
        build_node(
            root,
            tokenizer,
            &active_states,
            &matched_terminals,
            &node_terminal_ids,
            empty_terminals_id,
            &is_end,
            &flat_transitions,
            &self_loop_bytes,
            &mut terminal_sets,
            &mut segment_cache,
            &mut segment_outcome_tables,
            num_words,
        )
    }

    fn compute_trie_pm_root_exact_map_classes(
        tokenizer: &crate::automata::lexer::tokenizer::Tokenizer,
        root: &crate::ds::vocab_prefix_tree::VocabPrefixTreeNode,
        num_internal_tokens: u32,
    ) -> BuiltNodeClasses {
        type DensePossibleMatchKey = Vec<(u32, Vec<u64>)>;

        fn dense_map_key(map: &DensePossibleMatchMap) -> DensePossibleMatchKey {
            map.iter()
                .map(|(&terminal, bitmap)| (terminal, bitmap.to_vec()))
                .collect()
        }

        let num_words = (num_internal_tokens as usize + 63) / 64;
        let matched_terminals: Vec<Vec<u32>> = (0..tokenizer.num_states())
            .map(|state| tokenizer.matched_terminals_iter(state).collect())
            .collect();
        let is_end: Vec<bool> = (0..tokenizer.num_states())
            .map(|state| tokenizer.is_end(state))
            .collect();
        let flat_transitions: Vec<[u32; 256]> = (0..tokenizer.num_states() as usize)
            .map(|state_idx| {
                let dfa_state = &tokenizer.dfa.states()[state_idx];
                let mut flat = [u32::MAX; 256];
                for (byte, &target) in dfa_state.transitions.iter() {
                    flat[byte as usize] = target;
                }
                flat
            })
            .collect();
        let self_loop_bytes: Vec<crate::ds::u8set::U8Set> = (0..tokenizer.num_states() as usize)
            .map(|state_idx| {
                let dfa_state = &tokenizer.dfa.states()[state_idx];
                let mut bytes = crate::ds::u8set::U8Set::empty();
                for (byte, &target) in dfa_state.transitions.iter() {
                    if target == state_idx as u32 {
                        bytes.insert(byte);
                    }
                }
                bytes
            })
            .collect();
        let mut terminal_sets = TerminalSetInterner::default();
        let empty_terminals_id = terminal_sets.intern_slice(&[]);
        let node_terminal_ids: Vec<u32> = matched_terminals
            .iter()
            .map(|terminals| terminal_sets.intern_slice(terminals))
            .collect();
        let mut segment_cache = std::collections::HashMap::<Vec<u8>, usize>::new();
        let mut segment_outcome_tables = Vec::<Vec<SegmentWalkOutcome>>::new();

        fn build_node_exact(
            node: &crate::ds::vocab_prefix_tree::VocabPrefixTreeNode,
            tokenizer: &crate::automata::lexer::tokenizer::Tokenizer,
            active_states: &[u32],
            matched_terminals: &[Vec<u32>],
            node_terminal_ids: &[u32],
            empty_terminals_id: u32,
            is_end: &[bool],
            flat_transitions: &[[u32; 256]],
            self_loop_bytes: &[crate::ds::u8set::U8Set],
            terminal_sets: &mut TerminalSetInterner,
            segment_cache: &mut std::collections::HashMap<Vec<u8>, usize>,
            segment_outcome_tables: &mut Vec<Vec<SegmentWalkOutcome>>,
            num_words: usize,
        ) -> BuiltNodeClasses {
            struct ChildBuildData {
                segment_table_idx: usize,
                subtree_bytes: crate::ds::u8set::U8Set,
                reachable: Box<[u64]>,
                result: BuiltNodeClasses,
            }

            let mut child_data = Vec::new();
            for (segment, child) in node.iter_children() {
                let segment_table_idx = if let Some(&table_idx) = segment_cache.get(segment) {
                    table_idx
                } else {
                    let mut outcomes = vec![
                        SegmentWalkOutcome {
                            terminals_id: empty_terminals_id,
                            end_state: None,
                        };
                        tokenizer.num_states() as usize
                    ];

                    for start_state in 0..tokenizer.num_states() {
                        let mut current_state = start_state;
                        let mut blocked = false;
                        let mut encountered_terminals = Vec::new();

                        for &byte in segment {
                            let next_state = flat_transitions[current_state as usize][byte as usize];
                            if next_state == u32::MAX {
                                blocked = true;
                                break;
                            }
                            current_state = next_state;
                            encountered_terminals
                                .extend_from_slice(&matched_terminals[current_state as usize]);
                        }

                        if encountered_terminals.len() > 1 {
                            encountered_terminals.sort_unstable();
                            encountered_terminals.dedup();
                        }

                        let terminals_id = terminal_sets.intern_vec(encountered_terminals);
                        outcomes[start_state as usize] = SegmentWalkOutcome {
                            terminals_id,
                            end_state: (!blocked).then_some(current_state),
                        };
                    }

                    let table_idx = segment_outcome_tables.len();
                    segment_outcome_tables.push(outcomes);
                    segment_cache.insert(segment.to_vec(), table_idx);
                    table_idx
                };

                let subtree_bytes = crate::ds::u8set::U8Set::from_words(*child.subtree_bytes());
                let mut child_active_states = Vec::new();
                for &state in active_states {
                    let segment_outcome = segment_outcome_tables[segment_table_idx][state as usize];
                    if let Some(end_state) = segment_outcome.end_state {
                        if !is_end[end_state as usize]
                            && !subtree_bytes.is_subset(&self_loop_bytes[end_state as usize])
                        {
                            child_active_states.push(end_state);
                        }
                    }
                }
                child_active_states.sort_unstable();
                child_active_states.dedup();

                let result = build_node_exact(
                    child,
                    tokenizer,
                    &child_active_states,
                    matched_terminals,
                    node_terminal_ids,
                    empty_terminals_id,
                    is_end,
                    flat_transitions,
                    self_loop_bytes,
                    terminal_sets,
                    segment_cache,
                    segment_outcome_tables,
                    num_words,
                );

                child_data.push(ChildBuildData {
                    segment_table_idx,
                    subtree_bytes,
                    reachable: reachable_bitmap_for_test(child, num_words),
                    result,
                });
            }

            let mut map_class_ids = std::collections::HashMap::<DensePossibleMatchKey, u32>::new();
            let mut class_maps = Vec::<std::rc::Rc<DensePossibleMatchMap>>::new();
            let mut classes = vec![u32::MAX; tokenizer.num_states() as usize];

            for &state in active_states {
                let mut result = DensePossibleMatchMap::default();

                if node.has_token() {
                    let token_id = node.token_id() as u32;
                    for &terminal in terminal_sets.get(node_terminal_ids[state as usize]) {
                        let entry = result
                            .entry(terminal)
                            .or_insert_with(|| vec![0u64; num_words].into_boxed_slice());
                        entry[token_id as usize / 64] |= 1u64 << (token_id % 64);
                    }
                }

                for child in child_data.iter() {
                    let segment_outcome = segment_outcome_tables[child.segment_table_idx][state as usize];
                    for &terminal in terminal_sets.get(segment_outcome.terminals_id) {
                        let entry = result
                            .entry(terminal)
                            .or_insert_with(|| vec![0u64; num_words].into_boxed_slice());
                        merge_pm_bitmaps(entry, &child.reachable);
                    }

                    if let Some(end_state) = segment_outcome.end_state {
                        if is_end[end_state as usize]
                            || child.subtree_bytes.is_subset(&self_loop_bytes[end_state as usize])
                        {
                            continue;
                        }

                        let child_class_id = child.result.classes[end_state as usize];
                        if child_class_id != u32::MAX {
                            merge_pm_maps(
                                &mut result,
                                child.result.class_maps[child_class_id as usize].as_ref(),
                                num_words,
                            );
                        }
                    }
                }

                let key = dense_map_key(&result);
                let next_id = class_maps.len() as u32;
                let class_id = *map_class_ids.entry(key).or_insert_with(|| {
                    class_maps.push(std::rc::Rc::new(result));
                    next_id
                });
                classes[state as usize] = class_id;
            }

            BuiltNodeClasses { classes, class_maps }
        }

        let active_states = (0..tokenizer.num_states()).collect::<Vec<_>>();
        build_node_exact(
            root,
            tokenizer,
            &active_states,
            &matched_terminals,
            &node_terminal_ids,
            empty_terminals_id,
            &is_end,
            &flat_transitions,
            &self_loop_bytes,
            &mut terminal_sets,
            &mut segment_cache,
            &mut segment_outcome_tables,
            num_words,
        )
    }

    fn mask_has_token(mask: &[u32], token: u32) -> bool {
        let word = token as usize / 32;
        let bit = token as usize % 32;
        word < mask.len() && (mask[word] & (1u32 << bit)) != 0
    }

    fn kb814_normalized_schema_path() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/kb814_normalized_schema.json")
    }

    fn kb814_prepared_terminals_path() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/kb814_prepared_terminals.json")
    }

    fn gpt2_vocab_path() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../grammars2024/benchmarking/gpt2_vocab.json")
    }

    fn llama3_vocab_path() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join(".cache/vocab_cache/llama3_vocab.json")
    }

    fn o1051_schema_path() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join(
            "../constraint-framework-analysis/data/sources/jsonschemabench/data/Github_hard/o1051.json",
        )
    }

    fn decode_hex_bytes(hex: &str) -> Vec<u8> {
        assert_eq!(hex.len() % 2, 0, "hex payload must have even length");
        (0..hex.len())
            .step_by(2)
            .map(|offset| {
                u8::from_str_radix(&hex[offset..offset + 2], 16)
                    .unwrap_or_else(|err| panic!("invalid hex byte at offset {offset}: {err}"))
            })
            .collect()
    }

    fn gpt2_token_str_to_bytes(token_str: &str) -> Vec<u8> {
        token_str.chars().map(unicode_char_to_byte).collect()
    }

    fn unicode_char_to_byte(ch: char) -> u8 {
        if let Some(byte) = printable_byte(ch) {
            return byte;
        }

        let codepoint = ch as u32;
        let offset = codepoint
            .checked_sub(256)
            .expect("unsupported GPT-2 vocab char");
        for byte in 0u16..=255 {
            if printable_byte(char::from_u32(byte as u32).unwrap()).is_none() {
                let candidate_offset = non_printable_rank(byte as u8);
                if candidate_offset == offset as usize {
                    return byte as u8;
                }
            }
        }
        panic!("unable to decode GPT-2 vocab char: {ch:?}");
    }

    fn printable_byte(ch: char) -> Option<u8> {
        let codepoint = ch as u32;
        if (33..=126).contains(&codepoint)
            || (161..=172).contains(&codepoint)
            || (174..=255).contains(&codepoint)
        {
            Some(codepoint as u8)
        } else {
            None
        }
    }

    fn non_printable_rank(target: u8) -> usize {
        let mut rank = 0usize;
        for byte in 0u16..target as u16 {
            let byte = byte as u8;
            if printable_byte(char::from_u32(byte as u32).unwrap()).is_none() {
                rank += 1;
            }
        }
        rank
    }

    fn load_gpt2_vocab() -> Vocab {
        let vocab_path = gpt2_vocab_path();
        let vocab_json = fs::read_to_string(&vocab_path)
            .unwrap_or_else(|err| panic!("failed to read GPT-2 vocab at {}: {err}", vocab_path.display()));
        let vocab_map: serde_json::Map<String, serde_json::Value> =
            serde_json::from_str(&vocab_json).expect("parse GPT-2 vocab json");
        let entries = vocab_map
            .into_iter()
            .map(|(token_str, token_id)| {
                let token_id = token_id.as_u64().expect("token id must be integer") as u32;
                (token_id, gpt2_token_str_to_bytes(&token_str))
            })
            .collect();
        Vocab::new(entries, None)
    }

    fn load_llama3_vocab() -> Vocab {
        let vocab_path = llama3_vocab_path();
        let vocab_json = fs::read_to_string(&vocab_path)
            .unwrap_or_else(|err| panic!("failed to read Llama 3 vocab at {}: {err}", vocab_path.display()));
        let vocab_map: serde_json::Map<String, serde_json::Value> =
            serde_json::from_str(&vocab_json).expect("parse Llama 3 vocab json");
        let entries = vocab_map
            .into_iter()
            .map(|(token_id, token_hex)| {
                let token_id = token_id
                    .parse::<u32>()
                    .unwrap_or_else(|err| panic!("invalid token id {token_id:?}: {err}"));
                let token_hex = token_hex
                    .as_str()
                    .unwrap_or_else(|| panic!("token payload for {token_id} must be a hex string"));
                (token_id, decode_hex_bytes(token_hex))
            })
            .collect();
        Vocab::new(entries, None)
    }

    #[test]
    fn test_compile_simple_ab() {
        let gdef = simple_ab_grammar();
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())], None);

        let constraint = compile(&gdef, &vocab);
        assert!(constraint.parser_dwa.num_states() > 0);
        assert!(!constraint.possible_matches_for_state(0).is_empty());
    }

    #[test]
    fn test_possible_matches_union_covers_all_tokenizer_reachable_tokens() {
        let gdef = simple_ab_grammar();
        let vocab = Vocab::new(
            vec![
                (0, b"a".to_vec()),
                (1, b"b".to_vec()),
                (2, b"ab".to_vec()),
                (3, b"ba".to_vec()),
                (4, b"x".to_vec()),
            ],
            None,
        );

        let constraint = compile(&gdef, &vocab);

        for tokenizer_state in 0..constraint.tokenizer.num_states() {
            let mut expected = std::collections::BTreeSet::new();
            for (token_id, token_bytes) in &vocab.entries {
                let exec = constraint
                    .tokenizer
                    .execute_from_state(token_bytes, tokenizer_state);
                if !exec.matches.is_empty() {
                    expected.insert(*token_id);
                }
            }

            let actual: std::collections::BTreeSet<u32> = constraint
                .possible_matches_for_state(tokenizer_state)
                .values()
                .flat_map(|token_ids| token_ids.iter())
                .collect();

            assert!(
                expected.is_subset(&actual),
                "possible_matches union should cover all tokenizer-reachable tokens for state {} \
                 (expected {:?} ⊆ actual {:?})",
                tokenizer_state,
                expected,
                actual,
            );
        }
    }

    #[test]
    fn test_compile_choice() {
        let gdef = choice_grammar();
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())], None);

        let constraint = compile(&gdef, &vocab);
        assert!(constraint.parser_dwa.num_states() > 0);
    }

    #[test]
    fn test_compile_two_nt() {
        let gdef = two_nt_grammar();
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())], None);

        let constraint = compile(&gdef, &vocab);
        assert!(constraint.parser_dwa.num_states() > 0);
        assert!(constraint.table.num_states > 0);
    }

    #[test]
    fn test_compile_duplicate_token_bytes_expand_back_to_all_original_tokens() {
        let gdef = GrammarDef {
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            start: 0,
            terminals: vec![Terminal::Literal {
                id: 0,
                bytes: b"a".to_vec(),
            }],
            ..Default::default()
        };
        let vocab = Vocab::new(
            vec![
                (10, b"a".to_vec()),
                (20, b"a".to_vec()),
                (30, b"b".to_vec()),
            ],
            None,
        );

        let mask = compile(&gdef, &vocab).start().mask();
        assert!(mask_has_token(&mask, 10));
        assert!(mask_has_token(&mask, 20));
        assert!(!mask_has_token(&mask, 30));
    }

    #[test]
    fn test_compile_duplicate_token_bytes_are_represented_in_constraint_vocab_possible_matches() {
        let gdef = GrammarDef {
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            start: 0,
            terminals: vec![Terminal::Literal {
                id: 0,
                bytes: b"a".to_vec(),
            }],
            ..Default::default()
        };
        let vocab = Vocab::new(
            vec![
                (10, b"a".to_vec()),
                (20, b"a".to_vec()),
                (30, b"b".to_vec()),
            ],
            None,
        );

        let constraint = compile(&gdef, &vocab);
        let tokenizer_state = constraint.tokenizer.initial_state();
        let internal_tokens: std::collections::BTreeSet<u32> = [10u32, 20u32]
            .into_iter()
            .map(|token_id| constraint.internal_token_for_original(token_id))
            .collect();

        let internal_matches: std::collections::BTreeSet<u32> = constraint
            .possible_matches_for_state_internal(tokenizer_state)
            .into_iter()
            .flat_map(|m| m.into_values())
            .flat_map(|token_ids| token_ids.into_iter())
            .collect();
        assert_eq!(internal_matches, internal_tokens);

        let original_matches: std::collections::BTreeSet<u32> = constraint
            .possible_matches_for_state(tokenizer_state)
            .values()
            .flat_map(|token_ids| token_ids.iter())
            .collect();
        assert_eq!(original_matches, std::collections::BTreeSet::from([10, 20]));
    }

    #[test]
    fn test_build_tokenizer_projects_hidden_exclusion_groups() {
        let grammar = GrammarDef {
            rules: vec![],
            start: 0,
            terminals: vec![
                Terminal::Literal {
                    id: 0,
                    bytes: b"a".to_vec(),
                },
                Terminal::Expr {
                    id: 1,
                    expr: Expr::Exclude {
                        expr: Box::new(Expr::U8Class(crate::ds::u8set::U8Set::from_range(0, 255))),
                        exclude: Box::new(Expr::U8Seq(b"a".to_vec())),
                    },
                },
            ],
            ..Default::default()
        };

        let tokenizer = build_tokenizer(&grammar);

        assert_eq!(tokenizer.matched_terminals(tokenizer.run(b"a")), std::collections::BTreeSet::from([0]));
        assert_eq!(tokenizer.matched_terminals(tokenizer.run(b"b")), std::collections::BTreeSet::from([1]));
    }

    #[test]
    fn test_end_to_end_choice() {
        let gdef = choice_grammar();
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())], None);

        let constraint = compile(&gdef, &vocab);
        let mut state = constraint.start();

        let mask = state.mask();
        assert!(mask_has_token(&mask, 0), "token 'a' should be allowed");
        assert!(mask_has_token(&mask, 1), "token 'b' should be allowed");

        state
            .commit_token(0).unwrap();
        assert!(
            state.is_finished(),
            "parse should accept after 'a'"
        );
    }

    #[test]
    fn test_end_to_end_two_nt() {
        let gdef = two_nt_grammar();
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())], None);

        let constraint = compile(&gdef, &vocab);
        let mut state = constraint.start();

        let mask = state.mask();
        assert!(mask_has_token(&mask, 0), "token 'a' should be allowed initially");
        assert!(!mask_has_token(&mask, 1), "token 'b' should NOT be allowed initially");

        state
            .commit_token(0).unwrap();
        assert!(
            !state.is_finished(),
            "not yet accepting after 'a'"
        );

        let mask = state.mask();
        assert!(!mask_has_token(&mask, 0), "token 'a' should NOT be allowed after 'a'");
        assert!(mask_has_token(&mask, 1), "token 'b' should be allowed after 'a'");

        state
            .commit_token(1).unwrap();
        assert!(state.is_finished(), "should accept after 'ab'");
    }

    #[test]
    fn test_end_to_end_nested_nt() {
        let gdef = nested_nt_grammar();
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())], None);

        let constraint = compile(&gdef, &vocab);
        let mut state = constraint.start();

        let mask = state.mask();
        assert!(mask_has_token(&mask, 0), "token 'a' should be allowed initially");
        assert!(!mask_has_token(&mask, 1), "token 'b' should NOT be allowed initially");

        state
            .commit_token(0).unwrap();
        assert!(!state.is_finished(), "not accepting after 'a'");

        let mask = state.mask();
        assert!(!mask_has_token(&mask, 0), "token 'a' should NOT be allowed after 'a'");
        assert!(mask_has_token(&mask, 1), "token 'b' should be allowed after 'a'");

        state
            .commit_token(1).unwrap();
        assert!(state.is_finished(), "should accept after 'ab'");
    }

    #[test]
    fn test_end_to_end_three_terminals() {
        let gdef = three_terminal_grammar();
        let vocab = Vocab::new(
            vec![(0, b"a".to_vec()), (1, b"b".to_vec()), (2, b"c".to_vec())],
            None,
        );

        let constraint = compile(&gdef, &vocab);
        let mut state = constraint.start();

        let mask = state.mask();
        assert!(mask_has_token(&mask, 0), "token 'a' should be allowed initially");
        assert!(!mask_has_token(&mask, 1), "token 'b' should NOT be allowed initially");
        assert!(!mask_has_token(&mask, 2), "token 'c' should NOT be allowed initially");

        state.commit_token(0).unwrap();

        let mask = state.mask();
        assert!(!mask_has_token(&mask, 0), "no 'a' after 'a'");
        assert!(mask_has_token(&mask, 1), "'b' after 'a'");
        assert!(!mask_has_token(&mask, 2), "no 'c' after 'a'");

        state.commit_token(1).unwrap();

        let mask = state.mask();
        assert!(!mask_has_token(&mask, 0), "no 'a' after 'ab'");
        assert!(!mask_has_token(&mask, 1), "no 'b' after 'ab'");
        assert!(mask_has_token(&mask, 2), "'c' after 'ab'");

        state.commit_token(2).unwrap();
        assert!(state.is_finished(), "should accept after 'abc'");
    }

    #[test]
    fn test_end_to_end_nested_two_rhs() {
        let gdef = nested_two_rhs_grammar();
        let vocab = Vocab::new(
            vec![(0, b"a".to_vec()), (1, b"b".to_vec()), (2, b"c".to_vec())],
            None,
        );

        let constraint = compile(&gdef, &vocab);
        let mut state = constraint.start();

        let mask = state.mask();
        assert!(mask_has_token(&mask, 0), "token 'a' should be allowed initially");
        assert!(!mask_has_token(&mask, 1), "token 'b' should NOT be allowed initially");
        assert!(!mask_has_token(&mask, 2), "token 'c' should NOT be allowed initially");

        state.commit_token(0).unwrap();

        let mask = state.mask();
        assert!(!mask_has_token(&mask, 0), "no 'a' after 'a'");
        assert!(mask_has_token(&mask, 1), "'b' after 'a'");
        assert!(!mask_has_token(&mask, 2), "no 'c' after 'a'");

        state.commit_token(1).unwrap();

        let mask = state.mask();
        assert!(!mask_has_token(&mask, 0), "no 'a' after 'ab'");
        assert!(!mask_has_token(&mask, 1), "no 'b' after 'ab'");
        assert!(mask_has_token(&mask, 2), "'c' after 'ab'");

        state.commit_token(2).unwrap();
        assert!(state.is_finished(), "should accept after 'abc'");
    }

    #[test]
    fn test_commit_preserves_longer_terminal_continuation_after_shorter_match() {
        let gdef = GrammarDef {
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(1)],
            }],
            start: 0,
            terminals: vec![
                Terminal::Literal {
                    id: 0,
                    bytes: b"a".to_vec(),
                },
                Terminal::Literal {
                    id: 1,
                    bytes: b"ab".to_vec(),
                },
            ],
            ..Default::default()
        };
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())], None);

        let constraint = compile(&gdef, &vocab);
        let mut state = constraint.start();

        let mask = state.mask();
        assert!(mask_has_token(&mask, 0), "token 'a' should be allowed initially");
        assert!(!mask_has_token(&mask, 1), "token 'b' should not be allowed initially");

        state.commit_token(0).unwrap();
        assert!(
            !state.is_finished(),
            "the shorter literal 'a' should not complete a grammar expecting 'ab'"
        );

        let mask = state.mask();
        assert!(
            mask_has_token(&mask, 1),
            "token 'b' should remain allowed as a continuation of the longer literal 'ab'"
        );

        state.commit_token(1).unwrap();
        assert!(state.is_finished(), "should accept after committing 'ab' byte by byte");
    }

    // ── Nullable terminal expansion tests ───────────────────────────────────

    #[test]
    fn test_expand_nullable_terminals_no_nullables() {
        let gdef = simple_ab_grammar();
        let nullable = std::collections::BTreeSet::new();
        let mut rules = gdef.rules.clone();
        expand_nullable_terminals(&mut rules, &nullable);
        assert_eq!(rules.len(), gdef.rules.len());
        assert_eq!(rules[0].rhs, gdef.rules[0].rhs);
    }

    #[test]
    fn test_expand_nullable_terminals_single_nullable() {
        // Grammar: S → t0 t1, where t0 is nullable.
        // Expected: fresh NT2, S → NT2 t1, NT2 → ε, NT2 → t0
        let gdef = simple_ab_grammar(); // S → T0 T1, nonterminals: {0}
        let nullable = std::collections::BTreeSet::from([0u32]);
        let mut rules = gdef.rules.clone();
        expand_nullable_terminals(&mut rules, &nullable);

        // 1 rewritten original rule + 2 fresh-NT rules = 3 total.
        assert_eq!(rules.len(), 3);

        // The fresh NT id should be grammar.num_nonterminals() = 1.
        let fresh_nt = gdef.num_nonterminals();

        // S → NT_fresh t1
        assert_eq!(rules[0].lhs, 0);
        assert_eq!(
            rules[0].rhs,
            vec![Symbol::Nonterminal(fresh_nt), Symbol::Terminal(1)]
        );

        // NT_fresh → ε and NT_fresh → t0
        let fresh_rules: Vec<&Rule> =
            rules.iter().filter(|r| r.lhs == fresh_nt).collect();
        assert_eq!(fresh_rules.len(), 2);

        let rhs_set: std::collections::BTreeSet<Vec<Symbol>> =
            fresh_rules.iter().map(|r| r.rhs.clone()).collect();
        assert!(rhs_set.contains(&vec![])); // ε
        assert!(rhs_set.contains(&vec![Symbol::Terminal(0)])); // t0
    }

    #[test]
    fn test_expand_nullable_terminals_both_nullable() {
        // Grammar: S → t0 t1, where both are nullable.
        // Expected: fresh NT1 for t0, fresh NT2 for t1.
        // S → NT1 NT2, NT1 → ε | t0, NT2 → ε | t1
        let gdef = simple_ab_grammar();
        let nullable = std::collections::BTreeSet::from([0u32, 1u32]);
        let mut rules = gdef.rules.clone();
        expand_nullable_terminals(&mut rules, &nullable);

        // 1 rewritten rule + 2*2 fresh-NT rules = 5 total.
        assert_eq!(rules.len(), 5);

        let nt0 = gdef.num_nonterminals();     // fresh NT for t0
        let nt1 = gdef.num_nonterminals() + 1; // fresh NT for t1

        // S → NT0 NT1
        assert_eq!(
            rules[0].rhs,
            vec![Symbol::Nonterminal(nt0), Symbol::Nonterminal(nt1)]
        );
    }

    #[test]
    fn test_expand_nullable_terminals_nonterminal_untouched() {
        // Grammar: S → A t1, A → t0. If t0 is nullable:
        //   - Fresh NT for t0.
        //   - S → A t1 unchanged (A is a nonterminal, not touched).
        //   - A → NT_fresh (rewritten from A → t0).
        let gdef = two_nt_grammar(); // S → N1 T1, N1 → T0
        let nullable = std::collections::BTreeSet::from([0u32]);
        let mut rules = gdef.rules.clone();
        expand_nullable_terminals(&mut rules, &nullable);

        let fresh_nt = gdef.num_nonterminals(); // = 2

        // S → N1 T1 — N1 is a nonterminal, not rewritten.
        let s_rules: Vec<&Rule> = rules.iter().filter(|r| r.lhs == 0).collect();
        assert_eq!(s_rules.len(), 1);
        assert_eq!(
            s_rules[0].rhs,
            vec![Symbol::Nonterminal(1), Symbol::Terminal(1)]
        );

        // N1 → NT_fresh (was N1 → T0, T0 is nullable so replaced).
        let n1_rules: Vec<&Rule> = rules.iter().filter(|r| r.lhs == 1).collect();
        assert_eq!(n1_rules.len(), 1);
        assert_eq!(n1_rules[0].rhs, vec![Symbol::Nonterminal(fresh_nt)]);

        // Fresh NT → ε and Fresh NT → T0.
        let fresh_rules: Vec<&Rule> =
            rules.iter().filter(|r| r.lhs == fresh_nt).collect();
        assert_eq!(fresh_rules.len(), 2);
    }

    #[test]
    fn test_expand_nullable_terminals_multiple_occurrences() {
        // Grammar: S → t0 t0, where t0 is nullable.
        // Both occurrences should be replaced by the SAME fresh NT.
        let gdef = GrammarDef {
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0), Symbol::Terminal(0)],
            }],
            start: 0,
            terminals: vec![Terminal::Literal {
                id: 0,
                bytes: b"a".to_vec(),
            }],
            ..Default::default()
        };
        let nullable = std::collections::BTreeSet::from([0u32]);
        let mut rules = gdef.rules.clone();
        expand_nullable_terminals(&mut rules, &nullable);

        let fresh_nt = gdef.num_nonterminals(); // = 1
        // S → NT NT (same fresh NT for both positions) + 2 fresh-NT rules = 3.
        assert_eq!(rules.len(), 3);
        assert_eq!(
            rules[0].rhs,
            vec![Symbol::Nonterminal(fresh_nt), Symbol::Nonterminal(fresh_nt)]
        );
    }

    #[test]
    fn test_drain_nullable_terminals_from_tokenizer() {
        // Build a tokenizer with a nullable terminal (regex `a*` matches empty string).
        let exprs = vec![
            crate::automata::regex::Expr::Repeat {    // nullable: matches ""
                expr: Box::new(Expr::U8Seq(vec![b'a'])),
                min: 0,
                max: None,
            },
            Expr::U8Seq(b"b".to_vec()),                  // not nullable
        ];
        let mut tok = build_tokenizer_from_exprs(&exprs);

        // Before drain: terminal 0 should match at start state.
        assert!(
            tok.matched_terminals(tok.start_state()).contains(&0),
            "terminal 0 should be a start-state finalizer before drain"
        );

        let nullable = tok.isolate_start_state_and_drain_nullable_terminals();
        assert_eq!(nullable, std::collections::BTreeSet::from([0u32]));

        // After drain: terminal 0 should NOT match at start state.
        assert!(
            !tok.matched_terminals(tok.start_state()).contains(&0),
            "terminal 0 should be removed from start-state finalizers after drain"
        );
    }

    #[test]
    fn test_compile_with_nullable_terminal() {
        // S → opt_a b, where opt_a is `a*` (nullable).
        // The grammar should accept both "ab" and "b".
        let gdef = GrammarDef {
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0), Symbol::Terminal(1)],
            }],
            start: 0,
            terminals: vec![
                Terminal::Pattern {
                    id: 0,
                    pattern: "a*".to_string(),
                    utf8: true,
                },
                Terminal::Literal {
                    id: 1,
                    bytes: b"b".to_vec(),
                },
            ],
            ..Default::default()
        };
        let vocab = Vocab::new(
            vec![
                (0, b"a".to_vec()),
                (1, b"b".to_vec()),
                (2, b"aa".to_vec()),
            ],
            None,
        );
        let constraint = compile(&gdef, &vocab);
        assert!(constraint.parser_dwa.num_states() > 0);

        // "b" alone should be accepted (opt_a consumed nothing).
        let state = constraint.start();
        let mask = state.mask();
        assert!(mask_has_token(&mask, 1), "'b' should be allowed initially (opt_a is nullable)");
    }

    #[test]
    fn test_compact_unused_terminals_remaps_rules_and_terminal_ids() {
        let mut grammar = GrammarDef {
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0), Symbol::Terminal(2)],
            }],
            start: 0,
            terminals: vec![
                Terminal::Literal {
                    id: 0,
                    bytes: b"a".to_vec(),
                },
                Terminal::Literal {
                    id: 1,
                    bytes: b"dead".to_vec(),
                },
                Terminal::Literal {
                    id: 2,
                    bytes: b"b".to_vec(),
                },
            ],
            ..Default::default()
        };

        compact_unused_terminals(&mut grammar);

        assert_eq!(
            grammar.rules,
            vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0), Symbol::Terminal(1)],
            }],
            "used terminals should be renumbered densely when a dead terminal is removed from the middle"
        );
        assert_eq!(grammar.terminals.len(), 2);
        assert_eq!(grammar.terminals[0].id(), 0);
        assert_eq!(grammar.terminals[1].id(), 1);
        assert_eq!(grammar.terminals[0].name(), "a");
        assert_eq!(grammar.terminals[1].name(), "b");
        assert_eq!(grammar.ignore_terminal, None);
    }

    #[test]
    fn test_compact_unused_terminals_preserves_ignore_terminal_and_remaps_it() {
        let mut grammar = GrammarDef {
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0), Symbol::Terminal(3)],
            }],
            start: 0,
            terminals: vec![
                Terminal::Literal {
                    id: 0,
                    bytes: b"a".to_vec(),
                },
                Terminal::Literal {
                    id: 1,
                    bytes: b"dead".to_vec(),
                },
                Terminal::Pattern {
                    id: 2,
                    pattern: " +".to_string(),
                    utf8: true,
                },
                Terminal::Literal {
                    id: 3,
                    bytes: b"b".to_vec(),
                },
            ],
            ignore_terminal: Some(2),
            ..Default::default()
        };

        compact_unused_terminals(&mut grammar);

        assert_eq!(
            grammar.rules,
            vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0), Symbol::Terminal(2)],
            }],
            "used terminals should still be renumbered densely when an ignore terminal is retained"
        );
        assert_eq!(grammar.terminals.len(), 3);
        assert_eq!(grammar.terminals[0].name(), "a");
        assert_eq!(grammar.terminals[1].name(), " +");
        assert_eq!(grammar.terminals[2].name(), "b");
        assert_eq!(grammar.ignore_terminal, Some(1));
    }

    #[test]
    fn test_compact_unused_terminals_merges_identical_terminals() {
        // Terminals 0 and 2 are identical ("a"), terminal 1 is different ("b").
        // After compacting, terminals 0 and 2 should map to the same new ID.
        let mut grammar = GrammarDef {
            rules: vec![
                Rule { lhs: 0, rhs: vec![Symbol::Terminal(0), Symbol::Terminal(1)] },
                Rule { lhs: 0, rhs: vec![Symbol::Terminal(2)] },
            ],
            start: 0,
            terminals: vec![
                Terminal::Literal { id: 0, bytes: b"a".to_vec() },
                Terminal::Literal { id: 1, bytes: b"b".to_vec() },
                Terminal::Literal { id: 2, bytes: b"a".to_vec() },
            ],
            nonterminal_names: BTreeMap::new(),
            terminal_names: BTreeMap::new(),
            ignore_terminal: None,
        };
        compact_unused_terminals(&mut grammar);
        assert_eq!(grammar.terminals.len(), 2, "identical terminals should be merged");
        assert_eq!(grammar.terminals[0].name(), "a");
        assert_eq!(grammar.terminals[1].name(), "b");
        // Rule 1: T0 → merged "a" (id 0), T1 → "b" (id 1)
        assert_eq!(grammar.rules[0].rhs, vec![Symbol::Terminal(0), Symbol::Terminal(1)]);
        // Rule 2: T2 → merged "a" (id 0)
        assert_eq!(grammar.rules[1].rhs, vec![Symbol::Terminal(0)]);
    }

    #[test]
    fn test_compile_drops_unused_terminals_before_final_tokenizer_build() {
        let gdef = GrammarDef {
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0), Symbol::Terminal(2)],
            }],
            start: 0,
            terminals: vec![
                Terminal::Literal {
                    id: 0,
                    bytes: b"a".to_vec(),
                },
                Terminal::Pattern {
                    id: 1,
                    pattern: "x*".to_string(),
                    utf8: true,
                },
                Terminal::Literal {
                    id: 2,
                    bytes: b"b".to_vec(),
                },
            ],
            ..Default::default()
        };
        let vocab = Vocab::new(
            vec![
                (0, b"a".to_vec()),
                (1, b"b".to_vec()),
                (2, b"x".to_vec()),
            ],
            None,
        );

        let (normalized, tokenizer) = prepare_grammar_for_compile(&gdef);
        let constraint = compile_owned(gdef, &vocab);

        assert_eq!(
            tokenizer.num_terminals,
            2,
            "the final tokenizer should be built only from the live compacted terminals"
        );
        assert_eq!(normalized.terminals.len(), 2);
        assert_eq!(
            normalized
                .terminals
                .iter()
                .map(|terminal| terminal.name())
                .collect::<Vec<_>>(),
            vec!["a".to_string(), "b".to_string()],
            "the dead middle terminal should be absent from the normalized grammar"
        );
        assert_eq!(
            normalized.rules,
            vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0), Symbol::Terminal(1)],
            }],
            "rules should be remapped to the compacted terminal IDs"
        );

        let mut state = constraint.start();
        let mask = state.mask();
        assert!(mask_has_token(&mask, 0), "token 'a' should still be allowed initially");
        assert!(!mask_has_token(&mask, 1), "token 'b' should not be allowed initially");
        assert!(!mask_has_token(&mask, 2), "dead terminal token 'x' should not leak into the mask");

        state.commit_token(0).unwrap();
        let mask = state.mask();
        assert!(!mask_has_token(&mask, 0), "token 'a' should not be allowed after committing 'a'");
        assert!(mask_has_token(&mask, 1), "token 'b' should remain the live continuation after remapping");
        assert!(!mask_has_token(&mask, 2), "dead terminal token 'x' should remain absent after remapping");
    }

    #[test]
    fn test_compile_treats_ignore_terminal_as_epsilon_and_preserves_it_through_compaction() {
        let gdef = GrammarDef {
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0), Symbol::Terminal(3)],
            }],
            start: 0,
            terminals: vec![
                Terminal::Literal {
                    id: 0,
                    bytes: b"a".to_vec(),
                },
                Terminal::Literal {
                    id: 1,
                    bytes: b"dead".to_vec(),
                },
                Terminal::Pattern {
                    id: 2,
                    pattern: " +".to_string(),
                    utf8: true,
                },
                Terminal::Literal {
                    id: 3,
                    bytes: b"b".to_vec(),
                },
            ],
            ignore_terminal: Some(2),
            ..Default::default()
        };
        let vocab = Vocab::new(
            vec![
                (0, b"a".to_vec()),
                (1, b" ".to_vec()),
                (2, b"b".to_vec()),
                (3, b" a".to_vec()),
                (4, b" b".to_vec()),
            ],
            None,
        );

        let (normalized, _tokenizer) = prepare_grammar_for_compile(&gdef);
        let constraint = compile_owned(gdef, &vocab);

        assert_eq!(constraint.ignore_terminal, Some(1));
        assert_eq!(normalized.terminals.len(), 3);
        assert_eq!(normalized.ignore_terminal, Some(1));
        assert_eq!(
            normalized
                .terminals
                .iter()
                .map(|terminal| terminal.name())
                .collect::<Vec<_>>(),
            vec!["a".to_string(), " +".to_string(), "b".to_string()],
            "the dead terminal should be removed while the ignore terminal is preserved"
        );
        assert_eq!(
            normalized.rules,
            vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0), Symbol::Terminal(2)],
            }],
            "live grammar terminals should be remapped around the retained ignore terminal"
        );

        let mut state = constraint.start();
        let mask = state.mask();
        assert!(mask_has_token(&mask, 0), "token 'a' should be allowed initially");
        assert!(mask_has_token(&mask, 1), "ignore-only token ' ' should be allowed initially");
        assert!(!mask_has_token(&mask, 2), "token 'b' should not be allowed before 'a'");
        assert!(mask_has_token(&mask, 3), "token ' a' should be allowed via ignore+terminal composition");
        assert!(!mask_has_token(&mask, 4), "token ' b' should not be allowed before 'a'");

        state.commit_token(3).unwrap();
        assert!(!state.is_finished(), "consuming ignored space plus 'a' should still leave trailing 'b'");

        let mask = state.mask();
        assert!(!mask_has_token(&mask, 0), "token 'a' should no longer be allowed after 'a'");
        assert!(mask_has_token(&mask, 1), "ignore-only token ' ' should still be allowed between grammar terminals");
        assert!(mask_has_token(&mask, 2), "token 'b' should be allowed after 'a'");
        assert!(!mask_has_token(&mask, 3), "token ' a' should not be allowed once the grammar expects 'b'");
        assert!(mask_has_token(&mask, 4), "token ' b' should be allowed via ignore+terminal composition after 'a'");

        state.commit_token(4).unwrap();
        assert!(state.is_finished(), "consuming ignored space plus 'b' should finish the grammar");
    }

    #[test]
    fn test_prepare_grammar_for_compile_retains_and_remaps_names() {
        let grammar = GrammarDef {
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0), Symbol::Terminal(2)],
            }],
            start: 0,
            terminals: vec![
                Terminal::Literal {
                    id: 0,
                    bytes: b"a".to_vec(),
                },
                Terminal::Literal {
                    id: 1,
                    bytes: b"dead".to_vec(),
                },
                Terminal::Literal {
                    id: 2,
                    bytes: b"b".to_vec(),
                },
            ],
            nonterminal_names: std::collections::BTreeMap::from([(0, "start".to_string())]),
            terminal_names: std::collections::BTreeMap::from([
                (0, "A".to_string()),
                (1, "DEAD".to_string()),
                (2, "B".to_string()),
            ]),
            ignore_terminal: None,
        };

        let (normalized, _tokenizer) = prepare_grammar_for_compile(&grammar);

        assert_eq!(normalized.nonterminal_names.get(&0).map(String::as_str), Some("start"));
        assert_eq!(normalized.terminal_names.get(&0).map(String::as_str), Some("A"));
        assert_eq!(normalized.terminal_names.get(&1).map(String::as_str), Some("B"));
        assert!(!normalized.terminal_names.values().any(|name| name == "DEAD"));
    }

    #[test]
    fn test_inline_single_use_nonterminals_compacts_repetition_tail_chain() {
        let mut rules = vec![
            Rule {
                lhs: 0,
                rhs: vec![Symbol::Nonterminal(3)],
            },
            Rule {
                lhs: 3,
                rhs: vec![Symbol::Terminal(0), Symbol::Nonterminal(1), Symbol::Terminal(1)],
            },
            Rule {
                lhs: 3,
                rhs: vec![
                    Symbol::Terminal(0),
                    Symbol::Nonterminal(1),
                    Symbol::Nonterminal(4),
                    Symbol::Terminal(1),
                ],
            },
            Rule {
                lhs: 4,
                rhs: vec![Symbol::Nonterminal(5)],
            },
            Rule {
                lhs: 4,
                rhs: vec![Symbol::Nonterminal(4), Symbol::Nonterminal(5)],
            },
            Rule {
                lhs: 5,
                rhs: vec![Symbol::Nonterminal(6), Symbol::Nonterminal(7)],
            },
            Rule {
                lhs: 6,
                rhs: vec![Symbol::Terminal(2)],
            },
            Rule {
                lhs: 7,
                rhs: vec![Symbol::Nonterminal(1)],
            },
            Rule {
                lhs: 8,
                rhs: vec![Symbol::Nonterminal(9)],
            },
            Rule {
                lhs: 8,
                rhs: vec![Symbol::Nonterminal(8), Symbol::Nonterminal(9)],
            },
            Rule {
                lhs: 9,
                rhs: vec![Symbol::Nonterminal(6), Symbol::Nonterminal(2)],
            },
            Rule {
                lhs: 10,
                rhs: vec![Symbol::Terminal(3), Symbol::Nonterminal(2), Symbol::Nonterminal(8), Symbol::Terminal(4)],
            },
        ];
        let names = std::collections::BTreeMap::from([
            (0, "start".to_string()),
            (1, "json_kv".to_string()),
            (2, "json_value".to_string()),
            (3, "json_object".to_string()),
            (10, "json_array".to_string()),
        ]);

        let protected: std::collections::BTreeSet<NonterminalID> = names.keys().copied().chain(std::iter::once(0)).collect();

        inline_single_use_nonterminals(&mut rules, &protected);

        assert!(!rules.iter().any(|rule| matches!(rule.lhs, 6 | 7)));
        assert!(rules.contains(&Rule {
            lhs: 5,
            rhs: vec![Symbol::Terminal(2), Symbol::Nonterminal(1)],
        }));
        assert!(rules.contains(&Rule {
            lhs: 4,
            rhs: vec![Symbol::Nonterminal(5)],
        }));
        assert!(rules.contains(&Rule {
            lhs: 4,
            rhs: vec![Symbol::Nonterminal(4), Symbol::Nonterminal(5)],
        }));
        assert!(rules.contains(&Rule {
            lhs: 9,
            rhs: vec![Symbol::Terminal(2), Symbol::Nonterminal(2)],
        }));
        assert!(rules.contains(&Rule {
            lhs: 8,
            rhs: vec![Symbol::Nonterminal(9)],
        }));
        assert!(rules.contains(&Rule {
            lhs: 8,
            rhs: vec![Symbol::Nonterminal(8), Symbol::Nonterminal(9)],
        }));
    }

    #[test]
    #[should_panic]
    fn test_inline_single_use_nonterminals_keeps_multi_symbol_helper_with_multiple_occurrences() {
        let mut rules = vec![
            Rule {
                lhs: 0,
                rhs: vec![Symbol::Nonterminal(1)],
            },
            Rule {
                lhs: 1,
                rhs: vec![Symbol::Nonterminal(2), Symbol::Nonterminal(2)],
            },
            Rule {
                lhs: 2,
                rhs: vec![Symbol::Terminal(0), Symbol::Nonterminal(3)],
            },
            Rule {
                lhs: 3,
                rhs: vec![Symbol::Terminal(1)],
            },
        ];
        let names = std::collections::BTreeMap::from([
            (0, "start".to_string()),
            (1, "root".to_string()),
        ]);
        let protected: std::collections::BTreeSet<NonterminalID> = names.keys().copied().chain(std::iter::once(0)).collect();

        inline_single_use_nonterminals(&mut rules, &protected);

        assert!(rules.iter().any(|rule| rule.lhs == 2));
        assert!(rules.contains(&Rule {
            lhs: 1,
            rhs: vec![Symbol::Nonterminal(2), Symbol::Nonterminal(2)],
        }));
        assert!(rules.contains(&Rule {
            lhs: 2,
            rhs: vec![Symbol::Terminal(0), Symbol::Terminal(1)],
        }));
    }

    #[test]
    #[ignore = "fixture generation for kb814 tokenizer/equivalence benchmarking"]
    fn test_write_kb814_prepared_terminals_fixture() {
        let schema_path = kb814_normalized_schema_path();
        let schema_json = fs::read_to_string(&schema_path)
            .unwrap_or_else(|err| panic!("failed to read {}: {err}", schema_path.display()));
        let grammar = json_schema_to_grammar(&schema_json).expect("kb814 schema should import");
        let (prepared_grammar, _tokenizer) = prepare_owned_grammar_for_compile(grammar);

        let terminals_path = kb814_prepared_terminals_path();
        let payload = serde_json::to_vec(&prepared_grammar.terminals)
            .expect("serialize prepared terminals");
        fs::write(&terminals_path, payload)
            .unwrap_or_else(|err| panic!("failed to write {}: {err}", terminals_path.display()));

        eprintln!(
            "[kb814] wrote_prepared_terminals path={} terminals={}",
            terminals_path.display(),
            prepared_grammar.terminals.len(),
        );
    }

    #[test]
    #[ignore = "kb814 tokenizer/equivalence timing benchmark"]
    fn test_kb814_prepared_terminals_gpt2_timings() {
        let terminals_path = kb814_prepared_terminals_path();
        let terminals_json = fs::read_to_string(&terminals_path)
            .unwrap_or_else(|err| panic!("failed to read {}: {err}", terminals_path.display()));
        let terminals: Vec<Terminal> = serde_json::from_str(&terminals_json)
            .expect("parse prepared terminals json");
        let vocab = load_gpt2_vocab();
        let grammar = GrammarDef {
            terminals,
            ..Default::default()
        };

        let tokenizer_started_at = Instant::now();
        let tokenizer = build_tokenizer(&grammar);
        let tokenizer_ms = elapsed_ms(tokenizer_started_at);
        eprintln!(
            "[kb814] terminals_file={} vocab_file={} terminals={} tokenizer_states={} build_tokenizer_ms={:.3}",
            terminals_path.display(),
            gpt2_vocab_path().display(),
            grammar.terminals.len(),
            tokenizer.num_states(),
            tokenizer_ms,
        );

        unsafe {
            std::env::set_var("GLRMASK_PROFILE_COMPILE", "1");
        }
        let equivalence_started_at = Instant::now();
        let id_map = analyze_equivalences(
            &tokenizer,
            &vocab,
            &std::collections::BTreeMap::new(),
            None,
            None,
        );
        let equivalence_ms = elapsed_ms(equivalence_started_at);
        eprintln!(
            "[kb814] tokenizer_state_classes={} vocab_classes={} equivalence_ms={:.3}",
            id_map.tokenizer_states.internal_to_originals.len(),
            id_map.vocab_tokens.internal_to_originals.len(),
            equivalence_ms,
        );
    }

    #[test]
    #[ignore = "o1051 diagnostic: verify tokenizer-state classes preserve possible_matches"]
    fn test_o1051_llama3_tokenizer_classes_preserve_possible_matches() {
        let vocab = load_llama3_vocab();
        let schema_path = o1051_schema_path();
        let schema_json = fs::read_to_string(&schema_path)
            .unwrap_or_else(|err| panic!("failed to read {}: {err}", schema_path.display()));

        let constraint = crate::Constraint::from_json_schema(&schema_json, &vocab)
            .expect("o1051 should compile with the cached Llama 3 vocab");

        let mut merged_class_count = 0usize;
        let mut merged_state_count = 0usize;
        let mut max_class_size = 0usize;

        for states in &constraint.internal_tsid_to_states {
            if states.len() <= 1 {
                continue;
            }

            merged_class_count += 1;
            merged_state_count += states.len();
            max_class_size = max_class_size.max(states.len());

            let representative = constraint.possible_matches_for_state(states[0]);
            for &state in &states[1..] {
                let actual = constraint.possible_matches_for_state(state);
                assert_eq!(
                    actual,
                    representative,
                    "possible_matches diverged inside tokenizer-state class {} between states {} and {}",
                    constraint.internal_tsid_for_state(states[0]),
                    states[0],
                    state,
                );
            }
        }

        eprintln!(
            "[o1051/possible_matches_equiv] tokenizer_states={} internal_tsids={} merged_classes={} merged_states={} max_class_size={}",
            constraint.state_to_internal_tsid.len(),
            constraint.internal_tsid_to_states.len(),
            merged_class_count,
            merged_state_count,
            max_class_size,
        );
    }

    #[test]
    #[ignore = "o1051 diagnostic: measure PM-valid tokenizer DFA quotient"]
    fn test_o1051_llama3_pm_observable_tokenizer_classes() {
        let schema_path = o1051_schema_path();
        let schema_json = fs::read_to_string(&schema_path)
            .unwrap_or_else(|err| panic!("failed to read {}: {err}", schema_path.display()));
        let schema_value: serde_json::Value = serde_json::from_str(&schema_json)
            .expect("o1051 schema should parse as JSON");
        let schema_payload = schema_value.get("schema").unwrap_or(&schema_value);
        let grammar = json_schema_to_grammar(
            &serde_json::to_string(schema_payload).expect("o1051 schema should serialize"),
        )
            .expect("o1051 schema should import to a grammar");
        let (prepared_grammar, _nullable_terminals) = prepare_grammar_for_compile(&grammar);

        let tokenizer_started_at = Instant::now();
        let tokenizer = build_tokenizer(&prepared_grammar);
        let tokenizer_ms = elapsed_ms(tokenizer_started_at);

        let classes_started_at = Instant::now();
        let classes = compute_pm_observable_tokenizer_classes(&tokenizer);
        let classes_ms = elapsed_ms(classes_started_at);
        let num_classes = classes.iter().copied().max().map_or(0, |id| id + 1);

        eprintln!(
            "[o1051/pm_observable_classes] tokenizer_states={} pm_classes={} shrink={:.2}x tokenizer_ms={:.3} classes_ms={:.3}",
            tokenizer.num_states(),
            num_classes,
            tokenizer.num_states() as f64 / num_classes.max(1) as f64,
            tokenizer_ms,
            classes_ms,
        );
    }

    #[test]
    #[ignore = "o1051 diagnostic: measure trie-specific PM root quotient"]
    fn test_o1051_llama3_trie_specific_pm_root_classes() {
        let vocab = load_llama3_vocab();
        let schema_path = o1051_schema_path();
        let schema_json = fs::read_to_string(&schema_path)
            .unwrap_or_else(|err| panic!("failed to read {}: {err}", schema_path.display()));
        let schema_value: serde_json::Value = serde_json::from_str(&schema_json)
            .expect("o1051 schema should parse as JSON");
        let schema_payload = schema_value.get("schema").unwrap_or(&schema_value);
        let grammar = json_schema_to_grammar(
            &serde_json::to_string(schema_payload).expect("o1051 schema should serialize"),
        )
        .expect("o1051 schema should import to a grammar");
        let (prepared_grammar, _nullable_terminals) = prepare_grammar_for_compile(&grammar);

        let tokenizer_started_at = Instant::now();
        let tokenizer = build_tokenizer(&prepared_grammar);
        let tokenizer_ms = elapsed_ms(tokenizer_started_at);

        let trie_started_at = Instant::now();
        let trie = crate::ds::vocab_prefix_tree::VocabPrefixTree::build_owned(
            vocab.entries
                .iter()
                .map(|(&token_id, bytes)| (token_id as usize, bytes.clone()))
                .collect(),
        );
        let trie_ms = elapsed_ms(trie_started_at);

        let classes_started_at = Instant::now();
        let root_classes = compute_trie_pm_root_classes(&tokenizer, &trie.root);
        let classes_ms = elapsed_ms(classes_started_at);
        let num_classes = root_classes.iter().copied().max().map_or(0, |id| id + 1);

        for max_depth in [Some(0usize), Some(1), Some(2), Some(3)] {
            let localized_started_at = Instant::now();
            let localized_classes = compute_trie_pm_root_classes_with_depth(&tokenizer, &trie.root, max_depth);
            let localized_ms = elapsed_ms(localized_started_at);
            let localized_num_classes = localized_classes
                .iter()
                .copied()
                .max()
                .map_or(0, |id| id + 1);
            eprintln!(
                "[o1051/trie_pm_root_classes/localized] depth={} root_classes={} shrink={:.2}x classes_ms={:.3}",
                max_depth.unwrap(),
                localized_num_classes,
                tokenizer.num_states() as f64 / localized_num_classes.max(1) as f64,
                localized_ms,
            );
        }

        eprintln!(
            "[o1051/trie_pm_root_classes] tokenizer_states={} root_classes={} shrink={:.2}x tokenizer_ms={:.3} trie_ms={:.3} classes_ms={:.3}",
            tokenizer.num_states(),
            num_classes,
            tokenizer.num_states() as f64 / num_classes.max(1) as f64,
            tokenizer_ms,
            trie_ms,
            classes_ms,
        );
    }

    #[test]
    #[ignore = "o1051 diagnostic: compare bottom-up trie PM map build against collector"]
    fn test_o1051_llama3_trie_pm_bottom_up_map_build() {
        let vocab = load_llama3_vocab();
        let schema_path = o1051_schema_path();
        let schema_json = fs::read_to_string(&schema_path)
            .unwrap_or_else(|err| panic!("failed to read {}: {err}", schema_path.display()));
        let schema_value: serde_json::Value = serde_json::from_str(&schema_json)
            .expect("o1051 schema should parse as JSON");
        let schema_payload = schema_value.get("schema").unwrap_or(&schema_value);
        let grammar = json_schema_to_grammar(
            &serde_json::to_string(schema_payload).expect("o1051 schema should serialize"),
        )
        .expect("o1051 schema should import to a grammar");
        let (prepared_grammar, _nullable_terminals) = prepare_grammar_for_compile(&grammar);

        let tokenizer_started_at = Instant::now();
        let tokenizer = build_tokenizer(&prepared_grammar);
        let tokenizer_ms = elapsed_ms(tokenizer_started_at);

        let trie_started_at = Instant::now();
        let trie = crate::ds::vocab_prefix_tree::VocabPrefixTree::build_owned(
            vocab.entries
                .iter()
                .map(|(&token_id, bytes)| (token_id as usize, bytes.clone()))
                .collect(),
        );
        let trie_ms = elapsed_ms(trie_started_at);

        let bottom_up_started_at = Instant::now();
        let root_result = build_trie_pm_root_maps_by_class(
            &tokenizer,
            &trie.root,
            vocab.entries.len() as u32,
        );
        let bottom_up_ms = elapsed_ms(bottom_up_started_at);
        let root_class_count = root_result.class_maps.len();

        let mut representative_states = vec![u32::MAX; root_class_count];
        for state in 0..tokenizer.num_states() {
            let class_id = root_result.classes[state as usize] as usize;
            if representative_states[class_id] == u32::MAX {
                representative_states[class_id] = state;
            }
        }

        let baseline_started_at = Instant::now();
        let (baseline_maps, _profile) = crate::compiler::constraint_possible_matches::collector::collect_possible_matches_by_selected_original_tsid_dense(
            &tokenizer,
            &trie.root,
            vocab.entries.len() as u32,
            &representative_states,
        );
        let baseline_ms = elapsed_ms(baseline_started_at);

        for &state in representative_states.iter() {
            let class_id = root_result.classes[state as usize] as usize;
            assert_eq!(
                baseline_maps.get(&state).expect("collector should produce representative state map"),
                root_result.class_maps[class_id].as_ref(),
                "bottom-up class map diverged from collector for representative state {state}",
            );
        }

        eprintln!(
            "[o1051/trie_pm_bottom_up_maps] tokenizer_states={} root_classes={} reps={} tokenizer_ms={:.3} trie_ms={:.3} baseline_collector_ms={:.3} bottom_up_ms={:.3} speedup={:.2}x",
            tokenizer.num_states(),
            root_class_count,
            representative_states.len(),
            tokenizer_ms,
            trie_ms,
            baseline_ms,
            bottom_up_ms,
            baseline_ms / bottom_up_ms.max(0.001),
        );
    }

    #[test]
    #[ignore = "o1051 diagnostic: compare sampled dense possible-matches collector"]
    fn test_o1051_llama3_dense_possible_matches_collector() {
        let vocab = load_llama3_vocab();
        let schema_path = o1051_schema_path();
        let schema_json = fs::read_to_string(&schema_path)
            .unwrap_or_else(|err| panic!("failed to read {}: {err}", schema_path.display()));
        let schema_value: serde_json::Value = serde_json::from_str(&schema_json)
            .expect("o1051 schema should parse as JSON");
        let schema_payload = schema_value.get("schema").unwrap_or(&schema_value);
        let grammar = json_schema_to_grammar(
            &serde_json::to_string(schema_payload).expect("o1051 schema should serialize"),
        )
        .expect("o1051 schema should import to a grammar");
        let (prepared_grammar, _nullable_terminals) = prepare_grammar_for_compile(&grammar);

        let tokenizer_started_at = Instant::now();
        let tokenizer = build_tokenizer(&prepared_grammar);
        let tokenizer_ms = elapsed_ms(tokenizer_started_at);

        let trie_started_at = Instant::now();
        let trie = crate::ds::vocab_prefix_tree::VocabPrefixTree::build_owned(
            vocab.entries
                .iter()
                .map(|(&token_id, bytes)| (token_id as usize, bytes.clone()))
                .collect(),
        );
        let trie_ms = elapsed_ms(trie_started_at);

        let sample_state_limit = std::env::var("GLRMASK_PM_DIAG_SAMPLE_STATES")
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
            .unwrap_or(8192)
            .min(tokenizer.num_states());
        let entries: Vec<u32> = (0..sample_state_limit).collect();
        let state_chunk_parallel = std::env::var("GLRMASK_PM_STATE_CHUNK_PARALLEL")
            .map_or(false, |value| value == "1");
        let chunk_size = std::env::var("GLRMASK_PM_STATE_CHUNK_SIZE").ok();

        let collector_started_at = Instant::now();
        let (raw_maps, profile) = crate::compiler::constraint_possible_matches::collector::collect_possible_matches_by_selected_original_tsid_dense(
            &tokenizer,
            &trie.root,
            vocab.entries.len() as u32,
            &entries,
        );
        let collector_ms = elapsed_ms(collector_started_at);

        eprintln!(
            "[o1051/dense_possible_matches_collector] state_chunk_parallel={} chunk_size={:?} tokenizer_states={} sampled_states={} states_collected={} tokenizer_ms={:.3} trie_ms={:.3} collector_ms={:.3} root_compute_ms={:.3} materialize_output_ms={:.3}",
            state_chunk_parallel,
            chunk_size,
            tokenizer.num_states(),
            entries.len(),
            raw_maps.len(),
            tokenizer_ms,
            trie_ms,
            collector_ms,
            profile.root_compute_ms,
            profile.materialize_output_ms,
        );
    }

    #[test]
    #[ignore = "o1051 diagnostic: compare trie-class collector against current collector"]
    fn test_o1051_llama3_trie_class_dense_possible_matches_collector() {
        let vocab = load_llama3_vocab();
        let schema_path = o1051_schema_path();
        let schema_json = fs::read_to_string(&schema_path)
            .unwrap_or_else(|err| panic!("failed to read {}: {err}", schema_path.display()));
        let schema_value: serde_json::Value = serde_json::from_str(&schema_json)
            .expect("o1051 schema should parse as JSON");
        let schema_payload = schema_value.get("schema").unwrap_or(&schema_value);
        let grammar = json_schema_to_grammar(
            &serde_json::to_string(schema_payload).expect("o1051 schema should serialize"),
        )
        .expect("o1051 schema should import to a grammar");
        let (prepared_grammar, _nullable_terminals) = prepare_grammar_for_compile(&grammar);

        let tokenizer_started_at = Instant::now();
        let tokenizer = build_tokenizer(&prepared_grammar);
        let tokenizer_ms = elapsed_ms(tokenizer_started_at);

        let trie_started_at = Instant::now();
        let trie = crate::ds::vocab_prefix_tree::VocabPrefixTree::build_owned(
            vocab.entries
                .iter()
                .map(|(&token_id, bytes)| (token_id as usize, bytes.clone()))
                .collect(),
        );
        let trie_ms = elapsed_ms(trie_started_at);

        let sample_state_limit = std::env::var("GLRMASK_PM_DIAG_SAMPLE_STATES")
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
            .unwrap_or(8192)
            .min(tokenizer.num_states());
        let entries: Vec<u32> = (0..sample_state_limit).collect();

        let baseline_started_at = Instant::now();
        let (baseline_maps, baseline_profile) = crate::compiler::constraint_possible_matches::collector::collect_possible_matches_by_selected_original_tsid_dense(
            &tokenizer,
            &trie.root,
            vocab.entries.len() as u32,
            &entries,
        );
        let baseline_ms = elapsed_ms(baseline_started_at);

        let trie_class_started_at = Instant::now();
        let (trie_class_maps, trie_class_profile) = crate::compiler::constraint_possible_matches::collector::collect_possible_matches_dense_trie_class_build(
            &tokenizer,
            &trie.root,
            vocab.entries.len() as u32,
            &entries,
        );
        let trie_class_ms = elapsed_ms(trie_class_started_at);

        assert_eq!(trie_class_maps, baseline_maps);

        eprintln!(
            "[o1051/trie_class_dense_possible_matches_collector] tokenizer_states={} sampled_states={} tokenizer_ms={:.3} trie_ms={:.3} baseline_ms={:.3} trie_class_ms={:.3} speedup={:.2}x baseline_root_compute_ms={:.3} trie_class_root_compute_ms={:.3}",
            tokenizer.num_states(),
            entries.len(),
            tokenizer_ms,
            trie_ms,
            baseline_ms,
            trie_class_ms,
            baseline_ms / trie_class_ms.max(0.001),
            baseline_profile.root_compute_ms,
            trie_class_profile.root_compute_ms,
        );
    }

    #[test]
    #[ignore = "o1051 diagnostic: exact PM-map interning classes"]
    fn test_o1051_llama3_trie_pm_exact_map_classes() {
        let vocab = load_llama3_vocab();
        let schema_path = o1051_schema_path();
        let schema_json = fs::read_to_string(&schema_path)
            .unwrap_or_else(|err| panic!("failed to read {}: {err}", schema_path.display()));
        let schema_value: serde_json::Value = serde_json::from_str(&schema_json)
            .expect("o1051 schema should parse as JSON");
        let schema_payload = schema_value.get("schema").unwrap_or(&schema_value);
        let grammar = json_schema_to_grammar(
            &serde_json::to_string(schema_payload).expect("o1051 schema should serialize"),
        )
        .expect("o1051 schema should import to a grammar");
        let (prepared_grammar, _nullable_terminals) = prepare_grammar_for_compile(&grammar);

        let tokenizer_started_at = Instant::now();
        let tokenizer = build_tokenizer(&prepared_grammar);
        let tokenizer_ms = elapsed_ms(tokenizer_started_at);

        let trie_started_at = Instant::now();
        let trie = crate::ds::vocab_prefix_tree::VocabPrefixTree::build_owned(
            vocab.entries
                .iter()
                .map(|(&token_id, bytes)| (token_id as usize, bytes.clone()))
                .collect(),
        );
        let trie_ms = elapsed_ms(trie_started_at);

        let exact_started_at = Instant::now();
        let exact_result = compute_trie_pm_root_exact_map_classes(
            &tokenizer,
            &trie.root,
            vocab.entries.len() as u32,
        );
        let exact_ms = elapsed_ms(exact_started_at);

        let mut representative_states = vec![u32::MAX; exact_result.class_maps.len()];
        for state in 0..tokenizer.num_states() {
            let class_id = exact_result.classes[state as usize] as usize;
            if representative_states[class_id] == u32::MAX {
                representative_states[class_id] = state;
            }
        }

        let baseline_started_at = Instant::now();
        let (baseline_maps, _profile) = crate::compiler::constraint_possible_matches::collector::collect_possible_matches_by_selected_original_tsid_dense(
            &tokenizer,
            &trie.root,
            vocab.entries.len() as u32,
            &representative_states,
        );
        let baseline_ms = elapsed_ms(baseline_started_at);

        for &state in representative_states.iter() {
            let class_id = exact_result.classes[state as usize] as usize;
            assert_eq!(
                baseline_maps.get(&state).expect("collector should produce representative state map"),
                exact_result.class_maps[class_id].as_ref(),
                "exact PM-map class diverged from collector for representative state {state}",
            );
        }

        eprintln!(
            "[o1051/trie_pm_exact_map_classes] tokenizer_states={} root_classes={} reps={} shrink={:.2}x tokenizer_ms={:.3} trie_ms={:.3} classes_ms={:.3} baseline_reps_ms={:.3}",
            tokenizer.num_states(),
            exact_result.class_maps.len(),
            representative_states.len(),
            tokenizer.num_states() as f64 / exact_result.class_maps.len().max(1) as f64,
            tokenizer_ms,
            trie_ms,
            exact_ms,
            baseline_ms,
        );
    }

    /// Regression test for o76439: structural import with nested closed objects
    /// must accept cross-token terminal matches (e.g., ` {"` after `,`).
    #[test]
    #[ignore]
    fn test_o76439_gpt2_vocab_false_negative() {
        let vocab = load_gpt2_vocab();

        // Actual o76439 schema
        let schema = r#"{
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "ignoreSevertiesAtOrBelow": {
                    "type": "string",
                    "enum": ["negligible", "Negligible", "low", "Low",
                             "medium", "Medium", "high", "High"]
                },
                "vulnerabilities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "cveId": {"type": "string", "minLength": 1, "maxLength": 512},
                            "rationale": {"type": "string", "minLength": 1, "maxLength": 512}
                        },
                        "required": ["cveId", "rationale"],
                        "additionalProperties": false
                    }
                }
            },
            "required": ["ignoreSevertiesAtOrBelow"],
            "additionalProperties": false
        }"#;

        let c = crate::Constraint::from_json_schema(schema, &vocab).unwrap();
        let mut state = c.start();

        // Commit the prefix (token positions 0..49 from the mismatch report)
        let prefix = b"{\"ignoreSevertiesAtOrBelow\": \"Medium\", \"vulnerabilities\": [{\"cveId\": \"CVE-2022-1234\", \"rationale\": \"This vulnerability is not applicable to our system.\"},";
        state.commit_bytes(prefix).expect("prefix should commit");

        let mut state_clone = state.clone();
        state_clone
            .commit_bytes(b" {\"")
            .expect("token bytes ` {\"` should commit after the array-item separator");

        let target_bytes = b" {\"";
        let target_token_id = vocab
            .entries
            .iter()
            .find(|(_, bytes)| bytes.as_slice() == target_bytes)
            .map(|(&id, _)| id)
            .expect("GPT-2 vocab must contain ` {\"`");

        let mask = state.mask();
        assert!(
            mask_has_token(&mask, target_token_id),
            "token {} (` {{\"`) must be in the mask — false negative regression (o76439)",
            target_token_id
        );
    }

    #[test]
    fn test_cross_token_bridge_after_partial_literal_terminal() {
        let gdef = GrammarDef {
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0), Symbol::Terminal(1), Symbol::Terminal(2)],
            }],
            start: 0,
            terminals: vec![
                Terminal::Literal { id: 0, bytes: b"\"".to_vec() },
                Terminal::Literal { id: 1, bytes: b": ".to_vec() },
                Terminal::Literal { id: 2, bytes: b"true".to_vec() },
            ],
            ..Default::default()
        };
        let vocab = Vocab::new(
            vec![
                (0, b"\"".to_vec()),
                (1, b"\":".to_vec()),
                (2, b": ".to_vec()),
                (3, b" true".to_vec()),
                (4, b"true".to_vec()),
            ],
            None,
        );

        let constraint = compile(&gdef, &vocab);
        let mut state = constraint.start();

        let mask = state.mask();
        assert!(
            mask_has_token(&mask, 1),
            "token '\" :' must be allowed so the compile can stop mid-': ' and continue in the next token"
        );

        state.commit_token(1).unwrap();

        let mask = state.mask();
        assert!(
            mask_has_token(&mask, 3),
            "token ' true' must be allowed after '\" :' to bridge ': ' into 'true'"
        );
    }

    #[test]
    fn test_cross_token_bridge_after_complete_literal_terminal() {
        let gdef = GrammarDef {
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0), Symbol::Terminal(1)],
            }],
            start: 0,
            terminals: vec![
                Terminal::Literal { id: 0, bytes: b"}".to_vec() },
                Terminal::Literal { id: 1, bytes: b",".to_vec() },
            ],
            ..Default::default()
        };
        let vocab = Vocab::new(
            vec![
                (0, b"}".to_vec()),
                (1, b",".to_vec()),
                (2, b"},".to_vec()),
            ],
            None,
        );

        let constraint = compile(&gdef, &vocab);
        let mut state = constraint.start();

        let mask = state.mask();
        assert!(mask_has_token(&mask, 0), "token '}}' should be allowed initially");
        assert!(
            mask_has_token(&mask, 2),
            "token '}},' must be allowed to bridge a complete '}}' terminal into the following ',' terminal"
        );

        state.commit_token(2).unwrap();
        assert!(state.is_finished(), "state should finish after committing bridged token '}},'");
    }

    #[test]
    fn test_cross_token_bridge_across_reduction_boundary() {
        let gdef = GrammarDef {
            rules: vec![
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Nonterminal(1), Symbol::Terminal(1), Symbol::Nonterminal(1)],
                },
                Rule {
                    lhs: 1,
                    rhs: vec![Symbol::Terminal(0)],
                },
            ],
            start: 0,
            terminals: vec![
                Terminal::Literal { id: 0, bytes: b"}".to_vec() },
                Terminal::Literal { id: 1, bytes: b",".to_vec() },
            ],
            ..Default::default()
        };
        let vocab = Vocab::new(
            vec![
                (0, b"}".to_vec()),
                (1, b",".to_vec()),
                (2, b"},".to_vec()),
            ],
            None,
        );

        let constraint = compile(&gdef, &vocab);
        let mut state = constraint.start();

        let mask = state.mask();
        assert!(mask_has_token(&mask, 0), "token '}}' should be allowed initially");
        assert!(
            mask_has_token(&mask, 2),
            "token '}},' must be allowed even when the ',' only becomes legal after reducing the preceding '}}' item"
        );

        state.commit_token(2).unwrap();

        let mask = state.mask();
        assert!(mask_has_token(&mask, 0), "token '}}' should remain allowed for the trailing reduced item");

        state.commit_token(0).unwrap();
        assert!(state.is_finished(), "state should finish after the bridged token and final reduced item");
    }

    #[test]
    fn test_cross_token_bridge_across_nullable_inner_chain() {
        let gdef = GrammarDef {
            rules: vec![
                Rule {
                    lhs: 0,
                    rhs: vec![
                        Symbol::Terminal(0),
                        Symbol::Nonterminal(1),
                        Symbol::Terminal(3),
                        Symbol::Terminal(4),
                    ],
                },
                Rule {
                    lhs: 1,
                    rhs: vec![
                        Symbol::Terminal(1),
                        Symbol::Nonterminal(2),
                        Symbol::Terminal(2),
                    ],
                },
                Rule {
                    lhs: 2,
                    rhs: vec![],
                },
                Rule {
                    lhs: 2,
                    rhs: vec![Symbol::Terminal(5)],
                },
            ],
            start: 0,
            terminals: vec![
                Terminal::Literal { id: 0, bytes: b"{\"evt\":".to_vec() },
                Terminal::Literal { id: 1, bytes: b" {".to_vec() },
                Terminal::Literal { id: 2, bytes: b"}".to_vec() },
                Terminal::Literal { id: 3, bytes: b", ".to_vec() },
                Terminal::Literal { id: 4, bytes: b"\"next\":\"x\"}".to_vec() },
                Terminal::Literal { id: 5, bytes: b"\"k\":\"v\"".to_vec() },
            ],
            ..Default::default()
        };
        let vocab = Vocab::new(
            vec![
                (0, b"{\"evt\":".to_vec()),
                (1, b" {},".to_vec()),
                (2, b" \"next\":\"x\"}".to_vec()),
                (3, b" {\"k\":\"v\"},".to_vec()),
            ],
            None,
        );

        let (prepared_grammar, _) = prepare_grammar_for_compile(&gdef);
        assert!(
            prepared_grammar.rules.iter().any(|rule| {
                rule.rhs
                    .windows(2)
                    .any(|window| {
                        window == [Symbol::Terminal(1), Symbol::Terminal(2)]
                            || window == [Symbol::Terminal(2), Symbol::Terminal(3)]
                    })
            }),
            "prepared grammar should expose direct terminal adjacency across the nullable inner chain"
        );

        let constraint = compile(&gdef, &vocab);
        let mut state = constraint.start();
        let prefix_exec = constraint
            .tokenizer
            .execute_from_state(b"{\"evt\":", constraint.tokenizer.initial_state());
        let prefix_state = prefix_exec.end_state.expect("prefix should leave the tokenizer in a live state");
        let _possible_matches = constraint.possible_matches_for_state(prefix_state);

        state
            .commit_bytes(b"{\"evt\":")
            .expect("prefix token should advance the parser state");

        let mask = state.mask();
        assert!(mask_has_token(&mask, 1), "token ' {{}},' should be allowed after the key prefix when the inner object body reduces through epsilon before '}}'");
        assert!(mask_has_token(&mask, 3), "non-empty object token should also remain allowed");

        state.commit_token(1).unwrap();

        let mask = state.mask();
        assert!(mask_has_token(&mask, 2), "after the bridged empty-object token, the trailing sibling token should remain allowed");

        state.commit_token(2).unwrap();
        assert!(state.is_finished(), "state should finish after the bridged empty-object token and trailing sibling token");
    }

    #[test]
    fn test_cross_token_bridge_after_partial_key_prefix_through_nested_nullable_object_chain() {
        let gdef = GrammarDef {
            rules: vec![
                Rule {
                    lhs: 0,
                    rhs: vec![
                        Symbol::Terminal(0),
                        Symbol::Nonterminal(1),
                        Symbol::Terminal(5),
                        Symbol::Terminal(6),
                    ],
                },
                Rule {
                    lhs: 1,
                    rhs: vec![
                        Symbol::Terminal(1),
                        Symbol::Nonterminal(2),
                        Symbol::Terminal(4),
                    ],
                },
                Rule {
                    lhs: 2,
                    rhs: vec![Symbol::Terminal(2), Symbol::Nonterminal(3)],
                },
                Rule {
                    lhs: 2,
                    rhs: vec![Symbol::Nonterminal(4)],
                },
                Rule {
                    lhs: 3,
                    rhs: vec![Symbol::Terminal(3), Symbol::Terminal(7), Symbol::Nonterminal(3)],
                },
                Rule { lhs: 3, rhs: vec![] },
                Rule {
                    lhs: 4,
                    rhs: vec![Symbol::Terminal(7), Symbol::Nonterminal(3)],
                },
                Rule { lhs: 4, rhs: vec![] },
            ],
            start: 0,
            terminals: vec![
                Terminal::Literal { id: 0, bytes: b"{\"onRequestExternal\": ".to_vec() },
                Terminal::Literal { id: 1, bytes: b"{".to_vec() },
                Terminal::Literal { id: 2, bytes: b"\"removeRules\": \"x\"".to_vec() },
                Terminal::Literal { id: 3, bytes: b", ".to_vec() },
                Terminal::Literal { id: 4, bytes: b"}".to_vec() },
                Terminal::Literal { id: 5, bytes: b", ".to_vec() },
                Terminal::Literal { id: 6, bytes: b"\"next\":\"x\"}".to_vec() },
                Terminal::Literal { id: 7, bytes: b"\"extra\": \"y\"".to_vec() },
            ],
            ..Default::default()
        };
        let vocab = Vocab::new(
            vec![
                (0, b"{\"onRequestExternal\":".to_vec()),
                (1, b" {},".to_vec()),
                (2, b" \"next\":\"x\"}".to_vec()),
                (3, b" {\"removeRules\": \"x\"},".to_vec()),
            ],
            None,
        );

        let constraint = compile(&gdef, &vocab);
        let mut state = constraint.start();
        state
            .commit_bytes(b"{\"onRequestExternal\":")
            .expect("partial key-prefix token should advance the parser state");

        let mask = state.mask();
        assert!(
            mask_has_token(&mask, 1),
            "token ' {{}},' should remain allowed when the empty object closes only after a nested nullable chain and the comma-space separator continues in the next token"
        );

        state.commit_token(1).unwrap();
        let mask = state.mask();
        assert!(mask_has_token(&mask, 2), "the trailing sibling token should remain allowed after the bridged empty object token");
    }

    #[test]
    fn test_cross_token_bridge_with_regex_additional_property_alternative() {
        let gdef = GrammarDef {
            rules: vec![
                Rule {
                    lhs: 0,
                    rhs: vec![
                        Symbol::Terminal(0),
                        Symbol::Nonterminal(1),
                        Symbol::Terminal(3),
                        Symbol::Terminal(4),
                    ],
                },
                Rule {
                    lhs: 1,
                    rhs: vec![
                        Symbol::Terminal(1),
                        Symbol::Nonterminal(2),
                        Symbol::Terminal(2),
                    ],
                },
                Rule {
                    lhs: 2,
                    rhs: vec![
                        Symbol::Terminal(5),
                        Symbol::Terminal(6),
                        Symbol::Nonterminal(3),
                    ],
                },
                Rule { lhs: 2, rhs: vec![] },
                Rule {
                    lhs: 3,
                    rhs: vec![
                        Symbol::Terminal(3),
                        Symbol::Terminal(5),
                        Symbol::Terminal(6),
                        Symbol::Nonterminal(3),
                    ],
                },
                Rule { lhs: 3, rhs: vec![] },
            ],
            start: 0,
            terminals: vec![
                Terminal::Literal { id: 0, bytes: b"{\"sendRequest\":\"x\", \"onRequestExternal\": ".to_vec() },
                Terminal::Literal { id: 1, bytes: b"{".to_vec() },
                Terminal::Literal { id: 2, bytes: b"}".to_vec() },
                Terminal::Literal { id: 3, bytes: b", ".to_vec() },
                Terminal::Literal { id: 4, bytes: b"\"tail\":\"y\"}".to_vec() },
                Terminal::Pattern { id: 5, pattern: r#"\"(?:[^\"\\]|\\.)*\": ?"#.to_string(), utf8: false },
                Terminal::Literal { id: 6, bytes: b"\"z\"".to_vec() },
            ],
            ..Default::default()
        };
        let vocab = Vocab::new(
            vec![
                (0, b"{\"sendRequest\":\"x\", \"onRequestExternal\":".to_vec()),
                (1, b" {},".to_vec()),
                (2, b" \"tail\":\"y\"}".to_vec()),
                (3, b" {\"other\":\"z\"},".to_vec()),
            ],
            None,
        );

        let constraint = compile(&gdef, &vocab);
        let mut state = constraint.start();
        state
            .commit_bytes(b"{\"sendRequest\":\"x\", \"onRequestExternal\":")
            .expect("partial key-prefix token should advance the parser state");

        let mask = state.mask();
        assert!(
            mask_has_token(&mask, 1),
            "token ' {{}},' should remain allowed even when the empty object competes with a regex additional-property branch"
        );

        state.commit_token(1).unwrap();
        let mask = state.mask();
        assert!(mask_has_token(&mask, 2), "the trailing sibling token should remain allowed after the bridged empty object token");
    }

    #[test]
    fn test_json_schema_open_ordered_object_keeps_additional_property_continuation_after_shared_optional_ref() {
        let schema = r##"{
            "type": "object",
            "properties": {
                "extension": {
                    "type": "object",
                    "properties": {
                        "req0": { "instanceof": "function" },
                        "req1": { "instanceof": "function" },
                        "onRequest": { "$ref": "#/definitions/Event" },
                        "onRequestExternal": { "$ref": "#/definitions/Event" }
                    },
                    "required": ["req0", "req1"]
                }
            },
            "definitions": {
                "Event": {
                    "type": "object",
                    "properties": {
                        "addListener": { "instanceof": "function" },
                        "addRules": { "instanceof": "function" },
                        "getRules": { "instanceof": "function" },
                        "hasListener": { "instanceof": "function" },
                        "hasListeners": { "instanceof": "function" },
                        "removeListener": { "instanceof": "function" },
                        "removeRules": { "instanceof": "function" }
                    }
                }
            }
        }"##;

        let grammar = json_schema_to_grammar(schema).expect("schema should lower to a grammar");
        let vocab = Vocab::new(vec![(0, b" {},".to_vec())], None);
        let constraint = compile(&grammar, &vocab);
        let mut state = constraint.start();

        state
            .commit_bytes(
                b"{\"extension\": {\"req0\": \"function () {}\", \"req1\": \"function () {}\", \"onRequest\": {\"addListener\": \"function () {}\", \"addRules\": \"function () {}\", \"getRules\": \"function () {}\", \"hasListener\": \"function () {}\", \"hasListeners\": \"function () {}\", \"removeListener\": \"function () {}\", \"removeRules\": \"function () {}\"}, \"onRequestExternal\":"
            )
            .expect("prefix should keep the parser state live");

        let mask = state.mask();
        assert!(
            mask_has_token(&mask, 0),
            "open ordered objects should still allow an empty shared-ref object token followed by a comma when additional properties remain available"
        );

        state
            .commit_token(0)
            .expect("the bridged empty-object token should stay accepted");
    }

    #[test]
    fn test_json_schema_o62060_minimized_empty_object_bridge_up_to_w() {
        const PREFIX: &[u8] = b"{\"a\": 0, \"b\": 0, \"c\":";

        let tail = (b'e'..=b'w')
            .map(|key| format!("\"{}\":{{}}", key as char))
            .collect::<Vec<_>>()
            .join(",");
        let schema = [
            "{\"type\":\"object\",\"properties\":{\"a\":{},\"b\":{},\"d\":{},\"c\":{\"type\":\"object\"},",
            &tail,
            "},\"required\":[\"a\",\"b\",\"e\",\"c\"],\"additionalProperties\":false}",
        ]
            .concat();

        let grammar = json_schema_to_grammar(&schema).expect("schema should lower to a grammar");
        let vocab = Vocab::new(vec![(0u32, b" {},".to_vec())], None);
        let constraint = compile(&grammar, &vocab);
        let mut mask_state = constraint.start();

        mask_state
            .commit_bytes(PREFIX)
            .expect("minimized prefix bytes should advance the parser state");

        let mask = mask_state.mask();
        let mask_accepts = mask_has_token(&mask, 0);

        let mut commit_state = constraint.start();
        commit_state
            .commit_bytes(PREFIX)
            .expect("minimized prefix bytes should advance the parser state");
        let commit_accepts = commit_state.commit_token(0u32).is_ok();

        assert_eq!(
            (mask_accepts, commit_accepts),
            (true, true),
            "token ' {{}},' should remain both masked-in and committable after the minimized o62060 prefix witness"
        );
    }

    #[test]
    fn test_json_schema_o62060_minimized_empty_object_bridge_up_to_x() {
        const PREFIX: &[u8] = b"{\"a\": 0, \"b\": 0, \"c\":";

        let tail = (b'e'..=b'x')
            .map(|key| format!("\"{}\":{{}}", key as char))
            .collect::<Vec<_>>()
            .join(",");
        let schema = [
            "{\"type\":\"object\",\"properties\":{\"a\":{},\"b\":{},\"d\":{},\"c\":{\"type\":\"object\"},",
            &tail,
            "},\"required\":[\"a\",\"b\",\"e\",\"c\"],\"additionalProperties\":false}",
        ]
        .concat();

        let grammar = json_schema_to_grammar(&schema).expect("schema should lower to a grammar");
        let vocab = Vocab::new(vec![(0u32, b" {},".to_vec())], None);
        let constraint = compile(&grammar, &vocab);
        let mut mask_state = constraint.start();

        mask_state
            .commit_bytes(PREFIX)
            .expect("minimized prefix bytes should advance the parser state");

        let mask = mask_state.mask();
        let mask_accepts = mask_has_token(&mask, 0);

        let mut commit_state = constraint.start();
        commit_state
            .commit_bytes(PREFIX)
            .expect("minimized prefix bytes should advance the parser state");
        let commit_accepts = commit_state.commit_token(0u32).is_ok();

        println!("state before: {:?}", mask_state.debug_parser_stacks());
        println!("state after:  {:?}", commit_state.debug_parser_stacks());

        assert_eq!(
            (mask_accepts, commit_accepts),
            (true, true),
            "token ' {{}},' should remain both masked-in and committable after the minimized o62060 prefix witness"
        );
    }

    #[test]
    fn test_json_schema_o62060_minimized_empty_object_bridge_commit_full_prefix_and_comma_space() {
        const FULL_TERMINALS: &[u32] = &[
            7,  // {
            1,  // "
            12, // a"
            6,  // : 
            2,  // JSON_INTEGER
            8,  // , 
            1,  // "
            13, // b"
            6,  // : 
            2,  // JSON_INTEGER
            8,  // , 
            1,  // "
            15, // c"
            6,  // : 
            7,  // {
            9,  // }
            8,  // , 
        ];

        let grammar: GrammarDef = serde_json::from_str(include_str!("../../tests/data/o62060_minimized_empty_object_bridge_grammar.json"))
            .expect("fixture grammar json should parse");
        let (prepared, _tokenizer) = prepare_grammar_for_compile(&grammar);
        let analyzed = AnalyzedGrammar::from_grammar_def(&prepared);

        for (label, table) in [
            ("no_inline", GLRTable::build_with_unit_reduction_inlining(&analyzed, false)),
            ("inline", GLRTable::build_with_unit_reduction_inlining(&analyzed, true)),
        ] {
            let mut state = ParserGSS::from_stacks(&[(vec![0u32], TerminalsDisallowed::new())]);

            for (step, &terminal) in FULL_TERMINALS.iter().enumerate() {
                let before_tops = state.peek_values();
                state = advance_stacks(&table, &state, terminal);
                assert!(
                    !state.is_empty(),
                    "{} direct GLR table drive died at step {} on terminal {} with incoming tops {:?}",
                    label,
                    step,
                    terminal,
                    before_tops,
                );
            }

            assert!(
                !state.is_empty(),
                "{} direct GLR table drive should keep the full minimized terminal witness through ' {{}}, ' alive",
                label,
            );
        }
    }

    #[ignore = "known parser-DWA mask/commit mismatch repro for native JSON-schema required-open-object path"]
    #[test]
    fn test_json_schema_required_open_object_string_token_matches_commit() {
        let schema = r##"{
            "type": "object",
            "properties": {
                "aside": { "type": "boolean" },
                "autoplay": { "type": "boolean" },
                "css_class": {
                    "type": "string",
                    "pattern": "^[\\w\\s-]+$"
                },
                "description": {
                    "type": "string",
                    "minLength": 0,
                    "maxLength": 5000
                }
            },
            "required": ["id"],
            "additionalProperties": true
        }"##;

        let grammar = json_schema_to_grammar(schema).expect("schema should lower to a grammar");
        let vocab = Vocab::new(vec![(0u32, b"'];?>\"".to_vec()), (1u32, b" Vimeo".to_vec())], None);
        let constraint = compile(&grammar, &vocab);

        let mut prefix = Vec::from(
            b"{\"aside\": true, \"autoplay\": false, \"css_class\": \"vimeo-video-block\", \"description\": \"".as_slice(),
        );
        prefix.extend(std::iter::repeat(b"This is a Vimeo video block. ".as_slice()).take(79).flatten().copied());
        prefix.extend_from_slice(b"This is a");

        let mut mask_state = constraint.start();
        mask_state
            .commit_bytes(&prefix)
            .expect("prefix should keep the parser state live");
        let mask = mask_state.mask();
        let mask_accepts = mask_has_token(&mask, 0);

        let mut parser_accepts_candidate = false;
        let mut candidate_terminal_actions = Vec::new();
        for (&tokenizer_state, gss) in &mask_state.state {
            let terminals = constraint.possible_matches_for_state(tokenizer_state);
            for (terminal, tokens) in terminals {
                if !tokens.contains(0) || !stack_may_advance_on(&constraint.table, gss, terminal) {
                    continue;
                }
                parser_accepts_candidate = true;
                let mut actions = Vec::new();
                for parser_state in gss.peek_values() {
                    let action_kind = match constraint.table.action(parser_state, terminal) {
                        Some(Action::Shift(_, _)) => "shift",
                        Some(Action::StackShifts(_)) => "stack_shifts",
                        Some(Action::GuardedStackShifts(_)) => "guarded_stack_shifts",
                        Some(Action::Reduce(_, _)) => "reduce",
                        Some(Action::Split { .. }) => "split",
                        Some(Action::Accept) => "accept",
                        None => continue,
                    };
                    actions.push((parser_state, action_kind));
                }
                candidate_terminal_actions.push((tokenizer_state, terminal, actions));
            }
        }

        let mut commit_state = constraint.start();
        commit_state
            .commit_bytes(&prefix)
            .expect("prefix should keep the parser state live");
        let commit_accepts = commit_state.commit_bytes(b"'];?>\"").is_ok();

        let start_possible_matches = constraint
            .possible_matches_for_state(constraint.tokenizer.initial_state());
        let possible_matches_accept = start_possible_matches
            .values()
            .any(|tokens| tokens.contains(0));

        let (prepared, _tokenizer) = prepare_grammar_for_compile(&grammar);
        let analyzed = AnalyzedGrammar::from_grammar_def(&prepared);
        let table = GLRTable::build(&analyzed);
        let characterizations = characterize_terminals(&table, &analyzed);
        let templates = Templates::from_characterizations(&characterizations);
        let terminal_characterization = characterizations
            .get(&14)
            .expect("terminal 14 characterization should exist");
        let terminal_template = templates
            .by_terminal
            .get(&14)
            .expect("terminal 14 template should exist");
        let stack_shift_action = table.action(115, 14).cloned();
        let has_initial_escape_from_115 = terminal_characterization
            .escapes
            .iter()
            .any(|escape| matches!(escape.pop.first(), Some(StackMatcher::State(115))));
        let internal_tsid = constraint.internal_tsid_for_state(7311);
        let internal_token_0 = constraint.original_token_to_internal[0 as usize];
        let parser_dwa = constraint.parser_dwa();
        let mut parser_dwa_token_0_reachable = false;
        let mut parser_dwa_stack_walks = Vec::new();
        let mut template_accepting_walks = Vec::new();
        if let Some(gss) = mask_state.state.get(&7311) {
            if let Some((chain_states, _acc, _tail)) = gss.extract_chain_and_tail() {
                let mut template_state = terminal_template.start_state;
                let mut template_visited = vec![template_state];
                let mut template_accepting_states = Vec::new();
                let mut template_alive = true;
                for (index, parser_state) in chain_states.iter().copied().enumerate() {
                    if index == 0 && template_state == terminal_template.start_state && parser_state == 0 {
                        continue;
                    }
                    let template_node = &terminal_template.states[template_state as usize];
                    let positive_label = encode_positive_label(parser_state);
                    let Some(&target) = template_node
                        .transitions
                        .get(&positive_label)
                        .or_else(|| template_node.transitions.get(&DEFAULT_LABEL))
                    else {
                        template_alive = false;
                        break;
                    };
                    template_state = target;
                    template_visited.push(template_state);
                    if terminal_template.states[template_state as usize].is_accepting {
                        template_accepting_states.push((parser_state, template_state));
                    }
                }
                template_accepting_walks.push((chain_states.clone(), template_visited, template_accepting_states, template_alive));

                let mut wa_state = parser_dwa.start_state();
                let mut visited_states = vec![wa_state];
                let mut intermediate_final_tokens = Vec::new();
                let mut alive = true;

                for (index, parser_state) in chain_states.iter().copied().enumerate() {
                    if index == 0 && wa_state == parser_dwa.start_state() && parser_state == 0 {
                        continue;
                    }

                    let dwa_state = &parser_dwa.states()[wa_state as usize];
                    let positive_label = encode_positive_label(parser_state);
                    let Some((target, _weight)) = dwa_state
                        .transitions
                        .get(&positive_label)
                        .or_else(|| dwa_state.transitions.get(&DEFAULT_LABEL))
                    else {
                        alive = false;
                        break;
                    };
                    wa_state = *target;
                    visited_states.push(wa_state);

                    let final_tokens = parser_dwa.states()[wa_state as usize]
                        .final_weight
                        .as_ref()
                        .map(|weight| weight.tokens_for_tsid(internal_tsid))
                        .unwrap_or_default();
                    if final_tokens.contains(internal_token_0) {
                        parser_dwa_token_0_reachable = true;
                    }
                    intermediate_final_tokens.push((parser_state, wa_state, final_tokens));
                }

                let final_tokens = if alive {
                    parser_dwa.states()[wa_state as usize]
                        .final_weight
                        .as_ref()
                        .map(|weight| weight.tokens_for_tsid(internal_tsid))
                        .unwrap_or_default()
                } else {
                    Default::default()
                };
                if final_tokens.contains(internal_token_0) {
                    parser_dwa_token_0_reachable = true;
                }
                parser_dwa_stack_walks.push((chain_states, visited_states, intermediate_final_tokens, alive, final_tokens));
            }
        }

        assert!(
            parser_accepts_candidate,
            "expected at least one parser-actionable terminal containing token 0 after the prefix"
        );

        assert_eq!(
            (mask_accepts, commit_accepts),
            (true, true),
            "token b\"'];?>\\\"\" should remain both masked-in and committable; parser_accepts_candidate={parser_accepts_candidate} candidate_terminal_actions={candidate_terminal_actions:?} stack_shift_action={stack_shift_action:?} has_initial_escape_from_115={has_initial_escape_from_115} template_accepting_walks={template_accepting_walks:?} parser_dwa_token_0_reachable={parser_dwa_token_0_reachable} parser_dwa_stack_walks={parser_dwa_stack_walks:?} possible_matches_accept={possible_matches_accept} tokenizer_states={} parser_dwa_states={}",
            constraint.tokenizer.num_states(),
            constraint.parser_dwa().states().len(),
        );
    }

    #[test]
    fn diagnose_pattern_then_max_length_mask_commit_mismatch() {
        let disputed_token_id = 0;
        let disputed_token = b"aaaaeaga";
        let vocab = Vocab::new(vec![(disputed_token_id, disputed_token.to_vec())], None);
        let constraint = Constraint::from_glrm_grammar(
            r#"
            start start;

            internal t CHAR ::= /[^\x00-\x1f\x7f"\\]|\\u[0-9A-Fa-f]{4}/;
            t NUM ::= /0*(\.00+|e0)/;
            t PAT ::= CHAR* [a]{36} CHAR* "\"";
            t BOUNDED ::= CHAR{0,254} "\"";
            nt start ::= PAT (BOUNDED | NUM);
            "#,
            &vocab,
        )
        .expect("grammar should compile");
        let mut prefix = Vec::new();
        prefix.extend([b'a'; 36]);
        prefix.extend(br#"""#);
        prefix.extend([b'a'; 254]);

        let mut mask_state = constraint.start();
        mask_state
            .commit_bytes(&prefix)
            .expect("prefix should be accepted");
        let mask_accepts = mask_has_token(&mask_state.mask(), disputed_token_id);

        let mut commit_bytes_state = constraint.start();
        commit_bytes_state
            .commit_bytes(&prefix)
            .expect("prefix should be accepted");
        let commit_bytes_accepts = commit_bytes_state.commit_bytes(disputed_token).is_ok();

        assert_eq!(
            (mask_accepts, commit_bytes_accepts),
            (false, false),
            "the token would leave the bounded string in a tokenizer state whose future terminal cannot advance the parser stack",
        );
    }

    #[ignore = "diagnostic for the minimized split-boundary mask/commit mismatch"]
    #[test]
    fn diagnose_minimized_split_boundary_mask_commit_mismatch() {
        let vocab = Vocab::new(vec![(0, b"aa\"".to_vec())], None);
        let constraint = Constraint::from_glrm_grammar(r#"
start start;
t A_EXACT ::= "a"{32};
t A_UP_TO_32 ::= "a"{1,2} "\"";
nt start ::= (A_EXACT{4} | A_EXACT{5}) A_UP_TO_32;
"#, &vocab).expect("grammar should compile");

        let prefix = [b'a'; 159];

        let mut mask_state = constraint.start();
        mask_state
            .commit_bytes(&prefix)
            .expect("prefix should keep the parser state live");
        let mask = mask_state.mask();
        let mask_accepts = mask_has_token(&mask, 0);

        let mut commit_state = constraint.start();
        commit_state
            .commit_bytes(&prefix)
            .expect("prefix should keep commit path live");
        let commit_accepts = commit_state.commit_bytes(b"aa\"").is_ok();

        let prefix_possible_matches: Vec<_> = mask_state
            .state
            .iter()
            .map(|(&tokenizer_state, gss)| {
                let terminals = constraint.possible_matches_for_state(tokenizer_state);
                let matching_terminals: Vec<_> = terminals
                    .iter()
                    .filter(|(_, tokens)| tokens.contains(0))
                    .map(|(&terminal, tokens)| {
                        let stack_actionable = stack_may_advance_on(&constraint.table, gss, terminal);
                        let parser_actions: Vec<_> = gss
                            .peek_values()
                            .into_iter()
                            .filter_map(|parser_state| {
                                constraint.table.action(parser_state, terminal).map(|action| {
                                    let label = match action {
                                        Action::Shift(_, _) => "shift",
                                        Action::StackShifts(_) => "stack_shifts",
                                        Action::GuardedStackShifts(_) => "guarded_stack_shifts",
                                        Action::Reduce(_, _) => "reduce",
                                        Action::Split { .. } => "split",
                                        Action::Accept => "accept",
                                    };
                                    (parser_state, label)
                                })
                            })
                            .collect();
                        (terminal, stack_actionable, parser_actions, tokens.clone())
                    })
                    .collect();
                (tokenizer_state, matching_terminals)
            })
            .collect();

        let internal_token_0 = constraint.original_token_to_internal[0usize];
        let parser_dwa = constraint.parser_dwa();
        let mut parser_dwa_walks = Vec::new();
        let mut parser_dwa_token_0_reachable = false;
        for (&tokenizer_state, gss) in &mask_state.state {
            if let Some((chain_states, _acc, _tail)) = gss.extract_chain_and_tail() {
                let internal_tsid = constraint.internal_tsid_for_state(tokenizer_state);
                let mut wa_state = parser_dwa.start_state();
                let mut visited = vec![wa_state];
                let mut finals = Vec::new();
                let mut alive = true;

                for (index, parser_state) in chain_states.iter().copied().enumerate() {
                    if index == 0 && wa_state == parser_dwa.start_state() && parser_state == 0 {
                        continue;
                    }
                    let dwa_state = &parser_dwa.states()[wa_state as usize];
                    let positive_label = encode_positive_label(parser_state);
                    let Some((target, _weight)) = dwa_state
                        .transitions
                        .get(&positive_label)
                        .or_else(|| dwa_state.transitions.get(&DEFAULT_LABEL))
                    else {
                        alive = false;
                        break;
                    };
                    wa_state = *target;
                    visited.push(wa_state);
                    let final_tokens = parser_dwa.states()[wa_state as usize]
                        .final_weight
                        .as_ref()
                        .map(|weight| weight.tokens_for_tsid(internal_tsid))
                        .unwrap_or_default();
                    if final_tokens.contains(internal_token_0) {
                        parser_dwa_token_0_reachable = true;
                    }
                    finals.push((parser_state, wa_state, final_tokens));
                }

                parser_dwa_walks.push((tokenizer_state, chain_states, visited, alive, finals));
            }
        }

        assert_eq!(
            (mask_accepts, commit_accepts),
            (false, true),
            "expected the minimized witness to preserve the mismatch"
        );
        assert!(
            prefix_possible_matches
                .iter()
                .any(|(_, matches)| !matches.is_empty()),
            "expected the live tokenizer state to have at least one possible-match terminal containing token 0; prefix_possible_matches={prefix_possible_matches:?} parser_dwa_walks={parser_dwa_walks:?} parser_dwa_token_0_reachable={parser_dwa_token_0_reachable}"
        );
        assert!(
            !parser_dwa_token_0_reachable,
            "hypothesis failed: token 0 is already parser-DWA reachable; prefix_possible_matches={prefix_possible_matches:?} parser_dwa_walks={parser_dwa_walks:?}"
        );
    }

    #[ignore = "diagnostic for which parser-DWA stage drops the minimized split-boundary token"]
    #[test]
    fn diagnose_minimized_split_boundary_parser_dwa_stage() {
        fn run_case(disable_defaults: bool, disable_subtract_final: bool, disable_minimize: bool) -> (bool, bool) {
            fn set_flag(name: &str, enabled: bool) {
                if enabled {
                    unsafe { std::env::set_var(name, "1"); }
                } else {
                    unsafe { std::env::remove_var(name); }
                }
            }

            set_flag("GLRMASK_DISABLE_PARSER_DWA_DEFAULTS_OPT", disable_defaults);
            set_flag("GLRMASK_DISABLE_PARSER_DWA_SUBTRACT_FINAL", disable_subtract_final);
            set_flag("GLRMASK_DISABLE_PARSER_DWA_MINIMIZE", disable_minimize);

            let vocab = Vocab::new(vec![(0, b"aa\"".to_vec())], None);
            let constraint = Constraint::from_glrm_grammar(r#"
start start;
t A_EXACT ::= "a"{32};
t A_UP_TO_32 ::= "a"{1,2} "\"";
nt start ::= (A_EXACT{4} | A_EXACT{5}) A_UP_TO_32;
"#, &vocab).expect("grammar should compile");
            let prefix = [b'a'; 159];

            let mut mask_state = constraint.start();
            mask_state.commit_bytes(&prefix).unwrap();
            let mask_accepts = mask_has_token(&mask_state.mask(), 0);

            let internal_token_0 = constraint.original_token_to_internal[0usize];
            let parser_dwa = constraint.parser_dwa();
            let mut parser_dwa_token_0_reachable = false;
            for (&tokenizer_state, gss) in &mask_state.state {
                if let Some((chain_states, _acc, _tail)) = gss.extract_chain_and_tail() {
                    let internal_tsid = constraint.internal_tsid_for_state(tokenizer_state);
                    let mut wa_state = parser_dwa.start_state();
                    for (index, parser_state) in chain_states.iter().copied().enumerate() {
                        if index == 0 && wa_state == parser_dwa.start_state() && parser_state == 0 {
                            continue;
                        }
                        let dwa_state = &parser_dwa.states()[wa_state as usize];
                        let positive_label = encode_positive_label(parser_state);
                        let Some((target, _weight)) = dwa_state
                            .transitions
                            .get(&positive_label)
                            .or_else(|| dwa_state.transitions.get(&DEFAULT_LABEL))
                        else {
                            break;
                        };
                        wa_state = *target;
                        let final_tokens = parser_dwa.states()[wa_state as usize]
                            .final_weight
                            .as_ref()
                            .map(|weight| weight.tokens_for_tsid(internal_tsid))
                            .unwrap_or_default();
                        if final_tokens.contains(internal_token_0) {
                            parser_dwa_token_0_reachable = true;
                            break;
                        }
                    }
                }
            }

            set_flag("GLRMASK_DISABLE_PARSER_DWA_DEFAULTS_OPT", false);
            set_flag("GLRMASK_DISABLE_PARSER_DWA_SUBTRACT_FINAL", false);
            set_flag("GLRMASK_DISABLE_PARSER_DWA_MINIMIZE", false);

            (mask_accepts, parser_dwa_token_0_reachable)
        }

        let cases = [
            ("baseline", false, false, false),
            ("no_defaults", true, false, false),
            ("no_subtract_final", false, true, false),
            ("no_minimize", false, false, true),
            ("no_defaults_no_subtract", true, true, false),
            ("no_defaults_no_minimize", true, false, true),
            ("no_subtract_no_minimize", false, true, true),
            ("all_disabled", true, true, true),
        ];

        let mut results = Vec::new();
        for (label, disable_defaults, disable_subtract_final, disable_minimize) in cases {
            let (mask_accepts, parser_dwa_token_0_reachable) =
                run_case(disable_defaults, disable_subtract_final, disable_minimize);
            results.push((label, mask_accepts, parser_dwa_token_0_reachable));
        }

        assert_eq!(
            results,
            vec![
                ("baseline", false, false),
                ("no_defaults", false, false),
                ("no_subtract_final", false, false),
                ("no_minimize", false, false),
                ("no_defaults_no_subtract", false, false),
                ("no_defaults_no_minimize", false, false),
                ("no_subtract_no_minimize", false, false),
                ("all_disabled", false, false),
            ],
            "the token is already lost before defaults optimization, final-weight subtraction, and minimization; results={results:?}"
        );
    }

}
