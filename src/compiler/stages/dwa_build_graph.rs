//! Typed composition primitives for terminal-DWA and parser-DWA builds.
//!
//! This module deliberately does not choose a topology, read an environment
//! variable, or parse a configuration language. A caller explicitly takes
//! terminal leaves, merges terminal points, converts them to parser-NWA points,
//! then merges parser-NWA points. Parser points stay NWAs until the root because
//! parser-DWA DEFAULT_LABEL/fallback normalization is not an ordinary union.

use std::collections::BTreeMap;

use crate::automata::weighted::nwa::{NWA, NwaBody};
use crate::automata::weighted::terminal_automaton::TerminalAutomaton;
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::stages::id_map_and_terminal_dwa::merge::merge_id_maps_and_terminal_dwas;
use crate::compiler::stages::id_map_and_terminal_dwa::types::{
    LocalIdMapTerminalDwa, TerminalDwaBranch, compile_profile_enabled,
};
use crate::compiler::stages::id_map_and_terminal_dwa::{TerminalDwaLeaf, TerminalDwaLeaves};
use crate::compiler::stages::mapped_artifact::MappedArtifact;
use crate::compiler::stages::parser_dwa::build_parser_nwa_from_terminal_dwa_with_precomputed_templates;
use crate::compiler::stages::templates::Templates;

pub(crate) type TerminalDwaPoint = MappedArtifact<TerminalAutomaton>;
pub(crate) type ParserNwaPoint = MappedArtifact<NWA>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct TerminalDwaLeafKey {
    pub(crate) partition: usize,
    pub(crate) branch: TerminalDwaBranch,
}

/// Owns terminal leaves until a caller deliberately consumes them. A leaf can
/// be consumed exactly once, so duplicate or omitted subtrees fail immediately.
pub(crate) struct DwaBuildPoints {
    available: BTreeMap<TerminalDwaLeafKey, LocalIdMapTerminalDwa>,
    num_tokenizer_states: usize,
    max_token_id: u32,
}

impl DwaBuildPoints {
    pub(crate) fn from_terminal_leaves(leaves: TerminalDwaLeaves) -> Self {
        let mut available = BTreeMap::new();
        for TerminalDwaLeaf { partition, branch, output } in leaves.leaves {
            let key = TerminalDwaLeafKey { partition, branch };
            assert!(
                available.insert(key, output).is_none(),
                "duplicate terminal leaf {}",
                format_key(key),
            );
        }
        Self {
            available,
            num_tokenizer_states: leaves.num_tokenizer_states,
            max_token_id: leaves.max_token_id,
        }
    }

    pub(crate) fn take_branch(
        &mut self,
        partition: usize,
        branch: TerminalDwaBranch,
    ) -> TerminalDwaPoint {
        self.take_keys([TerminalDwaLeafKey { partition, branch }])
    }

    pub(crate) fn take_partition(&mut self, partition: usize) -> TerminalDwaPoint {
        let keys = self
            .available
            .keys()
            .filter(|key| key.partition == partition)
            .copied()
            .collect::<Vec<_>>();
        self.take_keys(keys)
    }

    pub(crate) fn take_all(&mut self) -> TerminalDwaPoint {
        self.take_keys(self.available.keys().copied().collect::<Vec<_>>())
    }

    /// Merge previously-built terminal points using this leaf collection's
    /// tokenizer metadata. This is the primitive for a hand-written merge tree.
    pub(crate) fn merge_terminal_points(
        &self,
        inputs: impl IntoIterator<Item = TerminalDwaPoint>,
    ) -> TerminalDwaPoint {
        merge_terminal_points(inputs, self.num_tokenizer_states, self.max_token_id)
    }

    pub(crate) fn remaining_leaf_keys(&self) -> Vec<TerminalDwaLeafKey> {
        self.available.keys().copied().collect()
    }

    pub(crate) fn remaining_partitions(&self) -> Vec<usize> {
        self.available
            .keys()
            .map(|key| key.partition)
            .fold(Vec::new(), |mut partitions, partition| {
                if partitions.last().copied() != Some(partition) {
                    partitions.push(partition);
                }
                partitions
            })
    }

    pub(crate) fn assert_exhausted(self) {
        assert!(
            self.available.is_empty(),
            "terminal leaves were not consumed: {}",
            self.available
                .keys()
                .copied()
                .map(format_key)
                .collect::<Vec<_>>()
                .join(", "),
        );
    }

    fn take_keys(
        &mut self,
        keys: impl IntoIterator<Item = TerminalDwaLeafKey>,
    ) -> TerminalDwaPoint {
        let inputs = keys
            .into_iter()
            .map(|key| {
                self.available.remove(&key).unwrap_or_else(|| {
                    panic!("terminal leaf {} is unavailable", format_key(key))
                })
            })
            .collect::<Vec<_>>();
        assert!(!inputs.is_empty(), "terminal point must have at least one leaf");
        merge_local_terminal_leaves(inputs, self.num_tokenizer_states, self.max_token_id)
    }
}

/// Merge any terminal-DWA points into a new terminal point. Calling this
/// recursively gives an arbitrary terminal merge tree.
pub(crate) fn merge_terminal_points(
    inputs: impl IntoIterator<Item = TerminalDwaPoint>,
    num_tokenizer_states: usize,
    max_token_id: u32,
) -> TerminalDwaPoint {
    let inputs = inputs.into_iter().collect::<Vec<_>>();
    assert!(!inputs.is_empty(), "terminal point must have at least one input");
    if inputs.len() == 1 {
        return inputs.into_iter().next().unwrap();
    }
    let inputs = inputs
        .into_iter()
        .map(|input| {
            let (automaton, id_map) = input.into_parts();
            let TerminalAutomaton::Dwa(dwa) = automaton else {
                panic!("terminal point composition requires DWA-backed automata");
            };
            LocalIdMapTerminalDwa { id_map, dwa, profile: Default::default() }
        })
        .collect();
    merge_local_terminal_leaves(inputs, num_tokenizer_states, max_token_id)
}

/// Convert a terminal point to a parser-NWA point without parser-DWA
/// finalization, so parser points can still be merged safely.
pub(crate) fn parser_point_from_terminal(
    terminal: TerminalDwaPoint,
    grammar: &AnalyzedGrammar,
    templates: &Templates,
) -> ParserNwaPoint {
    let (terminal_automaton, id_map) = terminal.into_parts();
    let parser_nwa = build_parser_nwa_from_terminal_dwa_with_precomputed_templates(
        &terminal_automaton,
        grammar,
        templates.clone(),
    );
    MappedArtifact::new(parser_nwa, id_map)
}

/// Merge parser-NWA points. The caller must finalize the resulting single NWA
/// through `finalize_parser_dwa_from_nwa` once at the root.
pub(crate) fn merge_parser_points(
    inputs: impl IntoIterator<Item = ParserNwaPoint>,
) -> ParserNwaPoint {
    let inputs = inputs.into_iter().collect::<Vec<_>>();
    assert!(!inputs.is_empty(), "parser point must have at least one input");
    if inputs.len() == 1 {
        return inputs.into_iter().next().unwrap();
    }
    let (nwas, id_map) = MappedArtifact::reconcile_vec(inputs).into_parts();
    let mut merged = NWA::new(0, 0);
    let mut body = NwaBody::default();
    for nwa in nwas {
        body = merged.union_in_place(&nwa, &body);
    }
    merged.set_start_states(body.start_states);
    MappedArtifact::new(merged, id_map)
}

/// Fixed code-level topologies for focused tests. This is not runtime
/// configuration; production remains on the historical global-terminal path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DwaBuildTopology {
    GlobalTerminal,
    PerPartitionParser,
    PerBranchParser,
    LeftDeepTerminal,
}

impl DwaBuildTopology {
    pub(crate) fn build_parser_point(
        self,
        leaves: TerminalDwaLeaves,
        grammar: &AnalyzedGrammar,
        templates: &Templates,
    ) -> ParserNwaPoint {
        let mut points = DwaBuildPoints::from_terminal_leaves(leaves);
        let parser = match self {
            Self::GlobalTerminal => parser_point_from_terminal(points.take_all(), grammar, templates),
            Self::PerPartitionParser => merge_parser_points(
                points
                    .remaining_partitions()
                    .into_iter()
                    .map(|partition| {
                        parser_point_from_terminal(points.take_partition(partition), grammar, templates)
                    })
                    .collect::<Vec<_>>(),
            ),
            Self::PerBranchParser => merge_parser_points(
                points
                    .remaining_leaf_keys()
                    .into_iter()
                    .map(|key| {
                        parser_point_from_terminal(
                            points.take_branch(key.partition, key.branch),
                            grammar,
                            templates,
                        )
                    })
                    .collect::<Vec<_>>(),
            ),
            Self::LeftDeepTerminal => {
                let mut keys = points.remaining_leaf_keys().into_iter();
                let first = keys.next().expect("terminal graph requires at least one leaf");
                let mut terminal = points.take_branch(first.partition, first.branch);
                for key in keys {
                    let next = points.take_branch(key.partition, key.branch);
                    terminal = points.merge_terminal_points([terminal, next]);
                }
                parser_point_from_terminal(terminal, grammar, templates)
            }
        };
        points.assert_exhausted();
        if compile_profile_enabled() {
            eprintln!(
                "[glrmask/profile][dwa_build_graph] topology={self:?} parser_nwa_states={} parser_nwa_transitions={}",
                parser.artifact().states().len(),
                parser.artifact().num_transitions(),
            );
        }
        parser
    }
}

fn merge_local_terminal_leaves(
    inputs: Vec<LocalIdMapTerminalDwa>,
    num_tokenizer_states: usize,
    max_token_id: u32,
) -> TerminalDwaPoint {
    assert!(!inputs.is_empty(), "terminal point must have at least one input");
    if inputs.len() == 1 {
        let input = inputs.into_iter().next().unwrap();
        return MappedArtifact::new(TerminalAutomaton::Dwa(input.dwa), input.id_map);
    }
    let merged = merge_id_maps_and_terminal_dwas(inputs, num_tokenizer_states, max_token_id);
    MappedArtifact::new(TerminalAutomaton::Dwa(merged.dwa), merged.id_map)
}

fn format_key(key: TerminalDwaLeafKey) -> String {
    format!("p{}.{}", key.partition, key.branch.as_str())
}
