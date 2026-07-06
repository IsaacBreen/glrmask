//! Declarative orchestration for terminal-DWA and parser-DWA experiments.
//!
//! A terminal point is either a direct `(partition, branch)` leaf or a merge of
//! terminal points.  A parser point converts one terminal point to a parser NWA
//! or merges parser points.  Parser points intentionally remain NWAs until the
//! root: the final parser-DWA DEFAULT_LABEL/fallback normalization needs the
//! merged support relation and is not an ordinary DWA union.
//!
//! `GLRMASK_DWA_BUILD_PLAN` accepts these presets:
//!
//! * `global_terminal`: merge all terminal leaves, then build one parser point.
//! * `partition_parser`: merge L1/L2P within each partition, build one parser point per partition, then merge parser points.
//! * `branch_parser`: build one parser point for every L1/L2P leaf, then merge all parser points.
//!
//! `legacy` and `off` retain the historical global-terminal build path.
//!
//! A custom expression uses `t(...)` for a terminal merge, `p(...)` to convert
//! a terminal point to a parser point, and `m(...)` for a parser merge.  Leaves
//! are `p<N>.l1`, `p<N>.l2p`, or `p<N>.all`; `all` denotes every remaining
//! terminal leaf.  Example:
//!
//! ```text
//! m(p(t(p0.l1,p0.l2p)),p(p1.all),p(p2.l1),p(p2.l2p))
//! ```

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

const PLAN_ENV: &str = "GLRMASK_DWA_BUILD_PLAN";

#[derive(Debug, Clone)]
pub(crate) struct DwaBuildPlan {
    source: String,
    kind: DwaBuildPlanKind,
}

#[derive(Debug, Clone)]
enum DwaBuildPlanKind {
    GlobalTerminal,
    PerPartitionParser,
    PerBranchParser,
    Custom(ParserPoint),
}

#[derive(Debug, Clone)]
enum TerminalPoint {
    Leaf(TerminalLeafSelector),
    Merge(Vec<TerminalPoint>),
}

#[derive(Debug, Clone)]
enum ParserPoint {
    FromTerminal(TerminalPoint),
    Merge(Vec<ParserPoint>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct TerminalLeafKey {
    partition: usize,
    branch: TerminalDwaBranch,
}

#[derive(Debug, Clone)]
enum TerminalLeafSelector {
    One(TerminalLeafKey),
    PartitionAll(usize),
    All,
}

impl DwaBuildPlan {
    /// No setting leaves the historical terminal-global build untouched.
    pub(crate) fn from_env() -> Option<Self> {
        let raw = std::env::var(PLAN_ENV).ok()?;
        let source = raw.trim();
        if source.is_empty()
            || source.eq_ignore_ascii_case("legacy")
            || source.eq_ignore_ascii_case("off")
        {
            return None;
        }
        if source.eq_ignore_ascii_case("global_terminal") {
            return Some(Self {
                source: source.to_owned(),
                kind: DwaBuildPlanKind::GlobalTerminal,
            });
        }

        let kind = match source.to_ascii_lowercase().as_str() {
            "partition_parser" | "per_partition_parser" | "partition" => {
                DwaBuildPlanKind::PerPartitionParser
            }
            "branch_parser" | "per_branch_parser" | "branch" => {
                DwaBuildPlanKind::PerBranchParser
            }
            _ => DwaBuildPlanKind::Custom(PlanParser::new(source).parse_root()),
        };
        Some(Self {
            source: source.to_owned(),
            kind,
        })
    }

    pub(crate) fn execute(
        &self,
        leaves: TerminalDwaLeaves,
        grammar: &AnalyzedGrammar,
        templates: &Templates,
    ) -> MappedArtifact<NWA> {
        let mut context = ExecutionContext::from_leaves(leaves);
        let root = match &self.kind {
            DwaBuildPlanKind::GlobalTerminal => ParserPoint::FromTerminal(TerminalPoint::Merge(vec![
                TerminalPoint::Leaf(TerminalLeafSelector::All),
            ])),
            DwaBuildPlanKind::PerPartitionParser => {
                let partitions = context.partitions();
                ParserPoint::Merge(
                    partitions
                        .into_iter()
                        .map(|partition| {
                            ParserPoint::FromTerminal(TerminalPoint::Merge(vec![
                                TerminalPoint::Leaf(TerminalLeafSelector::PartitionAll(partition)),
                            ]))
                        })
                        .collect(),
                )
            }
            DwaBuildPlanKind::PerBranchParser => ParserPoint::Merge(
                context
                    .leaf_keys()
                    .into_iter()
                    .map(|key| ParserPoint::FromTerminal(TerminalPoint::Leaf(TerminalLeafSelector::One(key))))
                    .collect(),
            ),
            DwaBuildPlanKind::Custom(root) => root.clone(),
        };

        let output = context.execute_parser(&root, grammar, templates);
        if !context.available.is_empty() {
            let remaining = context
                .available
                .keys()
                .map(format_leaf_key)
                .collect::<Vec<_>>()
                .join(", ");
            panic!(
                "{PLAN_ENV}={} left terminal leaves unconsumed: {remaining}",
                self.source
            );
        }
        if compile_profile_enabled() {
            eprintln!(
                "[glrmask/profile][dwa_build_plan] plan={} parser_nwa_states={} parser_nwa_transitions={}",
                self.source,
                output.artifact().states().len(),
                output.artifact().num_transitions(),
            );
        }
        output
    }
}

struct ExecutionContext {
    available: BTreeMap<TerminalLeafKey, LocalIdMapTerminalDwa>,
    num_tokenizer_states: usize,
    max_token_id: u32,
}

impl ExecutionContext {
    fn from_leaves(leaves: TerminalDwaLeaves) -> Self {
        let mut available = BTreeMap::new();
        for TerminalDwaLeaf {
            partition,
            branch,
            output,
        } in leaves.leaves
        {
            let key = TerminalLeafKey { partition, branch };
            assert!(
                available.insert(key, output).is_none(),
                "duplicate terminal leaf {}",
                format_leaf_key(&key),
            );
        }
        Self {
            available,
            num_tokenizer_states: leaves.num_tokenizer_states,
            max_token_id: leaves.max_token_id,
        }
    }

    fn leaf_keys(&self) -> Vec<TerminalLeafKey> {
        self.available.keys().copied().collect()
    }

    fn partitions(&self) -> Vec<usize> {
        self.available
            .keys()
            .map(|key| key.partition)
            .collect::<Vec<_>>()
            .into_iter()
            .fold(Vec::new(), |mut out, partition| {
                if out.last().copied() != Some(partition) {
                    out.push(partition);
                }
                out
            })
    }

    fn execute_parser(
        &mut self,
        point: &ParserPoint,
        grammar: &AnalyzedGrammar,
        templates: &Templates,
    ) -> MappedArtifact<NWA> {
        match point {
            ParserPoint::FromTerminal(terminal) => {
                let terminal = self.execute_terminal(terminal);
                let (terminal_automaton, id_map) = terminal.into_parts();
                let parser_nwa = build_parser_nwa_from_terminal_dwa_with_precomputed_templates(
                    &terminal_automaton,
                    grammar,
                    templates.clone(),
                );
                MappedArtifact::new(parser_nwa, id_map)
            }
            ParserPoint::Merge(children) => {
                assert!(!children.is_empty(), "parser merge point must have at least one child");
                let inputs = children
                    .iter()
                    .map(|child| self.execute_parser(child, grammar, templates))
                    .collect::<Vec<_>>();
                merge_parser_nwa_points(inputs)
            }
        }
    }

    fn execute_terminal(&mut self, point: &TerminalPoint) -> MappedArtifact<TerminalAutomaton> {
        match point {
            TerminalPoint::Leaf(selector) => {
                let inputs = self.take(selector);
                merge_terminal_dwa_points(inputs, self.num_tokenizer_states, self.max_token_id)
            }
            TerminalPoint::Merge(children) => {
                assert!(!children.is_empty(), "terminal merge point must have at least one child");
                let inputs = children
                    .iter()
                    .map(|child| self.execute_terminal(child))
                    .collect::<Vec<_>>();
                merge_terminal_artifacts(inputs, self.num_tokenizer_states, self.max_token_id)
            }
        }
    }

    fn take(&mut self, selector: &TerminalLeafSelector) -> Vec<LocalIdMapTerminalDwa> {
        let keys = match selector {
            TerminalLeafSelector::One(key) => vec![*key],
            TerminalLeafSelector::PartitionAll(partition) => self
                .available
                .keys()
                .filter(|key| key.partition == *partition)
                .copied()
                .collect(),
            TerminalLeafSelector::All => self.available.keys().copied().collect(),
        };
        assert!(
            !keys.is_empty(),
            "terminal selector {} resolved to no available leaves; available: {}",
            format_selector(selector),
            self.available
                .keys()
                .map(format_leaf_key)
                .collect::<Vec<_>>()
                .join(", "),
        );
        keys.into_iter()
            .map(|key| {
                self.available.remove(&key).unwrap_or_else(|| {
                    panic!(
                        "terminal leaf {} was consumed more than once",
                        format_leaf_key(&key),
                    )
                })
            })
            .collect()
    }
}

fn merge_terminal_dwa_points(
    inputs: Vec<LocalIdMapTerminalDwa>,
    num_tokenizer_states: usize,
    max_token_id: u32,
) -> MappedArtifact<TerminalAutomaton> {
    assert!(!inputs.is_empty(), "terminal point must have at least one input");
    if inputs.len() == 1 {
        let input = inputs.into_iter().next().unwrap();
        return MappedArtifact::new(TerminalAutomaton::Dwa(input.dwa), input.id_map);
    }
    let merged = merge_id_maps_and_terminal_dwas(inputs, num_tokenizer_states, max_token_id);
    MappedArtifact::new(TerminalAutomaton::Dwa(merged.dwa), merged.id_map)
}

fn merge_terminal_artifacts(
    inputs: Vec<MappedArtifact<TerminalAutomaton>>,
    num_tokenizer_states: usize,
    max_token_id: u32,
) -> MappedArtifact<TerminalAutomaton> {
    assert!(!inputs.is_empty(), "terminal point must have at least one input");
    if inputs.len() == 1 {
        return inputs.into_iter().next().unwrap();
    }
    let local_inputs = inputs
        .into_iter()
        .map(|input| {
            let (automaton, id_map) = input.into_parts();
            let TerminalAutomaton::Dwa(dwa) = automaton else {
                panic!("DWA build plans only merge DWA terminal points");
            };
            LocalIdMapTerminalDwa {
                id_map,
                dwa,
                profile: Default::default(),
            }
        })
        .collect::<Vec<_>>();
    merge_terminal_dwa_points(local_inputs, num_tokenizer_states, max_token_id)
}

fn merge_parser_nwa_points(inputs: Vec<MappedArtifact<NWA>>) -> MappedArtifact<NWA> {
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

fn format_leaf_key(key: &TerminalLeafKey) -> String {
    format!("p{}.{}", key.partition, key.branch.as_str())
}

fn format_selector(selector: &TerminalLeafSelector) -> String {
    match selector {
        TerminalLeafSelector::One(key) => format_leaf_key(key),
        TerminalLeafSelector::PartitionAll(partition) => format!("p{partition}.all"),
        TerminalLeafSelector::All => "all".to_owned(),
    }
}

struct PlanParser<'a> {
    source: &'a str,
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> PlanParser<'a> {
    fn new(source: &'a str) -> Self {
        Self {
            source,
            bytes: source.as_bytes(),
            pos: 0,
        }
    }

    fn parse_root(mut self) -> ParserPoint {
        let point = self.parse_parser_point();
        self.skip_ws();
        if self.pos != self.bytes.len() {
            self.error("unexpected trailing input");
        }
        point
    }

    fn parse_parser_point(&mut self) -> ParserPoint {
        match self.parse_ident().as_str() {
            "p" | "parser" => {
                self.expect(b'(');
                let terminal = self.parse_terminal_point();
                self.expect(b')');
                ParserPoint::FromTerminal(terminal)
            }
            "m" | "merge" => {
                self.expect(b'(');
                let children = self.parse_list(|parser| parser.parse_parser_point());
                ParserPoint::Merge(children)
            }
            other => self.error(&format!("expected parser point p(...) or m(...), got {other}")),
        }
    }

    fn parse_terminal_point(&mut self) -> TerminalPoint {
        self.skip_ws();
        if self.peek_ident_is("t") || self.peek_ident_is("terminal") {
            let _ = self.parse_ident();
            self.expect(b'(');
            let children = self.parse_list(|parser| parser.parse_terminal_point());
            return TerminalPoint::Merge(children);
        }
        TerminalPoint::Leaf(self.parse_leaf_selector())
    }

    fn parse_leaf_selector(&mut self) -> TerminalLeafSelector {
        let ident = self.parse_ident();
        if ident == "all" {
            return TerminalLeafSelector::All;
        }
        let partition = ident
            .strip_prefix('p')
            .and_then(|rest| rest.parse::<usize>().ok())
            .unwrap_or_else(|| self.error("expected terminal leaf p<N>.l1, p<N>.l2p, p<N>.all, or all"));
        self.expect(b'.');
        let branch = self.parse_ident();
        match branch.as_str() {
            "l1" => TerminalLeafSelector::One(TerminalLeafKey {
                partition,
                branch: TerminalDwaBranch::L1,
            }),
            "l2p" | "l2" => TerminalLeafSelector::One(TerminalLeafKey {
                partition,
                branch: TerminalDwaBranch::L2p,
            }),
            "all" => TerminalLeafSelector::PartitionAll(partition),
            _ => self.error("expected terminal branch l1, l2p, or all"),
        }
    }

    fn parse_list<T>(&mut self, mut parse_one: impl FnMut(&mut Self) -> T) -> Vec<T> {
        let mut items = Vec::new();
        loop {
            self.skip_ws();
            if self.consume(b')') {
                if items.is_empty() {
                    self.error("merge point cannot be empty");
                }
                return items;
            }
            items.push(parse_one(self));
            self.skip_ws();
            if self.consume(b',') {
                continue;
            }
            self.expect(b')');
            return items;
        }
    }

    fn parse_ident(&mut self) -> String {
        self.skip_ws();
        let start = self.pos;
        while self.pos < self.bytes.len()
            && (self.bytes[self.pos].is_ascii_alphanumeric() || self.bytes[self.pos] == b'_')
        {
            self.pos += 1;
        }
        if start == self.pos {
            self.error("expected identifier");
        }
        self.source[start..self.pos].to_ascii_lowercase()
    }

    fn peek_ident_is(&mut self, expected: &str) -> bool {
        let saved = self.pos;
        let found = self.parse_ident();
        self.pos = saved;
        found == expected
    }

    fn expect(&mut self, byte: u8) {
        self.skip_ws();
        if !self.consume(byte) {
            self.error(&format!("expected '{}'", byte as char));
        }
    }

    fn consume(&mut self, byte: u8) -> bool {
        if self.pos < self.bytes.len() && self.bytes[self.pos] == byte {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn skip_ws(&mut self) {
        while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }

    fn error(&self, message: &str) -> ! {
        panic!("invalid {PLAN_ENV} at byte {}: {message}; input={}", self.pos, self.source)
    }
}
