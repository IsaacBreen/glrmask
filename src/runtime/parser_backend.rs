//! Parser-independent runtime seam for finite acyclic stack transducers.
//!
//! Construction may borrow temporary LR analysis. Runtime and serialized
//! Constraints retain only native programs and table-free metadata.

use crate::ErrorExt as _;
pub(crate) mod wire;
pub(crate) mod composition;
pub(crate) mod embedding;
#[cfg(test)]
mod sparse_composition_tests;
#[cfg(test)]
mod explicit_lr_tests;
#[cfg(test)]
mod dynamic_backend_tests;
pub(crate) mod link;
pub(crate) mod link_program;
pub(crate) mod link_grammar;
pub(crate) mod scoped_program;
mod prepare;
#[cfg(test)]
mod rewrite_tests;

use std::collections::{BTreeMap, BTreeSet};
use std::ops::{ControlFlow, Deref, DerefMut};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use glrmask_parser_dwa::__private::templates::admissibility::{
    DomainProbe, TemplateDomain, TopAdmission,
};
use glrmask_parser_dwa::__private::templates::compile_dfa::try_split_commit_template_dfas;
use crate::automata::unweighted_u32::dfa::DFA;
use crate::compiler::glr::analysis::EOF;
use crate::compiler::glr::labels::encode_negative_label;
use crate::compiler::glr::parser::ParserGSS;
use crate::ds::leveled_gss::{LeveledGSS, Merge};
use crate::compiler::glr::table::{AdmissionPolicy, GLRTable};
use crate::ds::bitset::BitSet;
use crate::grammar::flat::{DirectRegularAutomaton, TerminalID};
use super::{CommitTemplateDfas, Constraint};

use prepare::top_certificate_rows;
pub(crate) use prepare::PreparedRelation;

/// Internal representation choice for the shared dynamic compiler. Ordinary
/// Dynamic retains LR by default; O2 selects Native explicitly. Resolve the
/// development override before entering compiler worker pools.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DynamicParserBackend {
    Native,
    LrTable,
}

impl DynamicParserBackend {
    #[inline]
    pub(crate) fn from_use_lr(use_lr: bool) -> Self {
        if use_lr { Self::LrTable } else { Self::Native }
    }
}

/// Exactly one executable parser representation. The empty variant is a
/// decode-time section placeholder,
/// never an implicit LR fallback. Native construction installs its immutable
/// program directly, without constructing or retaining an LR machine.
#[derive(Debug, Clone)]
pub(crate) enum ParserExecutable {
    SectionPending,
    ExplicitLr(GLRTable),
    LinkedLr(LinkedLrParser),
    Native(Arc<TemplateParser>),
}

/// The linked LR executor uses the scoped leaf action providers. The
/// coordinator owns no local stack states or action/goto rows.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct LinkedLrParser {
    pub(crate) terminal_count: u32,
    pub(crate) root_nullable: bool,
    pub(crate) embedded_end_token_ids: Vec<u32>,
    pub(crate) admission_policy: AdmissionPolicy,
}

impl ParserExecutable {
    pub(crate) fn section_pending() -> Self { Self::SectionPending }
    pub(crate) fn explicit(table: GLRTable) -> Self { Self::ExplicitLr(table) }
    pub(crate) fn is_present(&self) -> bool { matches!(self, Self::ExplicitLr(_)) }

    pub(crate) fn is_lr_backend(&self) -> bool {
        matches!(self, Self::ExplicitLr(_) | Self::LinkedLr(_))
    }

    pub(crate) fn linked_lr(&self) -> Option<&LinkedLrParser> {
        match self {
            Self::LinkedLr(metadata) => Some(metadata),
            _ => None,
        }
    }
    pub(crate) fn as_lr_mut(&mut self) -> Option<&mut GLRTable> {
        match self { Self::ExplicitLr(table) => Some(table), _ => None }
    }

    pub(crate) fn as_lr(&self) -> Option<&GLRTable> {
        match self { Self::ExplicitLr(table) => Some(table), _ => None }
    }
    pub(crate) fn clone_lr(&self) -> GLRTable { self.deref().clone() }
    pub(crate) fn into_lr(self) -> GLRTable {
        match self {
            Self::ExplicitLr(table) => table,
            _ => panic!("LR table requested from a template-only parser; this operation is not supported"),
        }
    }

    #[inline]
    fn native(&self) -> Option<&Arc<TemplateParser>> {
        match self { Self::Native(parser) => Some(parser), _ => None }
    }

    #[inline]
    fn native_mut(&mut self) -> Option<&mut Arc<TemplateParser>> {
        match self { Self::Native(parser) => Some(parser), _ => None }
    }
}
impl From<GLRTable> for ParserExecutable {
    #[track_caller]
    fn from(_table: GLRTable) -> Self {
        panic!("IMPLICIT LR-BACKED CONSTRAINT MATERIALIZATION IS FORBIDDEN: use the selected compiler representation boundary")
    }
}
impl Deref for ParserExecutable {
    type Target = GLRTable;
    #[track_caller]
    fn deref(&self) -> &GLRTable {
        self.as_lr().expect(
            "LR CONSTRAINT RUNTIME ACCESS IS FORBIDDEN: this parser has no retained LR table; route this operation through native parser programs",
        )
    }
}
impl DerefMut for ParserExecutable {
    #[track_caller]
    fn deref_mut(&mut self) -> &mut GLRTable {
        match self {
            Self::ExplicitLr(table) => table,
            _ => panic!("LR CONSTRAINT TABLE MUTATION IS FORBIDDEN: this parser has no retained LR table"),
        }
    }
}

pub(crate) mod table_serde {
    use super::ParserExecutable;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(
        table: &ParserExecutable, serializer: S,
    ) -> Result<S::Ok, S::Error> {
        if crate::compiler::glr::table::artifact_serde::external_serde_enabled() {
            return 0u8.serialize(serializer);
        }
        match table.as_lr() {
            Some(table) => crate::compiler::glr::table::artifact_serde::serialize(table, serializer),
            None => Err(serde::ser::Error::custom(
                "template-only table requires its dedicated artifact section"
            )),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<ParserExecutable, D::Error> {
        if crate::compiler::glr::table::artifact_serde::external_serde_enabled() {
            let marker = u8::deserialize(deserializer)?;
            if marker != 0 {
                return Err(serde::de::Error::custom("invalid external parser placeholder"));
            }
            return Ok(ParserExecutable::section_pending());
        }
        crate::compiler::glr::table::artifact_serde::deserialize(deserializer)
            .map(super::ParserExecutable::explicit)
    }
}

/// One exact admission relation. Linked programs already carry their own
/// local domains and completion, and never allocate global certificate rows.
#[derive(Debug, Clone)]
enum TemplateAdmission {
    Local(LocalAdmission),
    Linked(Arc<composition::TemplateComposition>),
}

#[derive(Debug, Clone)]
struct LocalAdmission {
    programs: Vec<Arc<PreparedRelation>>,
    completion: Arc<PreparedRelation>,
    top_rows: prepare::TopCertificateRows,
}

#[derive(Debug)]
pub(crate) struct TemplateParser {
    pub(crate) state_count: u32,
    pub(crate) terminal_count: u32,
    pub(crate) skip_terminals: BTreeSet<TerminalID>,
    admission: TemplateAdmission,
    pub(crate) embedding: Option<Arc<embedding::TemplateEmbedding>>,
    pub(crate) link_grammar: Option<Arc<link_grammar::LinkGrammar>>,
    profile: bool,
    advances: AtomicU64,
    admissions: AtomicU64,
    completions: AtomicU64,
}

pub(crate) struct PreparedTemplateParser {
    source_state_count: u32,
    source_terminal_count: u32,

    pub(crate) parser: TemplateParser,
}

impl PreparedTemplateParser {
    pub(crate) fn from_compiler_parts(
        table: &GLRTable,
        direct_regular: Option<&DirectRegularAutomaton>,
        retained_templates: &[Option<DFA>],
        ignore: Option<u32>,
        dynamic: bool,
        preserve_coordinate: bool,
    ) -> crate::Result<Self> {
        prepare::from_compiler_parts(
            table, direct_regular, retained_templates, ignore,
            dynamic, preserve_coordinate,
        )
    }
}

fn compile_domain(template: &CommitTemplateDfas) -> crate::Result<TemplateDomain> {
    TemplateDomain::compile(template).map_err(crate::Error::Compilation)
}

pub(super) fn matches_gss<A: Merge + Eq + std::hash::Hash>(domain: &TemplateDomain, stack: &LeveledGSS<u32, A>) -> bool {
    if stack.is_empty() { return false; }
    let cursor = match domain.start() {
        DomainProbe::Accept => return true,
        DomainProbe::Reject => return false,
        DomainProbe::NeedMore(cursor) => cursor,
    };
    stack.any_prefix_matching(cursor, |cursor, top| match domain.step(cursor, *top) {
        DomainProbe::Accept => ControlFlow::Break(true),
        DomainProbe::Reject => ControlFlow::Break(false),
        DomainProbe::NeedMore(cursor) => ControlFlow::Continue(cursor),
    })
}

fn intersecting_bits<'a>(
    left: &'a BitSet, right: &'a BitSet,
) -> impl Iterator<Item = usize> + 'a {
    left.words().iter().zip(right.words()).enumerate().flat_map(|(index, (&a, &b))| {
        let mut word = a & b;
        std::iter::from_fn(move || {
            if word == 0 { return None; }
            let bit = index * 64 + word.trailing_zeros() as usize;
            word &= word - 1;
            Some(bit)
        })
    })
}

impl TemplateParser {
    pub(crate) fn compile(
        state_count: u32,
        terminal_count: u32,
        skip_terminals: BTreeSet<TerminalID>,
        templates: &[Option<Arc<CommitTemplateDfas>>],
        completion_template: CommitTemplateDfas,
    ) -> crate::Result<Self> {
        prepare::compile_parser(
            state_count, terminal_count, skip_terminals,
            templates, completion_template, true,
        )
    }

    pub(crate) fn from_composition(
        state_count: u32,
        terminal_count: u32,
        composition: composition::TemplateComposition,
    ) -> crate::Result<Self> {
        if composition.outer_len() != terminal_count as usize {
            return Err(crate::Error::Compilation("outer scoped inventory count mismatch".into()));
        }

        Ok(Self {
            state_count,
            terminal_count,
            skip_terminals: BTreeSet::new(),
            admission: TemplateAdmission::Linked(Arc::new(composition)),
            embedding: None,
            link_grammar: None,
            profile: crate::env::var_os("GLRMASK_PROFILE_TEMPLATE_BACKEND").is_some(),
            advances: AtomicU64::new(0),
            admissions: AtomicU64::new(0),
            completions: AtomicU64::new(0),
        })
    }

    #[inline]
    pub(crate) fn composition(&self) -> Option<&composition::TemplateComposition> {
        match &self.admission {
            TemplateAdmission::Linked(composition) => Some(composition),
            TemplateAdmission::Local(_) => None,
        }
    }

    #[inline]
    fn local_admission(&self) -> Option<&LocalAdmission> {
        match &self.admission {
            TemplateAdmission::Local(local) => Some(local),
            TemplateAdmission::Linked(_) => None,
        }
    }

    pub(crate) fn completion_template(&self) -> &Arc<CommitTemplateDfas> {
        match &self.admission {
            TemplateAdmission::Local(local) => &local.completion.source,
            TemplateAdmission::Linked(linked) => &linked.completion_view.source,
        }
    }

    pub(crate) fn completion_domain(&self) -> &Arc<TemplateDomain> {
        match &self.admission {
            TemplateAdmission::Local(local) => &local.completion.domain,
            TemplateAdmission::Linked(linked) => &linked.completion_view.domain,
        }
    }

    pub(crate) fn local_domain(&self, terminal: usize) -> Option<&Arc<TemplateDomain>> {
        self.local_admission()?.programs.get(terminal).map(|p| &p.domain)
    }

    pub(crate) fn terminal_source(&self, terminal: usize) -> Option<&Arc<CommitTemplateDfas>> {
        match &self.admission {
            TemplateAdmission::Local(local) => local.programs.get(terminal).map(|p| &p.source),
            TemplateAdmission::Linked(linked) => linked.outer_view(terminal).map(|p| &p.source),
        }
    }

    pub(crate) fn local_relation(&self, terminal: usize) -> Option<&Arc<PreparedRelation>> {
        self.local_admission()?.programs.get(terminal)
    }

    /// Cold projection for compiler inputs and diagnostics; the executable
    /// relation inventory itself owns the programs and their derived views.
    pub(crate) fn terminal_sources(&self) -> crate::runtime::artifact::TemplateDfasByTerminal {
        (0..self.terminal_count as usize)
            .map(|terminal| self.terminal_source(terminal).cloned())
            .collect()
    }


    #[inline]
    fn possible_row(&self, top: u32) -> Option<&BitSet> {
        self.local_admission()?.top_rows.possible(top)
    }

    #[inline]
    fn unconditional_row(&self, top: u32) -> Option<&BitSet> {
        self.local_admission()?.top_rows.unconditional(top)
    }

    #[inline]
    pub(crate) fn record_advance(&self) {
        if self.profile { self.advances.fetch_add(1, Ordering::Relaxed); }
    }

    #[inline]
    pub(crate) fn admits<A: Merge + Eq + std::hash::Hash>(&self, stack: &LeveledGSS<u32, A>, terminal: TerminalID) -> bool {
        if let Some(provider) = self.composition_provider() {
            let closed = provider.close_native(stack);
            return if terminal == EOF {
                !closed.is_empty() && self.admits_control_closed(&closed, EOF)
            } else {
                provider.closed_admits_native(&closed, terminal)
            };
        }
        self.admits_control_closed(stack, terminal)
    }

    fn admits_control_closed<A: Merge + Eq + std::hash::Hash>(&self, stack: &LeveledGSS<u32, A>, terminal: TerminalID) -> bool {
        if self.profile { self.admissions.fetch_add(1, Ordering::Relaxed); }
        if let Some(composition) = self.composition() {
            return if terminal == EOF || terminal < self.terminal_count {
                composition.admits_outer(stack, terminal)
            } else {
                composition.admits(stack, terminal - self.terminal_count)
            };
        }
        let local = self.local_admission().expect("local admission");
        let (bit, domain) = if terminal == EOF {
            (self.terminal_count as usize, &local.completion.domain)
        } else {
            let Some(domain) = local.programs.get(terminal as usize).map(|p| &p.domain) else { return false; };
            (terminal as usize, domain)
        };
        if let Some(top) = stack.single_exclusive_top_value() {
            if local.top_rows.unconditional(top).is_some_and(|row| row.contains(bit)) {
                return true;
            }
            if local.top_rows.possible(top).is_some_and(|row| !row.contains(bit)) {
                return false;
            }
        } else {
            let tops = stack.peek_values();
            let mut possible = tops.is_empty();
            for top in tops {
                if local.top_rows.unconditional(top).is_some_and(|row| row.contains(bit)) {
                    return true;
                }
                possible |= local.top_rows.possible(top).is_none_or(|row| row.contains(bit));
            }
            if !possible { return false; }
        }
        matches_gss(domain, stack)
    }

    pub(crate) fn admits_any<A: Merge + Eq + std::hash::Hash>(&self, stack: &LeveledGSS<u32, A>, candidates: &BitSet) -> bool {
        if let Some(provider) = self.composition_provider() {
            let closed = provider.close_native(stack);
            return candidates.iter().any(|terminal| provider.closed_admits_native(&closed, terminal as u32));
        }
        let local = self.local_admission().expect("local admission");
        if let Some(top) = stack.single_exclusive_top_value()
            && let Some(possible) = local.top_rows.possible(top)
        {
            if self.profile { self.admissions.fetch_add(1, Ordering::Relaxed); }
            if local.top_rows.unconditional(top).is_some_and(|row| {
                row.words().iter().zip(candidates.words()).any(|(a, b)| a & b != 0)
            }) {
                return true;
            }
            return intersecting_bits(possible, candidates).any(|bit| {
                let domain = if bit == self.terminal_count as usize {
                    Some(&local.completion.domain)
                } else {
                    local.programs.get(bit).map(|p| &p.domain)
                };
                domain.is_some_and(|domain| matches_gss(domain, stack))
            });
        }
        for top in stack.peek_values() {
            if local.top_rows.unconditional(top).is_some_and(|row| {
                row.words().iter().zip(candidates.words()).any(|(a, b)| a & b != 0)
            }) {
                if self.profile { self.admissions.fetch_add(1, Ordering::Relaxed); }
                return true;
            }
        }
        for bit in candidates.iter() {
            let terminal = if bit == self.terminal_count as usize { EOF } else { bit as u32 };
            if self.admits(stack, terminal) { return true; }
        }
        false
    }

    pub(crate) fn admitted<A: Merge + Eq + std::hash::Hash>(&self, stack: &LeveledGSS<u32, A>, candidates: &BitSet) -> BitSet {
        if let Some(provider) = self.composition_provider() {
            let mut admitted = BitSet::new(candidates.len());
            let closed = provider.close_native(stack);
            for terminal in candidates.iter() {
                if provider.closed_admits_native(&closed, terminal as u32) {
                    admitted.set(terminal);
                }
            }
            return admitted;
        }
        let local = self.local_admission().expect("local admission");
        if self.profile { self.admissions.fetch_add(1, Ordering::Relaxed); }
        let mut result = BitSet::new(candidates.len());
        let mut possible = BitSet::new(candidates.len());
        let tops = stack.peek_values();
        if tops.is_empty() {
            possible = candidates.clone();
        }
        for top in tops {
            match local.top_rows.possible(top) {
                Some(row) => {
                    for ((dst, a), b) in possible.words_mut().iter_mut()
                        .zip(row.words()).zip(candidates.words())
                    {
                        *dst |= a & b;
                    }
                }
                None => possible = candidates.clone(),
            }
            if let Some(row) = local.top_rows.unconditional(top) {
                for ((dst, a), b) in result.words_mut().iter_mut()
                    .zip(row.words()).zip(candidates.words())
                {
                    *dst |= a & b;
                }
            }
        }
        for (word_index, &word) in possible.words().iter().enumerate() {
            let mut remaining = word & !result.words()[word_index];
            while remaining != 0 {
                let bit = word_index * 64 + remaining.trailing_zeros() as usize;
                remaining &= remaining - 1;
                let domain = if bit == self.terminal_count as usize {
                    Some(&local.completion.domain)
                } else {
                    local.programs.get(bit).map(|p| &p.domain)
                };
                if domain.is_some_and(|domain| matches_gss(domain, stack)) {
                    result.set(bit);
                }
            }
        }
        result
    }

    pub(crate) fn marker_top_partition(&self, terminal: u32)
        -> Option<(TopAdmission, impl Iterator<Item = (u32, TopAdmission)> + '_)> {
        if self.composition().is_some() { return None; }
        self.local_domain(terminal as usize).map(|domain| domain.top_admission_partition())
    }

    pub(crate) fn marker_top_admission(&self, top: u32, terminal: u32) -> TopAdmission {
        if let Some(composition) = self.composition() {
            return composition.outer_view(terminal as usize)
                .map_or(TopAdmission::Never, |view| view.classify_top(top));
        }
        self.local_domain(terminal as usize).map_or(TopAdmission::Never, |domain| domain.classify_top(top))
    }

    pub(crate) fn marker_admits_empty(&self, terminal: u32) -> bool {
        if self.composition().is_some() {
            let gss = ParserGSS::from_single_stack(Vec::new(),
                crate::compiler::glr::accumulator::TerminalsDisallowed::new());
            return self.admits(&gss, terminal);
        }
        self.local_domain(terminal as usize)
            .is_some_and(|domain| domain.start() == DomainProbe::Accept)
    }

    pub(crate) fn admits_flat_any(&self, stack: &[u32], candidates: &BitSet) -> bool {
        if self.composition().is_some() {
            let gss = ParserGSS::from_single_stack(
                stack.to_vec(),
                crate::compiler::glr::accumulator::TerminalsDisallowed::new(),
            );
            return self.admits_any(&gss, candidates);
        }
        let local = self.local_admission().expect("local admission");
        if self.profile { self.admissions.fetch_add(1, Ordering::Relaxed); }
        let top = stack.last().copied();
        if top.and_then(|top| local.top_rows.unconditional(top)).is_some_and(|row| {
            row.words().iter().zip(candidates.words()).any(|(a, b)| a & b != 0)
        }) {
            return true;
        }
        if let Some(possible) = top.and_then(|top| local.top_rows.possible(top)) {
            return intersecting_bits(possible, candidates).any(|bit| {
                let domain = if bit == self.terminal_count as usize {
                    Some(&local.completion.domain)
                } else {
                    local.programs.get(bit).map(|p| &p.domain)
                };
                domain.is_some_and(|domain| {
                    domain.matches_top_first(stack.iter().rev().copied())
                })
            });
        }
        for bit in candidates.iter() {
            if top.and_then(|top| local.top_rows.possible(top))
                .is_some_and(|row| !row.contains(bit))
            {
                continue;
            }
            let domain = if bit == self.terminal_count as usize {
                Some(&local.completion.domain)
            } else {
                local.programs.get(bit).map(|p| &p.domain)
            };
            if domain.is_some_and(|domain| {
                domain.matches_top_first(stack.iter().rev().copied())
            }) {
                return true;
            }
        }
        false
    }

    pub(crate) fn preserve_start_nullable(&mut self) {
        if self.completion_domain().matches_top_first([0]) { return; }
        let completion = link_program::with_nullable_return(self.completion_template(), 0)
            .expect("nullable completion must remain finite");
        let symbols = self.composition().map_or(self.state_count, |linked| linked.completion_view.symbols);
        let relation = Arc::new(PreparedRelation::prepare(
            completion, symbols, false,
        ).expect("validated nullable completion relation"));
        match &mut self.admission {
            TemplateAdmission::Local(local) => {
                local.completion = relation;
                local.top_rows = top_certificate_rows(
                    self.state_count,
                    &local.programs.iter().map(|p| Arc::clone(&p.domain)).collect::<Vec<_>>(),
                    &local.completion.domain,
                );
            }
            TemplateAdmission::Linked(linked) => {
                Arc::make_mut(linked).completion_view.relation = relation;
            }
        }
    }

    pub(crate) fn finished(&self, stack: &ParserGSS) -> bool {
        if self.profile { self.completions.fetch_add(1, Ordering::Relaxed); }
        self.admits(stack, EOF)
    }

    pub(crate) fn report(&self) -> serde_json::Value {
        let mut unique = rustc_hash::FxHashSet::default();
        let mut unique_heap = 0usize;
        let domains: Box<dyn Iterator<Item = &Arc<TemplateDomain>> + '_> = match &self.admission {
            TemplateAdmission::Local(local) => Box::new(local.programs.iter().map(|p| &p.domain)),
            TemplateAdmission::Linked(linked) => Box::new(linked.outer_views().map(|view| &view.domain)),
        };
        let mut domain_states = 0usize;
        let mut domain_edges = 0usize;
        let mut domain_heap = 0usize;
        for domain in domains {
            domain_states += domain.state_count();
            domain_edges += domain.edge_count();
            domain_heap += domain.heap_payload_bytes();
            if unique.insert(Arc::as_ptr(domain) as usize) {
                unique_heap += domain.heap_payload_bytes();
            }
        }
        serde_json::json!({
            "backend": "acyclic-template-dfa",
            "lr_table_present": false,
            "terminal_count": self.terminal_count,
            "stack_symbol_count": self.state_count,
            "domain_states": domain_states,
            "domain_edges": domain_edges,
            "domain_heap_payload_bytes": domain_heap,
            "unique_terminal_domains": unique.len(),
            "unique_domain_heap_payload_bytes": unique_heap,
            "completion_domain_states": self.completion_domain().state_count(),
            "composition": self.composition().map(|composition| composition.report()),
            "finite_embedding": self.embedding.is_some(),
            "counters_enabled": self.profile,
            "template_advances": self.advances.load(Ordering::Relaxed),
            "template_admission_queries": self.admissions.load(Ordering::Relaxed),
            "template_completion_queries": self.completions.load(Ordering::Relaxed),
        })
    }
}

fn sparse_regular_templates(
    automaton: &DirectRegularAutomaton,
    terminal_count: u32,
) -> crate::Result<(Vec<Option<Arc<CommitTemplateDfas>>>, CommitTemplateDfas, u32)> {
    let invalid = || crate::Error::Compilation(
        "invalid sparse regular automaton coordinate".into()
    );
    let state_count = automaton.states.len().checked_add(1)
        .and_then(|n| u32::try_from(n).ok())
        .ok_or_else(|| crate::Error::Compilation("regular stack alphabet too large".into()))?;
    if automaton.start_states.iter().any(|&state| state as usize >= automaton.states.len())
        || automaton.states.iter().any(|row| {
            row.epsilons.iter().any(|&state| state as usize >= automaton.states.len())
                || row.transitions.iter().any(|(&terminal, targets)| {
                    terminal >= terminal_count
                        || targets.iter().any(|&state| state as usize >= automaton.states.len())
                })
        })
    {
        return Err(invalid());
    }
    let mut by_terminal = vec![BTreeMap::<u32, Vec<u32>>::new(); terminal_count as usize];
    let mut finished_tops = Vec::new();
    for top in 0..state_count {
        let mut pending = if top == 0 {
            automaton.start_states.clone()
        } else {
            vec![top - 1]
        };
        let mut seen = BitSet::new(automaton.states.len());
        let mut targets = BTreeMap::<u32, BTreeSet<u32>>::new();
        let mut finished = false;
        while let Some(raw) = pending.pop() {
            if seen.contains(raw as usize) { continue; }
            let state = automaton.states.get(raw as usize).ok_or_else(|| {
                crate::Error::Compilation("invalid sparse regular state".into())
            })?;
            seen.set(raw as usize);
            finished |= state.is_accepting;
            for (&terminal, destinations) in &state.transitions {
                targets.entry(terminal).or_default()
                    .extend(destinations.iter().map(|raw| raw + 1));
            }
            pending.extend(state.epsilons.iter().copied());
        }
        if finished { finished_tops.push(top); }
        for (terminal, targets) in targets {
            by_terminal[terminal as usize].insert(top, targets.into_iter().collect());
        }
    }
    let mut templates = Vec::with_capacity(terminal_count as usize);
    for rows in by_terminal {
        let mut dfa = DFA::new();
        let accept = dfa.add_state();
        dfa.set_accepting(accept, true);
        let mut middles = BTreeMap::<Vec<u32>, u32>::new();
        for (top, targets) in rows {
            let middle = if let Some(&id) = middles.get(&targets) {
                id
            } else {
                let id = dfa.add_state();
                for &target in &targets {
                    dfa.add_transition(id, encode_negative_label(target), accept);
                }
                middles.insert(targets, id);
                id
            };
            dfa.add_transition(dfa.start_state, top as i32, middle);
        }
        let split = try_split_commit_template_dfas(&dfa).ok_or_else(|| {
            crate::Error::Compilation(
                "regular template did not satisfy the split phase contract".into()
            )
        })?;
        templates.push(Some(Arc::new(split)));
    }
    let mut completion = DFA::new();
    let accept = completion.add_state();
    completion.set_accepting(accept, true);
    for top in finished_tops {
        completion.add_transition(0, top as i32, accept);
    }
    let completion = try_split_commit_template_dfas(&completion).ok_or_else(|| {
        crate::Error::Compilation("regular completion template failed phase contract".into())
    })?;
    Ok((templates, completion, state_count))
}

impl Constraint {
    #[inline]
    pub(crate) fn template_parser(&self) -> Option<&Arc<TemplateParser>> {
        self.table.native()
    }

    #[inline]
    pub(crate) fn template_parser_mut(&mut self) -> Option<&mut Arc<TemplateParser>> {
        self.table.native_mut()
    }

    #[inline]
    pub(crate) fn has_template_parser(&self) -> bool {
        self.template_parser().is_some()
    }

    #[inline]
    pub(crate) fn parser_symbol_count(&self) -> u32 {
        match &self.table {
            ParserExecutable::Native(parser) => parser.state_count,
            ParserExecutable::ExplicitLr(table) => table.num_states,
            // The scoped coordinate lives in the recursive layout, not here.
            ParserExecutable::LinkedLr(_) => 0,
            ParserExecutable::SectionPending => panic!("parser section not installed"),
        }
    }

    #[inline]
    pub(crate) fn parser_terminal_count(&self) -> u32 {
        match &self.table {
            ParserExecutable::Native(parser) => parser.terminal_count,
            ParserExecutable::ExplicitLr(table) => table.num_terminals,
            ParserExecutable::LinkedLr(metadata) => metadata.terminal_count,
            ParserExecutable::SectionPending => panic!("parser section not installed"),
        }
    }

    #[inline]
    pub(crate) fn parser_has_controls(&self) -> bool {
        match &self.table {
            ParserExecutable::Native(parser) => parser.composition()
                .is_some_and(|composition| composition.has_controls()),
            ParserExecutable::ExplicitLr(table) => !table.control_terminals.is_empty(),
            // The scoped provider consumes its own controls; these are not
            // ordinary terminals on a global LR table.
            _ => false,
        }
    }

    #[inline]
    pub(crate) fn parser_skip_terminals(&self) -> &BTreeSet<TerminalID> {
        static EMPTY: BTreeSet<TerminalID> = BTreeSet::new();
        match &self.table {
            ParserExecutable::Native(parser) => &parser.skip_terminals,
            ParserExecutable::ExplicitLr(table) => &table.skip_terminals,
            _ => &EMPTY,
        }
    }

    pub(crate) fn parser_is_control_terminal(&self, terminal: TerminalID) -> bool {
        match &self.table {
            ParserExecutable::Native(parser) => parser.link_grammar.as_ref()
                .is_some_and(|grammar| grammar.control_terminals.contains(&terminal)),
            ParserExecutable::ExplicitLr(table) => table.control_terminals.contains(&terminal),
            _ => false,
        }
    }

    pub(crate) fn parser_control_terminals(&self) -> BTreeSet<u32> {
        match &self.table {
            ParserExecutable::Native(parser) => parser.link_grammar.as_ref()
                .map_or_else(BTreeSet::new, |grammar| grammar.control_terminals.clone()),
            ParserExecutable::ExplicitLr(table) => table.control_terminals.clone(),
            _ => BTreeSet::new(),
        }
    }

    #[inline]
    pub(crate) fn parser_admission_policy(&self) -> AdmissionPolicy {
        match &self.table {
            ParserExecutable::Native(_) => AdmissionPolicy::ExactSimulation,
            ParserExecutable::ExplicitLr(table) => table.admission_policy,
            ParserExecutable::LinkedLr(metadata) => metadata.admission_policy,
            ParserExecutable::SectionPending => panic!("parser section not installed"),
        }
    }

    #[inline]
    pub(crate) fn parser_advance_row(&self, top: u32) -> Option<&BitSet> {
        if self.template_composition_provider().is_some() { return None; }
        self.template_parser().map_or_else(
            || self.table.as_lr()?.advance_row(top),
            |parser| parser.possible_row(top),
        )
    }

    #[inline]
    pub(crate) fn parser_advance_row_allows(&self, top: u32, terminal: u32) -> bool {
        if self.template_composition_provider().is_some() {
            return top < self.parser_symbol_count();
        }
        if let Some(parser) = self.template_parser() {
            let bit = if terminal == EOF {
                parser.terminal_count as usize
            } else {
                terminal as usize
            };
            parser.possible_row(top).is_some_and(|row| row.contains(bit))
        } else {
            self.table.as_lr().is_some_and(|table| table.advance_row_allows(top, terminal))
        }
    }

    #[inline]
    pub(crate) fn parser_advance_row_intersects(&self, top: u32, terminals: &BitSet) -> bool {
        if self.template_composition_provider().is_some() {
            return top < self.parser_symbol_count() && !terminals.is_empty();
        }
        if let Some(parser) = self.template_parser() {
            parser.possible_row(top).is_some_and(|row| {
                row.words().iter().zip(terminals.words()).any(|(a, b)| a & b != 0)
            })
        } else {
            self.table.as_lr().is_some_and(|table| table.advance_row_intersects(top, terminals))
        }
    }

    #[inline]
    pub(crate) fn parser_unconditional_row(&self, top: u32) -> Option<&BitSet> {
        if self.template_composition_provider().is_some() { return None; }
        self.template_parser().map_or_else(
            || self.table.as_lr()?.unconditional_advance_row(top),
            |parser| parser.unconditional_row(top),
        )
    }

    pub(crate) fn install_template_parser(&mut self) -> crate::Result<()> {
        self.install_template_parser_in_coordinate(false)
    }

    pub(crate) fn install_template_parser_from_compile(&mut self) -> crate::Result<()> {
        if self.has_template_parser() { return Ok(()); }
        if self.uses_compact_segmented_parser_runtime() {
            return self.install_composition_template_parser();
        }
        self.install_template_parser_impl(true, false)
    }

    pub(crate) fn install_template_parser_in_coordinate(
        &mut self, preserve_coordinate: bool,
    ) -> crate::Result<()> {
        if self.has_template_parser() { return Ok(()); }
        if self.uses_compact_segmented_parser_runtime() {
            return self.install_composition_template_parser();
        }
        self.install_template_parser_impl(false, preserve_coordinate)
    }

    fn install_template_parser_impl(
        &mut self,
        reuse_compiler_templates: bool,
        preserve_coordinate: bool,
    ) -> crate::Result<()> {
        if self.has_template_parser() { return Ok(()); }
        let prepared = self.prepare_template_parser_impl(
            reuse_compiler_templates, preserve_coordinate,
        )?;
        self.install_prepared_template_parser(prepared)
    }

    pub(crate) fn prepare_template_parser(
        &self,
    ) -> crate::Result<Option<PreparedTemplateParser>> {
        if self.has_template_parser() { return Ok(None); }
        if self.uses_compact_segmented_parser_runtime() {
            return Err(crate::Error::Compilation(
                "prepared standalone template parser is not valid for segmented composition".into()
            ));
        }
        self.prepare_template_parser_impl(false, false).map(Some)
    }

    fn prepare_template_parser_impl(
        &self,
        reuse_compiler_templates: bool,
        preserve_coordinate: bool,
    ) -> crate::Result<PreparedTemplateParser> {
        if self.parser_has_controls() || self.uses_compact_segmented_parser_runtime() {
            return Err(crate::Error::Compilation(
                "template-only parser composition/control closure is not implemented; no LR fallback is permitted"
                    .into()
            ));
        }
        PreparedTemplateParser::from_compiler_parts(
            &self.table,
            self.direct_regular_automaton.as_ref(),
            if reuse_compiler_templates {
                &self.composition_parser_templates_by_terminal
            } else {
                &[]
            },
            self.ignore_terminal,
            self.uses_dynamic_runtime(),
            preserve_coordinate,
        )
    }

    pub(crate) fn install_prepared_template_parser(
        &mut self, prepared: PreparedTemplateParser,
    ) -> crate::Result<()> {
        if self.has_template_parser() || !self.table.is_present()
            || self.table.num_states != prepared.source_state_count
            || self.table.num_terminals != prepared.source_terminal_count
            || self.table.skip_terminals != prepared.parser.skip_terminals
        {
            return Err(crate::Error::Compilation(
                "prepared template parser no longer matches its source constraint".into()
            ));
        }
        self.serialized_artifact_cache = None;
        self.deferred_table_rules_blob = None;
        self.deferred_table_rules = Default::default();
        self.table = ParserExecutable::Native(Arc::new(prepared.parser));
        if let Some(queries) = &self.marker_queries {
            let terminals = queries.terminals.to_vec();
            self.install_marker_terminals(terminals).map_err(crate::Error::Compilation)?;
        }
        Ok(())
    }

    #[cfg(feature = "internal-api")]
    pub(crate) fn save_without_effect_metadata_for_diagnostic(&self) -> Vec<u8> {
        fn strip(constraint: &mut Constraint) {
            if let Some(parser) = constraint.template_parser_mut() {
                let source = parser.as_ref();
                let mut replacement = TemplateParser {
                    state_count: source.state_count,
                    terminal_count: source.terminal_count,
                    skip_terminals: source.skip_terminals.clone(),
                    admission: source.admission.clone(),
                    embedding: source.embedding.clone(),
                    link_grammar: source.link_grammar.clone(),

                    profile: source.profile,
                    advances: AtomicU64::new(source.advances.load(Ordering::Relaxed)),
                    admissions: AtomicU64::new(source.admissions.load(Ordering::Relaxed)),
                    completions: AtomicU64::new(source.completions.load(Ordering::Relaxed)),
                };
                if let Some(grammar) = replacement.link_grammar.as_mut() {
                    Arc::make_mut(grammar).set_stack_effects(None);
                }
                *parser = Arc::new(replacement);
            }
            constraint.serialized_artifact_cache = None;
            if let Some(overlay) = constraint.static_dynamic_overlay.as_mut() {
                for component in &mut overlay.segmented_parser_components {
                    strip(Arc::make_mut(&mut component.constraint));
                }
            }
        }
        let mut isolated = self.clone();
        strip(&mut isolated);
        isolated.save()
    }

    pub(crate) fn parser_backend_report(&self) -> serde_json::Value {
        match self.template_parser() {
            Some(parser) => {
                assert!(!self.table.is_present(), "template-only constraint retained an LR table");
                let mut report = parser.report();
                if let Some(summary) = parser.link_grammar.as_ref()
                    .and_then(|grammar| grammar.stack_effects())
                {
                    let entry_effects = summary.entries.values().map(Vec::len).sum::<usize>();
                    let pushes = summary.effects.iter()
                        .chain(summary.entries.values().flatten())
                        .map(|effect| effect.pushes.len()).sum::<usize>();
                    let effect_size = summary.effects.first().map_or(0, std::mem::size_of_val);
                    let vector_bytes = std::mem::size_of_val(summary)
                        + summary.effects.capacity() * effect_size
                        + summary.accepting.capacity() * std::mem::size_of::<u32>()
                        + summary.entries.values()
                            .map(|row| row.capacity() * effect_size).sum::<usize>()
                        + summary.effects.iter().chain(summary.entries.values().flatten())
                            .map(|effect| effect.pushes.capacity() * std::mem::size_of::<u32>())
                            .sum::<usize>();
                    report["compiler_effect_metadata"] = serde_json::json!({
                        "states": summary.states,
                        "terminals": summary.terminals,
                        "effects": summary.effects.len(),
                        "entry_terminals": summary.entries.len(),
                        "entry_effects": entry_effects,
                        "accepting_states": summary.accepting.len(),
                        "pushed_symbols": pushes,
                        "serialized_bytes": bincode::serialized_size(summary).ok(),
                        "owned_vector_bytes_excluding_btree_and_allocator": vector_bytes
                    });
                }
                report["sparse_regular"] = self.direct_regular_automaton.is_some().into();
                report["virtual_lexer"] = self.tokenizer.has_any_virtual_runtime().into();
                if let Some(overlay) = &self.static_dynamic_overlay {
                    report["finite_observation_leaves"] = overlay.recursive_static_observation.as_ref()
                        .map_or(0, |observation| observation.leaf_offsets.len().saturating_sub(1))
                        .into();
                    if let Some(observation) = &overlay.recursive_static_observation {
                        report["finite_observation_offsets"] =
                            serde_json::json!(observation.leaf_offsets);
                    }
                    report["component_parsers"] = serde_json::Value::Array(
                        overlay.segmented_parser_components.iter()
                            .map(|component| component.constraint.parser_backend_report())
                            .collect()
                    );
                    report["packed_lr_compiler_table_present"] =
                        false.into();
                    report["static_boundary_shards"] = overlay.segmented_parser_components.iter()
                        .filter(|component| matches!(
                            component.boundary.as_ref().map(|shard| &shard.backend),
                            Some(super::SegmentedBoundaryShardBackend::StaticParser(_))
                        )).count().into();
                    report["dynamic_boundary_shards"] = overlay.segmented_parser_components.iter()
                        .filter(|component| matches!(
                            component.boundary.as_ref().map(|shard| &shard.backend),
                            Some(super::SegmentedBoundaryShardBackend::DynamicDirect)
                        )).count().into();
                    report["boundary_candidate_tokens"] = serde_json::Value::Array(
                        overlay.segmented_parser_components.iter().enumerate()
                            .map(|(index, component)| serde_json::json!({
                                "component": index,
                                "count": component.boundary.as_ref()
                                    .and_then(|shard| shard.candidate_tokens.as_ref())
                                    .map(|ids| ids.len())
                            })).collect()
                    );
                    if let Some(composition) = parser.composition() {
                        let mut checked = 0usize;
                        let mut sources = 0usize;
                        let mut domains = 0usize;
                        let mut identities = 0usize;
                        let mut offset = 0usize;
                        for component in &overlay.segmented_parser_components {
                            let local = component.constraint.template_parser()
                                .expect("native component");
                            for terminal in 0..local.terminal_count as usize {
                                let view = composition.outer_view(offset + terminal).unwrap();
                                if let Some(nested) = local.composition() {
                                    checked += 1;
                                    sources += usize::from(Arc::ptr_eq(
                                        &view.source, &nested.outer_view(terminal).unwrap().source,
                                    ));
                                    domains += usize::from(Arc::ptr_eq(
                                        &view.domain, &nested.outer_view(terminal).unwrap().domain,
                                    ));
                                } else if component.constraint.ignore_terminal == Some(terminal as u32)
                                    || local.skip_terminals.contains(&(terminal as u32))
                                {
                                    identities += 1;
                                } else {
                                    checked += 1;
                                    sources += usize::from(Arc::ptr_eq(
                                        &view.source,
                                        local.terminal_source(terminal).unwrap(),
                                    ));
                                    domains += usize::from(Arc::ptr_eq(
                                        &view.domain, local.local_domain(terminal).expect("local domain"),
                                    ));
                                }
                            }
                            offset += local.terminal_count as usize;
                        }
                        report["component_sharing"] = serde_json::json!({
                            "checked_retained_terminal_relations": checked,
                            "shared_retained_sources": sources,
                            "shared_retained_domains": domains,
                            "synthetic_ignore_or_skip_relations": identities,
                            "dense_global_certificate_rows":
                                parser.local_admission().map_or(0, |local| local.top_rows.stored_rows())
                        });
                    }
                }
                report
            }
            None => serde_json::json!({
                "backend": "lr-table",
                "lr_table_present": self.table.is_present(),
                "linked_lr_provider": self.table.linked_lr().is_some(),
                "sparse_regular": self.uses_sparse_direct_regular_runtime()
            }),
        }
    }
}
