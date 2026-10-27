//! Passive grammar-boundary observations. Markers never enter the model-token
//! inventory, never execute parser actions, and never participate in mask walks.
#[cfg(test)]
use crate::test_modes::MaskMode;
use std::sync::Arc;
use crate::automata::lexer::Lexer;

use crate::compiler::glr::parser::ParserGSS;
use crate::grammar::flat::{GrammarDef, Terminal};
use crate::runtime::Constraint;
use crate::compiler::glr::accumulator::TerminalsDisallowed;
use crate::ds::leveled_gss::VirtualStack;
use super::commit::TemplateAdvanceRuntime;
use glrmask_parser_dwa::__private::templates::admissibility::{TemplateDomain, TopAdmission};
use glrmask_parser_dwa::__private::templates::native::{NativeTableIndex, ProgramCompiler};
use crate::compiler::glr::table::AdmissionPolicy;

/// A marker ID, from `@mark(id)` in the grammar.
#[doc(hidden)]
pub type MarkerId = u32;

/// A marker passed by a commit.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MarkerEvent {
    /// The marker's ID.
    pub id: MarkerId,
    /// Where the marker is, in bytes from the start of the sequence.
    pub byte_offset: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
pub(crate) struct MarkerTerminal {
    pub(crate) terminal_id: u32,
    pub(crate) marker_id: u32,
}

#[derive(Debug, Clone, Copy)]
struct Candidate {
    terminal: MarkerTerminal,
    unconditional: bool,
}

/// Immutable CSR candidate index. IDs may be sparse; never allocate by MarkerId.
#[derive(Debug)]
pub(crate) struct MarkerQueries {
    pub(crate) terminals: Box<[MarkerTerminal]>,
    ids: Box<[MarkerId]>,
    offsets: Box<[usize]>,
    candidates: Box<[Candidate]>,
    empty: Box<[MarkerId]>,
    // Read-only predicates over the retained LR parser's existing stack alphabet.
    // No marker transitions or alternate executable parser are retained here.
    lr_domains: Box<[Option<Arc<TemplateDomain>>]>,
}

impl MarkerQueries {
    fn build(constraint: &Constraint, mut terminals: Vec<MarkerTerminal>) -> Result<Self, String> {
        terminals.sort_unstable();
        terminals.dedup();
        let terminal_count = constraint.parser_terminal_count();
        if terminals.iter().any(|t| t.terminal_id >= terminal_count) {
            return Err("marker terminal lies outside the parser inventory".into());
        }
        if terminals.windows(2).any(|pair| pair[0].terminal_id == pair[1].terminal_id) {
            return Err("one probe terminal cannot have two marker IDs".into());
        }
        for marker in &terminals {
            if constraint.ignore_terminal == Some(marker.terminal_id)
                || constraint.parser_skip_terminals().contains(&marker.terminal_id)
                || constraint.parser_is_control_terminal(marker.terminal_id)
                || constraint.special_token_terminals.iter().any(|special| special.terminal_id == marker.terminal_id)
                || constraint.tokenizer.possible_future_terminals(constraint.tokenizer.initial_state())
                    .contains(marker.terminal_id as usize) {
                return Err("marker probe must not be bound to lexical input, a model token, or parser control".into());
            }
        }
        let mut ids = terminals.iter().map(|t| t.marker_id).collect::<Vec<_>>();
        ids.sort_unstable(); ids.dedup();
        let lr_domains = Self::compile_lr_domains(constraint, &terminals)?;
        let states = constraint.parser_symbol_count() as usize;
        const MAX_INDEX_BYTES: usize = 64 * 1024 * 1024;
        let base_bytes = states.checked_add(1).and_then(|n| n.checked_mul(2 * std::mem::size_of::<usize>()))
            .filter(|&bytes| bytes <= MAX_INDEX_BYTES).ok_or("marker index alphabet exceeds its memory budget")?;
        let max_pairs = (MAX_INDEX_BYTES - base_bytes)
            / (2 * std::mem::size_of::<(u32, Candidate)>() + std::mem::size_of::<Candidate>());
        let mut pairs = Vec::<(u32, Candidate)>::new();
        let mut counts = vec![0usize; states + 1];
        let mut empty = Vec::new();
        let mut record = |top: u32, terminal: MarkerTerminal, certificate: TopAdmission| -> Result<(), String> {
            if top as usize >= states || certificate == TopAdmission::Never { return Ok(()); }
            if pairs.len() >= max_pairs { return Err("marker candidate index exceeds its memory budget".into()); }
            counts[top as usize + 1] += 1;
            pairs.push((top, Candidate { terminal, unconditional: certificate == TopAdmission::Always }));
            Ok(())
        };
        if let Some(parser) = &constraint.template_parser {
            for &terminal in &terminals {
                if parser.marker_admits_flat(&[], terminal.terminal_id) { empty.push(terminal.marker_id); }
                if let Some((default, exceptions)) = parser.marker_top_partition(terminal.terminal_id) {
                    if default == TopAdmission::Never {
                        for (top, certificate) in exceptions { record(top, terminal, certificate)?; }
                    } else {
                        let mut exceptions = exceptions.peekable();
                        for top in 0..states as u32 {
                            let certificate = if exceptions.peek().is_some_and(|&(key, _)| key == top) {
                                exceptions.next().unwrap().1
                            } else { default };
                            record(top, terminal, certificate)?;
                        }
                    }
                } else {
                    // Scoped views may map owner domains nontrivially. Keep the
                    // exact classifier rather than assuming a contiguous child
                    // alphabet, and bound this uncommon construction explicitly.
                    if states.checked_mul(terminals.len()).is_none_or(|n| n > 32_000_000) {
                        return Err("scoped marker index exceeds its construction budget".into());
                    }
                    for top in 0..states as u32 { record(top, terminal, parser.marker_top_admission(top, terminal.terminal_id))?; }
                }
            }
        } else if !lr_domains.is_empty() {
            // ExactSimulation needs a stack-domain proof, not merely a row.
            // Compile that proof once instead of replaying reductions per event.
            for &terminal in &terminals {
                let domain = lr_domains[terminal.terminal_id as usize].as_ref()
                    .expect("selected marker domain is complete");
                if domain.matches_top_first(std::iter::empty()) { empty.push(terminal.marker_id); }
                let (default, exceptions) = domain.top_admission_partition();
                if default == TopAdmission::Never {
                    for (top, certificate) in exceptions { record(top, terminal, certificate)?; }
                } else {
                    let mut exceptions = exceptions.peekable();
                    for top in 0..states as u32 {
                        let certificate = if exceptions.peek().is_some_and(|&(key, _)| key == top) {
                            exceptions.next().unwrap().1
                        } else { default };
                        record(top, terminal, certificate)?;
                    }
                }
            }
        } else {
            // Retained LR already has sparse admitted-terminal bitsets. Walk
            // present entries, not every marker for every state.
            let by_terminal = terminals.iter().map(|t| (t.terminal_id, *t))
                .collect::<rustc_hash::FxHashMap<_, _>>();
            for top in 0..states as u32 {
                if let Some(row) = constraint.table.advance_row(top) {
                    for terminal in row.iter().filter_map(|id| by_terminal.get(&(id as u32))) {
                        let certificate = if constraint.table.admission_policy == AdmissionPolicy::RowPresenceExact
                            || constraint.table.unconditional_advance_row(top)
                                .is_some_and(|row| row.contains(terminal.terminal_id as usize)) {
                            TopAdmission::Always
                        } else { TopAdmission::DependsOnSuffix };
                        record(top, *terminal, certificate)?;
                    }
                }
            }
        }
        for top in 1..counts.len() { counts[top] += counts[top - 1]; }
        let mut write = counts[..states].to_vec();
        let mut candidates = vec![Candidate { terminal: MarkerTerminal { terminal_id: 0, marker_id: 0 }, unconditional: false }; pairs.len()];
        for (top, candidate) in pairs {
            candidates[write[top as usize]] = candidate;
            write[top as usize] += 1;
        }
        empty.sort_unstable(); empty.dedup();
        Ok(Self { terminals: terminals.into(), ids: ids.into(), offsets: counts.into(), candidates: candidates.into(), empty: empty.into(), lr_domains })
    }

    fn compile_lr_domains(constraint: &Constraint, terminals: &[MarkerTerminal])
        -> Result<Box<[Option<Arc<TemplateDomain>>]>, String>
    {
        if constraint.has_template_parser()
            || constraint.table.admission_policy == AdmissionPolicy::RowPresenceExact {
            return Ok(Box::new([]));
        }
        let count = constraint.parser_terminal_count() as usize;
        if count > (64 * 1024 * 1024) / std::mem::size_of::<Option<Arc<TemplateDomain>>>() {
            return Err("marker predicate inventory exceeds its memory budget".into());
        }
        let mut selected = vec![false; count];
        for terminal in terminals { selected[terminal.terminal_id as usize] = true; }
        let index = NativeTableIndex::new(&constraint.table, &selected)?;
        let compiler = ProgramCompiler::new();
        let compiled = index.compile_selected(|characterization| {
            let program = compiler.compile(characterization)?;
            TemplateDomain::compile(&program)
        }).map_err(|error| format!("marker admission predicate: {error}"))?;
        let mut domains = vec![None; count];
        let mut bytes = count * std::mem::size_of::<Option<Arc<TemplateDomain>>>();
        for group in compiled.groups {
            bytes = bytes.checked_add(group.program.heap_payload_bytes())
                .filter(|&size| size <= 64 * 1024 * 1024)
                .ok_or("marker predicates exceed their memory budget")?;
            for terminal in group.terminals {
                domains[terminal as usize] = Some(Arc::clone(&group.program));
            }
        }
        Ok(domains.into_boxed_slice())
    }

    #[inline]
    fn admits_gss(&self, constraint: &Constraint, stack: &ParserGSS, terminal: u32) -> bool {
        if let Some(Some(domain)) = self.lr_domains.get(terminal as usize) {
            super::parser_backend::matches_gss(domain, stack)
        } else {
            super::commit::parser_may_advance_on(constraint, stack, terminal)
        }
    }

    #[inline]
    fn for_top(&self, top: u32) -> &[Candidate] {
        let top = top as usize;
        if top + 1 >= self.offsets.len() { return &[]; }
        &self.candidates[self.offsets[top]..self.offsets[top + 1]]
    }

    /// Union observations over represented parser alternatives, with no output
    /// stack construction for native predicates. Duplicate observations are
    /// removed when the public batch is completed.
    pub(crate) fn observe_gss(&self, constraint: &Constraint, stack: &ParserGSS,
        offset: u64, out: &mut Vec<MarkerEvent>) {
        if stack.is_empty() { return; }
        if !self.empty.is_empty() && stack.contains_empty_stack() {
            out.extend(self.empty.iter().map(|&id| MarkerEvent { id, byte_offset: offset }));
        }
        if let Some(top) = stack.single_exclusive_top_value() {
            self.observe_top(constraint, stack, top, offset, out);
        } else {
            stack.for_each_top_value(|top| self.observe_top(constraint, stack, top, offset, out));
        }
    }

    #[inline]
    fn observe_top(&self, constraint: &Constraint, stack: &ParserGSS,
        top: u32, offset: u64, out: &mut Vec<MarkerEvent>) {
        for candidate in self.for_top(top) {
            if candidate.unconditional || self.admits_gss(
                constraint, stack, candidate.terminal.terminal_id) {
                out.push(MarkerEvent { id: candidate.terminal.marker_id, byte_offset: offset });
            }
        }
    }

    pub(crate) fn observe_flat(&self, constraint: &Constraint, stack: &[u32],
        offset: u64, out: &mut Vec<MarkerEvent>) {
        let Some(&top) = stack.last() else {
            out.extend(self.empty.iter().map(|&id| MarkerEvent { id, byte_offset: offset }));
            return;
        };
        let mut fallback = None;
        for candidate in self.for_top(top) {
            let admitted = candidate.unconditional || if let Some(Some(domain)) = self.lr_domains.get(candidate.terminal.terminal_id as usize) {
                domain.matches_top_first(stack.iter().rev().copied())
            } else if let Some(parser) = &constraint.template_parser {
                parser.marker_admits_flat(stack, candidate.terminal.terminal_id)
            } else {
                // Only a genuinely suffix-dependent retained-LR query needs a
                // borrowed-stack adapter. Top-certified markers allocate none.
                let gss = fallback.get_or_insert_with(|| ParserGSS::from_single_stack(stack.to_vec(),
                    crate::compiler::glr::accumulator::TerminalsDisallowed::new()));
                crate::runtime::commit::parser_may_advance_on(constraint, gss, candidate.terminal.terminal_id)
            };
            if admitted { out.push(MarkerEvent { id: candidate.terminal.marker_id, byte_offset: offset }); }
        }
    }
}

impl Constraint {
    /// The IDs of the grammar's markers, in ascending order.
    #[cfg(any(test, feature = "internal-api"))]
    #[doc(hidden)]
    pub fn marker_ids(&self) -> &[MarkerId] {
        self.marker_queries.as_deref().map_or(&[], |queries| queries.ids.as_ref())
    }

    pub(crate) fn install_marker_terminals(&mut self, terminals: Vec<MarkerTerminal>) -> Result<(), String> {
        self.marker_queries = if terminals.is_empty() { None }
            else { Some(Arc::new(MarkerQueries::build(self, terminals)?)) };
        Ok(())
    }

    pub(crate) fn install_grammar_markers(&mut self, grammar: &GrammarDef) -> Result<(), String> {
        let terminals = grammar.terminals.iter().filter_map(|terminal| match terminal {
            Terminal::MarkerProbe { id, marker_id } => Some(MarkerTerminal { terminal_id: *id, marker_id: *marker_id }),
            _ => None,
        }).collect();
        self.install_marker_terminals(terminals)
    }
}

/// Reused per-state storage. Absent entirely for unannotated constraints.
#[derive(Debug, Clone)]
pub(crate) struct MarkerState {
    pub(crate) byte_offset: u64,
    pub(crate) events: Vec<MarkerEvent>,
    last_positions: Vec<Option<u64>>,
}

impl MarkerState {
    fn new(queries: &MarkerQueries) -> Self {
        Self { byte_offset: 0, events: Vec::with_capacity(queries.ids.len().min(16)),
            last_positions: vec![None; queries.ids.len()] }
    }

    pub(crate) fn begin(&mut self) { self.events.clear(); }

    pub(crate) fn finish(&mut self, queries: &MarkerQueries, next_offset: u64, success: bool) {
        if !success { self.events.clear(); return; }
        if self.events.windows(2).any(|p| (p[0].byte_offset, p[0].id) > (p[1].byte_offset, p[1].id)) {
            self.events.sort_unstable_by_key(|e| (e.byte_offset, e.id));
        }
        self.events.dedup();
        self.events.retain(|event| {
            let slot = queries.ids.binary_search(&event.id).expect("compiled marker ID");
            if self.last_positions[slot] == Some(event.byte_offset) { return false; }
            self.last_positions[slot] = Some(event.byte_offset);
            true
        });
        self.byte_offset = next_offset;
    }
}

/// Explicit compile-time effect parameter: ordinary commits and all internal
/// mask probes instantiate NoMarkers, so their inner loops contain no callback,
/// pointer chase, marker branch, or event-buffer operation.
pub(crate) trait CommitObserver {
    const ENABLED: bool;
    #[inline] fn checkpoint(&self) -> usize { 0 }
    #[inline] fn rewind(&mut self, _checkpoint: usize) {}
    #[inline] fn stack(&mut self, _constraint: &Constraint, _stack: &ParserGSS, _offset: usize) {}
    #[inline] fn flat(&mut self, _constraint: &Constraint, _stack: &[u32], _offset: usize) {}
    #[inline] fn carried(&mut self, _constraint: &Constraint,
        _stack: &VirtualStack<u32, TerminalsDisallowed>, _offset: usize) {}
    #[inline] fn language(&mut self, _constraint: &Constraint,
        _runtime: &mut TemplateAdvanceRuntime, _language: u32,
        _accumulator: &TerminalsDisallowed, _offset: usize) {}
    #[inline] fn frontier(&mut self, constraint: &Constraint, frontier: &super::state::ParserStateMap, offset: usize) {
        if Self::ENABLED {
            for (&lexer, stack) in frontier.iter() {
                if super::commit::runtime_tokenizer_is_reset_state(constraint, lexer) {
                    self.stack(constraint, stack, offset);
                }
            }
        }
    }
    #[inline] fn attempt<T>(&mut self, f: impl FnOnce(&mut Self) -> Option<T>) -> Option<T> where Self: Sized {
        let checkpoint = self.checkpoint();
        let result = f(self);
        if result.is_none() { self.rewind(checkpoint); }
        result
    }
    #[inline] fn shifted<T>(&mut self, _offset: usize, f: impl FnOnce(&mut Self) -> T) -> T where Self: Sized { f(self) }
}

pub(crate) struct NoMarkers;
impl CommitObserver for NoMarkers { const ENABLED: bool = false; }

pub(crate) struct LiveMarkers<'a> {
    pub(crate) queries: &'a MarkerQueries,
    pub(crate) state: &'a mut MarkerState,
    pub(crate) base: u64,
}

impl CommitObserver for LiveMarkers<'_> {
    const ENABLED: bool = true;
    #[inline] fn checkpoint(&self) -> usize { self.state.events.len() }
    #[inline] fn rewind(&mut self, checkpoint: usize) { self.state.events.truncate(checkpoint); }
    #[inline] fn stack(&mut self, constraint: &Constraint, stack: &ParserGSS, offset: usize) {
        self.queries.observe_gss(constraint, stack, self.base + offset as u64, &mut self.state.events);
    }
    #[inline] fn flat(&mut self, constraint: &Constraint, stack: &[u32], offset: usize) {
        self.queries.observe_flat(constraint, stack, self.base + offset as u64, &mut self.state.events);
    }
    #[inline] fn carried(&mut self, constraint: &Constraint,
        stack: &VirtualStack<u32, TerminalsDisallowed>, offset: usize) {
        let Some(&top) = stack.top() else {
            // A drained virtual prefix can expose an empty word or a shared
            // nonempty floor. Query its exact existing language, never advance.
            self.queries.observe_gss(constraint, &stack.clone().into_gss(),
                self.base + offset as u64, &mut self.state.events);
            return;
        };
        let mut fallback = None;
        for candidate in self.queries.for_top(top) {
            if candidate.unconditional || {
                let gss = fallback.get_or_insert_with(|| stack.clone().into_gss());
                self.queries.admits_gss(constraint, gss, candidate.terminal.terminal_id)
            } {
                self.state.events.push(MarkerEvent { id: candidate.terminal.marker_id,
                    byte_offset: self.base + offset as u64 });
            }
        }
    }
    #[inline] fn language(&mut self, constraint: &Constraint,
        runtime: &mut TemplateAdvanceRuntime, language: u32,
        accumulator: &TerminalsDisallowed, offset: usize) {
        if !self.queries.empty.is_empty() && runtime.language_contains_empty_stack(language) {
            self.state.events.extend(self.queries.empty.iter().map(|&id| MarkerEvent {
                id, byte_offset: self.base + offset as u64,
            }));
        }
        let mut fallback = None;
        for top in runtime.language_top_states(language) {
            for candidate in self.queries.for_top(top) {
                if candidate.unconditional || {
                    let gss = fallback.get_or_insert_with(|| runtime.gss_from_language(language, accumulator.clone()));
                    self.queries.admits_gss(constraint, gss, candidate.terminal.terminal_id)
                } {
                    self.state.events.push(MarkerEvent { id: candidate.terminal.marker_id,
                        byte_offset: self.base + offset as u64 });
                }
            }
        }
    }
    #[inline] fn shifted<T>(&mut self, offset: usize, f: impl FnOnce(&mut Self) -> T) -> T {
        self.base += offset as u64;
        let result = f(self);
        self.base -= offset as u64;
        result
    }
}

impl super::ConstraintState<'_> {
    pub(crate) fn initialize_markers(&mut self) {
        let Some(queries) = self.constraint.marker_queries.as_deref() else { return; };
        let mut markers = Box::new(MarkerState::new(queries));
        LiveMarkers { queries, state: &mut markers, base: 0 }.frontier(self.constraint, &self.state, 0);
        markers.finish(queries, 0, true);
        self.markers = Some(markers);
    }

    /// The markers passed by the most recent commit, or by
    /// [`Constraint::start`], in byte order.
    ///
    /// A marker means some parse reached that point. That parse may still fail
    /// later.
    #[cfg(any(test, feature = "internal-api"))]
    #[doc(hidden)]
    pub fn marker_events(&self) -> &[MarkerEvent] {
        self.markers.as_deref().map_or(&[], |markers| markers.events.as_slice())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ConstraintState;
    use crate::{Grammar, Vocab};

    fn observations(state: &ConstraintState<'_>) -> Vec<u32> {
        let mut events = Vec::new();
        if let Some(queries) = &state.constraint.marker_queries {
            for (&lexer, stack) in state.state.iter() {
                if crate::runtime::commit::runtime_tokenizer_is_reset_state(state.constraint, lexer) {
                    queries.observe_gss(state.constraint, stack, 0, &mut events);
                }
            }
        }
        let mut ids = events.iter().map(|e| e.id).collect::<Vec<_>>();
        ids.sort_unstable(); ids.dedup(); ids
    }

    #[test]
    fn passive_probe_admission_and_masks_match_erased_markers_in_all_modes() {
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec()), (2, b"ab".to_vec())]);
        for mode in MaskMode::ALL {
            let marked = Grammar::from_glrm("start root\nnt root = @mark(0) \"a\" @mark(7) \"b\" @mark(42)\n").unwrap()
                .compile_with_build_options(&vocab, mode.build_options()).unwrap();
            let ordinary = Grammar::from_glrm("start root\nnt root = \"a\" \"b\"\n").unwrap()
                .compile_with_build_options(&vocab, mode.build_options()).unwrap();
            assert_eq!(marked.marker_ids(), [0, 7, 42], "{mode:?}");
            let mut state = marked.start(); let mut reference = ordinary.start();
            assert_eq!(observations(&state), [0], "initial {mode:?}");
            assert_eq!(state.mask(), reference.mask(), "initial mask {mode:?}");
            for (token, marker) in [(0, 7), (1, 42)] {
                assert!(state.commit_token(token)); assert!(reference.commit_token(token));
                let before = state.state.clone();
                assert_eq!(observations(&state), [marker], "{mode:?} token {token}");
                assert_eq!(state.state, before, "marker query mutated the parser");
                assert_eq!(state.mask(), reference.mask(), "mask {mode:?} token {token}");
                assert_eq!(state.is_accepting(), reference.is_accepting());
            }
        }
    }

    #[test]
    fn observed_commits_preserve_exact_frontier_and_masks_across_ambiguous_grammars() {
        let pieces: &[&[u8]] = &[b"a", b"b", b"c", b"x", b"ab", b"aa", b"bb", b"abc",
            b"(", b")", b"(x)", b"((x))", b" ", b"  ", b"a b", b":", b"a:a", b"aaaa", b"aaabbb"];
        let vocab = Vocab::new(pieces.iter().enumerate().map(|(i,b)| (i as u32,b.to_vec())).collect());
        let sources = [
            "start root\nnt root = (@mark(1) \"a\" @mark(2) \"b\" @mark(3))*\n",
            "start root\nnt root = \"x\" | \"(\" @mark(1) root @mark(2) \")\"\n",
            "start root\nnt root = \"a\" @mark(1) root \"b\" @mark(2) | eps\n",
            "start root\nnt root = (\"a\" @mark(1) \"b\" | \"a\" @mark(2) \"c\")*\n",
            "start root\nignore WS\nt WS = \" \"\nnt root = (\"a\" @mark(1) \"b\" @mark(2))*\n",
            "start root\nt WORD = /[a-c]+/\nnt root = @mark(1) WORD @mark(2) \":\" WORD @mark(3)\n",
            "start root\nnt root = @mark(1)* (\"a\" @mark(2))* @mark(3)\n",
        ];
        let mut comparisons = 0usize;
        for mode in MaskMode::ALL {
            for source in sources {
                let constraint = Grammar::from_glrm(source).unwrap().compile_with_build_options(&vocab,
                    mode.build_options()).unwrap();
                for seed in 0..16u64 {
                    let mut observed = constraint.start();
                    let mut silent = observed.clone();
                    silent.markers = None;
                    let mut rng = seed + 123;
                    let mut seen = std::collections::BTreeSet::new();
                    seen.extend(observed.marker_events().iter().map(|e| (e.id,e.byte_offset)));
                    for step in 0..32 {
                        let mask = observed.mask();
                        assert_eq!(mask, silent.mask(), "{mode:?} {source} seed {seed} step {step}");
                        let allowed = (0..pieces.len()).filter(|&id| mask[id/32] & (1<<(id%32)) != 0)
                            .collect::<Vec<_>>();
                        if allowed.is_empty() { break; }
                        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                        let token = allowed[rng as usize % allowed.len()];
                        let (left,right) = if seed % 2 == 0 {
                            (observed.commit_token(token as u32),silent.commit_token(token as u32))
                        } else { (observed.commit_bytes(pieces[token]),silent.commit_bytes(pieces[token])) };
                        assert_eq!(left,right);
                        assert_eq!(observed.state,silent.state,"observer changed frontier: {mode:?} {source} seed {seed} step {step}");
                        assert_eq!(observed.is_accepting(),silent.is_accepting());
                        // Independent reference: query the ordinary parser,
                        // not the marker's precompiled certificate/domain.
                        if let Some(queries) = &constraint.marker_queries {
                            let mut oracle = Vec::new();
                            for marker in &queries.terminals {
                                if observed.state.iter().any(|(&lexer,gss)|
                                    super::super::commit::runtime_tokenizer_is_reset_state(&constraint,lexer)
                                    && super::super::commit::parser_may_advance_on(&constraint,gss,marker.terminal_id)) {
                                    oracle.push(marker.marker_id);
                                }
                            }
                            oracle.sort_unstable(); oracle.dedup();
                            assert_eq!(observations(&observed),oracle,
                                "passive predicate changed admission: {mode:?} {source} seed {seed} step {step}");
                        }
                        for event in observed.marker_events() {
                            assert!(seen.insert((event.id,event.byte_offset)),"duplicate marker across batches");
                        }
                        comparisons += 1;
                        if !left { break; }
                    }
                }
            }
        }
        assert!(comparisons > 3_000, "only {comparisons} observed/silent commits compared");
        eprintln!("MARKER_FRONTIER_PARITY comparisons={comparisons}");
    }

    #[test]
    fn empty_word_is_distinct_from_no_parser_stack() {
        let empty = ParserGSS::empty();
        let epsilon = ParserGSS::from_single_stack(Vec::new(), TerminalsDisallowed::new());
        let one = ParserGSS::from_single_stack(vec![0], TerminalsDisallowed::new());
        assert!(empty.is_empty()); assert!(!empty.contains_empty_stack());
        assert!(!epsilon.is_empty()); assert!(epsilon.contains_empty_stack());
        assert!(!one.is_empty()); assert!(!one.contains_empty_stack());
        assert!(epsilon.merge(&one).contains_empty_stack());
        assert!(!one.merge(&empty).contains_empty_stack());
    }
    #[test]
    fn observer_adapters_preserve_empty_and_mixed_stack_languages() {
        let vocab = Vocab::new(vec![(0, b"a".to_vec())]);
        let constraint = Grammar::from_ebnf(r#"start ::= "a""#).unwrap().compile(&vocab).unwrap();
        // A valid admission index whose sole predicate accepts the empty word.
        // This isolates adapter semantics from any particular grammar frontend.
        let queries = MarkerQueries { terminals: Box::new([]), ids: vec![7].into(),
            offsets: vec![0].into(), candidates: Box::new([]), empty: vec![7].into(),
            lr_domains: Box::new([]) };
        let acc = TerminalsDisallowed::new();
        let empty = ParserGSS::from_single_stack(vec![], acc.clone());
        let nonempty = ParserGSS::from_single_stack(vec![0], acc.clone());
        let mixed = ParserGSS::from_stacks(&[(vec![],acc.clone()),(vec![0],acc.clone())]);
        for stack in [&empty, &nonempty, &mixed] {
            let mut runtime = TemplateAdvanceRuntime::default();
            let (language, accumulator) = runtime.language_from_uniform_gss(stack).unwrap();
            let mut marker_state = MarkerState::new(&queries);
            LiveMarkers { queries: &queries, state: &mut marker_state, base: 10 }
                .language(&constraint, &mut runtime, language, &accumulator, 2);
            marker_state.finish(&queries,12,true);
            let expected = if stack.contains_empty_stack() {
                vec![MarkerEvent { id:7,byte_offset:12 }]
            } else { vec![] };
            assert_eq!(marker_state.events,expected);
        }
        let mut runtime = TemplateAdvanceRuntime::default();
        let mut rejected = MarkerState::new(&queries);
        LiveMarkers { queries:&queries,state:&mut rejected,base:0 }
            .language(&constraint,&mut runtime,0,&acc,0);
        assert!(rejected.events.is_empty());
        let mut carried = nonempty.try_virtual_stack().unwrap();
        assert_eq!(carried.pop(1),0);
        assert!(carried.top().is_none());
        assert!(carried.clone().into_gss().contains_empty_stack());
        let mut result = MarkerState::new(&queries);
        LiveMarkers { queries:&queries,state:&mut result,base:10 }
            .carried(&constraint,&carried,2);
        result.finish(&queries,12,true);
        assert_eq!(result.events,[MarkerEvent {id:7,byte_offset:12}]);
    }

}
