//! Compositional fixed-vocabulary interface-tail summaries.
//!
//! A reusable boundary whitelist only asks whether some positive proper prefix
//! of a model token can reach this component's current public interface.  The
//! history before the token is existential, so carrying `(raw lexer state,
//! parser history position)` through every byte preserves far more information
//! than that observation requires.
//!
//! This module instead propagates typed normal-return/public-exit tails through
//! the grammar.  The production hierarchy currently keeps one-byte tails for
//! ordinary leaf preparation and a two-byte compositional refinement. Unknown
//! interiors widen locally; known parent postambles remain exact.

use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use rayon::prelude::*;

use crate::automata::lexer::ast::Expr;
use crate::automata::lexer::tokenizer::Lexer;
use crate::grammar::flat::{NonterminalID, Rule, Symbol, TerminalID};
use crate::runtime::Constraint;

const DEFAULT_FIXED_POINT_LIMIT: usize = 256;
const FINITE_EXPR_WORD_CAP: usize = 4096;

fn finite_literal_words(expr: &Expr, cap: usize) -> Option<Vec<Vec<u8>>> {
    fn combine(left: Vec<Vec<u8>>, right: Vec<Vec<u8>>, cap: usize) -> Option<Vec<Vec<u8>>> {
        if left.len().saturating_mul(right.len()) > cap {
            return None;
        }
        let mut out = Vec::with_capacity(left.len().saturating_mul(right.len()));
        for a in &left {
            for b in &right {
                let mut joined = Vec::with_capacity(a.len() + b.len());
                joined.extend_from_slice(a);
                joined.extend_from_slice(b);
                out.push(joined);
            }
        }
        out.sort_unstable();
        out.dedup();
        (out.len() <= cap).then_some(out)
    }

    match expr {
        Expr::U8Seq(bytes) => Some(vec![bytes.clone()]),
        Expr::U8Class(class) => {
            let words = class.iter().map(|byte| vec![byte]).collect::<Vec<_>>();
            (words.len() <= cap).then_some(words)
        }
        Expr::Epsilon => Some(vec![Vec::new()]),
        Expr::Shared(inner) => finite_literal_words(inner, cap),
        Expr::Seq(parts) => {
            let mut words = vec![Vec::new()];
            for part in parts {
                words = combine(words, finite_literal_words(part, cap)?, cap)?;
            }
            Some(words)
        }
        Expr::Choice(options) => {
            let mut words = Vec::new();
            for option in options {
                words.extend(finite_literal_words(option, cap)?);
                if words.len() > cap {
                    return None;
                }
            }
            words.sort_unstable();
            words.dedup();
            (words.len() <= cap).then_some(words)
        }
        Expr::Repeat { expr, min, max: Some(max) }
            if *max <= 16 && max.saturating_sub(*min) <= 8 => {
            let unit = finite_literal_words(expr, cap)?;
            let mut all = Vec::new();
            for count in *min..=*max {
                let mut words = vec![Vec::new()];
                for _ in 0..count {
                    words = combine(words, unit.clone(), cap)?;
                }
                all.extend(words);
                if all.len() > cap {
                    return None;
                }
            }
            all.sort_unstable();
            all.dedup();
            (all.len() <= cap).then_some(all)
        }
        Expr::Dfa(_)
        | Expr::Intersect { .. }
        | Expr::Exclude { .. }
        | Expr::Repeat { .. } => None,
    }
}

fn outward_terminals(constraint: &Constraint) -> BTreeSet<TerminalID> {
    constraint
        .unbound_grammar_placeholders
        .values()
        .copied()
        .chain(constraint.late_grammar_slots.iter().map(|slot| slot.terminal_id))
        .chain(
            constraint
                .special_token_terminals
                .iter()
                .filter(|special| {
                    constraint.is_late_grammar_placeholder_terminal(special.terminal_id)
                        || constraint.token_bytes_for_id(special.token_id).is_none()
                })
                .map(|special| special.terminal_id),
        )
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct ByteSet([u64; 4]);

impl ByteSet {
    fn full() -> Self {
        Self([u64::MAX; 4])
    }

    fn insert(&mut self, byte: u8) {
        self.0[byte as usize >> 6] |= 1u64 << (byte & 63);
    }

    fn contains(&self, byte: u8) -> bool {
        self.0[byte as usize >> 6] & (1u64 << (byte & 63)) != 0
    }

    fn union_with(&mut self, other: Self) {
        for (left, right) in self.0.iter_mut().zip(other.0) {
            *left |= right;
        }
    }

    fn count(&self) -> usize {
        self.0.iter().map(|word| word.count_ones() as usize).sum()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct ByteLanguage {
    epsilon: bool,
    bytes: ByteSet,
}

impl ByteLanguage {
    fn empty() -> Self {
        Self::default()
    }

    fn epsilon() -> Self {
        Self { epsilon: true, bytes: ByteSet::default() }
    }

    fn top() -> Self {
        Self { epsilon: true, bytes: ByteSet::full() }
    }

    fn union_with(&mut self, other: Self) {
        self.epsilon |= other.epsilon;
        self.bytes.union_with(other.bytes);
    }

    fn concat_tail(self, right: Self) -> Self {
        // r=1 suffix algebra. Any one-byte observation produced by the right
        // operand hides the left. Only an epsilon completion of the right can
        // expose the left operand's final byte.
        let mut bytes = right.bytes;
        if right.epsilon {
            bytes.union_with(self.bytes);
        }
        Self { epsilon: self.epsilon && right.epsilon, bytes }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BytePhaseSummary {
    historical_productive: bool,
    normal: ByteLanguage,
    event_from_entry: ByteLanguage,
    tail_to_return: ByteLanguage,
    tail_to_event: ByteLanguage,
}

impl BytePhaseSummary {
    fn bottom() -> Self {
        Self {
            historical_productive: false,
            normal: ByteLanguage::empty(),
            event_from_entry: ByteLanguage::empty(),
            tail_to_return: ByteLanguage::empty(),
            tail_to_event: ByteLanguage::empty(),
        }
    }

    fn identity() -> Self {
        Self {
            historical_productive: true,
            normal: ByteLanguage::epsilon(),
            event_from_entry: ByteLanguage::empty(),
            tail_to_return: ByteLanguage::empty(),
            tail_to_event: ByteLanguage::empty(),
        }
    }

    fn union_with(&mut self, other: Self) {
        self.historical_productive |= other.historical_productive;
        self.normal.union_with(other.normal);
        self.event_from_entry.union_with(other.event_from_entry);
        self.tail_to_return.union_with(other.tail_to_return);
        self.tail_to_event.union_with(other.tail_to_event);
    }

    fn concat(self, right: Self) -> Self {
        let normal = self.normal.concat_tail(right.normal);
        let mut event_from_entry = self.event_from_entry;
        event_from_entry.union_with(self.normal.concat_tail(right.event_from_entry));

        let mut tail_to_return = self.tail_to_return.concat_tail(right.normal);
        if self.historical_productive {
            tail_to_return.union_with(right.tail_to_return);
        }

        let mut tail_to_event = self.tail_to_event;
        tail_to_event.union_with(self.tail_to_return.concat_tail(right.event_from_entry));
        if self.historical_productive {
            tail_to_event.union_with(right.tail_to_event);
        }

        Self {
            historical_productive: self.historical_productive && right.historical_productive,
            normal,
            event_from_entry,
            tail_to_return,
            tail_to_event,
        }
    }
}


fn expr_last_byte_language(expr: &Expr) -> ByteLanguage {
    match expr {
        Expr::U8Seq(bytes) => {
            if let Some(&last) = bytes.last() {
                let mut set = ByteSet::default();
                set.insert(last);
                ByteLanguage { epsilon: false, bytes: set }
            } else {
                ByteLanguage::epsilon()
            }
        }
        Expr::U8Class(class) => {
            let mut set = ByteSet::default();
            for byte in class.iter() {
                set.insert(byte);
            }
            ByteLanguage { epsilon: false, bytes: set }
        }
        Expr::Dfa(dfa) => {
            // `Expr::Dfa` is normally epsilon-free. If a hand-built or legacy
            // expression carries epsilon edges, direct transition/finalizer
            // inspection is not enough to certify its final byte. Widen the
            // lexical atom rather than risk an under-approximation.
            if dfa.has_epsilon_transitions() {
                return ByteLanguage::top();
            }
            let mut set = ByteSet::default();
            for state in 0..dfa.num_states() as u32 {
                for (byte, target) in dfa.transitions(state) {
                    if !dfa.finalizers(target).is_empty() {
                        set.insert(byte);
                    }
                }
            }
            ByteLanguage {
                epsilon: !dfa.finalizers(0).is_empty(),
                bytes: set,
            }
        }
        Expr::Intersect { expr, intersect } => {
            let left = expr_last_byte_language(expr);
            let right = expr_last_byte_language(intersect);
            let mut bytes = ByteSet::default();
            for i in 0..4 {
                bytes.0[i] = left.bytes.0[i] & right.bytes.0[i];
            }
            ByteLanguage {
                epsilon: left.epsilon && right.epsilon,
                bytes,
            }
        }
        Expr::Exclude { expr, exclude } => {
            let base = expr_last_byte_language(expr);
            // Exclusion can only remove words, so retaining the base last-byte
            // set is a sound upper bound. Nullability can be decided exactly.
            ByteLanguage {
                epsilon: base.epsilon && !exclude.is_nullable(),
                bytes: base.bytes,
            }
        }
        Expr::Seq(parts) => {
            let mut out = ByteLanguage::epsilon();
            for part in parts {
                out = out.concat_tail(expr_last_byte_language(part));
            }
            out
        }
        Expr::Choice(options) => {
            let mut out = ByteLanguage::empty();
            for option in options {
                out.union_with(expr_last_byte_language(option));
            }
            out
        }
        Expr::Repeat { expr, min, max } => {
            if *max == Some(0) {
                return ByteLanguage::epsilon();
            }
            let inner = expr_last_byte_language(expr);
            ByteLanguage {
                epsilon: *min == 0 || inner.epsilon,
                bytes: inner.bytes,
            }
        }
        Expr::Shared(inner) => expr_last_byte_language(inner),
        Expr::Epsilon => ByteLanguage::epsilon(),
    }
}

fn byte_terminal_summary(
    constraint: &Constraint,
    terminal: TerminalID,
    outward: &BTreeSet<TerminalID>,
    child_overrides: &BTreeMap<TerminalID, BytePhaseSummary>,
) -> BytePhaseSummary {
    if let Some(child) = child_overrides.get(&terminal) {
        return *child;
    }
    if outward.contains(&terminal) {
        return BytePhaseSummary {
            historical_productive: true,
            normal: ByteLanguage::empty(),
            event_from_entry: ByteLanguage::epsilon(),
            tail_to_return: ByteLanguage::epsilon(),
            tail_to_event: ByteLanguage::epsilon(),
        };
    }
    if constraint.table.control_terminals.contains(&terminal) {
        return BytePhaseSummary {
            historical_productive: true,
            normal: ByteLanguage::epsilon(),
            event_from_entry: ByteLanguage::empty(),
            tail_to_return: ByteLanguage::epsilon(),
            tail_to_event: ByteLanguage::empty(),
        };
    }
    if constraint
        .special_token_terminals
        .iter()
        .any(|special| special.terminal_id == terminal)
    {
        return BytePhaseSummary {
            historical_productive: true,
            normal: ByteLanguage::top(),
            event_from_entry: ByteLanguage::empty(),
            tail_to_return: ByteLanguage::top(),
            tail_to_event: ByteLanguage::empty(),
        };
    }

    let language = constraint
        .retained_terminal_expr(terminal)
        .map(expr_last_byte_language)
        .unwrap_or_else(ByteLanguage::top);
    BytePhaseSummary {
        historical_productive: language.epsilon || language.bytes.count() != 0,
        normal: language,
        event_from_entry: ByteLanguage::empty(),
        // A cut at the lexeme end contributes epsilon; any nonempty residual
        // completion has a final byte in the terminal's complete last-byte set.
        tail_to_return: ByteLanguage {
            epsilon: true,
            bytes: language.bytes,
        },
        tail_to_event: ByteLanguage::empty(),
    }
}

fn byte_rule_summary(
    rule: &Rule,
    nt: &BTreeMap<NonterminalID, BytePhaseSummary>,
    terminal: &[BytePhaseSummary],
) -> BytePhaseSummary {
    rule.rhs.iter().fold(BytePhaseSummary::identity(), |acc, symbol| {
        let right = match symbol {
            Symbol::Terminal(id) => terminal
                .get(*id as usize)
                .copied()
                .unwrap_or_else(BytePhaseSummary::bottom),
            Symbol::Nonterminal(id) => nt.get(id).copied().unwrap_or_else(BytePhaseSummary::bottom),
        };
        acc.concat(right)
    })
}

fn summarize_rules_module_r1(
    constraint: &Constraint,
    child_overrides: &BTreeMap<TerminalID, BytePhaseSummary>,
    bound_slots: &BTreeSet<TerminalID>,
) -> Result<(BytePhaseSummary, usize, bool), String> {
    let rules = constraint.retained_table_rules()?;
    if rules.is_empty() {
        return Err("r1 interface-tail summary requires retained grammar rules".to_owned());
    }
    let mut outward = outward_terminals(constraint);
    for terminal in bound_slots {
        outward.remove(terminal);
    }
    let terminals = (0..constraint.tokenizer.num_terminals())
        .map(|terminal| {
            byte_terminal_summary(
                constraint,
                terminal,
                &outward,
                child_overrides,
            )
        })
        .collect::<Vec<_>>();

    let mut nts = BTreeMap::<NonterminalID, BytePhaseSummary>::new();
    for rule in rules {
        nts.entry(rule.lhs).or_insert_with(BytePhaseSummary::bottom);
        for symbol in &rule.rhs {
            if let Symbol::Nonterminal(id) = symbol {
                nts.entry(*id).or_insert_with(BytePhaseSummary::bottom);
            }
        }
    }

    let mut converged = false;
    let mut iterations = 0usize;
    for iteration in 0..DEFAULT_FIXED_POINT_LIMIT {
        iterations = iteration + 1;
        let snapshot = nts.clone();
        let mut next = snapshot.clone();
        for rule in rules {
            let summary = byte_rule_summary(rule, &snapshot, &terminals);
            next.entry(rule.lhs)
                .or_insert_with(BytePhaseSummary::bottom)
                .union_with(summary);
        }
        if next == nts {
            converged = true;
            break;
        }
        nts = next;
    }
    if !converged {
        let top = ByteLanguage::top();
        return Ok((
            BytePhaseSummary {
                historical_productive: true,
                normal: top,
                event_from_entry: top,
                tail_to_return: top,
                tail_to_event: top,
            },
            iterations,
            true,
        ));
    }

    let mut module = byte_rule_summary(&rules[0], &nts, &terminals);

    // Local Skip/IGNORE is modeled conservatively at module exits. This keeps
    // real ignore-before-public-event/root-return witnesses while avoiding any
    // assumption about a future parent's reset domain.
    let mut ignore_ids = constraint.table.skip_terminals.clone();
    if let Some(ignore) = constraint.ignore_terminal {
        ignore_ids.insert(ignore);
    }
    for ignore in ignore_ids {
        let bytes = if let Some(child) = child_overrides.get(&ignore) {
            child.tail_to_return.bytes
        } else if let Some(expr) = constraint.retained_terminal_expr(ignore) {
            expr_last_byte_language(expr).bytes
        } else {
            ByteSet::full()
        };
        module.tail_to_return.bytes.union_with(bytes);
        if !module.event_from_entry.bytes.0.iter().all(|word| *word == 0)
            || module.event_from_entry.epsilon
            || !module.tail_to_event.bytes.0.iter().all(|word| *word == 0)
            || module.tail_to_event.epsilon
        {
            module.tail_to_event.bytes.union_with(bytes);
        }
    }
    Ok((module, iterations, false))
}

fn has_recursive_public_exit(constraint: &Constraint) -> bool {
    if !outward_terminals(constraint).is_empty() {
        return true;
    }
    constraint
        .static_dynamic_overlay
        .as_ref()
        .is_some_and(|overlay| {
            overlay
                .segmented_parser_components
                .iter()
                .any(|component| has_recursive_public_exit(component.constraint.as_ref()))
        })
}

fn opaque_child_r1(constraint: &Constraint) -> BytePhaseSummary {
    let top = ByteLanguage::top();
    let public = has_recursive_public_exit(constraint);
    BytePhaseSummary {
        historical_productive: true,
        normal: top,
        event_from_entry: if public { top } else { ByteLanguage::empty() },
        tail_to_return: top,
        tail_to_event: if public { top } else { ByteLanguage::empty() },
    }
}

fn summarize_composition_envelope_r1(
    parent: &Constraint,
    bindings: &[(TerminalID, &Constraint)],
) -> Result<(BytePhaseSummary, usize, bool), String> {
    let mut overrides = BTreeMap::<TerminalID, BytePhaseSummary>::new();
    let mut bound_slots = BTreeSet::<TerminalID>::new();
    for &(slot, child) in bindings {
        bound_slots.insert(slot);
        overrides.insert(slot, opaque_child_r1(child));
    }
    summarize_rules_module_r1(parent, &overrides, &bound_slots)
}

fn summarize_constraint_module_r1(
    constraint: &Constraint,
) -> Result<(BytePhaseSummary, usize, bool), String> {
    let Some(overlay) = constraint.static_dynamic_overlay.as_ref() else {
        return summarize_rules_module_r1(constraint, &BTreeMap::new(), &BTreeSet::new());
    };
    if overlay.segmented_parser_components.is_empty() {
        return summarize_rules_module_r1(constraint, &BTreeMap::new(), &BTreeSet::new());
    }

    let parent_component = overlay
        .segmented_parser_components
        .first()
        .ok_or_else(|| "composed constraint is missing root component".to_owned())?;
    let parent = parent_component.constraint.as_ref();
    let mut child_by_component = BTreeMap::<u32, (BytePhaseSummary, usize, bool)>::new();
    let mut overrides = BTreeMap::<TerminalID, BytePhaseSummary>::new();
    let mut bound_slots = BTreeSet::<TerminalID>::new();
    let mut iterations = 0usize;
    let mut widened = false;

    for link in &overlay.segmented_parser_links {
        if link.parent_component != 0 {
            // Immediate-level overlays should only point from the local parent;
            // nested structure lives inside the child constraint itself. If a
            // future artifact violates that shape, do not publish a partially
            // interpreted summary: widen every observable field explicitly.
            let top = ByteLanguage::top();
            return Ok((
                BytePhaseSummary {
                    historical_productive: true,
                    normal: top,
                    event_from_entry: top,
                    tail_to_return: top,
                    tail_to_event: top,
                },
                iterations,
                true,
            ));
        }
        let child = child_by_component.entry(link.child_component).or_insert_with(|| {
            overlay
                .segmented_parser_components
                .get(link.child_component as usize)
                .and_then(|component| summarize_constraint_module_r1(component.constraint.as_ref()).ok())
                .unwrap_or((
                    BytePhaseSummary {
                        historical_productive: true,
                        normal: ByteLanguage::top(),
                        event_from_entry: ByteLanguage::top(),
                        tail_to_return: ByteLanguage::top(),
                        tail_to_event: ByteLanguage::top(),
                    },
                    0,
                    true,
                ))
        });
        iterations = iterations.saturating_add(child.1);
        widened |= child.2;
        bound_slots.insert(link.slot_terminal);
        overrides.insert(link.slot_terminal, child.0);
    }

    let (module, local_iterations, local_widened) =
        summarize_rules_module_r1(parent, &overrides, &bound_slots)?;
    iterations = iterations.saturating_add(local_iterations);
    widened |= local_widened;
    Ok((module, iterations, widened))
}

fn summarize_root_r1(constraint: &Constraint) -> Result<(ByteLanguage, usize, bool), String> {
    let (module, iterations, widened) = summarize_constraint_module_r1(constraint)?;
    let mut exits = module.tail_to_return;
    exits.union_with(module.tail_to_event);
    Ok((exits, iterations, widened))
}

fn candidate_ids_for_r1(vocab: &crate::Vocab, language: ByteLanguage) -> Vec<u32> {
    let mut ids = Vec::new();
    for (id, bytes) in vocab.iter() {
        if bytes.len() < 2 {
            continue;
        }
        if bytes[..bytes.len() - 1]
            .iter()
            .copied()
            .any(|byte| language.bytes.contains(byte))
        {
            ids.push(id);
        }
    }
    ids.sort_unstable();
    ids.dedup();
    ids
}

pub(crate) fn build_composition_boundary_tail_r1(
    parent: &Constraint,
    bindings: &[(TerminalID, &Constraint)],
    vocab: &crate::Vocab,
) -> Result<BoundaryTailR1Result, String> {
    let summary_started = Instant::now();
    let (module, fixed_point_iterations, fixed_point_widened) =
        summarize_composition_envelope_r1(parent, bindings)?;
    let mut language = module.tail_to_return;
    language.union_with(module.tail_to_event);
    let summary_ms = summary_started.elapsed().as_secs_f64() * 1000.0;
    let map_started = Instant::now();
    let candidate_ids = candidate_ids_for_r1(vocab, language);
    let map_ms = map_started.elapsed().as_secs_f64() * 1000.0;
    Ok(BoundaryTailR1Result {
        candidate_ids,
        exit_byte_count: language.bytes.count(),
        fixed_point_iterations,
        fixed_point_widened,
        summary_ms,
        map_ms,
    })
}

#[derive(Debug, Clone)]
pub(crate) struct BoundaryTailR1Result {
    pub(crate) candidate_ids: Vec<u32>,
    pub(crate) exit_byte_count: usize,
    pub(crate) fixed_point_iterations: usize,
    pub(crate) fixed_point_widened: bool,
    pub(crate) summary_ms: f64,
    pub(crate) map_ms: f64,
}

pub(crate) fn build_boundary_tail_r1(
    constraint: &Constraint,
    vocab: &crate::Vocab,
) -> Result<BoundaryTailR1Result, String> {
    let summary_started = Instant::now();
    let (language, fixed_point_iterations, fixed_point_widened) = summarize_root_r1(constraint)?;
    let summary_ms = summary_started.elapsed().as_secs_f64() * 1000.0;
    let map_started = Instant::now();
    let candidate_ids = candidate_ids_for_r1(vocab, language);
    let map_ms = map_started.elapsed().as_secs_f64() * 1000.0;
    Ok(BoundaryTailR1Result {
        candidate_ids,
        exit_byte_count: language.bytes.count(),
        fixed_point_iterations,
        fixed_point_widened,
        summary_ms,
        map_ms,
    })

}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PairSet {
    // Row indexed by first byte; each row contains possible second bytes.
    rows: Box<[ByteSet; 256]>,
}

impl Default for PairSet {
    fn default() -> Self {
        Self {
            rows: Box::new([ByteSet::default(); 256]),
        }
    }
}

impl PairSet {
    fn insert(&mut self, first: u8, second: u8) {
        self.rows[first as usize].insert(second);
    }

    fn contains(&self, first: u8, second: u8) -> bool {
        self.rows[first as usize].contains(second)
    }

    fn union_with(&mut self, other: &Self) {
        for (left, right) in self.rows.iter_mut().zip(other.rows.iter()) {
            left.union_with(*right);
        }
    }

    fn cartesian(left: ByteSet, right: ByteSet) -> Self {
        let mut out = Self::default();
        for first in 0u16..=255 {
            let first = first as u8;
            if left.contains(first) {
                out.rows[first as usize] = right;
            }
        }
        out
    }

    fn any_first_ending_in(last: ByteSet) -> Self {
        Self {
            rows: Box::new([last; 256]),
        }
    }

    fn count(&self) -> usize {
        self.rows.iter().map(ByteSet::count).sum()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct Tail2Language {
    epsilon: bool,
    /// Last byte of every non-empty word.
    last1: ByteSet,
    /// Exact one-byte words. Needed for proper-prefix cuts shorter than r.
    len1: ByteSet,
    /// Last two bytes of every word of length >= 2.
    last2: PairSet,
}

impl Tail2Language {
    fn empty() -> Self {
        Self::default()
    }

    fn epsilon() -> Self {
        Self {
            epsilon: true,
            ..Self::default()
        }
    }

    fn from_words(words: &[Vec<u8>]) -> Self {
        let mut out = Self::empty();
        for word in words {
            match word.as_slice() {
                [] => out.epsilon = true,
                [byte] => {
                    out.last1.insert(*byte);
                    out.len1.insert(*byte);
                }
                _ => {
                    let n = word.len();
                    out.last1.insert(word[n - 1]);
                    out.last2.insert(word[n - 2], word[n - 1]);
                }
            }
        }
        out
    }

    /// Maximally precise r=2 upper envelope available from an r=1 language
    /// alone: preserve epsilon and final-byte support, but allow every possible
    /// preceding byte for words of length >=2 and conservatively allow a
    /// one-byte word for every supported final byte.
    fn from_r1_upper(r1: ByteLanguage) -> Self {
        Self {
            epsilon: r1.epsilon,
            last1: r1.bytes,
            len1: r1.bytes,
            last2: PairSet::any_first_ending_in(r1.bytes),
        }
    }

    fn union_with(&mut self, other: &Self) {
        self.epsilon |= other.epsilon;
        self.last1.union_with(other.last1);
        self.len1.union_with(other.len1);
        self.last2.union_with(&other.last2);
    }

    fn concat_tail(&self, right: &Self) -> Self {
        let epsilon = self.epsilon && right.epsilon;

        let mut last1 = right.last1;
        if right.epsilon {
            last1.union_with(self.last1);
        }

        let mut len1 = ByteSet::default();
        if self.epsilon {
            len1.union_with(right.len1);
        }
        if right.epsilon {
            len1.union_with(self.len1);
        }

        let mut last2 = right.last2.clone();
        // If the right word has exactly one byte, the final pair is the last
        // byte of a non-empty left word followed by that one right byte.
        last2.union_with(&PairSet::cartesian(self.last1, right.len1));
        if right.epsilon {
            last2.union_with(&self.last2);
        }

        Self {
            epsilon,
            last1,
            len1,
            last2,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PhaseSummary2 {
    historical_productive: bool,
    normal: Tail2Language,
    event_from_entry: Tail2Language,
    tail_to_return: Tail2Language,
    tail_to_event: Tail2Language,
}

impl PhaseSummary2 {
    fn bottom() -> Self {
        Self {
            historical_productive: false,
            normal: Tail2Language::empty(),
            event_from_entry: Tail2Language::empty(),
            tail_to_return: Tail2Language::empty(),
            tail_to_event: Tail2Language::empty(),
        }
    }

    fn identity() -> Self {
        Self {
            historical_productive: true,
            normal: Tail2Language::epsilon(),
            event_from_entry: Tail2Language::empty(),
            tail_to_return: Tail2Language::empty(),
            tail_to_event: Tail2Language::empty(),
        }
    }

    fn from_r1_upper(summary: BytePhaseSummary) -> Self {
        Self {
            historical_productive: summary.historical_productive,
            normal: Tail2Language::from_r1_upper(summary.normal),
            event_from_entry: Tail2Language::from_r1_upper(summary.event_from_entry),
            tail_to_return: Tail2Language::from_r1_upper(summary.tail_to_return),
            tail_to_event: Tail2Language::from_r1_upper(summary.tail_to_event),
        }
    }

    fn union_with(&mut self, other: &Self) {
        self.historical_productive |= other.historical_productive;
        self.normal.union_with(&other.normal);
        self.event_from_entry.union_with(&other.event_from_entry);
        self.tail_to_return.union_with(&other.tail_to_return);
        self.tail_to_event.union_with(&other.tail_to_event);
    }

    fn concat(&self, right: &Self) -> Self {
        let normal = self.normal.concat_tail(&right.normal);

        let mut event_from_entry = self.event_from_entry.clone();
        event_from_entry.union_with(&self.normal.concat_tail(&right.event_from_entry));

        let mut tail_to_return = self.tail_to_return.concat_tail(&right.normal);
        if self.historical_productive {
            tail_to_return.union_with(&right.tail_to_return);
        }

        let mut tail_to_event = self.tail_to_event.clone();
        tail_to_event.union_with(&self.tail_to_return.concat_tail(&right.event_from_entry));
        if self.historical_productive {
            tail_to_event.union_with(&right.tail_to_event);
        }

        Self {
            historical_productive: self.historical_productive && right.historical_productive,
            normal,
            event_from_entry,
            tail_to_return,
            tail_to_event,
        }
    }
}

fn terminal_summary2(
    constraint: &Constraint,
    terminal: TerminalID,
    outward: &BTreeSet<TerminalID>,
    child_overrides: &BTreeMap<TerminalID, PhaseSummary2>,
) -> PhaseSummary2 {
    if let Some(child) = child_overrides.get(&terminal) {
        return child.clone();
    }
    if outward.contains(&terminal) {
        let eps = Tail2Language::epsilon();
        return PhaseSummary2 {
            historical_productive: true,
            normal: Tail2Language::empty(),
            event_from_entry: eps.clone(),
            tail_to_return: eps.clone(),
            tail_to_event: eps,
        };
    }
    if constraint.table.control_terminals.contains(&terminal) {
        let eps = Tail2Language::epsilon();
        return PhaseSummary2 {
            historical_productive: true,
            normal: eps.clone(),
            event_from_entry: Tail2Language::empty(),
            tail_to_return: eps,
            tail_to_event: Tail2Language::empty(),
        };
    }
    if let Some(expr) = constraint.retained_terminal_expr(terminal) {
        if let Some(words) = finite_literal_words(expr, FINITE_EXPR_WORD_CAP) {
            let normal = Tail2Language::from_words(&words);
            let mut tails = Tail2Language::epsilon();
            for word in &words {
                for cut in 0..word.len() {
                    let suffix = word[cut..].to_vec();
                    tails.union_with(&Tail2Language::from_words(&[suffix]));
                }
            }
            return PhaseSummary2 {
                historical_productive: !words.is_empty(),
                normal,
                event_from_entry: Tail2Language::empty(),
                tail_to_return: tails,
                tail_to_event: Tail2Language::empty(),
            };
        }
    }
    // Unsupported lexical atom: retain the r=1 facts and widen only the
    // unknown byte to the left.
    PhaseSummary2::from_r1_upper(byte_terminal_summary(
        constraint,
        terminal,
        outward,
        &BTreeMap::new(),
    ))
}

fn rule_summary2(
    rule: &Rule,
    nt: &BTreeMap<NonterminalID, PhaseSummary2>,
    terminals: &[PhaseSummary2],
) -> PhaseSummary2 {
    rule.rhs.iter().fold(PhaseSummary2::identity(), |acc, symbol| {
        let right = match symbol {
            Symbol::Terminal(id) => terminals
                .get(*id as usize)
                .cloned()
                .unwrap_or_else(PhaseSummary2::bottom),
            Symbol::Nonterminal(id) => nt.get(id).cloned().unwrap_or_else(PhaseSummary2::bottom),
        };
        acc.concat(&right)
    })
}

fn summarize_rules_module_r2(
    constraint: &Constraint,
    child_overrides: &BTreeMap<TerminalID, PhaseSummary2>,
    bound_slots: &BTreeSet<TerminalID>,
) -> Result<(PhaseSummary2, usize, bool), String> {
    let rules = constraint.retained_table_rules()?;
    if rules.is_empty() {
        return Err("r2 interface-tail summary requires retained grammar rules".to_owned());
    }
    let mut outward = outward_terminals(constraint);
    for terminal in bound_slots {
        outward.remove(terminal);
    }
    let terminals = (0..constraint.tokenizer.num_terminals())
        .map(|terminal| terminal_summary2(constraint, terminal, &outward, child_overrides))
        .collect::<Vec<_>>();
    let mut nts = BTreeMap::<NonterminalID, PhaseSummary2>::new();
    for rule in rules {
        nts.entry(rule.lhs).or_insert_with(PhaseSummary2::bottom);
        for symbol in &rule.rhs {
            if let Symbol::Nonterminal(id) = symbol {
                nts.entry(*id).or_insert_with(PhaseSummary2::bottom);
            }
        }
    }
    let mut iterations = 0usize;
    for iteration in 0..DEFAULT_FIXED_POINT_LIMIT {
        iterations = iteration + 1;
        let snapshot = nts.clone();
        let mut next = snapshot.clone();
        for rule in rules {
            let summary = rule_summary2(rule, &snapshot, &terminals);
            next.entry(rule.lhs)
                .or_insert_with(PhaseSummary2::bottom)
                .union_with(&summary);
        }
        if next == nts {
            let module = rule_summary2(&rules[0], &next, &terminals);
            return Ok((module, iterations, false));
        }
        nts = next;
    }
    Ok((
        PhaseSummary2::from_r1_upper(BytePhaseSummary {
            historical_productive: true,
            normal: ByteLanguage::top(),
            event_from_entry: ByteLanguage::top(),
            tail_to_return: ByteLanguage::top(),
            tail_to_event: ByteLanguage::top(),
        }),
        iterations,
        true,
    ))
}

fn candidate_ids_for_r2(vocab: &crate::Vocab, language: &Tail2Language) -> Vec<u32> {
    let mut ids = Vec::new();
    'tokens: for (id, bytes) in vocab.iter() {
        if bytes.len() < 2 {
            continue;
        }
        // k=1 (<r): the entire one-byte prefix must be an exact exit word.
        if language.len1.contains(bytes[0]) {
            ids.push(id);
            continue;
        }
        // k>=2: the final two bytes immediately before the proper cut must be
        // a supported exit tail.
        for cut in 2..bytes.len() {
            if language.last2.contains(bytes[cut - 2], bytes[cut - 1]) {
                ids.push(id);
                continue 'tokens;
            }
        }
    }
    ids.sort_unstable();
    ids.dedup();
    ids
}

pub(crate) fn build_composition_boundary_tail_r2(
    parent: &Constraint,
    bindings: &[(TerminalID, &Constraint)],
    vocab: &crate::Vocab,
) -> Result<BoundaryTailR2Result, String> {
    // The r=1 layer explicitly widens for local Skip/IGNORE consumption before
    // an outward event. The r=2 pair algebra below intentionally models only
    // grammar-symbol concatenation, so decline rather than risk forgetting an
    // ignore byte between the historical cut and the interface. Composition
    // automatically falls back to the sound and faster r=1 summary.
    if parent.ignore_terminal.is_some() || !parent.table.skip_terminals.is_empty() {
        return Err("r2 interface-tail summary declines parent skip/ignore; use r1".to_owned());
    }
    let started = Instant::now();
    let child_started = Instant::now();
    // One child may be called from several parent terminals.  Its module
    // summary is independent of the call site, so compute it once and fan the
    // result back out to every bound slot.
    let mut unique_children = BTreeMap::<usize, (&Constraint, Vec<TerminalID>)>::new();
    for &(slot, child) in bindings {
        unique_children
            .entry(child as *const Constraint as usize)
            .or_insert_with(|| (child, Vec::new()))
            .1
            .push(slot);
    }
    let child_rows = unique_children
        .into_values()
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|(child, slots)| {
            let (r1, _, widened) = summarize_constraint_module_r1(child)?;
            let mut r2 = PhaseSummary2::from_r1_upper(r1);
            if widened {
                r2.historical_productive = true;
            }
            Ok::<_, String>((slots, r2))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let child_summary_ms = child_started.elapsed().as_secs_f64() * 1000.0;
    let mut overrides = BTreeMap::<TerminalID, PhaseSummary2>::new();
    let mut bound_slots = BTreeSet::<TerminalID>::new();
    for (slots, r2) in child_rows {
        for slot in slots {
            bound_slots.insert(slot);
            overrides.insert(slot, r2.clone());
        }
    }
    let (module, fixed_point_iterations, fixed_point_widened) =
        summarize_rules_module_r2(parent, &overrides, &bound_slots)?;
    let mut exits = module.tail_to_return;
    exits.union_with(&module.tail_to_event);
    let summary_ms = started.elapsed().as_secs_f64() * 1000.0;
    let map_started = Instant::now();
    let candidate_ids = candidate_ids_for_r2(vocab, &exits);
    let map_ms = map_started.elapsed().as_secs_f64() * 1000.0;
    Ok(BoundaryTailR2Result {
        candidate_ids,
        exit_last1_count: exits.last1.count(),
        exit_pair_count: exits.last2.count(),
        fixed_point_iterations,
        fixed_point_widened,
        child_summary_ms,
        summary_ms,
        map_ms,
    })
}

#[derive(Debug, Clone)]
pub(crate) struct BoundaryTailR2Result {
    pub(crate) candidate_ids: Vec<u32>,
    pub(crate) exit_last1_count: usize,
    pub(crate) exit_pair_count: usize,
    pub(crate) fixed_point_iterations: usize,
    pub(crate) fixed_point_widened: bool,
    pub(crate) child_summary_ms: f64,
    pub(crate) summary_ms: f64,
    pub(crate) map_ms: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vocab(entries: &[(u32, &[u8])]) -> crate::Vocab {
        crate::Vocab::new(
            entries
                .iter()
                .map(|(id, bytes)| (*id, bytes.to_vec()))
                .collect(),
        )
    }

    #[test]
    fn r1_root_completion_is_strict_proper_prefix_and_preserves_duplicate_ids() {
        let vocab = vocab(&[(0, b"a"), (1, b"ab"), (2, b"ab"), (3, b"b")]);
        let constraint = Constraint::from_glrm_grammar(
            r#"
                start document;
                nt document ::= "a";
            "#,
            &vocab,
        )
        .unwrap();
        let probe = build_boundary_tail_r1(&constraint, &vocab).unwrap();
        assert!(
            !probe.candidate_ids.contains(&0),
            "token-end completion is not a proper-prefix event"
        );
        assert!(probe.candidate_ids.contains(&1));
        assert!(probe.candidate_ids.contains(&2));
    }

    #[test]
    fn r1_ignore_before_public_slot_keeps_positive_ignore_prefix() {
        let vocab = vocab(&[(0, b" x"), (1, b" xq"), (2, b"xq"), (3, b" ")]);
        let constraint = Constraint::from_glrm_grammar(
            r#"
                start document;
                ignore WS;
                t WS ::= " "+;
                t SUB ::= @token(999);
                nt document ::= SUB;
            "#,
            &vocab,
        )
        .unwrap();
        let probe = build_boundary_tail_r1(&constraint, &vocab).unwrap();
        assert!(probe.candidate_ids.contains(&0));
        assert!(probe.candidate_ids.contains(&1));
        assert!(
            !probe.candidate_ids.contains(&3),
            "one-byte token has no proper internal cut"
        );
    }

    #[test]
    fn r1_opaque_bound_child_preserves_mandatory_postamble() {
        let vocab = vocab(&[
            (0, b")x"),
            (1, b"abc)x"),
            (2, b"abcx"),
            (3, b")"),
            (4, b"x)"),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t SUB ::= @token(999);
                nt document ::= "prefix(" SUB ")";
            "#,
            &vocab,
        )
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start document;
                nt document ::= "anything";
            "#,
            &vocab,
        )
        .unwrap();
        let slot = parent
            .terminal_display_names
            .iter()
            .position(|name| name == "SUB")
            .unwrap() as u32;
        let probe =
            build_composition_boundary_tail_r1(&parent, &[(slot, &child)], &vocab).unwrap();
        assert_eq!(probe.exit_byte_count, 1);
        assert!(probe.candidate_ids.contains(&0));
        assert!(probe.candidate_ids.contains(&1));
        assert!(!probe.candidate_ids.contains(&2));
        assert!(
            !probe.candidate_ids.contains(&3),
            "token-end ')' is not a proper-prefix cut"
        );
        assert!(
            !probe.candidate_ids.contains(&4),
            "final ')' is not an internal cut"
        );
    }

    #[test]
    fn r2_child_final_byte_plus_postamble_prunes_unrelated_pairs() {
        let vocab = vocab(&[
            (0, b")x"),
            (1, b"})x"),
            (2, b"a)x"),
            (3, b"}x"),
            (4, b")"),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t SUB ::= @token(999);
                nt document ::= SUB ")";
            "#,
            &vocab,
        )
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start document;
                nt document ::= "}";
            "#,
            &vocab,
        )
        .unwrap();
        let slot = parent
            .terminal_display_names
            .iter()
            .position(|name| name == "SUB")
            .unwrap() as u32;
        let probe =
            build_composition_boundary_tail_r2(&parent, &[(slot, &child)], &vocab).unwrap();
        assert!(
            probe.candidate_ids.contains(&0),
            "token may start in the postamble itself"
        );
        assert!(
            probe.candidate_ids.contains(&1),
            "child-final '}}' followed by ')' is a real two-byte tail"
        );
        assert!(
            !probe.candidate_ids.contains(&2),
            "unrelated byte before ')' must be pruned"
        );
        assert!(
            !probe.candidate_ids.contains(&3),
            "child return alone has not yet reached parent root"
        );
        assert!(
            !probe.candidate_ids.contains(&4),
            "token-end ')' is not a proper-prefix cut"
        );
    }
    #[test]
    fn r2_parent_ignore_declines_to_r1_layer() {
        let vocab = vocab(&[(0, b" x"), (1, b" )x"), (2, b")x")]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                ignore WS;
                t WS ::= " "+;
                t SUB ::= @token(999);
                nt document ::= SUB ")";
            "#,
            &vocab,
        )
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start document;
                nt document ::= "anything";
            "#,
            &vocab,
        )
        .unwrap();
        let slot = parent
            .terminal_display_names
            .iter()
            .position(|name| name == "SUB")
            .unwrap() as u32;
        assert!(build_composition_boundary_tail_r2(&parent, &[(slot, &child)], &vocab).is_err());
    }

}
