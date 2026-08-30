//! General lazy regex residuals for dynamic tokenization.
//!
//! The key invariant is that bounded repetition remains a symbolic
//! `(body, min, max)` node. Byte derivatives decrement those integers only
//! when a body copy is actually consumed; construction never allocates in
//! proportion to `max`.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use rustc_hash::{FxHashMap, FxHashSet};
use rayon::prelude::*;

use super::ast::Expr;
use super::compile::{compile_terminal_expr_dfa, expression_contains_large_bounded_repeat, VocabularyRepeatHorizonCache};
use super::dfa::DFA;
use super::runtime_repeat_product::{VirtualRuntimeStateOwners, VirtualStateAllocator};
use crate::ds::bitset::BitSet;
use crate::ds::u8set::U8Set;
use crate::grammar::flat::TerminalID;
use crate::Vocab;

pub(crate) type ResidualId = u32;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum ResidualNode {
    Empty,
    Epsilon,
    SigmaStar,
    Literal { bytes: Arc<[u8]>, offset: u32 },
    Class(U8Set),
    Dfa { dfa: Arc<DFA>, states: Box<[u32]> },
    Seq(Box<[ResidualId]>),
    Choice(Box<[ResidualId]>),
    Intersect(ResidualId, ResidualId),
    Exclude(ResidualId, ResidualId),
    Repeat {
        body: ResidualId,
        min: usize,
        max: Option<usize>,
    },
}

#[derive(Debug)]
pub(crate) struct ResidualArena {
    nodes: Vec<ResidualNode>,
    ids: FxHashMap<ResidualNode, ResidualId>,
    nullable: Vec<bool>,
    transitions: Vec<Option<Box<[u32; 256]>>>,
    first_bytes_cache: Vec<Option<U8Set>>,
    nonempty_cache: Vec<Option<bool>>,
    empty: ResidualId,
    epsilon: ResidualId,
    sigma_star: ResidualId,
}

const TRANSITION_UNKNOWN: u32 = u32::MAX;
const DEFAULT_LIVENESS_STATE_BUDGET: usize = 262_144;
const MAX_FINITE_MASK_DENSE_STATES: usize = 8 * 1024 * 1024;
const DEFAULT_LIVENESS_TRANSITION_BUDGET: usize = 4_194_304;

struct ResidualLivenessBudget {
    state_limit: usize,
    transition_limit: usize,
    states_used: usize,
    transitions_used: usize,
}

impl ResidualLivenessBudget {
    fn new(state_limit: usize, transition_limit: usize) -> Self {
        Self {
            state_limit,
            transition_limit,
            states_used: 0,
            transitions_used: 0,
        }
    }

    fn consume_state(&mut self) -> Result<(), String> {
        self.states_used = self
            .states_used
            .checked_add(1)
            .ok_or_else(|| "dynamic residual liveness state count overflow".to_owned())?;
        if self.states_used > self.state_limit {
            return Err(format!(
                "dynamic residual liveness exceeded state budget ({})",
                self.state_limit
            ));
        }
        Ok(())
    }

    fn consume_transition(&mut self) -> Result<(), String> {
        self.transitions_used = self
            .transitions_used
            .checked_add(1)
            .ok_or_else(|| "dynamic residual liveness transition count overflow".to_owned())?;
        if self.transitions_used > self.transition_limit {
            return Err(format!(
                "dynamic residual liveness exceeded transition budget ({})",
                self.transition_limit
            ));
        }
        Ok(())
    }
}

impl ResidualArena {
    pub(crate) fn from_expr(expr: &Expr) -> Option<(Self, ResidualId)> {
        let mut arena = Self {
            nodes: Vec::new(),
            ids: FxHashMap::default(),
            nullable: Vec::new(),
            transitions: Vec::new(),
            first_bytes_cache: Vec::new(),
            nonempty_cache: Vec::new(),
            empty: 0,
            epsilon: 0,
            sigma_star: 0,
        };
        arena.empty = arena.intern_raw(ResidualNode::Empty, false)?;
        arena.epsilon = arena.intern_raw(ResidualNode::Epsilon, true)?;
        arena.sigma_star = arena.intern_raw(ResidualNode::SigmaStar, true)?;
        let root = arena.compile_expr(expr)?;
        Some((arena, root))
    }

    pub(crate) fn state_count(&self) -> usize {
        self.nodes.len()
    }

    pub(crate) fn is_nullable(&self, id: ResidualId) -> bool {
        self.nullable[id as usize]
    }

    pub(crate) fn is_empty(&self, id: ResidualId) -> bool {
        id == self.empty
    }

    fn intern_raw(&mut self, node: ResidualNode, nullable: bool) -> Option<ResidualId> {
        if let Some(&id) = self.ids.get(&node) {
            return Some(id);
        }
        let id = u32::try_from(self.nodes.len()).ok()?;
        self.ids.insert(node.clone(), id);
        self.nodes.push(node);
        self.nullable.push(nullable);
        self.transitions.push(None);
        self.first_bytes_cache.push(None);
        self.nonempty_cache.push(None);
        Some(id)
    }

    fn compile_expr(&mut self, expr: &Expr) -> Option<ResidualId> {
        match expr {
            Expr::U8Seq(bytes) => self.literal(bytes),
            Expr::U8Class(bytes) => self.class(*bytes),
            Expr::Dfa(dfa) => {
                if dfa.num_states() == 0 {
                    return None;
                }
                self.dfa(Arc::clone(dfa), &[0])
            }
            Expr::Intersect { expr, intersect } => {
                let left = self.compile_expr(expr)?;
                let right = self.compile_expr(intersect)?;
                self.intersect(left, right)
            }
            Expr::Seq(parts) => {
                let parts = parts
                    .iter()
                    .map(|part| self.compile_expr(part))
                    .collect::<Option<Vec<_>>>()?;
                self.seq(parts)
            }
            Expr::Choice(parts) => {
                let parts = parts
                    .iter()
                    .map(|part| self.compile_expr(part))
                    .collect::<Option<Vec<_>>>()?;
                self.choice(parts)
            }
            Expr::Exclude { expr, exclude } => {
                let left = self.compile_expr(expr)?;
                let right = self.compile_expr(exclude)?;
                self.exclude(left, right)
            }
            Expr::Repeat { expr, min, max } => {
                let body = self.compile_expr(expr)?;
                self.repeat(body, *min, *max)
            }
            Expr::Shared(inner) => self.compile_expr(inner),
            Expr::Epsilon => Some(self.epsilon),
        }
    }

    fn literal(&mut self, bytes: &[u8]) -> Option<ResidualId> {
        if bytes.is_empty() {
            return Some(self.epsilon);
        }
        self.intern_raw(
            ResidualNode::Literal {
                bytes: Arc::from(bytes),
                offset: 0,
            },
            false,
        )
    }

    fn literal_at(&mut self, bytes: Arc<[u8]>, offset: u32) -> Option<ResidualId> {
        if offset as usize >= bytes.len() {
            return Some(self.epsilon);
        }
        self.intern_raw(ResidualNode::Literal { bytes, offset }, false)
    }

    fn class(&mut self, bytes: U8Set) -> Option<ResidualId> {
        if bytes.is_empty() {
            Some(self.empty)
        } else {
            self.intern_raw(ResidualNode::Class(bytes), false)
        }
    }

    fn dfa(&mut self, dfa: Arc<DFA>, roots: &[u32]) -> Option<ResidualId> {
        if roots.iter().any(|&state| state as usize >= dfa.num_states()) {
            return Some(self.empty);
        }
        let mut states = dfa.epsilon_closure(roots);
        states.sort_unstable();
        states.dedup();
        if states.is_empty() {
            return Some(self.empty);
        }
        let nullable = states.iter().any(|&state| !dfa.finalizers(state).is_empty());
        self.intern_raw(
            ResidualNode::Dfa {
                dfa,
                states: states.into_vec().into_boxed_slice(),
            },
            nullable,
        )
    }

    fn seq(&mut self, parts: Vec<ResidualId>) -> Option<ResidualId> {
        let mut flat = Vec::new();
        for part in parts {
            if part == self.empty {
                return Some(self.empty);
            }
            if part == self.epsilon {
                continue;
            }
            match self.nodes[part as usize].clone() {
                ResidualNode::Seq(children) => flat.extend(children.iter().copied()),
                _ => flat.push(part),
            }
        }
        match flat.len() {
            0 => Some(self.epsilon),
            1 => Some(flat[0]),
            _ => {
                let nullable = flat.iter().all(|&id| self.is_nullable(id));
                self.intern_raw(ResidualNode::Seq(flat.into_boxed_slice()), nullable)
            }
        }
    }

    fn choice(&mut self, parts: Vec<ResidualId>) -> Option<ResidualId> {
        let mut flat = Vec::new();
        for part in parts {
            if part == self.sigma_star {
                return Some(self.sigma_star);
            }
            if part == self.empty {
                continue;
            }
            match self.nodes[part as usize].clone() {
                ResidualNode::Choice(children) => flat.extend(children.iter().copied()),
                _ => flat.push(part),
            }
        }
        flat.sort_unstable();
        flat.dedup();
        match flat.len() {
            0 => Some(self.empty),
            1 => Some(flat[0]),
            _ => {
                let nullable = flat.iter().any(|&id| self.is_nullable(id));
                self.intern_raw(ResidualNode::Choice(flat.into_boxed_slice()), nullable)
            }
        }
    }

    fn intersect(&mut self, mut left: ResidualId, mut right: ResidualId) -> Option<ResidualId> {
        if left == self.empty || right == self.empty {
            return Some(self.empty);
        }
        if left == self.sigma_star {
            return Some(right);
        }
        if right == self.sigma_star {
            return Some(left);
        }
        if left == right {
            return Some(left);
        }
        if left == self.epsilon {
            return Some(if self.is_nullable(right) { self.epsilon } else { self.empty });
        }
        if right == self.epsilon {
            return Some(if self.is_nullable(left) { self.epsilon } else { self.empty });
        }
        if right < left {
            std::mem::swap(&mut left, &mut right);
        }
        self.intern_raw(
            ResidualNode::Intersect(left, right),
            self.is_nullable(left) && self.is_nullable(right),
        )
    }

    fn exclude(&mut self, left: ResidualId, right: ResidualId) -> Option<ResidualId> {
        if left == self.empty || left == right {
            return Some(self.empty);
        }
        if right == self.sigma_star {
            return Some(self.empty);
        }
        if right == self.empty {
            return Some(left);
        }
        if left == self.epsilon {
            return Some(if self.is_nullable(right) { self.empty } else { self.epsilon });
        }
        if right == self.epsilon && !self.is_nullable(left) {
            return Some(left);
        }
        self.intern_raw(
            ResidualNode::Exclude(left, right),
            self.is_nullable(left) && !self.is_nullable(right),
        )
    }

    fn repeat(
        &mut self,
        mut body: ResidualId,
        mut min: usize,
        max: Option<usize>,
    ) -> Option<ResidualId> {
        if max.is_some_and(|max| min > max) {
            return Some(self.empty);
        }
        if max == Some(0) {
            return Some(if min == 0 { self.epsilon } else { self.empty });
        }
        if body == self.empty {
            return Some(if min == 0 { self.epsilon } else { self.empty });
        }
        if body == self.epsilon {
            return Some(self.epsilon);
        }
        if body == self.sigma_star {
            return Some(self.sigma_star);
        }

        if min == 0
            && max.is_none()
            && matches!(self.nodes[body as usize], ResidualNode::Class(bytes) if bytes.is_full())
        {
            return Some(self.sigma_star);
        }

        // If B is nullable, B^[m,n] equals (B\\{epsilon})^[0,n]. Empty
        // copies can satisfy the lower bound without consuming input. This is
        // the crucial normalization that prevents a derivative from skipping
        // O(n) nullable copies when n is enormous.
        if self.is_nullable(body) {
            body = self.exclude(body, self.epsilon)?;
            min = 0;
            if body == self.empty {
                return Some(self.epsilon);
            }
        }

        if min == 1 && max == Some(1) {
            return Some(body);
        }
        self.intern_raw(
            ResidualNode::Repeat { body, min, max },
            min == 0,
        )
    }

    pub(crate) fn step(&mut self, id: ResidualId, byte: u8) -> Option<ResidualId> {
        if let Some(row) = self.transitions[id as usize].as_ref() {
            let cached = row[byte as usize];
            if cached != TRANSITION_UNKNOWN {
                return Some(cached);
            }
        }
        let target = self.derive_uncached(id, byte)?;
        let row = self.transitions[id as usize]
            .get_or_insert_with(|| Box::new([TRANSITION_UNKNOWN; 256]));
        row[byte as usize] = target;
        Some(target)
    }

    fn sparse_step(
        &mut self,
        id: ResidualId,
        byte: u8,
        budget: &mut ResidualLivenessBudget,
        cache: &mut FxHashMap<u64, ResidualId>,
    ) -> Result<ResidualId, String> {
        let key = (u64::from(id) << 8) | u64::from(byte);
        if let Some(&target) = cache.get(&key) {
            return Ok(target);
        }
        // Charge every actual derivative computation, including recursive
        // child derivatives. This makes the transition budget a real bound on
        // the temporary sparse cache/work rather than only on outer BFS edges.
        budget.consume_transition()?;
        let target = self.derive_uncached_sparse(id, byte, budget, cache)?;
        cache.insert(key, target);
        Ok(target)
    }

    /// Derivative used only by exact liveness reachability. Unlike `step`, it
    /// must not populate the persistent dense 256-entry transition row for
    /// every explored residual: a hard Boolean liveness proof can visit many
    /// states that normal token traversal will never touch. A temporary sparse
    /// cache keeps repeated sub-derivatives cheap without retaining
    /// `O(256 * visited_states)` memory after the query completes.
    fn derive_uncached_sparse(
        &mut self,
        id: ResidualId,
        byte: u8,
        budget: &mut ResidualLivenessBudget,
        cache: &mut FxHashMap<u64, ResidualId>,
    ) -> Result<ResidualId, String> {
        let overflow = || "dynamic residual state-id overflow".to_owned();
        match self.nodes[id as usize].clone() {
            ResidualNode::Empty | ResidualNode::Epsilon => Ok(self.empty),
            ResidualNode::SigmaStar => Ok(self.sigma_star),
            ResidualNode::Literal { bytes, offset } => {
                if bytes[offset as usize] != byte {
                    Ok(self.empty)
                } else {
                    self.literal_at(bytes, offset + 1).ok_or_else(overflow)
                }
            }
            ResidualNode::Class(bytes) => Ok(if bytes.contains(byte) {
                self.epsilon
            } else {
                self.empty
            }),
            ResidualNode::Dfa { dfa, states } => {
                let mut targets = Vec::new();
                for &state in states.iter() {
                    if let Some(target) = dfa.step(state, byte) {
                        targets.push(target);
                    }
                }
                self.dfa(dfa, &targets).ok_or_else(overflow)
            }
            ResidualNode::Choice(parts) => {
                let mut derivatives = Vec::with_capacity(parts.len());
                for &part in parts.iter() {
                    derivatives.push(self.sparse_step(part, byte, budget, cache)?);
                }
                self.choice(derivatives).ok_or_else(overflow)
            }
            ResidualNode::Seq(parts) => {
                let mut alternatives = Vec::new();
                for index in 0..parts.len() {
                    let head = parts[index];
                    let derivative = self.sparse_step(head, byte, budget, cache)?;
                    if derivative != self.empty {
                        let mut sequence = Vec::with_capacity(parts.len() - index);
                        sequence.push(derivative);
                        sequence.extend_from_slice(&parts[index + 1..]);
                        alternatives.push(self.seq(sequence).ok_or_else(overflow)?);
                    }
                    if !self.is_nullable(head) {
                        break;
                    }
                }
                self.choice(alternatives).ok_or_else(overflow)
            }
            ResidualNode::Intersect(left, right) => {
                let left = self.sparse_step(left, byte, budget, cache)?;
                let right = self.sparse_step(right, byte, budget, cache)?;
                self.intersect(left, right).ok_or_else(overflow)
            }
            ResidualNode::Exclude(left, right) => {
                let left = self.sparse_step(left, byte, budget, cache)?;
                let right = self.sparse_step(right, byte, budget, cache)?;
                self.exclude(left, right).ok_or_else(overflow)
            }
            ResidualNode::Repeat { body, min, max } => {
                let derivative = self.sparse_step(body, byte, budget, cache)?;
                if derivative == self.empty {
                    return Ok(self.empty);
                }
                let next_max = max.map(|max| max - 1);
                let tail = self
                    .repeat(body, min.saturating_sub(1), next_max)
                    .ok_or_else(overflow)?;
                self.seq(vec![derivative, tail]).ok_or_else(overflow)
            }
        }
    }

    fn derive_uncached(&mut self, id: ResidualId, byte: u8) -> Option<ResidualId> {
        match self.nodes[id as usize].clone() {
            ResidualNode::Empty | ResidualNode::Epsilon => Some(self.empty),
            ResidualNode::SigmaStar => Some(self.sigma_star),
            ResidualNode::Literal { bytes, offset } => {
                if bytes[offset as usize] != byte {
                    Some(self.empty)
                } else {
                    self.literal_at(bytes, offset + 1)
                }
            }
            ResidualNode::Class(bytes) => {
                Some(if bytes.contains(byte) { self.epsilon } else { self.empty })
            }
            ResidualNode::Dfa { dfa, states } => {
                let mut targets = Vec::new();
                for &state in states.iter() {
                    if let Some(target) = dfa.step(state, byte) {
                        targets.push(target);
                    }
                }
                self.dfa(dfa, &targets)
            }
            ResidualNode::Choice(parts) => {
                let derivatives = parts
                    .iter()
                    .map(|&part| self.step(part, byte))
                    .collect::<Option<Vec<_>>>()?;
                self.choice(derivatives)
            }
            ResidualNode::Seq(parts) => {
                let mut alternatives = Vec::new();
                for index in 0..parts.len() {
                    let head = parts[index];
                    let derivative = self.step(head, byte)?;
                    if derivative != self.empty {
                        let mut sequence = Vec::with_capacity(parts.len() - index);
                        sequence.push(derivative);
                        sequence.extend_from_slice(&parts[index + 1..]);
                        alternatives.push(self.seq(sequence)?);
                    }
                    if !self.is_nullable(head) {
                        break;
                    }
                }
                self.choice(alternatives)
            }
            ResidualNode::Intersect(left, right) => {
                let left = self.step(left, byte)?;
                let right = self.step(right, byte)?;
                self.intersect(left, right)
            }
            ResidualNode::Exclude(left, right) => {
                let left = self.step(left, byte)?;
                let right = self.step(right, byte)?;
                self.exclude(left, right)
            }
            ResidualNode::Repeat { body, min, max } => {
                let derivative = self.step(body, byte)?;
                if derivative == self.empty {
                    return Some(self.empty);
                }
                let next_max = max.map(|max| max - 1);
                let tail = self.repeat(body, min.saturating_sub(1), next_max)?;
                self.seq(vec![derivative, tail])
            }
        }
    }

    fn has_nonempty_fast(&mut self, id: ResidualId) -> Option<bool> {
        match self.nodes[id as usize].clone() {
            ResidualNode::Empty | ResidualNode::Epsilon => Some(false),
            ResidualNode::SigmaStar => Some(true),
            ResidualNode::Literal { .. } | ResidualNode::Class(_) => Some(true),
            // `Expr::Dfa` is allowed to carry stale derived future metadata,
            // so there is no metadata-only fast proof here. Let the ordinary
            // bounded derivative reachability below answer this exactly; that
            // keeps arbitrary embedded-DFA graphs under the same hard work
            // ceilings as every other non-structural residual.
            ResidualNode::Dfa { .. } => None,
            ResidualNode::Choice(parts) => {
                let mut unknown = false;
                for &part in parts.iter() {
                    match self.has_nonempty_fast(part) {
                        Some(true) => return Some(true),
                        Some(false) => {}
                        None => unknown = true,
                    }
                }
                (!unknown).then_some(false)
            }
            ResidualNode::Seq(parts) => {
                let mut any_nonempty = false;
                let mut unknown = false;
                for &part in parts.iter() {
                    let any_word = if self.is_nullable(part) {
                        Some(true)
                    } else {
                        self.has_nonempty_fast(part)
                    };
                    match any_word {
                        Some(false) => return Some(false),
                        Some(true) => {}
                        None => unknown = true,
                    }
                    match self.has_nonempty_fast(part) {
                        Some(true) => any_nonempty = true,
                        Some(false) => {}
                        None => unknown = true,
                    }
                }
                if !unknown {
                    Some(any_nonempty)
                } else if !any_nonempty {
                    None
                } else {
                    None
                }
            }
            ResidualNode::Repeat { body, max, .. } => {
                if max == Some(0) {
                    Some(false)
                } else {
                    self.has_nonempty_fast(body)
                }
            }
            ResidualNode::Intersect(_, _) | ResidualNode::Exclude(_, _) => None,
        }
    }

    /// Byte alphabet that can possibly begin a nonempty word of this
    /// residual. For exclusion this is deliberately an over-approximation
    /// (`FIRST(left)`), which is sufficient for exact reachability because the
    /// derivative itself still decides whether the byte is actually live.
    fn first_bytes(&mut self, id: ResidualId) -> Option<U8Set> {
        if let Some(bytes) = self.first_bytes_cache[id as usize] {
            return Some(bytes);
        }
        let bytes = match self.nodes[id as usize].clone() {
            ResidualNode::Empty | ResidualNode::Epsilon => U8Set::empty(),
            ResidualNode::SigmaStar => U8Set::all(),
            ResidualNode::Literal { bytes, offset } => {
                U8Set::single(*bytes.get(offset as usize)?)
            }
            ResidualNode::Class(bytes) => bytes,
            ResidualNode::Dfa { dfa, states } => {
                let mut bytes = U8Set::empty();
                for &state in states.iter() {
                    for (byte, _) in dfa.states().get(state as usize)?.transitions.iter() {
                        bytes.insert(byte);
                    }
                }
                bytes
            }
            ResidualNode::Choice(parts) => {
                let mut bytes = U8Set::empty();
                for &part in parts.iter() {
                    bytes |= self.first_bytes(part)?;
                }
                bytes
            }
            ResidualNode::Seq(parts) => {
                let mut bytes = U8Set::empty();
                for &part in parts.iter() {
                    bytes |= self.first_bytes(part)?;
                    if !self.is_nullable(part) {
                        break;
                    }
                }
                bytes
            }
            ResidualNode::Intersect(left, right) => {
                self.first_bytes(left)?.intersection(&self.first_bytes(right)?)
            }
            ResidualNode::Exclude(left, _) => self.first_bytes(left)?,
            ResidualNode::Repeat { body, max, .. } => {
                if max == Some(0) {
                    U8Set::empty()
                } else {
                    self.first_bytes(body)?
                }
            }
        };
        self.first_bytes_cache[id as usize] = Some(bytes);
        Some(bytes)
    }

    /// Cheap exact answer when structural recursion is sufficient, otherwise
    /// a conservative `true`. This is suitable for the infallible tokenizer
    /// future bitset: dynamic mask/commit perform the fallible exact check at
    /// token boundaries before retaining a symbolic residual.
    fn conservative_has_future(&mut self, id: ResidualId) -> bool {
        if let Some(value) = self.nonempty_cache[id as usize] {
            return value;
        }
        self.has_nonempty_fast(id).unwrap_or(true)
    }

    /// Exact existence of a nonempty accepted continuation. Most expressions,
    /// including giant bounded repeats, resolve structurally in O(AST) time.
    /// Boolean combinations fall back to lazy derivative-graph reachability.
    /// The search has a hard resource ceiling; exceeding it is an error, never
    /// a semantic "dead" result.
    pub(crate) fn has_future(&mut self, id: ResidualId) -> Result<bool, String> {
        self.has_future_with_budget(
            id,
            DEFAULT_LIVENESS_STATE_BUDGET,
            DEFAULT_LIVENESS_TRANSITION_BUDGET,
        )
    }

    fn has_future_with_budget(
        &mut self,
        id: ResidualId,
        state_budget: usize,
        transition_budget: usize,
    ) -> Result<bool, String> {
        let mut budget = ResidualLivenessBudget::new(state_budget, transition_budget);
        self.has_future_with_work_budget(id, &mut budget)
    }

    fn has_future_with_work_budget(
        &mut self,
        id: ResidualId,
        budget: &mut ResidualLivenessBudget,
    ) -> Result<bool, String> {
        if let Some(value) = self.nonempty_cache[id as usize] {
            return Ok(value);
        }
        if let Some(value) = self.has_nonempty_fast(id) {
            self.nonempty_cache[id as usize] = Some(value);
            return Ok(value);
        }

        // Positive-word existence composes exactly through these regular
        // operators. Resolve their children independently so a giant repeat
        // never turns a hard *body* liveness question into a search over its
        // repetition counter. Boolean language relations remain on the general
        // derivative-graph fallback below.
        let structural = match self.nodes[id as usize].clone() {
            ResidualNode::Choice(parts) => {
                let mut live = false;
                for &part in parts.iter() {
                    if self.has_future_with_work_budget(part, budget)? {
                        live = true;
                        break;
                    }
                }
                Some(live)
            }
            ResidualNode::Seq(parts) => {
                let has_nonnullable = parts.iter().any(|&part| !self.is_nullable(part));
                if has_nonnullable {
                    let mut live = true;
                    for &part in parts.iter() {
                        if self.is_nullable(part) {
                            continue;
                        }
                        if !self.has_future_with_work_budget(part, budget)? {
                            live = false;
                            break;
                        }
                    }
                    // Every live nonnullable component contributes at least one
                    // byte, while nullable siblings can always contribute
                    // epsilon. Their positive-word languages are irrelevant.
                    Some(live)
                } else {
                    let mut live = false;
                    for &part in parts.iter() {
                        if self.has_future_with_work_budget(part, budget)? {
                            live = true;
                            break;
                        }
                    }
                    Some(live)
                }
            }
            ResidualNode::Repeat { body, max, .. } => Some(
                max != Some(0) && self.has_future_with_work_budget(body, budget)?,
            ),
            _ => None,
        };
        if let Some(value) = structural {
            self.nonempty_cache[id as usize] = Some(value);
            return Ok(value);
        }

        let mut seen = FxHashSet::<ResidualId>::default();
        let mut queue = VecDeque::from([id]);
        let mut sparse_transitions = FxHashMap::<u64, ResidualId>::default();
        seen.insert(id);
        budget.consume_state()?;
        while let Some(state) = queue.pop_front() {
            let first_bytes = self
                .first_bytes(state)
                .ok_or_else(|| "dynamic residual FIRST-set construction overflow".to_owned())?;
            for byte in first_bytes.iter() {
                let target =
                    self.sparse_step(state, byte, budget, &mut sparse_transitions)?;
                if target == self.empty {
                    continue;
                }
                if self.is_nullable(target) {
                    self.nonempty_cache[id as usize] = Some(true);
                    return Ok(true);
                }
                if seen.insert(target) {
                    budget.consume_state()?;
                    queue.push_back(target);
                }
            }
        }
        self.nonempty_cache[id as usize] = Some(false);
        Ok(false)
    }
}

// Exact liveness oracle for the important bounded-code intersection shape
// emitted by the JSON Schema string importer:
//
//     pattern_language ∩ prefix · C^[min,max] · suffix
//
// `C` must be a deterministic, non-nullable prefix code and the first suffix
// byte must not begin a productive C word.  JSON_STRING_CHAR satisfies these
// conditions.  The exact byte residual remains authoritative for transitions;
// this sidecar proves only the Boolean observation "some nonempty accepted
// continuation exists".
//
// At a C boundary, consuming one complete C word induces a finite relation on
// states of the independently compiled pattern DFA.  Therefore future
// liveness is exactly existence of a path whose number of relation edges lies
// in the remaining repetition interval.  Binary relation doubling answers
// that interval query in O(log max) relation applications without expanding
// the repeat counter.

const MAX_BOUNDED_CODE_ORACLE_PATTERN_STATES: usize = 4_096;
const MAX_BOUNDED_CODE_ORACLE_BODY_PRODUCT_CELLS: usize = 2_000_000;
const MAX_BOUNDED_CODE_ORACLE_RELATION_BYTES: usize = 128 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum BoundedCodeEnvelopeState {
    Prefix { next: usize },
    Body { completed: usize, body_state: u32 },
    Suffix { next: usize },
    Done,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct BoundedCodeOracleCoordinate {
    pattern_state: u32,
    envelope: BoundedCodeEnvelopeState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BoundedCodeOracleSlot {
    Unknown,
    Exact(BoundedCodeOracleCoordinate),
    Ambiguous,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct BoolRelation {
    rows: Vec<BitSet>,
}

impl BoolRelation {
    fn identity(states: usize) -> Self {
        let mut rows = Vec::with_capacity(states);
        for state in 0..states {
            let mut row = BitSet::new(states);
            row.set(state);
            rows.push(row);
        }
        Self { rows }
    }

    fn apply(&self, states: &BitSet) -> BitSet {
        let mut out = BitSet::new(self.rows.len());
        for state in states.iter() {
            out.union_with(&self.rows[state]);
        }
        out
    }

    /// Relation composition in execution order: first `self`, then `next`.
    fn then(&self, next: &Self) -> Self {
        debug_assert_eq!(self.rows.len(), next.rows.len());
        let rows = self
            .rows
            .iter()
            .map(|row| next.apply(row))
            .collect::<Vec<_>>();
        Self { rows }
    }

    fn union(&self, other: &Self) -> Self {
        debug_assert_eq!(self.rows.len(), other.rows.len());
        let rows = self
            .rows
            .iter()
            .zip(&other.rows)
            .map(|(left, right)| left.union(right))
            .collect::<Vec<_>>();
        Self { rows }
    }
}

/// Return whether the exact bounded-code liveness oracle can be constructed
/// for this expression without changing its language. Dynamic compilation uses
/// this as a proof-backed representation selector for bounded intersections
/// that would otherwise be eagerly materialized below the generic giant-repeat
/// threshold.
pub(crate) fn expression_supports_bounded_code_liveness_oracle(expr: &Expr) -> bool {
    BoundedCodeIntersectionOracle::from_expr(expr).is_some()
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct BoundedCodeIntersectionOracle {
    pattern: Arc<DFA>,
    body: Arc<DFA>,
    body_productive: Box<[bool]>,
    prefix: Arc<[u8]>,
    suffix: Arc<[u8]>,
    min: usize,
    max: usize,
    suffix_accepting: BitSet,
    completion_relations: Vec<Option<BoolRelation>>,
    exact_powers: Vec<BoolRelation>,
    prefix_sums: Vec<BoolRelation>,
}

fn canonicalize_bounded_code_oracle_dfa(dfa: DFA) -> DFA {
    // Oracle coordinates are persisted indirectly through Static TSID tables.
    // Make their component state IDs depend only on DFA language/topology, not
    // hash-map insertion order during an independent compile/load rebuild.
    let (dfa, _) = dfa.minimize_with_state_mapping();
    let n = dfa.num_states();
    if n <= 1 {
        return dfa;
    }

    let mut old_to_new = vec![u32::MAX; n];
    let mut order = Vec::with_capacity(n);
    let mut queue = VecDeque::new();
    old_to_new[0] = 0;
    order.push(0u32);
    queue.push_back(0u32);
    while let Some(source) = queue.pop_front() {
        for byte in 0u16..=255 {
            let Some(target) = dfa.step(source, byte as u8) else { continue; };
            if old_to_new[target as usize] == u32::MAX {
                let next = order.len() as u32;
                old_to_new[target as usize] = next;
                order.push(target);
                queue.push_back(target);
            }
        }
    }
    // `compile_terminal_expr_dfa` should already be root-reachable, but retain
    // any unexpected disconnected states deterministically rather than making
    // this canonicalizer lossy.
    for old in 0..n as u32 {
        if old_to_new[old as usize] == u32::MAX {
            old_to_new[old as usize] = order.len() as u32;
            order.push(old);
        }
    }

    let mut out = DFA::new(n);
    out.ensure_group_capacity(dfa.num_groups());
    for group in 0..dfa.num_groups() as u32 {
        out.set_group_u8set(group, *dfa.group_id_to_u8set(group));
    }
    for (new_state, &old_state) in order.iter().enumerate() {
        out.overwrite_state_metadata(
            new_state as u32,
            dfa.finalizers(old_state).clone(),
            BitSet::new(dfa.num_groups()),
        );
        for byte in 0u16..=255 {
            if let Some(old_target) = dfa.step(old_state, byte as u8) {
                out.add_transition(
                    new_state as u32,
                    byte as u8,
                    old_to_new[old_target as usize],
                );
            }
        }
    }
    out.recompute_possible_futures();
    out
}

impl BoundedCodeIntersectionOracle {
    fn from_expr(expr: &Expr) -> Option<Self> {
        let mut operands = Vec::new();
        flatten_intersection_operands(expr, &mut operands);
        if operands.len() < 2 {
            return None;
        }

        let mut envelope: Option<(Vec<u8>, Expr, usize, usize, Vec<u8>)> = None;
        let mut pattern_operands = Vec::new();
        for operand in operands {
            if let Some((prefix, body, min, max, suffix)) = bounded_code_envelope(operand) {
                match &mut envelope {
                    None => {
                        envelope = Some((prefix, body, min, max, suffix));
                        continue;
                    }
                    Some((existing_prefix, existing_body, existing_min, existing_max, existing_suffix))
                        if *existing_prefix == prefix
                            && *existing_body == body
                            && *existing_suffix == suffix =>
                    {
                        // Intersecting identical code envelopes is exactly an
                        // intersection of their copy-count intervals. This
                        // occurs naturally when JSON Schema `allOf` contains
                        // multiple differently-patterned bounded strings.
                        *existing_min = (*existing_min).max(min);
                        *existing_max = (*existing_max).min(max);
                        continue;
                    }
                    Some(_) => {}
                }
            }
            pattern_operands.push(operand.clone());
        }
        let (prefix, body_expr, min, max, suffix) = envelope?;
        if pattern_operands.is_empty() || max == usize::MAX {
            return None;
        }
        let pattern_expr = pattern_operands
            .into_iter()
            .reduce(|expr, intersect| Expr::Intersect {
                expr: Box::new(expr),
                intersect: Box::new(intersect),
            })?;
        // The oracle is a sidecar for avoiding giant-repeat materialization, so
        // its own proof construction must never eagerly materialize a giant
        // bounded repeat hidden inside either finite coordinate. The outer
        // envelope repeat is represented by the relation-doubling counter and
        // is intentionally not part of this check.
        if expression_contains_large_bounded_repeat(&pattern_expr)
            || expression_contains_large_bounded_repeat(&body_expr)
        {
            return None;
        }
        let pattern = Arc::new(canonicalize_bounded_code_oracle_dfa(
            compile_terminal_expr_dfa(&pattern_expr),
        ));
        let body = Arc::new(canonicalize_bounded_code_oracle_dfa(
            compile_terminal_expr_dfa(&body_expr),
        ));
        if pattern.num_states() == 0
            || pattern.num_states() > MAX_BOUNDED_CODE_ORACLE_PATTERN_STATES
            || pattern
                .states()
                .iter()
                .any(|state| !state.epsilon_transitions.is_empty())
            || body.num_states() == 0
            || body
                .states()
                .iter()
                .any(|state| !state.epsilon_transitions.is_empty())
        {
            return None;
        }

        let body_productive = exact_productive_states(&body);
        if !body.finalizers(0).is_empty() || !dfa_language_is_prefix_free(&body, &body_productive) {
            return None;
        }
        // At a repetition boundary a suffix byte must choose exactly one of
        // "start another code word" and "start the suffix".  A transition
        // into a semantically dead body state does not create ambiguity.
        if body
            .step(0, suffix[0])
            .is_some_and(|target| body_productive[target as usize])
        {
            return None;
        }

        let pattern_states = pattern.num_states();
        let body_states = body.num_states();
        if pattern_states.checked_mul(body_states)?
            > MAX_BOUNDED_CODE_ORACLE_BODY_PRODUCT_CELLS
        {
            return None;
        }
        // `apply_up_to(k)` represents the inclusive range 0..=k as `k + 1`
        // binary blocks.  Size the doubling table for `max + 1`, not `max`:
        // when max = 2^n - 1 the inclusive range needs the R^(2^n) block even
        // though an exact count <= max does not. `usize::MAX` was rejected
        // above, so the addition is exact.
        let range_block_count = max + 1;
        let bits = usize::BITS as usize - range_block_count.leading_zeros() as usize;
        let words_per_row = pattern_states.div_ceil(64);
        let relation_bytes = pattern_states
            .checked_mul(words_per_row)?
            .checked_mul(std::mem::size_of::<u64>())?;
        let estimated_relation_bytes = relation_bytes
            .checked_mul(body_states.checked_add(bits.checked_mul(2)?)?)?;
        if estimated_relation_bytes > MAX_BOUNDED_CODE_ORACLE_RELATION_BYTES {
            return None;
        }

        let mut suffix_accepting = BitSet::new(pattern_states);
        for state in 0..pattern_states as u32 {
            if let Some(end) = step_fixed_bytes(&pattern, state, &suffix)
                && !pattern.finalizers(end).is_empty()
            {
                suffix_accepting.set(state as usize);
            }
        }

        let mut oracle = Self {
            pattern,
            body,
            body_productive: body_productive.into_boxed_slice(),
            prefix: Arc::from(prefix.into_boxed_slice()),
            suffix: Arc::from(suffix.into_boxed_slice()),
            min,
            max,
            suffix_accepting,
            completion_relations: vec![None; body_states],
            exact_powers: Vec::new(),
            prefix_sums: Vec::new(),
        };
        let one_code = oracle.completion_relation(0).clone();
        oracle.exact_powers.push(one_code);
        oracle
            .prefix_sums
            .push(BoolRelation::identity(pattern_states));
        oracle.ensure_power(bits.saturating_sub(1));
        Some(oracle)
    }

    fn root_coordinate(&self) -> BoundedCodeOracleCoordinate {
        BoundedCodeOracleCoordinate {
            pattern_state: 0,
            envelope: BoundedCodeEnvelopeState::Prefix { next: 0 },
        }
    }

    fn completion_relation(&mut self, body_state: u32) -> &BoolRelation {
        let index = body_state as usize;
        if self.completion_relations[index].is_none() {
            let pattern_states = self.pattern.num_states();
            let body_states = self.body.num_states();
            let mut rows = Vec::with_capacity(pattern_states);
            for pattern_start in 0..pattern_states as u32 {
                let mut targets = BitSet::new(pattern_states);
                let mut seen = FxHashSet::<u64>::default();
                let mut queue = VecDeque::from([(pattern_start, body_state)]);
                seen.insert((u64::from(pattern_start) << 32) | u64::from(body_state));
                while let Some((pattern_state, code_state)) = queue.pop_front() {
                    for (byte, &code_target) in
                        self.body.states()[code_state as usize].transitions.iter()
                    {
                        let Some(pattern_target) = self.pattern.step(pattern_state, byte) else {
                            continue;
                        };
                        if !self.body.finalizers(code_target).is_empty() {
                            targets.set(pattern_target as usize);
                            continue;
                        }
                        if !self.body_productive[code_target as usize] {
                            continue;
                        }
                        debug_assert!((code_target as usize) < body_states);
                        let key = (u64::from(pattern_target) << 32) | u64::from(code_target);
                        if seen.insert(key) {
                            queue.push_back((pattern_target, code_target));
                        }
                    }
                }
                rows.push(targets);
            }
            self.completion_relations[index] = Some(BoolRelation { rows });
        }
        self.completion_relations[index].as_ref().unwrap()
    }

    fn ensure_power(&mut self, bit: usize) {
        while self.exact_powers.len() <= bit {
            let previous_power = self.exact_powers.last().unwrap().clone();
            let previous_sum = self.prefix_sums.last().unwrap().clone();
            let next_power = previous_power.then(&previous_power);
            let shifted_sum = previous_power.then(&previous_sum);
            self.exact_powers.push(next_power);
            self.prefix_sums.push(previous_sum.union(&shifted_sum));
        }
    }

    fn apply_exact_count(&self, mut states: BitSet, count: usize) -> BitSet {
        let mut remaining = count;
        let mut bit = 0usize;
        while remaining != 0 {
            if remaining & 1 != 0 {
                states = self.exact_powers[bit].apply(&states);
                if states.is_empty() {
                    break;
                }
            }
            remaining >>= 1;
            bit += 1;
        }
        states
    }

    /// Union states reachable after any number of whole code words in
    /// `[0, max_extra]`.
    fn apply_up_to(&self, states: BitSet, max_extra: usize) -> BitSet {
        let mut exact_offset = states;
        let mut union = BitSet::new(self.pattern.num_states());
        let mut block_count = max_extra.checked_add(1).unwrap();
        let mut bit = 0usize;
        while block_count != 0 {
            if block_count & 1 != 0 {
                union.union_with(&self.prefix_sums[bit].apply(&exact_offset));
                exact_offset = self.exact_powers[bit].apply(&exact_offset);
            }
            block_count >>= 1;
            bit += 1;
        }
        union
    }

    fn range_reaches_suffix(
        &self,
        starts: BitSet,
        completed: usize,
    ) -> bool {
        if completed > self.max {
            return false;
        }
        let low = self.min.saturating_sub(completed);
        let high = self.max - completed;
        if low > high {
            return false;
        }
        let after_low = self.apply_exact_count(starts, low);
        if after_low.is_empty() {
            return false;
        }
        let reachable = self.apply_up_to(after_low, high - low);
        !reachable.is_disjoint(&self.suffix_accepting)
    }

    fn step_coordinate(
        &self,
        coordinate: BoundedCodeOracleCoordinate,
        byte: u8,
    ) -> Option<BoundedCodeOracleCoordinate> {
        let pattern_state = self.pattern.step(coordinate.pattern_state, byte)?;
        let envelope = match coordinate.envelope {
            BoundedCodeEnvelopeState::Prefix { next } => {
                if self.prefix.get(next).copied()? != byte {
                    return None;
                }
                if next + 1 == self.prefix.len() {
                    BoundedCodeEnvelopeState::Body {
                        completed: 0,
                        body_state: 0,
                    }
                } else {
                    BoundedCodeEnvelopeState::Prefix { next: next + 1 }
                }
            }
            BoundedCodeEnvelopeState::Body {
                completed,
                body_state,
            } => {
                if body_state == 0
                    && completed >= self.min
                    && self.suffix[0] == byte
                {
                    if self.suffix.len() == 1 {
                        BoundedCodeEnvelopeState::Done
                    } else {
                        BoundedCodeEnvelopeState::Suffix { next: 1 }
                    }
                } else {
                    if completed >= self.max {
                        return None;
                    }
                    let target = self.body.step(body_state, byte)?;
                    if !self.body.finalizers(target).is_empty() {
                        BoundedCodeEnvelopeState::Body {
                            completed: completed.checked_add(1)?,
                            body_state: 0,
                        }
                    } else if self.body_productive[target as usize] {
                        BoundedCodeEnvelopeState::Body {
                            completed,
                            body_state: target,
                        }
                    } else {
                        return None;
                    }
                }
            }
            BoundedCodeEnvelopeState::Suffix { next } => {
                if self.suffix.get(next).copied()? != byte {
                    return None;
                }
                if next + 1 == self.suffix.len() {
                    BoundedCodeEnvelopeState::Done
                } else {
                    BoundedCodeEnvelopeState::Suffix { next: next + 1 }
                }
            }
            BoundedCodeEnvelopeState::Done => return None,
        };
        Some(BoundedCodeOracleCoordinate {
            pattern_state,
            envelope,
        })
    }

    fn has_future(&mut self, coordinate: BoundedCodeOracleCoordinate) -> bool {
        match coordinate.envelope {
            BoundedCodeEnvelopeState::Done => false,
            BoundedCodeEnvelopeState::Prefix { next } => {
                let Some(pattern_state) =
                    step_fixed_bytes(&self.pattern, coordinate.pattern_state, &self.prefix[next..])
                else {
                    return false;
                };
                let mut starts = BitSet::new(self.pattern.num_states());
                starts.set(pattern_state as usize);
                self.range_reaches_suffix(starts, 0)
            }
            BoundedCodeEnvelopeState::Body {
                completed,
                body_state,
            } if body_state == 0 => {
                let mut starts = BitSet::new(self.pattern.num_states());
                starts.set(coordinate.pattern_state as usize);
                self.range_reaches_suffix(starts, completed)
            }
            BoundedCodeEnvelopeState::Body {
                completed,
                body_state,
            } => {
                if completed >= self.max {
                    return false;
                }
                let mut starts = BitSet::new(self.pattern.num_states());
                starts.set(coordinate.pattern_state as usize);
                let after_current = self.completion_relation(body_state).apply(&starts);
                if after_current.is_empty() {
                    return false;
                }
                self.range_reaches_suffix(after_current, completed + 1)
            }
            BoundedCodeEnvelopeState::Suffix { next } => {
                step_fixed_bytes(&self.pattern, coordinate.pattern_state, &self.suffix[next..])
                    .is_some_and(|state| !self.pattern.finalizers(state).is_empty())
            }
        }
    }
}


#[derive(Debug, Clone)]
#[doc(hidden)]
pub struct VirtualResidualMaskProjection {
    runtime: Arc<VirtualResidualRuntime>,
    state_offset: u32,
    pattern_states: usize,
    body_states: usize,
    prefix_len: usize,
    suffix_len: usize,
    min: usize,
    full_max: usize,
    mask_max: usize,
    crossed_boundaries: usize,
    local_to_mask_state: Arc<[u32]>,
}

/// Serialized half of a [`VirtualResidualMaskProjection`].
///
/// The exact residual runtime is already reconstructed from the terminal
/// expression + virtual-runtime metadata. Persist only the compiled finite
/// observation transport and reattach it to that exact runtime after load.
#[derive(Debug, Clone, serde::Deserialize)]
#[doc(hidden)]
pub struct VirtualResidualMaskProjectionArtifact {
    terminal: TerminalID,
    state_offset: u32,
    local_to_mask_state: Vec<u32>,
    oracle_bytes: Vec<u8>,
    #[serde(skip)]
    runtime_expr_bytes: Vec<u8>,
    #[serde(skip)]
    compiled_mask_max: usize,
    #[serde(skip)]
    compiled_crossed_boundaries: usize,
}

#[derive(serde::Serialize)]
#[doc(hidden)]
pub struct VirtualResidualMaskProjectionArtifactRef<'a> {
    terminal: TerminalID,
    state_offset: u32,
    local_to_mask_state: &'a [u32],
    oracle_bytes: Vec<u8>,
}

impl VirtualResidualMaskProjectionArtifact {
    #[doc(hidden)]
    pub fn state_offset(&self) -> u32 {
        self.state_offset
    }

    #[doc(hidden)]
    pub fn terminal(&self) -> TerminalID {
        self.terminal
    }

    #[doc(hidden)]
    pub fn oracle_bytes(&self) -> &[u8] {
        &self.oracle_bytes
    }

    #[doc(hidden)]
    pub fn runtime_expr_bytes(&self) -> &[u8] {
        &self.runtime_expr_bytes
    }

    #[doc(hidden)]
    pub fn compiled_mask_max(&self) -> usize { self.compiled_mask_max }

    #[doc(hidden)]
    pub fn compiled_crossed_boundaries(&self) -> usize { self.compiled_crossed_boundaries }

    #[doc(hidden)]
    pub fn from_wire(
        terminal: TerminalID,
        state_offset: u32,
        local_to_mask_state: Vec<u32>,
        oracle_bytes: Vec<u8>,
        runtime_expr_bytes: Vec<u8>,
        compiled_mask_max: usize,
        compiled_crossed_boundaries: usize,
    ) -> Self {
        Self {
            terminal, state_offset, local_to_mask_state, oracle_bytes, runtime_expr_bytes,
            compiled_mask_max, compiled_crossed_boundaries,
        }
    }
}

impl VirtualResidualMaskProjection {
    #[doc(hidden)]
    pub fn artifact_ref(&self) -> VirtualResidualMaskProjectionArtifactRef<'_> {
        VirtualResidualMaskProjectionArtifactRef {
            terminal: self.runtime.terminal(),
            state_offset: self.state_offset,
            local_to_mask_state: self.local_to_mask_state.as_ref(),
            oracle_bytes: self.runtime.serialized_bounded_code_oracle(),
        }
    }

    #[doc(hidden)]
    pub fn artifact_wire_parts(&self) -> (TerminalID, u32, &[u32], Vec<u8>, usize, usize) {
        (
            self.runtime.terminal(),
            self.state_offset,
            self.local_to_mask_state.as_ref(),
            self.runtime.serialized_bounded_code_oracle(),
            self.mask_max,
            self.crossed_boundaries,
        )
    }

    fn local_state_for_coordinate(&self, coordinate: BoundedCodeOracleCoordinate) -> Option<u32> {
        let pattern = coordinate.pattern_state as usize;
        if pattern >= self.pattern_states {
            return None;
        }
        let prefix_block = self.prefix_len.checked_mul(self.pattern_states)?;
        let body_layer = self.body_states.checked_mul(self.pattern_states)?;
        let body_block = self.mask_max.checked_add(1)?.checked_mul(body_layer)?;
        let suffix_slots = self.suffix_len.saturating_sub(1);
        let suffix_block = suffix_slots.checked_mul(self.pattern_states)?;
        let local = match coordinate.envelope {
            BoundedCodeEnvelopeState::Prefix { next } => {
                if next >= self.prefix_len { return None; }
                next.checked_mul(self.pattern_states)?.checked_add(pattern)?
            }
            BoundedCodeEnvelopeState::Body { completed, body_state } => {
                let body_state = body_state as usize;
                if body_state >= self.body_states || completed > self.full_max {
                    return None;
                }
                let distance_to_upper = self.full_max - completed;
                let mapped_completed = if completed < self.min {
                    completed
                } else if distance_to_upper <= self.crossed_boundaries {
                    self.mask_max.checked_sub(distance_to_upper)?
                } else {
                    self.min
                };
                if mapped_completed > self.mask_max { return None; }
                prefix_block
                    .checked_add(mapped_completed.checked_mul(body_layer)?)?
                    .checked_add(body_state.checked_mul(self.pattern_states)?)?
                    .checked_add(pattern)?
            }
            BoundedCodeEnvelopeState::Suffix { next } => {
                if next == 0 || next >= self.suffix_len { return None; }
                prefix_block
                    .checked_add(body_block)?
                    .checked_add((next - 1).checked_mul(self.pattern_states)?)?
                    .checked_add(pattern)?
            }
            BoundedCodeEnvelopeState::Done => prefix_block
                .checked_add(body_block)?
                .checked_add(suffix_block)?
                .checked_add(pattern)?,
        };
        u32::try_from(local).ok()
    }

    #[inline]
    pub fn project(&self, full_state: u32) -> Option<u32> {
        if !self.runtime.handles_state(full_state) {
            return None;
        }
        let coordinate = self.runtime.oracle_coordinate(full_state)?;
        let local = self.local_state_for_coordinate(coordinate)? as usize;
        let mapped = *self.local_to_mask_state.get(local)?;
        (mapped != u32::MAX)
            .then_some(mapped)?
            .checked_add(self.state_offset)
    }

    pub(super) fn set_state_offset(&mut self, state_offset: u32) {
        self.state_offset = state_offset;
    }

    pub fn physical_state_count(&self) -> u32 {
        self.runtime.physical_state_count()
    }
}

impl BoundedCodeIntersectionOracle {
    fn finite_mask_dense_state_count(&self, mask_max: usize) -> Option<usize> {
        if mask_max < self.min || mask_max > self.max {
            return None;
        }
        let pattern_states = self.pattern.num_states();
        let body_states = self.body.num_states();
        let prefix_states = self.prefix.len().checked_mul(pattern_states)?;
        let body_states_total = mask_max
            .checked_add(1)?
            .checked_mul(body_states)?
            .checked_mul(pattern_states)?;
        let suffix_states = self
            .suffix
            .len()
            .saturating_sub(1)
            .checked_mul(pattern_states)?;
        prefix_states
            .checked_add(body_states_total)?
            .checked_add(suffix_states)?
            .checked_add(pattern_states)
    }

    fn coordinate_local_state(
        &self,
        coordinate: BoundedCodeOracleCoordinate,
        mask_max: usize,
    ) -> Option<u32> {
        let pattern_states = self.pattern.num_states();
        let body_states = self.body.num_states();
        let pattern = coordinate.pattern_state as usize;
        if pattern >= pattern_states { return None; }
        let prefix_block = self.prefix.len().checked_mul(pattern_states)?;
        let body_layer = body_states.checked_mul(pattern_states)?;
        let body_block = mask_max.checked_add(1)?.checked_mul(body_layer)?;
        let suffix_slots = self.suffix.len().saturating_sub(1);
        let suffix_block = suffix_slots.checked_mul(pattern_states)?;
        let local = match coordinate.envelope {
            BoundedCodeEnvelopeState::Prefix { next } => {
                if next >= self.prefix.len() { return None; }
                next.checked_mul(pattern_states)?.checked_add(pattern)?
            }
            BoundedCodeEnvelopeState::Body { completed, body_state } => {
                let body_state = body_state as usize;
                if completed > mask_max || body_state >= body_states { return None; }
                prefix_block
                    .checked_add(completed.checked_mul(body_layer)?)?
                    .checked_add(body_state.checked_mul(pattern_states)?)?
                    .checked_add(pattern)?
            }
            BoundedCodeEnvelopeState::Suffix { next } => {
                if next == 0 || next >= self.suffix.len() { return None; }
                prefix_block
                    .checked_add(body_block)?
                    .checked_add((next - 1).checked_mul(pattern_states)?)?
                    .checked_add(pattern)?
            }
            BoundedCodeEnvelopeState::Done => prefix_block
                .checked_add(body_block)?
                .checked_add(suffix_block)?
                .checked_add(pattern)?,
        };
        u32::try_from(local).ok()
    }

    fn finite_mask_dfa(&self, mask_max: usize) -> Option<(DFA, u32, Vec<u32>)> {
        let profile = std::env::var_os("GLRMASK_PROFILE_TOKENIZER_TIMING").is_some();
        let total_started = std::time::Instant::now();
        let pattern_states = self.pattern.num_states();
        let dense_state_count = self.finite_mask_dense_state_count(mask_max)?;
        if dense_state_count == 0 || dense_state_count > MAX_FINITE_MASK_DENSE_STATES {
            return None;
        }
        let mask_oracle = BoundedCodeIntersectionOracle {
            pattern: Arc::clone(&self.pattern),
            body: Arc::clone(&self.body),
            body_productive: self.body_productive.clone(),
            prefix: Arc::clone(&self.prefix),
            suffix: Arc::clone(&self.suffix),
            min: self.min,
            max: mask_max,
            suffix_accepting: self.suffix_accepting.clone(),
            completion_relations: self.completion_relations.clone(),
            exact_powers: self.exact_powers.clone(),
            prefix_sums: self.prefix_sums.clone(),
        };

        // The finite envelope deliberately identifies the huge exact count
        // interval with a small one-token stencil. A state that is a valid
        // projection source after a long exact input need not be reachable from
        // the finite root at the corresponding small count, because the format
        // DFA can be in a state only reachable after many code words. Seed the
        // finite graph with every *actually possible* projected code-word
        // boundary, then retain the transition closure of those roots. This
        // avoids the full count × body × pattern Cartesian product while still
        // covering every exact token-boundary state.
        let crossed_boundaries = mask_max.checked_sub(self.min)?.checked_sub(1)?;
        let after_prefix = step_fixed_bytes(&self.pattern, 0, &self.prefix)?;
        let mut pattern_start = BitSet::new(pattern_states);
        pattern_start.set(after_prefix as usize);
        let mut boundary_classes = Vec::<(usize, BitSet)>::new();

        // Counts below min are observed exactly.
        for completed in 0..self.min {
            let states = self.apply_exact_count(pattern_start.clone(), completed);
            if !states.is_empty() {
                boundary_classes.push((completed, states));
            }
        }

        // The deep interior collapses to the first accepting count. Compute the
        // exact union of format states reachable at any full count represented
        // by that interior layer using the existing relation-doubling oracle.
        let interior_high = self
            .max
            .checked_sub(crossed_boundaries.saturating_add(1));
        if let Some(interior_high) = interior_high.filter(|&high| high >= self.min) {
            let after_low = self.apply_exact_count(pattern_start.clone(), self.min);
            if !after_low.is_empty() {
                let states = self.apply_up_to(after_low, interior_high - self.min);
                if !states.is_empty() {
                    boundary_classes.push((self.min, states));
                }
            }
        }

        // Near the true upper bound, preserve distance-to-upper exactly.
        let upper_start = self.max.saturating_sub(crossed_boundaries).max(self.min);
        for completed in upper_start..=self.max {
            let distance_to_upper = self.max - completed;
            let mapped_completed = mask_max.checked_sub(distance_to_upper)?;
            let states = self.apply_exact_count(pattern_start.clone(), completed);
            if !states.is_empty() {
                boundary_classes.push((mapped_completed, states));
            }
        }

        let seeds_started = std::time::Instant::now();
        let mut dense_to_sparse = vec![u32::MAX; dense_state_count];
        let mut coordinates = Vec::<BoundedCodeOracleCoordinate>::new();
        let mut seed_count = 0usize;
        let mut add_seed = |coordinate: BoundedCodeOracleCoordinate| -> Option<()> {
            let dense = mask_oracle.coordinate_local_state(coordinate, mask_max)? as usize;
            if dense_to_sparse[dense] == u32::MAX {
                dense_to_sparse[dense] = u32::try_from(coordinates.len()).ok()?;
                coordinates.push(coordinate);
                seed_count += 1;
            }
            Some(())
        };
        add_seed(mask_oracle.root_coordinate())?;
        for (mapped_completed, states) in boundary_classes {
            for pattern_state in states.iter_ones() {
                add_seed(BoundedCodeOracleCoordinate {
                    pattern_state: u32::try_from(pattern_state).ok()?,
                    envelope: BoundedCodeEnvelopeState::Body {
                        completed: mapped_completed,
                        body_state: 0,
                    },
                })?;
            }
        }
        drop(add_seed);
        let seeds_ms = seeds_started.elapsed().as_secs_f64() * 1000.0;

        let expand_started = std::time::Instant::now();
        let mut dfa = DFA::new(coordinates.len());
        dfa.ensure_group_capacity(1);
        dfa.set_group_u8set(0, crate::ds::u8set::U8Set::all());
        let mut source = 0usize;
        const EXPANSION_BATCH: usize = 1_024;
        while source < coordinates.len() {
            while dfa.num_states() < coordinates.len() {
                dfa.add_state();
            }
            let batch_end = coordinates.len().min(source.saturating_add(EXPANSION_BATCH));
            let batch = coordinates[source..batch_end].to_vec();
            let rows = batch
                .par_iter()
                .map(|&coordinate| {
                    let accepting = matches!(coordinate.envelope, BoundedCodeEnvelopeState::Done)
                        && !mask_oracle.pattern.finalizers(coordinate.pattern_state).is_empty();
                    let transitions = (0u16..=255)
                        .filter_map(|byte| {
                            let byte = byte as u8;
                            let next = mask_oracle.step_coordinate(coordinate, byte)?;
                            let dense = mask_oracle.coordinate_local_state(next, mask_max)? as usize;
                            Some((byte, next, dense))
                        })
                        .collect::<Vec<_>>();
                    Some((accepting, transitions))
                })
                .collect::<Option<Vec<_>>>()?;

            for (row_offset, (accepting, transitions)) in rows.into_iter().enumerate() {
                let source_state = (source + row_offset) as u32;
                let mut finalizers = BitSet::new(1);
                if accepting {
                    finalizers.set(0);
                }
                dfa.overwrite_state_metadata(source_state, finalizers, BitSet::new(1));
                for (byte, next, dense) in transitions {
                    let target = if dense_to_sparse[dense] == u32::MAX {
                        let target = u32::try_from(coordinates.len()).ok()?;
                        dense_to_sparse[dense] = target;
                        coordinates.push(next);
                        target
                    } else {
                        dense_to_sparse[dense]
                    };
                    while dfa.num_states() <= target as usize {
                        dfa.add_state();
                    }
                    dfa.add_transition(source_state, byte, target);
                }
            }
            source = batch_end;
        }
        dfa.recompute_possible_futures();
        let expand_ms = expand_started.elapsed().as_secs_f64() * 1000.0;

        let sparse_state_count = dfa.num_states();
        let root_dense = mask_oracle
            .coordinate_local_state(mask_oracle.root_coordinate(), mask_max)? as usize;
        let root_sparse = dense_to_sparse[root_dense];
        if root_sparse == u32::MAX {
            return None;
        }
        // Every sparse state is reachable from at least one exact projection
        // seed, but many seeds are intentionally disconnected from state 0.
        // Preserve those roots while quotienting language-equivalent states.
        let minimize_started = std::time::Instant::now();
        let (dfa, sparse_to_minimized) = dfa.minimize_with_state_mapping_preserve_unreachable();
        let minimize_ms = minimize_started.elapsed().as_secs_f64() * 1000.0;
        let root = *sparse_to_minimized.get(root_sparse as usize)?;
        if root == u32::MAX {
            return None;
        }
        let remap_started = std::time::Instant::now();
        let dense_to_minimized = dense_to_sparse
            .into_iter()
            .map(|sparse| {
                if sparse == u32::MAX {
                    u32::MAX
                } else {
                    sparse_to_minimized[sparse as usize]
                }
            })
            .collect::<Vec<_>>();
        let remap_ms = remap_started.elapsed().as_secs_f64() * 1000.0;
        if profile {
            eprintln!(
                "[glrmask/profile][residual_mask_symbolic_sources] dense_states={} seeds={} sparse_states={} minimized_states={} seeds_ms={:.3} expand_ms={:.3} minimize_ms={:.3} remap_ms={:.3} total_ms={:.3}",
                dense_state_count, seed_count, sparse_state_count, dfa.num_states(),
                seeds_ms, expand_ms, minimize_ms, remap_ms, total_started.elapsed().as_secs_f64() * 1000.0,
            );
        }
        Some((dfa, root, dense_to_minimized))
    }
}

fn unwrap_shared_expr(mut expr: &Expr) -> &Expr {
    while let Expr::Shared(inner) = expr {
        expr = inner;
    }
    expr
}

fn flatten_intersection_operands<'a>(expr: &'a Expr, out: &mut Vec<&'a Expr>) {
    match unwrap_shared_expr(expr) {
        Expr::Intersect { expr, intersect } => {
            flatten_intersection_operands(expr, out);
            flatten_intersection_operands(intersect, out);
        }
        other => out.push(other),
    }
}

fn flatten_sequence_operands<'a>(expr: &'a Expr, out: &mut Vec<&'a Expr>) {
    match unwrap_shared_expr(expr) {
        Expr::Seq(parts) => {
            for part in parts {
                flatten_sequence_operands(part, out);
            }
        }
        other => out.push(other),
    }
}

fn bounded_code_envelope(expr: &Expr) -> Option<(Vec<u8>, Expr, usize, usize, Vec<u8>)> {
    let mut parts = Vec::new();
    flatten_sequence_operands(expr, &mut parts);
    let repeat_index = parts
        .iter()
        .position(|part| matches!(unwrap_shared_expr(part), Expr::Repeat { max: Some(_), .. }))?;
    if parts
        .iter()
        .enumerate()
        .any(|(index, part)| {
            index != repeat_index
                && !matches!(unwrap_shared_expr(part), Expr::U8Seq(bytes) if !bytes.is_empty())
        })
    {
        return None;
    }
    if parts[repeat_index + 1..]
        .iter()
        .any(|part| matches!(unwrap_shared_expr(part), Expr::Repeat { .. }))
    {
        return None;
    }
    let mut prefix = Vec::new();
    for part in &parts[..repeat_index] {
        let Expr::U8Seq(bytes) = unwrap_shared_expr(part) else {
            return None;
        };
        prefix.extend_from_slice(bytes);
    }
    let mut suffix = Vec::new();
    for part in &parts[repeat_index + 1..] {
        let Expr::U8Seq(bytes) = unwrap_shared_expr(part) else {
            return None;
        };
        suffix.extend_from_slice(bytes);
    }
    if prefix.is_empty() || suffix.is_empty() {
        return None;
    }
    let Expr::Repeat {
        expr: body,
        min,
        max: Some(max),
    } = unwrap_shared_expr(parts[repeat_index])
    else {
        return None;
    };
    (*min <= *max).then(|| (prefix, unwrap_shared_expr(body).clone(), *min, *max, suffix))
}

fn exact_productive_states(dfa: &DFA) -> Vec<bool> {
    let mut reverse = vec![Vec::<u32>::new(); dfa.num_states()];
    for (source, state) in dfa.states().iter().enumerate() {
        for (_, &target) in state.transitions.iter() {
            reverse[target as usize].push(source as u32);
        }
    }
    let mut productive = vec![false; dfa.num_states()];
    let mut stack = Vec::new();
    for state in 0..dfa.num_states() as u32 {
        if !dfa.finalizers(state).is_empty() {
            productive[state as usize] = true;
            stack.push(state);
        }
    }
    while let Some(state) = stack.pop() {
        for &predecessor in &reverse[state as usize] {
            if !productive[predecessor as usize] {
                productive[predecessor as usize] = true;
                stack.push(predecessor);
            }
        }
    }
    productive
}

fn dfa_language_is_prefix_free(dfa: &DFA, productive: &[bool]) -> bool {
    for state in dfa.states() {
        if state.finalizers.is_empty() {
            continue;
        }
        if state
            .transitions
            .iter()
            .any(|(_, &target)| productive[target as usize])
        {
            return false;
        }
    }
    true
}

fn step_fixed_bytes(dfa: &DFA, mut state: u32, bytes: &[u8]) -> Option<u32> {
    for &byte in bytes {
        state = dfa.step(state, byte)?;
    }
    Some(state)
}

#[derive(Debug)]
struct ResidualRuntimeStore {
    arena: ResidualArena,
    root: ResidualId,
    state_by_residual: Vec<u32>,
    residual_by_state: FxHashMap<u32, ResidualId>,
    state_by_residual_coordinate:
        FxHashMap<(ResidualId, BoundedCodeOracleCoordinate), u32>,
    coordinate_by_state: FxHashMap<u32, BoundedCodeOracleCoordinate>,
    oracle_future_by_state: FxHashMap<u32, bool>,
    liveness_oracle: Option<BoundedCodeIntersectionOracle>,
    oracle_coordinates: Vec<BoundedCodeOracleSlot>,
    oracle_futures: Vec<Option<bool>>,
}

/// Exact general symbolic tokenizer component. The regex upper bounds live in
/// `ResidualNode::Repeat`; this store grows only when runtime bytes discover a
/// new canonical language residual.
#[derive(Debug)]
pub(super) struct VirtualResidualRuntime {
    runtime_index: u32,
    terminal: TerminalID,
    physical_state_count: u32,
    root_state: u32,
    root_has_future: bool,
    preserve_oracle_coordinate: bool,
    state_allocator: Arc<VirtualStateAllocator>,
    state_owners: Arc<VirtualRuntimeStateOwners>,
    accepting: BitSet,
    live: BitSet,
    dead: BitSet,
    accepting_list: Box<[TerminalID]>,
    store: Mutex<ResidualRuntimeStore>,
}

impl VirtualResidualRuntime {
    pub(super) fn new(
        expr: &Expr,
        runtime_index: u32,
        terminal: TerminalID,
        num_terminals: u32,
        physical_state_count: u32,
        root_state: u32,
        state_allocator: Arc<VirtualStateAllocator>,
        state_owners: Arc<VirtualRuntimeStateOwners>,
    ) -> Option<Self> {
        Self::new_impl(
            expr,
            runtime_index,
            terminal,
            num_terminals,
            physical_state_count,
            root_state,
            state_allocator,
            state_owners,
            false,
        )
    }

    pub(super) fn new_preserving_oracle_coordinate(
        expr: &Expr,
        runtime_index: u32,
        terminal: TerminalID,
        num_terminals: u32,
        physical_state_count: u32,
        root_state: u32,
        state_allocator: Arc<VirtualStateAllocator>,
        state_owners: Arc<VirtualRuntimeStateOwners>,
    ) -> Option<Self> {
        Self::new_impl(
            expr,
            runtime_index,
            terminal,
            num_terminals,
            physical_state_count,
            root_state,
            state_allocator,
            state_owners,
            true,
        )
    }

    fn new_impl(
        expr: &Expr,
        runtime_index: u32,
        terminal: TerminalID,
        num_terminals: u32,
        physical_state_count: u32,
        root_state: u32,
        state_allocator: Arc<VirtualStateAllocator>,
        state_owners: Arc<VirtualRuntimeStateOwners>,
        preserve_oracle_coordinate: bool,
    ) -> Option<Self> {
        if terminal >= num_terminals
            || physical_state_count == 0
            || root_state >= physical_state_count
            || state_owners.owner_index(root_state) != Some(runtime_index as usize)
        {
            return None;
        }
        let (mut arena, root) = ResidualArena::from_expr(expr)?;
        let liveness_oracle = BoundedCodeIntersectionOracle::from_expr(expr);
        let root_oracle_coordinate = liveness_oracle
            .as_ref()
            .map(BoundedCodeIntersectionOracle::root_coordinate);
        // Keep the physical proxy root's serialized future bit conservative.
        // Old artifacts and load-time validation use that physical metadata, so
        // installing an exact liveness sidecar must not silently change the wire
        // contract. At runtime `observation()` overlays an exact future bit for
        // residuals carrying a certified bounded-code coordinate; uncertified
        // Boolean residuals retain the conservative bit and are resolved through
        // the fallible `exact_has_future` boundary. Construction therefore does
        // not run the generic potentially expensive emptiness solver merely to
        // populate serialized proxy metadata.
        let root_live = arena.conservative_has_future(root);
        let mut state_by_residual = vec![u32::MAX; root as usize + 1];
        state_by_residual[root as usize] = root_state;
        let mut oracle_coordinates = vec![BoundedCodeOracleSlot::Unknown; arena.state_count()];
        let oracle_futures = vec![None; arena.state_count()];
        if let Some(coordinate) = root_oracle_coordinate {
            oracle_coordinates[root as usize] = BoundedCodeOracleSlot::Exact(coordinate);
        }
        let mut state_by_residual_coordinate = FxHashMap::default();
        let mut coordinate_by_state = FxHashMap::default();
        if preserve_oracle_coordinate {
            let coordinate = root_oracle_coordinate?;
            state_by_residual_coordinate.insert((root, coordinate), root_state);
            coordinate_by_state.insert(root_state, coordinate);
        }
        let mut accepting = BitSet::new(num_terminals as usize);
        accepting.set(terminal as usize);
        let live = accepting.clone();
        Some(Self {
            runtime_index,
            terminal,
            physical_state_count,
            root_state,
            root_has_future: root_live,
            preserve_oracle_coordinate,
            state_allocator,
            state_owners,
            accepting,
            live,
            dead: BitSet::new(num_terminals as usize),
            accepting_list: vec![terminal].into_boxed_slice(),
            store: Mutex::new(ResidualRuntimeStore {
                arena,
                root,
                state_by_residual,
                residual_by_state: FxHashMap::default(),
                state_by_residual_coordinate,
                coordinate_by_state,
                oracle_future_by_state: FxHashMap::default(),
                liveness_oracle,
                oracle_coordinates,
                oracle_futures,
            }),
        })
    }

    pub(super) fn terminal(&self) -> TerminalID {
        self.terminal
    }

    pub(super) fn root_state(&self) -> u32 {
        self.root_state
    }

    pub(super) fn physical_state_count(&self) -> u32 {
        self.physical_state_count
    }

    fn residual_for_state(store: &ResidualRuntimeStore, root_state: u32, state: u32) -> Option<ResidualId> {
        if state == root_state {
            Some(store.root)
        } else {
            store.residual_by_state.get(&state).copied()
        }
    }

    fn intern_locked(
        &self,
        store: &mut ResidualRuntimeStore,
        residual: ResidualId,
        coordinate: Option<BoundedCodeOracleCoordinate>,
    ) -> Option<u32> {
        if self.preserve_oracle_coordinate {
            let coordinate = coordinate?;
            if let Some(&state) = store
                .state_by_residual_coordinate
                .get(&(residual, coordinate))
            {
                return Some(state);
            }
            let state = self.state_allocator.allocate().expect(
                "exact residual tokenizer state-id space exhausted below the dynamic-NFA high-bit tag",
            );
            self.state_owners
                .register_virtual(state, self.runtime_index)
                .expect("residual virtual state owner index must follow shared allocator");
            store
                .state_by_residual_coordinate
                .insert((residual, coordinate), state);
            store.coordinate_by_state.insert(state, coordinate);
            store.residual_by_state.insert(state, residual);
            return Some(state);
        }

        let residual_index = residual as usize;
        if store.state_by_residual.len() <= residual_index {
            store.state_by_residual.resize(residual_index + 1, u32::MAX);
        }
        let state = store.state_by_residual[residual_index];
        if state != u32::MAX {
            return Some(state);
        }
        let state = self.state_allocator.allocate().expect(
            "exact residual tokenizer state-id space exhausted below the dynamic-NFA high-bit tag",
        );
        self.state_owners
            .register_virtual(state, self.runtime_index)
            .expect("residual virtual state owner index must follow shared allocator");
        store.state_by_residual[residual_index] = state;
        store.residual_by_state.insert(state, residual);
        Some(state)
    }

    pub(super) fn handles_state(&self, state: u32) -> bool {
        self.state_owners.owner_index(state) == Some(self.runtime_index as usize)
    }

    pub(super) fn owner_index(&self, state: u32) -> Option<usize> {
        self.state_owners.owner_index(state)
    }

    fn step_residual_locked(
        &self,
        store: &mut ResidualRuntimeStore,
        state: u32,
        residual: ResidualId,
        byte: u8,
    ) -> Option<u32> {
        let source_coordinate = if self.preserve_oracle_coordinate {
            store
                .coordinate_by_state
                .get(&state)
                .copied()
                .map(BoundedCodeOracleSlot::Exact)
                .unwrap_or(BoundedCodeOracleSlot::Unknown)
        } else {
            store
                .oracle_coordinates
                .get(residual as usize)
                .copied()
                .unwrap_or(BoundedCodeOracleSlot::Unknown)
        };
        let target = store.arena.step(residual, byte)?;
        if store.arena.is_empty(target) {
            return None;
        }
        if store.oracle_coordinates.len() < store.arena.state_count() {
            store.oracle_coordinates.resize(
                store.arena.state_count(),
                BoundedCodeOracleSlot::Unknown,
            );
            store.oracle_futures.resize(store.arena.state_count(), None);
        }
        if let Some(oracle) = store.liveness_oracle.as_ref() {
            let target_slot = match source_coordinate {
                BoundedCodeOracleSlot::Exact(source_coordinate) => {
                    if let Some(target_coordinate) =
                        oracle.step_coordinate(source_coordinate, byte)
                    {
                        BoundedCodeOracleSlot::Exact(target_coordinate)
                    } else {
                        // A structurally non-empty residual can still denote
                        // the empty language. The generic dynamic lane falls
                        // back to the exact residual solver; the Static
                        // coordinate-preserving lane treats this as a violated
                        // construction invariant because it cannot project an
                        // unmodelled state into a precompiled TSID coordinate.
                        if self.preserve_oracle_coordinate {
                            panic!(
                                "coordinate-preserving bounded-code runtime lost its exact coordinate on byte {byte}"
                            );
                        }
                        BoundedCodeOracleSlot::Ambiguous
                    }
                }
                BoundedCodeOracleSlot::Ambiguous | BoundedCodeOracleSlot::Unknown => {
                    if self.preserve_oracle_coordinate {
                        panic!(
                            "coordinate-preserving bounded-code runtime reached a state without an exact source coordinate"
                        );
                    }
                    BoundedCodeOracleSlot::Ambiguous
                }
            };

            if self.preserve_oracle_coordinate {
                let BoundedCodeOracleSlot::Exact(target_coordinate) = target_slot else {
                    unreachable!("coordinate-preserving target is exact or panics above");
                };
                return self.intern_locked(store, target, Some(target_coordinate));
            }

            let target_index = target as usize;
            let slot = &mut store.oracle_coordinates[target_index];
            let previous = *slot;
            *slot = match (previous, target_slot) {
                (BoundedCodeOracleSlot::Unknown, next) => next,
                (BoundedCodeOracleSlot::Exact(existing), BoundedCodeOracleSlot::Exact(next))
                    if existing == next =>
                {
                    BoundedCodeOracleSlot::Exact(existing)
                }
                (BoundedCodeOracleSlot::Ambiguous, _)
                | (_, BoundedCodeOracleSlot::Ambiguous)
                | (BoundedCodeOracleSlot::Exact(_), BoundedCodeOracleSlot::Exact(_)) => {
                    BoundedCodeOracleSlot::Ambiguous
                }
                (existing, BoundedCodeOracleSlot::Unknown) => existing,
            };
            if *slot != previous {
                store.oracle_futures[target_index] = None;
            }
        }
        self.intern_locked(store, target, None)
    }

    pub(super) fn step(&self, state: u32, byte: u8) -> Option<u32> {
        if state == self.root_state && !self.root_has_future {
            return None;
        }
        let mut store = self.store.lock().unwrap();
        let residual = Self::residual_for_state(&store, self.root_state, state)?;
        self.step_residual_locked(&mut store, state, residual, byte)
    }

    fn certified_oracle_future(
        store: &mut ResidualRuntimeStore,
        residual: ResidualId,
    ) -> Option<bool> {
        let index = residual as usize;
        let BoundedCodeOracleSlot::Exact(coordinate) = store
            .oracle_coordinates
            .get(index)
            .copied()
            .unwrap_or(BoundedCodeOracleSlot::Unknown)
        else {
            return None;
        };
        if let Some(cached) = store.oracle_futures.get(index).copied().flatten() {
            return Some(cached);
        }
        let future = store.liveness_oracle.as_mut()?.has_future(coordinate);
        store.oracle_futures[index] = Some(future);
        Some(future)
    }

    fn certified_oracle_future_for_state(
        &self,
        store: &mut ResidualRuntimeStore,
        state: u32,
        residual: ResidualId,
    ) -> Option<bool> {
        if !self.preserve_oracle_coordinate {
            return Self::certified_oracle_future(store, residual);
        }
        if let Some(&future) = store.oracle_future_by_state.get(&state) {
            return Some(future);
        }
        let coordinate = *store.coordinate_by_state.get(&state)?;
        let future = store.liveness_oracle.as_mut()?.has_future(coordinate);
        store.oracle_future_by_state.insert(state, future);
        Some(future)
    }

    fn observation(&self, state: u32) -> Option<(bool, bool)> {
        let mut store = self.store.lock().unwrap();
        let residual = Self::residual_for_state(&store, self.root_state, state)?;
        // Match the existing virtual-runtime convention: the physical proxy
        // root is the drained zero-byte configuration and must not emit a
        // terminal match before any input is consumed.
        let accepting = state != self.root_state && store.arena.is_nullable(residual);
        // A certified bounded-code coordinate has an infallible exact future
        // query once its oracle has been constructed. Surface that answer
        // through the ordinary tokenizer metadata path as well as through
        // `exact_has_future`: dynamic mask/commit contain several generic
        // consumers of `possible_future_terminals`, and those consumers should
        // observe the same exact liveness bit for this certified family.
        //
        // Unknown/ambiguous coordinates deliberately retain the old
        // conservative contract. Their exact query remains fallible and is
        // resolved only at the explicit dynamic residual boundary.
        let future = if let Some(future) =
            self.certified_oracle_future_for_state(&mut store, state, residual)
        {
            future
        } else if state == self.root_state {
            self.root_has_future
        } else {
            store.arena.conservative_has_future(residual)
        };
        Some((accepting, future))
    }

    pub(super) fn vocabulary_repeat_boundary_horizon(
        &self,
        vocab: &Vocab,
        cache: &VocabularyRepeatHorizonCache,
    ) -> Option<usize> {
        let store = self.store.lock().ok()?;
        let oracle = store.liveness_oracle.as_ref()?;
        cache.horizon_for_dfa(oracle.body.as_ref(), vocab)
    }

    pub(super) fn serialized_bounded_code_oracle(&self) -> Vec<u8> {
        let store = self.store.lock().unwrap();
        let oracle = store
            .liveness_oracle
            .as_ref()
            .expect("Static residual projection requires bounded-code oracle");
        bincode::serialize(oracle).expect("bounded-code oracle serialization should succeed")
    }

    pub(super) fn new_preserving_oracle_coordinate_from_oracle_bytes(
        expr: &Expr,
        oracle_bytes: &[u8],
        runtime_index: u32,
        terminal: TerminalID,
        num_terminals: u32,
        physical_state_count: u32,
        root_state: u32,
        state_allocator: Arc<VirtualStateAllocator>,
        state_owners: Arc<VirtualRuntimeStateOwners>,
    ) -> Option<Self> {
        if terminal >= num_terminals
            || physical_state_count == 0
            || root_state >= physical_state_count
            || state_owners.owner_index(root_state) != Some(runtime_index as usize)
        {
            return None;
        }
        let (mut arena, root) = ResidualArena::from_expr(expr)?;
        let liveness_oracle: BoundedCodeIntersectionOracle =
            bincode::deserialize(oracle_bytes).ok()?;
        let root_oracle_coordinate = liveness_oracle.root_coordinate();
        // Structural checks sufficient for compiled-artifact restoration. The
        // exact residual derivative remains authoritative for transitions and
        // will fail closed if a coordinate ceases to track it.
        if liveness_oracle.pattern.num_states() == 0
            || liveness_oracle.pattern.num_states() > MAX_BOUNDED_CODE_ORACLE_PATTERN_STATES
            || liveness_oracle.body.num_states() == 0
            || liveness_oracle.body_productive.len() != liveness_oracle.body.num_states()
            || liveness_oracle.completion_relations.len() != liveness_oracle.body.num_states()
            || liveness_oracle.min > liveness_oracle.max
            || liveness_oracle.prefix.is_empty()
            || liveness_oracle.suffix.is_empty()
        {
            return None;
        }
        let pattern_states = liveness_oracle.pattern.num_states();
        let relation_valid = |relation: &BoolRelation| {
            relation.rows.len() == pattern_states
                && relation.rows.iter().all(|row| row.iter().all(|state| state < pattern_states))
        };
        if liveness_oracle
            .completion_relations
            .iter()
            .flatten()
            .any(|relation| !relation_valid(relation))
            || liveness_oracle.exact_powers.iter().any(|relation| !relation_valid(relation))
            || liveness_oracle.prefix_sums.iter().any(|relation| !relation_valid(relation))
        {
            return None;
        }
        let root_live = arena.conservative_has_future(root);
        let mut state_by_residual = vec![u32::MAX; root as usize + 1];
        state_by_residual[root as usize] = root_state;
        let mut oracle_coordinates = vec![BoundedCodeOracleSlot::Unknown; arena.state_count()];
        let oracle_futures = vec![None; arena.state_count()];
        oracle_coordinates[root as usize] = BoundedCodeOracleSlot::Exact(root_oracle_coordinate);
        let mut state_by_residual_coordinate = FxHashMap::default();
        let mut coordinate_by_state = FxHashMap::default();
        state_by_residual_coordinate.insert((root, root_oracle_coordinate), root_state);
        coordinate_by_state.insert(root_state, root_oracle_coordinate);
        let mut accepting = BitSet::new(num_terminals as usize);
        accepting.set(terminal as usize);
        let live = accepting.clone();
        Some(Self {
            runtime_index, terminal, physical_state_count, root_state, root_has_future: root_live,
            preserve_oracle_coordinate: true, state_allocator, state_owners, accepting, live,
            dead: BitSet::new(num_terminals as usize),
            accepting_list: vec![terminal].into_boxed_slice(),
            store: Mutex::new(ResidualRuntimeStore {
                arena, root, state_by_residual, residual_by_state: FxHashMap::default(),
                state_by_residual_coordinate, coordinate_by_state,
                oracle_future_by_state: FxHashMap::default(),
                liveness_oracle: Some(liveness_oracle), oracle_coordinates, oracle_futures,
            }),
        })
    }

    pub(super) fn root_has_future(&self) -> bool {
        self.root_has_future
    }

    /// Exact nonempty-continuation query for a state owned by this runtime.
    /// Resource exhaustion is propagated to dynamic mask/commit instead of
    /// being collapsed into a dead transition.
    pub(super) fn exact_has_future(&self, state: u32) -> Result<Option<bool>, String> {
        let mut store = self.store.lock().unwrap();
        let Some(residual) = Self::residual_for_state(&store, self.root_state, state) else {
            return Ok(None);
        };
        if let Some(future) =
            self.certified_oracle_future_for_state(&mut store, state, residual)
        {
            return Ok(Some(future));
        }
        store.arena.has_future(residual).map(Some)
    }

    pub(super) fn has_bounded_code_liveness_oracle(&self) -> bool {
        self.store.lock().unwrap().liveness_oracle.is_some()
    }

    pub(super) fn finalizers(&self, state: u32) -> Option<&BitSet> {
        let (accepting, _) = self.observation(state)?;
        Some(if accepting { &self.accepting } else { &self.dead })
    }

    pub(super) fn finalizer_list(&self, state: u32) -> Option<&[TerminalID]> {
        let (accepting, _) = self.observation(state)?;
        Some(if accepting { self.accepting_list.as_ref() } else { &[] })
    }

    pub(super) fn futures(&self, state: u32) -> Option<&BitSet> {
        let (_, future) = self.observation(state)?;
        Some(if future { &self.live } else { &self.dead })
    }

    pub(super) fn transitions(&self, state: u32) -> Option<Vec<(u8, u32)>> {
        if !self.handles_state(state) {
            return None;
        }
        if state == self.root_state && !self.root_has_future {
            return Some(Vec::new());
        }
        let mut store = self.store.lock().unwrap();
        let residual = Self::residual_for_state(&store, self.root_state, state)?;
        let bytes = store.arena.first_bytes(residual)?;
        let mut out = Vec::new();
        for byte in bytes.iter() {
            if let Some(target) = self.step_residual_locked(&mut store, state, residual, byte) {
                out.push((byte, target));
            }
        }
        Some(out)
    }

    fn oracle_coordinate(&self, state: u32) -> Option<BoundedCodeOracleCoordinate> {
        let store = self.store.lock().unwrap();
        if self.preserve_oracle_coordinate {
            return store.coordinate_by_state.get(&state).copied();
        }
        let residual = Self::residual_for_state(&store, self.root_state, state)?;
        match store.oracle_coordinates.get(residual as usize).copied()? {
            BoundedCodeOracleSlot::Exact(coordinate) => Some(coordinate),
            BoundedCodeOracleSlot::Unknown | BoundedCodeOracleSlot::Ambiguous => None,
        }
    }

    pub(super) fn restore_compiled_finite_mask_projection(
        self: &Arc<Self>,
        component_state_count: u32,
        artifact: VirtualResidualMaskProjectionArtifact,
    ) -> Result<VirtualResidualMaskProjection, String> {
        if artifact.terminal != self.terminal {
            return Err(format!(
                "virtual residual projection terminal mismatch: artifact={} runtime={}",
                artifact.terminal, self.terminal,
            ));
        }
        if !self.preserve_oracle_coordinate {
            return Err("compiled virtual residual projection requires coordinate-preserving runtime".to_owned());
        }
        let store = self.store.lock().map_err(|_| "virtual residual runtime lock poisoned".to_owned())?;
        let oracle = store.liveness_oracle.as_ref().ok_or_else(|| "compiled virtual residual projection has no bounded-code oracle".to_owned())?;
        let mask_max = artifact.compiled_mask_max;
        let crossed_boundaries = artifact.compiled_crossed_boundaries;
        let desired_mask_max = oracle
            .min
            .checked_add(crossed_boundaries)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| "compiled virtual residual projection stencil overflow".to_owned())?;
        if crossed_boundaries == 0 || mask_max != oracle.max.min(desired_mask_max) {
            return Err("compiled virtual residual projection stencil is inconsistent".to_owned());
        }
        let expected_dense_states = oracle
            .finite_mask_dense_state_count(mask_max)
            .filter(|&states| states <= MAX_FINITE_MASK_DENSE_STATES)
            .ok_or_else(|| "compiled virtual residual projection dense coordinate is invalid".to_owned())?;
        if artifact.local_to_mask_state.len() != expected_dense_states {
            return Err(format!(
                "compiled virtual residual projection map has {} entries, expected {}",
                artifact.local_to_mask_state.len(), expected_dense_states,
            ));
        }
        if artifact.local_to_mask_state.iter().any(|&state| state != u32::MAX && state >= component_state_count) {
            return Err(format!(
                "compiled virtual residual projection references state outside component width {}",
                component_state_count,
            ));
        }
        Ok(VirtualResidualMaskProjection {
            runtime: Arc::clone(self),
            state_offset: artifact.state_offset,
            pattern_states: oracle.pattern.num_states(),
            body_states: oracle.body.num_states(),
            prefix_len: oracle.prefix.len(),
            suffix_len: oracle.suffix.len(),
            min: oracle.min,
            full_max: oracle.max,
            mask_max,
            crossed_boundaries,
            local_to_mask_state: Arc::from(artifact.local_to_mask_state.into_boxed_slice()),
        })
    }

    pub(super) fn restore_finite_mask_projection(
        self: &Arc<Self>,
        max_token_len: usize,
        component_state_count: u32,
        artifact: VirtualResidualMaskProjectionArtifact,
    ) -> Result<VirtualResidualMaskProjection, String> {
        if artifact.terminal != self.terminal {
            return Err(format!(
                "virtual residual projection terminal mismatch: artifact={} runtime={}",
                artifact.terminal, self.terminal,
            ));
        }
        if !self.preserve_oracle_coordinate {
            return Err(
                "virtual residual projection requires coordinate-preserving residual runtime"
                    .to_owned(),
            );
        }
        let store = self
            .store
            .lock()
            .map_err(|_| "virtual residual runtime lock poisoned".to_owned())?;
        let oracle = store
            .liveness_oracle
            .as_ref()
            .ok_or_else(|| "virtual residual projection has no bounded-code oracle".to_owned())?;
        let minimum_body_width = oracle
            .body
            .min_match_byte_len()
            .ok_or_else(|| "virtual residual projection body has no minimum byte width".to_owned())?
            .max(1);
        let crossed_boundaries = max_token_len
            .div_ceil(minimum_body_width)
            .saturating_add(1);
        if oracle.min > crossed_boundaries.saturating_add(1) {
            return Err("virtual residual projection lower bound exceeds finite stencil".to_owned());
        }
        let desired_mask_max = oracle
            .min
            .checked_add(crossed_boundaries)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| "virtual residual projection stencil overflow".to_owned())?;
        let mask_max = oracle.max.min(desired_mask_max);
        let expected_dense_states = oracle
            .finite_mask_dense_state_count(mask_max)
            .filter(|&states| states <= MAX_FINITE_MASK_DENSE_STATES)
            .ok_or_else(|| "virtual residual projection dense coordinate is invalid".to_owned())?;
        if artifact.local_to_mask_state.len() != expected_dense_states {
            return Err(format!(
                "virtual residual projection map has {} entries, expected {}",
                artifact.local_to_mask_state.len(), expected_dense_states,
            ));
        }
        if artifact.local_to_mask_state.iter().any(|&state| {
            state != u32::MAX && state >= component_state_count
        }) {
            return Err(format!(
                "virtual residual projection references state outside component width {}",
                component_state_count,
            ));
        }
        Ok(VirtualResidualMaskProjection {
            runtime: Arc::clone(self),
            state_offset: artifact.state_offset,
            pattern_states: oracle.pattern.num_states(),
            body_states: oracle.body.num_states(),
            prefix_len: oracle.prefix.len(),
            suffix_len: oracle.suffix.len(),
            min: oracle.min,
            full_max: oracle.max,
            mask_max,
            crossed_boundaries,
            local_to_mask_state: Arc::from(artifact.local_to_mask_state.into_boxed_slice()),
        })
    }

    pub(super) fn build_finite_mask_projection(
        self: &Arc<Self>,
        max_token_len: usize,
        state_offset: u32,
    ) -> Option<(DFA, u32, VirtualResidualMaskProjection)> {
        let minimum_body_width = {
            let store = self.store.lock().unwrap();
            store.liveness_oracle.as_ref()?.body.min_match_byte_len()?.max(1)
        };
        // A token that begins in the middle of one body copy can complete at
        // most ceil(token_bytes / minimum_body_width) copies.
        let crossed_boundaries = max_token_len
            .div_ceil(minimum_body_width)
            .saturating_add(1);
        self.build_finite_mask_projection_for_crossed_boundaries(crossed_boundaries, state_offset)
    }

    pub(super) fn build_finite_mask_projection_for_crossed_boundaries(
        self: &Arc<Self>,
        crossed_boundaries: usize,
        state_offset: u32,
    ) -> Option<(DFA, u32, VirtualResidualMaskProjection)> {
        let store = self.store.lock().unwrap();
        let oracle = store.liveness_oracle.as_ref()?;
        // Keep the first accepting layer plus a full upper-bound token stencil.
        // Large lower minima need their own lower-bound abstraction; decline
        // rather than making this first exact lane scale with minLength.
        if oracle.min > crossed_boundaries.saturating_add(1) {
            return None;
        }
        let desired_mask_max = oracle
            .min
            .checked_add(crossed_boundaries)?
            .checked_add(1)?;
        let mask_max = oracle.max.min(desired_mask_max);
        // Even when the declared upper bound already fits inside one model-token
        // stencil, keep using the finite oracle coordinate. The absence of a
        // truncating stencil does not imply that eagerly materializing the
        // original pattern × length product is cheap.
        let (dfa, root, local_to_mask_state) = oracle.finite_mask_dfa(mask_max)?;
        let projection = VirtualResidualMaskProjection {
            runtime: Arc::clone(self),
            state_offset,
            pattern_states: oracle.pattern.num_states(),
            body_states: oracle.body.num_states(),
            prefix_len: oracle.prefix.len(),
            suffix_len: oracle.suffix.len(),
            min: oracle.min,
            full_max: oracle.max,
            mask_max,
            crossed_boundaries,
            local_to_mask_state: Arc::from(local_to_mask_state.into_boxed_slice()),
        };
        Some((dfa, root, projection))
    }

    pub(super) fn interned_state_count(&self) -> usize {
        self.store.lock().unwrap().residual_by_state.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn bytes(value: &[u8]) -> Expr {
        Expr::U8Seq(value.to_vec())
    }

    fn accepts(arena: &mut ResidualArena, mut state: ResidualId, input: &[u8]) -> bool {
        for &byte in input {
            state = arena.step(state, byte).unwrap();
        }
        arena.is_nullable(state)
    }

    fn materialized_accepts(dfa: &DFA, input: &[u8]) -> bool {
        let mut state = 0;
        for &byte in input {
            let Some(target) = dfa.step(state, byte) else {
                return false;
            };
            state = target;
        }
        dfa.finalizers(state).contains(0)
    }

    fn random_small_expr(rng: &mut StdRng, depth: usize) -> Expr {
        let atom = |rng: &mut StdRng| match rng.gen_range(0..4) {
            0 => Expr::U8Seq(vec![b'a' + rng.gen_range(0..3)]),
            1 => Expr::U8Seq(
                (0..rng.gen_range(1..=3))
                    .map(|_| b'a' + rng.gen_range(0..3))
                    .collect(),
            ),
            2 => Expr::U8Class(U8Set::from_bytes(match rng.gen_range(0..3) {
                0 => b"ab",
                1 => b"bc",
                _ => b"abc",
            })),
            _ => Expr::Epsilon,
        };

        if depth == 0 {
            return atom(rng);
        }
        match rng.gen_range(0..9) {
            0..=2 => atom(rng),
            3 => Expr::Choice(vec![
                random_small_expr(rng, depth - 1),
                random_small_expr(rng, depth - 1),
            ]),
            4 => Expr::Seq(vec![
                random_small_expr(rng, depth - 1),
                random_small_expr(rng, depth - 1),
            ]),
            5 => Expr::Repeat {
                expr: Box::new(random_small_expr(rng, depth - 1)),
                min: rng.gen_range(0..=2),
                max: Some(rng.gen_range(2..=4)),
            },
            6 => Expr::Repeat {
                expr: Box::new(atom(rng)),
                min: rng.gen_range(0..=1),
                max: None,
            },
            7 => Expr::Exclude {
                expr: Box::new(random_small_expr(rng, depth - 1)),
                exclude: Box::new(random_small_expr(rng, depth - 1)),
            },
            _ => Expr::Intersect {
                expr: Box::new(random_small_expr(rng, depth - 1)),
                intersect: Box::new(random_small_expr(rng, depth - 1)),
            },
        }
    }

    fn all_words(alphabet: &[u8], max_len: usize) -> Vec<Vec<u8>> {
        let mut words = vec![Vec::new()];
        let mut frontier = vec![Vec::new()];
        for _ in 0..max_len {
            let mut next = Vec::new();
            for prefix in frontier {
                for &byte in alphabet {
                    let mut word = prefix.clone();
                    word.push(byte);
                    words.push(word.clone());
                    next.push(word);
                }
            }
            frontier = next;
        }
        words
    }

    #[test]
    fn giant_repeat_bound_stays_symbolic() {
        let expr = Expr::Repeat {
            expr: Box::new(bytes(b"ab")),
            min: 3,
            max: Some(1_000_000_000),
        };
        let (mut arena, root) = ResidualArena::from_expr(&expr).unwrap();
        let initial_states = arena.state_count();
        assert!(arena.has_future(root).unwrap());
        assert!(arena.state_count() <= initial_states + 2);

        let mut state = root;
        for _ in 0..3 {
            state = arena.step(state, b'a').unwrap();
            state = arena.step(state, b'b').unwrap();
        }
        assert!(arena.is_nullable(state));
        assert!(arena.has_future(state).unwrap());
        assert!(arena.state_count() < 32);
    }

    #[test]
    fn repeat_suffix_boundary_is_general_derivative_nondeterminism() {
        let expr = Expr::Seq(vec![
            Expr::Repeat {
                expr: Box::new(bytes(b"a")),
                min: 0,
                max: Some(1_000_000_000),
            },
            bytes(b"ab"),
        ]);
        let (mut arena, root) = ResidualArena::from_expr(&expr).unwrap();
        assert!(accepts(&mut arena, root, b"ab"));
        assert!(accepts(&mut arena, root, b"aab"));
        assert!(accepts(&mut arena, root, b"aaaaab"));
        assert!(!accepts(&mut arena, root, b"b"));
        assert!(arena.state_count() < 64);
    }

    #[test]
    fn nullable_repeat_body_does_not_walk_the_bound() {
        let body = Expr::Choice(vec![Expr::Epsilon, bytes(b"a")]);
        let expr = Expr::Repeat {
            expr: Box::new(body),
            min: 500_000_000,
            max: Some(1_000_000_000),
        };
        let (mut arena, root) = ResidualArena::from_expr(&expr).unwrap();
        assert!(arena.is_nullable(root));
        assert!(arena.has_future(root).unwrap());
        assert!(accepts(&mut arena, root, b"aaa"));
        assert!(arena.state_count() < 32);
    }

    #[test]
    fn boolean_residuals_derive_compositionally() {
        let left = Expr::Seq(vec![
            Expr::Repeat {
                expr: Box::new(bytes(b"a")),
                min: 0,
                max: Some(32),
            },
            bytes(b"b"),
        ]);
        let right = Expr::Choice(vec![bytes(b"b"), bytes(b"aab"), bytes(b"c")]);
        let expr = Expr::Intersect {
            expr: Box::new(left),
            intersect: Box::new(right),
        };
        let (mut arena, root) = ResidualArena::from_expr(&expr).unwrap();
        assert!(arena.has_future(root).unwrap());
        assert!(accepts(&mut arena, root, b"b"));
        assert!(accepts(&mut arena, root, b"aab"));
        assert!(!accepts(&mut arena, root, b"ab"));
        assert!(!accepts(&mut arena, root, b"c"));
    }

    #[test]
    fn embedded_dfa_epsilon_closure_is_a_compositional_residual() {
        let mut dfa = DFA::new(4);
        dfa.ensure_group_capacity(1);
        dfa.add_epsilon_transition(0, 1);
        dfa.add_transition(1, b'a', 2);
        dfa.add_epsilon_transition(2, 3);
        let mut accepting = BitSet::new(1);
        accepting.set(0);
        dfa.overwrite_state_metadata(3, accepting, BitSet::new(1));

        let expr = Expr::Dfa(Arc::new(dfa));
        let (mut arena, root) = ResidualArena::from_expr(&expr)
            .expect("epsilon-bearing embedded DFA must stay in the general residual algebra");
        assert!(!arena.is_nullable(root));
        assert!(arena.has_future(root).unwrap());
        assert!(accepts(&mut arena, root, b"a"));
        assert!(!accepts(&mut arena, root, b""));
        assert!(!accepts(&mut arena, root, b"aa"));
    }

    #[test]
    fn boolean_liveness_ceiling_is_error_not_dead() {
        let expr = Expr::Intersect {
            expr: Box::new(Expr::Repeat {
                expr: Box::new(bytes(b"a")),
                min: 100,
                max: Some(100),
            }),
            intersect: Box::new(Expr::Repeat {
                expr: Box::new(bytes(b"aa")),
                min: 50,
                max: Some(50),
            }),
        };
        let (mut arena, root) = ResidualArena::from_expr(&expr).unwrap();
        let error = arena
            .has_future_with_budget(root, 8, 64)
            .expect_err("a deliberately tiny resource ceiling must not become a false dead result");
        assert!(error.contains("budget"), "unexpected liveness error: {error}");
        assert!(arena.has_future_with_budget(root, 256, 512).unwrap());
    }

    #[test]
    fn boolean_liveness_does_not_retain_dense_transition_rows() {
        let expr = Expr::Intersect {
            expr: Box::new(Expr::Repeat {
                expr: Box::new(bytes(b"a")),
                min: 2,
                max: Some(4),
            }),
            intersect: Box::new(bytes(b"aa")),
        };
        let (mut arena, root) = ResidualArena::from_expr(&expr).unwrap();
        assert!(arena.transitions.iter().all(Option::is_none));
        assert!(arena.has_future(root).unwrap());
        assert!(
            arena.transitions.iter().all(Option::is_none),
            "exact liveness must keep derivative caching query-local rather than retaining dense rows",
        );

        // Normal runtime stepping deliberately keeps the dense hot-path cache.
        assert_ne!(arena.step(root, b'a').unwrap(), arena.empty);
        assert!(arena.transitions[root as usize].is_some());
    }

    #[test]
    fn boolean_liveness_budget_charges_recursive_sparse_derivatives() {
        let expr = Expr::Intersect {
            expr: Box::new(Expr::Choice(vec![bytes(b"a"), bytes(b"b"), bytes(b"c")])),
            intersect: Box::new(Expr::Choice(vec![bytes(b"a"), bytes(b"b")])),
        };
        let (mut arena, root) = ResidualArena::from_expr(&expr).unwrap();
        let error = arena
            .has_future_with_budget(root, 16, 1)
            .expect_err("recursive sparse derivative work must consume the transition budget");
        assert!(error.contains("transition budget"), "unexpected error: {error}");
        assert!(arena.has_future_with_budget(root, 16, 32).unwrap());
    }

    #[test]
    fn giant_repeat_liveness_solves_embedded_body_once() {
        let mut body = DFA::new(3);
        body.ensure_group_capacity(1);
        body.add_transition(0, b'a', 1);
        body.add_transition(1, b'b', 2);
        let mut accepting = BitSet::new(1);
        accepting.set(0);
        body.overwrite_state_metadata(2, accepting, BitSet::new(1));
        // Deliberately leave derived future metadata stale: the residual
        // engine must reason from the DFA graph, not from precomputed labels.

        let expr = Expr::Repeat {
            expr: Box::new(Expr::Dfa(Arc::new(body))),
            min: 1_000_000_000,
            max: Some(1_000_000_000),
        };
        let (mut arena, root) = ResidualArena::from_expr(&expr).unwrap();
        assert!(
            arena.has_future_with_budget(root, 8, 16).unwrap(),
            "repeat liveness must solve the body language once rather than walk the billion-copy counter",
        );
    }

    #[test]
    fn sigma_star_identity_eliminates_trivial_boolean_counter_search() {
        let mut body = DFA::new(2);
        body.ensure_group_capacity(1);
        body.add_transition(0, b'a', 1);
        let mut accepting = BitSet::new(1);
        accepting.set(0);
        body.overwrite_state_metadata(1, accepting, BitSet::new(1));

        let counted = Expr::Repeat {
            expr: Box::new(Expr::Dfa(Arc::new(body))),
            min: 1_000_000_000,
            max: Some(1_000_000_000),
        };
        let sigma_star = Expr::Repeat {
            expr: Box::new(Expr::U8Class(U8Set::all())),
            min: 0,
            max: None,
        };
        let expr = Expr::Intersect {
            expr: Box::new(counted),
            intersect: Box::new(sigma_star.clone()),
        };
        let (mut arena, root) = ResidualArena::from_expr(&expr).unwrap();
        assert!(
            arena.has_future_with_budget(root, 4, 4).unwrap(),
            "intersection with sigma-star must simplify before walking the billion-copy counter",
        );

        let choice = Expr::Choice(vec![bytes(b"literal"), sigma_star.clone()]);
        let (mut arena, root) = ResidualArena::from_expr(&choice).unwrap();
        assert_eq!(root, arena.sigma_star);
        assert!(accepts(&mut arena, root, b"anything\0goes"));

        let excluded = Expr::Exclude {
            expr: Box::new(bytes(b"literal")),
            exclude: Box::new(sigma_star),
        };
        let (arena, root) = ResidualArena::from_expr(&excluded).unwrap();
        assert!(arena.is_empty(root));
    }

    #[test]
    fn seeded_residual_algebra_matches_materialized_dfa() {
        let mut rng = StdRng::seed_from_u64(0x5E51_DA1A_2026_0826);
        let words = all_words(b"abc", 4);
        for case in 0..256 {
            let expr = random_small_expr(&mut rng, 3);
            let dfa = super::super::compile::compile_terminal_expr_dfa(&expr);
            let (mut arena, root) = ResidualArena::from_expr(&expr)
                .unwrap_or_else(|| panic!("residual compilation failed for case {case}: {expr:?}"));

            for word in &words {
                assert_eq!(
                    accepts(&mut arena, root, word),
                    materialized_accepts(&dfa, word),
                    "residual/materialized language mismatch in case {case}, expr={expr:?}, word={word:?}",
                );
            }

            assert_eq!(
                arena.has_future(root).unwrap(),
                dfa.possible_future_group_ids(0).contains(0),
                "residual/materialized root liveness mismatch in case {case}, expr={expr:?}",
            );
        }
    }

    #[test]
    fn sequence_liveness_skips_hard_nullable_siblings() {
        let hard_nullable = Expr::Repeat {
            expr: Box::new(Expr::Intersect {
                expr: Box::new(Expr::Repeat {
                    expr: Box::new(bytes(b"a")),
                    min: 100,
                    max: Some(100),
                }),
                intersect: Box::new(Expr::Repeat {
                    expr: Box::new(bytes(b"aa")),
                    min: 50,
                    max: Some(50),
                }),
            }),
            min: 0,
            max: Some(1_000_000_000),
        };
        let expr = Expr::Seq(vec![hard_nullable, bytes(b"z")]);
        let (mut arena, root) = ResidualArena::from_expr(&expr).unwrap();
        assert!(
            arena.has_future_with_budget(root, 0, 0).unwrap(),
            "the nonnullable literal proves a positive sequence word without solving the nullable sibling",
        );
    }

    #[test]
    fn runtime_future_bit_is_conservative_until_exact_boundary_check() {
        // The whole intersection accepts "c", but after consuming 'a' its
        // residual is exactly b ∩ c: syntactically nonempty, semantically dead.
        let expr = Expr::Intersect {
            expr: Box::new(Expr::Choice(vec![bytes(b"ab"), bytes(b"c")])),
            intersect: Box::new(Expr::Choice(vec![bytes(b"ac"), bytes(b"c")])),
        };
        let allocator = Arc::new(VirtualStateAllocator::new(2).unwrap());
        let owners = Arc::new(VirtualRuntimeStateOwners::new(2, &[1]).unwrap());
        let runtime =
            VirtualResidualRuntime::new(&expr, 0, 0, 1, 2, 1, allocator, owners).unwrap();
        let dead_prefix = runtime
            .step(1, b'a')
            .expect("syntactic derivative is retained until the exact boundary check");
        assert!(runtime.futures(dead_prefix).unwrap().contains(0));
        assert_eq!(runtime.exact_has_future(dead_prefix).unwrap(), Some(false));

        let accepting = runtime.step(1, b'c').unwrap();
        assert!(runtime.finalizers(accepting).unwrap().contains(0));
        assert_eq!(runtime.exact_has_future(accepting).unwrap(), Some(false));
    }

    #[test]
    fn empty_boolean_root_is_conservative_until_exact_boundary_check() {
        let a_star = || Expr::Repeat {
            expr: Box::new(bytes(b"a")),
            min: 0,
            max: None,
        };
        let expr = Expr::Intersect {
            expr: Box::new(Expr::Seq(vec![a_star(), bytes(b"b")])),
            intersect: Box::new(Expr::Seq(vec![a_star(), bytes(b"c")])),
        };
        let allocator = Arc::new(VirtualStateAllocator::new(2).unwrap());
        let owners = Arc::new(VirtualRuntimeStateOwners::new(2, &[1]).unwrap());
        let runtime =
            VirtualResidualRuntime::new(&expr, 0, 0, 1, 2, 1, allocator, owners).unwrap();
        assert!(runtime.root_has_future());
        assert!(runtime.futures(1).unwrap().contains(0));
        assert_eq!(runtime.exact_has_future(1).unwrap(), Some(false));
        assert_eq!(
            runtime.step(1, b'a'),
            Some(1),
            "a syntactically continuing dead Boolean residual may remain as a conservative proxy until exact boundary pruning",
        );
    }

    #[test]
    fn runtime_construction_does_not_force_boolean_liveness_search() {
        let expr = Expr::Intersect {
            expr: Box::new(Expr::Repeat {
                expr: Box::new(bytes(b"a")),
                min: 100,
                max: Some(100),
            }),
            intersect: Box::new(Expr::Repeat {
                expr: Box::new(bytes(b"aa")),
                min: 50,
                max: Some(50),
            }),
        };
        let allocator = Arc::new(VirtualStateAllocator::new(2).unwrap());
        let owners = Arc::new(VirtualRuntimeStateOwners::new(2, &[1]).unwrap());
        let runtime =
            VirtualResidualRuntime::new(&expr, 0, 0, 1, 2, 1, allocator, owners).unwrap();
        assert!(runtime.root_has_future());

        let store = runtime.store.lock().unwrap();
        assert_eq!(
            store.arena.nonempty_cache[store.root as usize],
            None,
            "constructing the runtime must not eagerly solve a hard Boolean liveness problem",
        );
    }

    fn bounded_code_body() -> Expr {
        Expr::Choice(vec![bytes(b"a"), bytes(b"bc")])
    }

    fn bounded_code_envelope_expr(min: usize, max: usize) -> Expr {
        Expr::Seq(vec![
            bytes(b"<"),
            Expr::Repeat {
                expr: Box::new(bounded_code_body()),
                min,
                max: Some(max),
            },
            bytes(b">"),
        ])
    }

    fn bounded_code_envelope_with_body(body: Expr, min: usize, max: usize) -> Expr {
        Expr::Seq(vec![
            bytes(b"<"),
            Expr::Repeat {
                expr: Box::new(body),
                min,
                max: Some(max),
            },
            bytes(b">"),
        ])
    }

    fn exact_code_count_pattern(count: usize) -> Expr {
        let mut parts = Vec::with_capacity(count + 2);
        parts.push(bytes(b"<"));
        parts.extend((0..count).map(|_| bounded_code_body()));
        parts.push(bytes(b">"));
        Expr::Seq(parts)
    }

    #[test]
    fn bounded_code_intersection_oracle_respects_gapped_copy_counts() {
        // The pattern admits exactly two or four code words.  An envelope of
        // exactly three words is therefore dead even though each operand is
        // individually live.  This is the counterexample that rules out a
        // simple min/max-distance liveness approximation.
        let pattern = Expr::Choice(vec![exact_code_count_pattern(2), exact_code_count_pattern(4)]);
        let dead = Expr::Intersect {
            expr: Box::new(pattern.clone()),
            intersect: Box::new(bounded_code_envelope_expr(3, 3)),
        };
        let mut dead_oracle = BoundedCodeIntersectionOracle::from_expr(&dead)
            .expect("prefix-code bounded intersection should certify");
        assert!(!dead_oracle.has_future(dead_oracle.root_coordinate()));

        let live = Expr::Intersect {
            expr: Box::new(pattern),
            intersect: Box::new(bounded_code_envelope_expr(3, 4)),
        };
        let mut live_oracle = BoundedCodeIntersectionOracle::from_expr(&live)
            .expect("prefix-code bounded intersection should certify");
        assert!(live_oracle.has_future(live_oracle.root_coordinate()));
    }

    #[test]
    fn bounded_code_oracle_coalesces_identical_envelope_intervals_exactly() {
        let pattern = exact_code_count_pattern(3);
        let expr = Expr::Intersect {
            expr: Box::new(Expr::Intersect {
                expr: Box::new(pattern),
                intersect: Box::new(bounded_code_envelope_expr(1, 4)),
            }),
            intersect: Box::new(bounded_code_envelope_expr(3, 6)),
        };
        let mut oracle = BoundedCodeIntersectionOracle::from_expr(&expr)
            .expect("identical bounded-code envelopes should coalesce");
        assert_eq!((oracle.min, oracle.max), (3, 4));
        assert!(oracle.has_future(oracle.root_coordinate()));

        let materialized = compile_terminal_expr_dfa(&expr);
        assert!(
            materialized
                .possible_future_group_ids(0)
                .contains(0),
            "materialized intersection must agree that the root has a future",
        );
    }

    #[test]
    fn bounded_code_oracle_coalesces_disjoint_identical_envelopes_to_dead() {
        let pattern = Expr::Choice(vec![exact_code_count_pattern(2), exact_code_count_pattern(4)]);
        let expr = Expr::Intersect {
            expr: Box::new(Expr::Intersect {
                expr: Box::new(pattern),
                intersect: Box::new(bounded_code_envelope_expr(1, 2)),
            }),
            intersect: Box::new(bounded_code_envelope_expr(4, 5)),
        };
        let mut oracle = BoundedCodeIntersectionOracle::from_expr(&expr)
            .expect("disjoint identical envelopes should still admit an exact dead certificate");
        assert_eq!((oracle.min, oracle.max), (4, 2));
        assert!(!oracle.has_future(oracle.root_coordinate()));

        let materialized = compile_terminal_expr_dfa(&expr);
        assert!(
            !materialized
                .possible_future_group_ids(0)
                .contains(0),
            "materialized disjoint intersection must also be dead",
        );
    }

    #[test]
    fn bounded_code_oracle_rejects_ambiguous_code_boundaries() {
        let pattern = Expr::Seq(vec![
            bytes(b"<"),
            Expr::Repeat {
                expr: Box::new(Expr::U8Class(U8Set::all())),
                min: 0,
                max: None,
            },
            bytes(b">"),
        ]);

        // "a" is a prefix of "ab", so greedily treating the first accepting
        // body state as one completed code word would not be exact.
        let non_prefix_free = Expr::Intersect {
            expr: Box::new(pattern.clone()),
            intersect: Box::new(bounded_code_envelope_with_body(
                Expr::Choice(vec![bytes(b"a"), bytes(b"ab")]),
                0,
                4,
            )),
        };
        assert!(BoundedCodeIntersectionOracle::from_expr(&non_prefix_free).is_none());

        // At a code boundary, '>' could either begin another productive body
        // word or begin the fixed suffix. The sidecar deliberately refuses
        // such an envelope rather than choosing one interpretation.
        let suffix_ambiguous = Expr::Intersect {
            expr: Box::new(pattern),
            intersect: Box::new(bounded_code_envelope_with_body(
                Expr::Choice(vec![bytes(b"a"), bytes(b">x")]),
                0,
                4,
            )),
        };
        assert!(BoundedCodeIntersectionOracle::from_expr(&suffix_ambiguous).is_none());
    }

    #[test]
    fn bounded_code_oracle_does_not_materialize_nested_giant_repeats() {
        let pattern = Expr::Seq(vec![
            bytes(b"<"),
            Expr::Repeat {
                expr: Box::new(Expr::U8Class(U8Set::all())),
                min: 0,
                max: None,
            },
            bytes(b">"),
        ]);
        let giant_body = Expr::Repeat {
            expr: Box::new(bytes(b"a")),
            min: 4_096,
            max: Some(4_096),
        };
        let body_giant = Expr::Intersect {
            expr: Box::new(pattern),
            intersect: Box::new(bounded_code_envelope_with_body(giant_body, 0, 5_000)),
        };
        assert!(BoundedCodeIntersectionOracle::from_expr(&body_giant).is_none());

        // A giant bounded repeat can also hide inside the independently
        // compiled pattern operand. Even when simplification could make a
        // particular example cheap (epsilon repeated many times), the oracle
        // must not rely on eagerly discovering that after materialization.
        let giant_pattern = Expr::Choice(vec![
            Expr::Seq(vec![
                bytes(b"<"),
                Expr::Repeat {
                    expr: Box::new(Expr::Epsilon),
                    min: 0,
                    max: Some(4_096),
                },
                bytes(b">"),
            ]),
            bytes(b"x"),
        ]);
        let pattern_giant = Expr::Intersect {
            expr: Box::new(giant_pattern),
            intersect: Box::new(bounded_code_envelope_expr(0, 5_000)),
        };
        assert!(BoundedCodeIntersectionOracle::from_expr(&pattern_giant).is_none());
    }

    #[test]
    fn bounded_code_oracle_matches_materialized_future_at_every_small_prefix() {
        let pattern = Expr::Choice(vec![
            exact_code_count_pattern(1),
            exact_code_count_pattern(3),
            exact_code_count_pattern(4),
        ]);
        let expr = Expr::Intersect {
            expr: Box::new(pattern),
            intersect: Box::new(bounded_code_envelope_expr(1, 4)),
        };
        let materialized = compile_terminal_expr_dfa(&expr);
        let allocator = Arc::new(VirtualStateAllocator::new(2).unwrap());
        let owners = Arc::new(VirtualRuntimeStateOwners::new(2, &[1]).unwrap());
        let runtime =
            VirtualResidualRuntime::new(&expr, 0, 0, 1, 2, 1, allocator, owners).unwrap();
        assert!(runtime.has_bounded_code_liveness_oracle());

        let alphabet = [b'<', b'>', b'a', b'b', b'c'];
        let mut queue = VecDeque::from([(Vec::<u8>::new(), 1u32, 0u32)]);
        let mut seen = FxHashSet::<(u32, u32)>::default();
        seen.insert((1, 0));
        while let Some((prefix, residual_state, materialized_state)) = queue.pop_front() {
            assert_eq!(
                runtime.exact_has_future(residual_state).unwrap(),
                Some(
                    materialized
                        .possible_future_group_ids(materialized_state)
                        .contains(0)
                ),
                "future mismatch after prefix {:?}",
                String::from_utf8_lossy(&prefix),
            );
            assert_eq!(
                runtime
                    .futures(residual_state)
                    .expect("reached residual state must have metadata")
                    .contains(0),
                materialized
                    .possible_future_group_ids(materialized_state)
                    .contains(0),
                "ordinary future metadata mismatch after prefix {:?}",
                String::from_utf8_lossy(&prefix),
            );
            if prefix.len() >= 12 {
                continue;
            }
            for &byte in &alphabet {
                let Some(materialized_target) = materialized.step(materialized_state, byte) else {
                    continue;
                };
                let Some(residual_target) = runtime.step(residual_state, byte) else {
                    continue;
                };
                if seen.insert((residual_target, materialized_target)) {
                    let mut next_prefix = prefix.clone();
                    next_prefix.push(byte);
                    queue.push_back((next_prefix, residual_target, materialized_target));
                }
            }
        }
    }

    #[test]
    fn bounded_code_oracle_keeps_billion_bound_logarithmic() {
        let expr = Expr::Intersect {
            expr: Box::new(exact_code_count_pattern(2)),
            intersect: Box::new(bounded_code_envelope_expr(0, 1_000_000_000)),
        };
        let mut oracle = BoundedCodeIntersectionOracle::from_expr(&expr)
            .expect("billion-copy prefix-code envelope should certify");
        assert!(oracle.has_future(oracle.root_coordinate()));
        assert!(
            oracle.exact_powers.len() <= 31,
            "doubling table must scale with log2(max), got {} layers",
            oracle.exact_powers.len(),
        );
    }

    #[test]
    fn bounded_code_oracle_sizes_doubling_for_inclusive_power_of_two_ranges() {
        for max in [0usize, 1, 3, 7, 15] {
            let expr = Expr::Intersect {
                expr: Box::new(exact_code_count_pattern(max)),
                intersect: Box::new(bounded_code_envelope_expr(0, max)),
            };
            let mut oracle = BoundedCodeIntersectionOracle::from_expr(&expr)
                .unwrap_or_else(|| panic!("bounded-code oracle should certify max={max}"));
            assert!(
                oracle.has_future(oracle.root_coordinate()),
                "exactly {max} code words must be reachable inside 0..={max}",
            );
        }
    }

    #[test]
    fn bounded_code_oracle_ambiguity_propagates_to_existing_successors() {
        let expr = Expr::Intersect {
            expr: Box::new(exact_code_count_pattern(1)),
            intersect: Box::new(bounded_code_envelope_expr(0, 4)),
        };
        let allocator = Arc::new(VirtualStateAllocator::new(2).unwrap());
        let owners = Arc::new(VirtualRuntimeStateOwners::new(2, &[1]).unwrap());
        let runtime =
            VirtualResidualRuntime::new(&expr, 0, 0, 1, 2, 1, allocator, owners).unwrap();

        let body_boundary = runtime.step(1, b'<').unwrap();
        let after_one = runtime.step(body_boundary, b'a').unwrap();
        assert!(runtime.futures(after_one).unwrap().contains(0));

        {
            let mut store = runtime.store.lock().unwrap();
            let source = VirtualResidualRuntime::residual_for_state(
                &store,
                runtime.root_state,
                body_boundary,
            )
            .unwrap() as usize;
            let target = VirtualResidualRuntime::residual_for_state(
                &store,
                runtime.root_state,
                after_one,
            )
            .unwrap() as usize;
            assert!(matches!(
                store.oracle_coordinates[target],
                BoundedCodeOracleSlot::Exact(_)
            ));
            assert_eq!(store.oracle_futures[target], Some(true));
            store.oracle_coordinates[source] = BoundedCodeOracleSlot::Ambiguous;
        }

        assert_eq!(runtime.step(body_boundary, b'a'), Some(after_one));
        let store = runtime.store.lock().unwrap();
        let target = VirtualResidualRuntime::residual_for_state(
            &store,
            runtime.root_state,
            after_one,
        )
        .unwrap() as usize;
        assert_eq!(
            store.oracle_coordinates[target],
            BoundedCodeOracleSlot::Ambiguous
        );
        assert_eq!(store.oracle_futures[target], None);
    }

    #[test]
    fn bounded_code_runtime_liveness_does_not_fall_back_to_boolean_search() {
        let expr = Expr::Intersect {
            expr: Box::new(Expr::Choice(vec![
                exact_code_count_pattern(2),
                exact_code_count_pattern(4),
            ])),
            intersect: Box::new(bounded_code_envelope_expr(3, 3)),
        };
        let allocator = Arc::new(VirtualStateAllocator::new(2).unwrap());
        let owners = Arc::new(VirtualRuntimeStateOwners::new(2, &[1]).unwrap());
        let runtime =
            VirtualResidualRuntime::new(&expr, 0, 0, 1, 2, 1, allocator, owners).unwrap();
        assert!(runtime.has_bounded_code_liveness_oracle());
        assert_eq!(runtime.exact_has_future(1).unwrap(), Some(false));
        assert!(
            runtime.futures(1).unwrap().is_empty(),
            "certified exact liveness must be visible through ordinary tokenizer future metadata",
        );
        let store = runtime.store.lock().unwrap();
        assert_eq!(
            store.arena.nonempty_cache[store.root as usize],
            None,
            "certified bounded-code liveness must not invoke generic Boolean reachability",
        );
    }
}
