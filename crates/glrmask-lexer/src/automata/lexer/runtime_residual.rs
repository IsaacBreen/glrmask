//! General lazy regex residuals for dynamic tokenization.
//!
//! The key invariant is that bounded repetition remains a symbolic
//! `(body, min, max)` node. Byte derivatives decrement those integers only
//! when a body copy is actually consumed; construction never allocates in
//! proportion to `max`.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use rustc_hash::{FxHashMap, FxHashSet};

use super::ast::Expr;
use super::compile::compile_terminal_expr_dfa;
use super::dfa::DFA;
use super::runtime_repeat_product::{VirtualRuntimeStateOwners, VirtualStateAllocator};
use crate::ds::bitset::BitSet;
use crate::ds::u8set::U8Set;
use crate::grammar::flat::TerminalID;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BoundedCodeEnvelopeState {
    Prefix { next: usize },
    Body { completed: usize, body_state: u32 },
    Suffix { next: usize },
    Done,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

#[derive(Debug, Clone)]
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

#[derive(Debug)]
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

impl BoundedCodeIntersectionOracle {
    fn from_expr(expr: &Expr) -> Option<Self> {
        let mut operands = Vec::new();
        flatten_intersection_operands(expr, &mut operands);
        if operands.len() < 2 {
            return None;
        }

        let mut envelope = None;
        let mut pattern_operands = Vec::new();
        for operand in operands {
            if envelope.is_none()
                && let Some(candidate) = bounded_code_envelope(operand)
            {
                envelope = Some(candidate);
            } else {
                pattern_operands.push(operand.clone());
            }
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
        let pattern = Arc::new(compile_terminal_expr_dfa(&pattern_expr));
        let body = Arc::new(compile_terminal_expr_dfa(&body_expr));
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
        let bits = usize::BITS as usize - max.leading_zeros() as usize;
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
    liveness_oracle: Option<BoundedCodeIntersectionOracle>,
    oracle_coordinates: Vec<BoundedCodeOracleSlot>,
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
        // Root metadata follows the same contract as every other residual:
        // cheap structural proofs are exact, while hard Boolean liveness is a
        // conservative `true`. Dynamic mask/commit resolve that uncertainty
        // through `exact_has_future` at their fallible residual boundaries.
        // Do not make construction solve a potentially expensive emptiness
        // problem merely to populate an infallible tokenizer metadata bit.
        // Keep the infallible tokenizer metadata contract unchanged even when
        // this runtime has an exact liveness sidecar.  Artifacts serialize the
        // conservative root future bit, while dynamic mask/commit call
        // `exact_has_future` at their fallible boundary.  In particular, do
        // not turn installing a new proof oracle into an artifact-version
        // change.
        let root_live = arena.conservative_has_future(root);
        let mut state_by_residual = vec![u32::MAX; root as usize + 1];
        state_by_residual[root as usize] = root_state;
        let mut oracle_coordinates = vec![BoundedCodeOracleSlot::Unknown; arena.state_count()];
        if let Some(coordinate) = root_oracle_coordinate {
            oracle_coordinates[root as usize] = BoundedCodeOracleSlot::Exact(coordinate);
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
                liveness_oracle,
                oracle_coordinates,
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

    fn intern_locked(&self, store: &mut ResidualRuntimeStore, residual: ResidualId) -> Option<u32> {
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
        residual: ResidualId,
        byte: u8,
    ) -> Option<u32> {
        let source_coordinate = store
            .oracle_coordinates
            .get(residual as usize)
            .copied()
            .unwrap_or(BoundedCodeOracleSlot::Unknown);
        let target = store.arena.step(residual, byte)?;
        if store.arena.is_empty(target) {
            return None;
        }
        if store.oracle_coordinates.len() < store.arena.state_count() {
            store.oracle_coordinates.resize(
                store.arena.state_count(),
                BoundedCodeOracleSlot::Unknown,
            );
        }
        if let BoundedCodeOracleSlot::Exact(source_coordinate) = source_coordinate
            && let Some(oracle) = store.liveness_oracle.as_ref()
        {
            let target_slot = if let Some(target_coordinate) =
                oracle.step_coordinate(source_coordinate, byte)
            {
                BoundedCodeOracleSlot::Exact(target_coordinate)
            } else {
                // A structurally non-empty residual can still denote the empty
                // language.  Do not trust that mismatch as a dead proof here;
                // merely stop using the sidecar for this residual and let the
                // exact general solver decide it.
                BoundedCodeOracleSlot::Ambiguous
            };
            let slot = &mut store.oracle_coordinates[target as usize];
            *slot = match (*slot, target_slot) {
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
        }
        self.intern_locked(store, target)
    }

    pub(super) fn step(&self, state: u32, byte: u8) -> Option<u32> {
        if state == self.root_state && !self.root_has_future {
            return None;
        }
        let mut store = self.store.lock().unwrap();
        let residual = Self::residual_for_state(&store, self.root_state, state)?;
        self.step_residual_locked(&mut store, residual, byte)
    }

    fn observation(&self, state: u32) -> Option<(bool, bool)> {
        let mut store = self.store.lock().unwrap();
        let residual = Self::residual_for_state(&store, self.root_state, state)?;
        // Match the existing virtual-runtime convention: the physical proxy
        // root is the drained zero-byte configuration and must not emit a
        // terminal match before any input is consumed.
        let accepting = state != self.root_state && store.arena.is_nullable(residual);
        let future = if state == self.root_state {
            self.root_has_future
        } else {
            store.arena.conservative_has_future(residual)
        };
        Some((accepting, future))
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
        let coordinate = store
            .oracle_coordinates
            .get(residual as usize)
            .copied()
            .unwrap_or(BoundedCodeOracleSlot::Unknown);
        if let BoundedCodeOracleSlot::Exact(coordinate) = coordinate
            && let Some(oracle) = store.liveness_oracle.as_mut()
        {
            return Ok(Some(oracle.has_future(coordinate)));
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
            if let Some(target) = self.step_residual_locked(&mut store, residual, byte) {
                out.push((byte, target));
            }
        }
        Some(out)
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
        let store = runtime.store.lock().unwrap();
        assert_eq!(
            store.arena.nonempty_cache[store.root as usize],
            None,
            "certified bounded-code liveness must not invoke generic Boolean reachability",
        );
    }
}
