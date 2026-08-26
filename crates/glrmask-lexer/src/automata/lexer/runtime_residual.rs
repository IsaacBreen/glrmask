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
}

const TRANSITION_UNKNOWN: u32 = u32::MAX;
const DEFAULT_LIVENESS_STATE_BUDGET: usize = 262_144;
const DEFAULT_LIVENESS_TRANSITION_BUDGET: usize = 4_194_304;

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
        };
        arena.empty = arena.intern_raw(ResidualNode::Empty, false)?;
        arena.epsilon = arena.intern_raw(ResidualNode::Epsilon, true)?;
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

    fn derive_uncached(&mut self, id: ResidualId, byte: u8) -> Option<ResidualId> {
        match self.nodes[id as usize].clone() {
            ResidualNode::Empty | ResidualNode::Epsilon => Some(self.empty),
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

    /// Exact existence of a positive-length word accepted from one embedded
    /// DFA residual. Do not trust `possible_future_group_ids` here: `Expr::Dfa`
    /// can be constructed directly, and its graph can be valid even when the
    /// derived future metadata has not been recomputed by the caller.
    fn dfa_has_nonempty_word(dfa: &DFA, start: &[u32]) -> bool {
        let mut seen = FxHashSet::<u32>::default();
        let mut queue = VecDeque::<u32>::new();

        let enqueue_after_byte =
            |target: u32, seen: &mut FxHashSet<u32>, queue: &mut VecDeque<u32>| -> bool {
                for state in dfa.epsilon_closure(&[target]) {
                    if !dfa.finalizers(state).is_empty() {
                        return true;
                    }
                    if seen.insert(state) {
                        queue.push_back(state);
                    }
                }
                false
            };

        seen.extend(start.iter().copied());
        for &state in start {
            for (_, &target) in dfa.states()[state as usize].transitions.iter() {
                if enqueue_after_byte(target, &mut seen, &mut queue) {
                    return true;
                }
            }
        }

        while let Some(state) = queue.pop_front() {
            for (_, &target) in dfa.states()[state as usize].transitions.iter() {
                if enqueue_after_byte(target, &mut seen, &mut queue) {
                    return true;
                }
            }
        }
        false
    }

    fn has_nonempty_fast(&mut self, id: ResidualId) -> Option<bool> {
        match self.nodes[id as usize].clone() {
            ResidualNode::Empty | ResidualNode::Epsilon => Some(false),
            ResidualNode::Literal { .. } | ResidualNode::Class(_) => Some(true),
            ResidualNode::Dfa { dfa, states } => {
                Some(Self::dfa_has_nonempty_word(dfa.as_ref(), states.as_ref()))
            }
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
        if let Some(value) = self.nonempty_cache[id as usize] {
            return Ok(value);
        }
        if let Some(value) = self.has_nonempty_fast(id) {
            self.nonempty_cache[id as usize] = Some(value);
            return Ok(value);
        }

        let mut seen = FxHashMap::<ResidualId, ()>::default();
        let mut queue = VecDeque::from([id]);
        seen.insert(id, ());
        let mut transitions = 0usize;
        while let Some(state) = queue.pop_front() {
            let first_bytes = self
                .first_bytes(state)
                .ok_or_else(|| "dynamic residual FIRST-set construction overflow".to_owned())?;
            for byte in first_bytes.iter() {
                transitions = transitions
                    .checked_add(1)
                    .ok_or_else(|| "dynamic residual liveness transition count overflow".to_owned())?;
                if transitions > transition_budget {
                    return Err(format!(
                        "dynamic residual liveness exceeded transition budget ({transition_budget})"
                    ));
                }
                let target = self
                    .step(state, byte)
                    .ok_or_else(|| "dynamic residual state-id overflow".to_owned())?;
                if target == self.empty {
                    continue;
                }
                if self.is_nullable(target) {
                    self.nonempty_cache[id as usize] = Some(true);
                    return Ok(true);
                }
                if seen.insert(target, ()).is_none() {
                    if seen.len() > state_budget {
                        return Err(format!(
                            "dynamic residual liveness exceeded state budget ({state_budget})"
                        ));
                    }
                    queue.push_back(target);
                }
            }
        }
        self.nonempty_cache[id as usize] = Some(false);
        Ok(false)
    }
}

#[derive(Debug)]
struct ResidualRuntimeStore {
    arena: ResidualArena,
    root: ResidualId,
    state_by_residual: FxHashMap<ResidualId, u32>,
    residual_by_state: FxHashMap<u32, ResidualId>,
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
        // Force exact root liveness now under the general resource ceiling.
        // A hard Boolean case that exceeds the ceiling rejects installation;
        // an exactly empty nonempty-language is still a valid lexer component
        // and is represented by a proxy root with no future.
        let root_live = arena.has_future(root).ok()?;
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
                state_by_residual: FxHashMap::default(),
                residual_by_state: FxHashMap::default(),
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
        if residual == store.root {
            return Some(self.root_state);
        }
        if let Some(&state) = store.state_by_residual.get(&residual) {
            return Some(state);
        }
        let state = self.state_allocator.allocate().expect(
            "exact residual tokenizer state-id space exhausted below the dynamic-NFA high-bit tag",
        );
        self.state_owners
            .register_virtual(state, self.runtime_index)
            .expect("residual virtual state owner index must follow shared allocator");
        store.state_by_residual.insert(residual, state);
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
        let target = store.arena.step(residual, byte)?;
        if store.arena.is_empty(target) {
            return None;
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
        store.arena.has_future(residual).map(Some)
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
        self.store.lock().unwrap().state_by_residual.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes(value: &[u8]) -> Expr {
        Expr::U8Seq(value.to_vec())
    }

    fn accepts(arena: &mut ResidualArena, mut state: ResidualId, input: &[u8]) -> bool {
        for &byte in input {
            state = arena.step(state, byte).unwrap();
        }
        arena.is_nullable(state)
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
    fn empty_boolean_root_is_a_valid_dead_proxy() {
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
        assert!(!runtime.root_has_future());
        assert!(runtime.step(1, b'a').is_none());
        assert!(runtime.transitions(1).unwrap().is_empty());
        assert!(!runtime.futures(1).unwrap().contains(0));
    }
}
