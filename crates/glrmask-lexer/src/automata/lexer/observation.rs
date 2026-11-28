//! Shared exact byte-action layouts for compiler observations.
use std::sync::Arc;
use super::{Lexer, tokenizer::{Tokenizer, TokenizerStateSet}};
use smallvec::SmallVec;
/// An exact byte action on the represented alphabet. Dense physical rows and
/// finite-vocabulary quotient rows differ only in their column map. The rows
/// always exist; derived transposes must never stand in for missing topology.
#[derive(Debug, Clone)]
pub struct ByteRows<C = u32> {
    pub values: Arc<[C]>,
    pub byte_to_column: [u8; 256],
    pub width: usize,
}

fn column_count(classes: &[u8; 256]) -> usize {
    let mut seen = [false; 256];
    let mut width = 0;
    for &class in classes {
        let index = class as usize;
        seen[index] = true;
        width = width.max(index + 1);
    }
    assert!(seen[..width].iter().all(|&present| present), "byte classes must be contiguous");
    width
}

pub fn column_representatives(classes: &[u8; 256]) -> Vec<u8> {
    let width = column_count(classes);
    let mut representatives = vec![0; width];
    for byte in (0..=255u8).rev() {
        representatives[classes[byte as usize] as usize] = byte;
    }
    representatives
}

/// Give every relevant byte a column, reserving zero for the unused domain.
/// A wider alphabet can use the dense representation without expansion.
pub fn conservative_byte_classes(relevant: &[bool; 256]) -> Option<[u8; 256]> {
    if relevant.iter().filter(|&&active| active).count() >= 255 {
        return None;
    }
    let mut classes = [0; 256];
    let mut next = 1u8;
    for (byte, &active) in relevant.iter().enumerate() {
        if active {
            classes[byte] = next;
            next += 1;
        }
    }
    Some(classes)
}

/// Exact column refinement. Each live row splits current classes by target;
/// absent edges retain the dead subgroup. At most 256 classes can exist.
struct ColumnPartition {
    classes: [u8; 256],
    sizes: [u16; 256],
    relevant: [bool; 256],
    next: usize,
    limit: usize,
    groups: rustc_hash::FxHashMap<(u8, u32), u8>,
}

impl ColumnPartition {
    fn new(relevant: &[bool; 256], reserve_unobserved: bool) -> Self {
        let offset = usize::from(reserve_unobserved);
        let count = relevant.iter().filter(|&&yes| yes).count();
        let mut sizes = [0; 256];
        sizes[offset] = count as u16;
        Self {
            classes: relevant.map(|yes| if yes { offset as u8 } else { 0 }),
            sizes,
            relevant: *relevant,
            next: offset + usize::from(count != 0),
            limit: offset + count,
            groups: Default::default(),
        }
    }

    fn complete(&self) -> bool { self.next == self.limit }

    fn refine(&mut self, edges: impl IntoIterator<Item = (u8, u32)>) {
        if self.complete() { return; }
        // At most one edge per byte; singleton classes need no more work.
        let mut live = [(0u8, 0u8, 0u32); 256];
        let mut len = 0;
        let mut counts = [0u16; 256];
        let mut first = [u32::MAX; 256];
        let mut mixed = [false; 256];
        for (byte, target) in edges {
            let class = self.classes[byte as usize];
            if self.relevant[byte as usize] && target != u32::MAX && self.sizes[class as usize] > 1 {
                live[len] = (byte, class, target);
                len += 1;
                let index = class as usize;
                if counts[index] == 0 { first[index] = target; }
                else { mixed[index] |= first[index] != target; }
                counts[index] += 1;
            }
        }
        if len == 0 { return; }
        // The first live subgroup can keep its ID only when no byte is dead.
        let mut can_retain: [bool; 256] = std::array::from_fn(|class|
            counts[class] == self.sizes[class]);
        let unchanged: [bool; 256] = std::array::from_fn(|class|
            !mixed[class] && can_retain[class]);
        self.groups.clear();
        for &(byte, old, target) in &live[..len] {
            if unchanged[old as usize] {
                continue;
            }
            let class = if let Some(&class) = self.groups.get(&(old, target)) {
                class
            } else {
                let class = if std::mem::take(&mut can_retain[old as usize]) {
                    old
                } else {
                    let new = u8::try_from(self.next).expect("byte class domain");
                    self.next += 1;
                    new
                };
                self.groups.insert((old, target), class);
                class
            };
            if class != old {
                self.sizes[old as usize] -= 1;
                self.sizes[class as usize] += 1;
                self.classes[byte as usize] = class;
            }
        }
    }
}

/// One congruence for the concatenated exact observations of several lexers.
pub fn exact_byte_classes_for_tokenizers(tokenizers: &[&Tokenizer], relevant: &[bool; 256]) -> [u8; 256] {
    let mut partition = ColumnPartition::new(relevant, relevant.iter().any(|&yes| !yes));
    for tokenizer in tokenizers {
        for state in 0..tokenizer.num_states() {
            if partition.complete() { return partition.classes; }
            partition.refine(tokenizer.transitions_from(state));
        }
    }
    partition.classes
}

pub fn compute_tokenizer_byte_classes(tokenizer: &Tokenizer) -> [u8; 256] {
    exact_byte_classes_for_tokenizers(&[tokenizer], &[true; 256])
}

impl ByteRows {
    /// Exact physical action, independent of any finite token alphabet.
    pub fn from_tokenizer(tokenizer: &Tokenizer) -> Self {
        ByteRows::from_tokenizer_mapped(tokenizer, |target| target)
    }
}

impl<C: Copy + PartialEq> ByteRows<C> {
    /// Encode the exact physical action in a different cell width without
    /// materializing a second u32 slab. The encoding must be a function of
    /// the target (including the dead sentinel), never of the source byte.
    pub fn from_tokenizer_mapped(tokenizer: &Tokenizer, encode: impl Fn(u32) -> C + Sync) -> Self
    where C: Send + Sync {
        let virtual_rows = tokenizer.has_any_virtual_runtime();
        let classes = if virtual_rows {
            std::array::from_fn(|byte| byte as u8)
        } else {
            compute_tokenizer_byte_classes(tokenizer)
        };
        let representatives = column_representatives(&classes);
        let width = representatives.len();
        use rayon::prelude::*;
        let states = tokenizer.num_states() as usize;
        let mut values = vec![encode(u32::MAX); states.checked_mul(width).expect("byte row domain")];
        let build = |(state, output): (usize, &mut [C])| {
            if virtual_rows || width >= 32 {
                let mut row = [u32::MAX; 256];
                tokenizer.fill_transition_row(state as u32, &mut row);
                for (cell, &byte) in output.iter_mut().zip(&representatives) {
                    *cell = encode(row[byte as usize]);
                }
            } else {
                for (cell, &byte) in output.iter_mut().zip(&representatives) {
                    *cell = encode(tokenizer.get_transition(state as u32, byte));
                }
            }
        };
        if states >= 1024 && rayon::current_num_threads() > 1 {
            values.par_chunks_mut(width).enumerate().for_each(build);
        } else {
            values.chunks_mut(width).enumerate().for_each(build);
        }
        Self::new(Arc::from(values), classes)
    }

    pub fn dense(values: Arc<[C]>) -> Self {
        Self::new(values, std::array::from_fn(|byte| byte as u8))
    }

    pub fn new(values: Arc<[C]>, byte_to_column: [u8; 256]) -> Self {
        let width = column_count(&byte_to_column);
        assert_eq!(values.len() % width, 0, "invalid byte rows");
        Self { values, byte_to_column, width }
    }

    pub fn num_states(&self) -> usize {
        self.values.len() / self.width
    }

    #[inline]
    pub fn row(&self, state: usize) -> &[C] {
        let base = state * self.width;
        &self.values[base..base + self.width]
    }

    #[inline]
    pub fn trans(&self, state: usize, byte: usize) -> C {
        self.values[state * self.width + self.byte_to_column[byte] as usize]
    }

    pub fn same_topology(&self, other: &Self) -> bool {
        self.byte_to_column == other.byte_to_column
            && (Arc::ptr_eq(&self.values, &other.values) || self.values == other.values)
    }
}

impl ByteRows {
    /// Exact quotient after projecting targets. Refine one representative
    /// per already equal source column, then expand only the byte map.
    pub fn observed_classes(&self, relevant: &[bool; 256], project: impl Fn(usize, u32) -> u32) -> [u8; 256] {
        let mut by_column = [None; 256];
        let mut selected = [false; 256];
        for byte in (0..=255u8).rev() {
            if relevant[byte as usize] {
                by_column[self.byte_to_column[byte as usize] as usize] = Some(byte);
            }
        }
        let columns: Vec<_> = by_column.iter().enumerate()
            .filter_map(|(column, &byte)| byte.map(|byte| {
                selected[byte as usize] = true;
                (column, byte)
            })).collect();
        let mut partition = ColumnPartition::new(&selected, relevant.iter().any(|&yes| !yes));
        for (state, row) in self.values.chunks_exact(self.width).enumerate() {
            if partition.complete() { break; }
            partition.refine(columns.iter().map(|&(column, byte)| (byte, project(state, row[column]))));
        }
        std::array::from_fn(|byte| {
            if relevant[byte] {
                let representative = by_column[self.byte_to_column[byte] as usize].unwrap();
                partition.classes[representative as usize]
            } else { 0 }
        })
    }

    /// Inspect a compressed transition once, not every byte it represents.
    pub fn self_loop_byte_sets(&self, relevant: &[bool; 256]) -> Vec<crate::ds::u8set::U8Set> {
        use crate::ds::u8set::U8Set;
        let mut by_column = vec![U8Set::empty(); self.width];
        for byte in 0..=255u8 {
            if relevant[byte as usize] {
                by_column[self.byte_to_column[byte as usize] as usize].insert(byte);
            }
        }
        self.values.chunks_exact(self.width).enumerate().map(|(state, row)| {
            let mut result = U8Set::empty();
            for (&target, &bytes) in row.iter().zip(&by_column) {
                if target == state as u32 { result |= bytes; }
            }
            result
        }).collect()
    }

    /// Repack the entire represented action. The caller must have proved
    /// equality of the source columns for every pair of bytes in a class.
    /// This does not turn a finite-alphabet source into a physical lexer.
    pub fn repack(&self, classes: &[u8; 256], transposed: bool) -> Arc<[u32]> {
        self.repack_relevant(classes, &[true; 256], transposed)
    }

    /// Repack a certified byte partition on an explicit observation domain.
    /// Classes must be contiguous; only pairs of observed bytes in each class
    /// must have equal source actions. Classes with no observed byte are dead.
    /// In particular the unused finite-vocabulary catch-all cannot acquire the
    /// action of an arbitrary unobserved physical byte.
    pub fn repack_relevant(
        &self, classes: &[u8; 256], relevant: &[bool; 256], transposed: bool,
    ) -> Arc<[u32]> {
        let width = column_count(classes);
        let mut representatives = vec![None; width];
        for byte in (0..=255u8).rev() {
            if relevant[byte as usize] {
                representatives[classes[byte as usize] as usize] = Some(byte);
            }
        }
        if !transposed && self.byte_to_column == *classes {
            let unused = representatives.iter().enumerate()
                .filter_map(|(class, representative)| representative.is_none().then_some(class))
                .collect::<Vec<_>>();
            if unused.is_empty() || self.values.chunks_exact(width)
                .all(|row| unused.iter().all(|&class| row[class] == u32::MAX))
            {
                return Arc::clone(&self.values);
            }
        }
        let states = self.num_states();
        let mut values = vec![u32::MAX; states * width];
        let columns = representatives.iter().enumerate()
            .filter_map(|(class, byte)| byte.map(|b| (class, self.byte_to_column[b as usize] as usize)))
            .collect::<Vec<_>>();
        for state in 0..states {
            let row = self.row(state);
            for &(class, source_column) in &columns {
                let index = if transposed { class * states + state } else { state * width + class };
                values[index] = row[source_column];
            }
        }
        Arc::from(values)
    }
}


/// Exact interning of canonical state configurations. The hash index owns
/// only candidate IDs; a configuration has one allocation, and collisions
/// are resolved by equality of its complete state sequence.
#[derive(Default)]
pub struct ConfigInterner {
    ids_by_hash: rustc_hash::FxHashMap<u64, smallvec::SmallVec<[u32; 1]>>,
    configs: Vec<Box<[u32]>>,
    singleton_ids: Vec<u32>,
}

impl std::ops::Deref for ConfigInterner {
    type Target = [Box<[u32]>];
    fn deref(&self) -> &Self::Target { &self.configs }
}

impl ConfigInterner {
    /// Dense singleton indexing is useful for scanners seeded at every raw
    /// state; sparse powerset consumers keep the ordinary hash-only layout.
    pub fn with_singletons(state_count: usize) -> Self {
        Self { singleton_ids: vec![u32::MAX; state_count], ..Self::default() }
    }

    fn intern_singleton(&mut self, state: u32) -> Option<u32> {
        let existing = *self.singleton_ids.get(state as usize)?;
        if existing != u32::MAX { return Some(existing); }
        let id = u32::try_from(self.configs.len()).expect("configuration ID overflow");
        assert_ne!(id, u32::MAX, "configuration ID exhausted");
        self.configs.push(Box::new([state]));
        self.singleton_ids[state as usize] = id;
        Some(id)
    }

    fn hash(states: &[u32]) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = rustc_hash::FxHasher::default();
        states.hash(&mut hasher);
        hasher.finish()
    }

    fn lookup(&self, hash: u64, states: &[u32]) -> Option<u32> {
        self.ids_by_hash.get(&hash)?.iter().copied()
            .find(|&id| self.configs[id as usize].as_ref() == states)
    }

    fn insert(&mut self, hash: u64, states: Box<[u32]>) -> u32 {
        let id = u32::try_from(self.configs.len()).expect("configuration ID overflow");
        assert_ne!(id, u32::MAX, "configuration ID exhausted");
        self.configs.push(states);
        self.ids_by_hash.entry(hash).or_default().push(id);
        id
    }

    /// Query the canonical state-set identity without publishing a row.
    /// Bounded observers can admit a new configuration only after their
    /// independent state and memory budgets have been checked.
    pub fn find(&self, states: &[u32]) -> Option<u32> {
        if let [state] = states {
            if let Some(&id) = self.singleton_ids.get(*state as usize) {
                return (id != u32::MAX).then_some(id);
            }
        }
        self.lookup(Self::hash(states), states)
    }

    pub fn intern(&mut self, states: Vec<u32>) -> u32 {
        if let [state] = states.as_slice() {
            if let Some(id) = self.intern_singleton(*state) { return id; }
        }
        let hash = Self::hash(&states);
        if let Some(id) = self.lookup(hash, &states) { return id; }
        self.insert(hash, states.into_boxed_slice())
    }

    pub fn intern_slice(&mut self, states: &[u32]) -> u32 {
        if let [state] = states {
            if let Some(id) = self.intern_singleton(*state) { return id; }
        }
        let hash = Self::hash(states);
        if let Some(id) = self.lookup(hash, states) { return id; }
        self.insert(hash, states.into())
    }

    pub fn into_configs(self) -> Vec<Box<[u32]>> { self.configs }
}

/// A lazy exact configuration action. The caller supplies the byte domain
/// and, optionally, a physical-column equivalence proof. Bytes outside that
/// domain may still be evaluated but cannot read or write a memo row.
/// Observed configurations have a single canonical ID and one transition row.
pub struct ConfigActionCache {
    configs: ConfigInterner,
    byte_slot: [u16; 256],
    width: usize,
    transitions: ConfigMemoRows,
}

enum ConfigMemoRows {
    Flat(Vec<u32>),
    Lazy(Vec<Option<Box<[u32]>>>),
}

impl ConfigActionCache {
    pub const DEAD: u32 = u32::MAX;
    const UNKNOWN: u32 = u32::MAX - 1;

    pub fn new(
        bytes: impl IntoIterator<Item = u8>,
        physical_class: impl Fn(u8) -> usize,
        singleton_state_count: Option<usize>,
    ) -> Self {
        Self::new_with_layout(bytes, physical_class, singleton_state_count, false)
    }

    pub fn new_lazy(
        bytes: impl IntoIterator<Item = u8>,
        physical_class: impl Fn(u8) -> usize,
    ) -> Self {
        Self::new_with_layout(bytes, physical_class, None, true)
    }

    fn new_with_layout(
        bytes: impl IntoIterator<Item = u8>,
        physical_class: impl Fn(u8) -> usize,
        singleton_state_count: Option<usize>,
        lazy: bool,
    ) -> Self {
        let mut byte_slot = [u16::MAX; 256];
        let mut physical_slots = [u16::MAX; 256];
        let mut width = 0usize;
        for byte in bytes {
            let slot = &mut physical_slots[physical_class(byte)];
            if *slot == u16::MAX {
                *slot = width as u16;
                width += 1;
            }
            byte_slot[byte as usize] = *slot;
        }
        let configs = singleton_state_count.map_or_else(
            ConfigInterner::default, ConfigInterner::with_singletons,
        );
        Self {
            configs, byte_slot, width,
            transitions: if lazy {
                ConfigMemoRows::Lazy(Vec::new())
            } else {
                ConfigMemoRows::Flat(Vec::with_capacity(singleton_state_count.unwrap_or(0).saturating_mul(width)))
            },
        }
    }

    pub fn configs(&self) -> &ConfigInterner { &self.configs }

    fn publish_config(&mut self, before: usize, id: u32) -> (u32, bool) {
        assert!(id < Self::UNKNOWN, "configuration ID exhausted");
        let inserted = self.configs.len() != before;
        if inserted {
            match &mut self.transitions {
                ConfigMemoRows::Flat(rows) => rows.resize(self.configs.len() * self.width, Self::UNKNOWN),
                ConfigMemoRows::Lazy(rows) => rows.resize_with(self.configs.len(), || None),
            }
        }
        (id, inserted)
    }

    pub fn intern(&mut self, mut states: Vec<u32>) -> (u32, bool) {
        if states.is_empty() { return (Self::DEAD, false); }
        states.sort_unstable();
        states.dedup();
        let before = self.configs.len();
        let id = self.configs.intern(states);
        self.publish_config(before, id)
    }

    /// Only use for a sorted, deduplicated configuration.
    pub fn intern_canonical_slice(&mut self, states: &[u32]) -> (u32, bool) {
        if states.is_empty() { return (Self::DEAD, false); }
        debug_assert!(states.windows(2).all(|w| w[0] < w[1]));
        let before = self.configs.len();
        let id = self.configs.intern_slice(states);
        self.publish_config(before, id)
    }

    #[inline]
    fn column(&self, byte: u8) -> Option<usize> {
        let slot = self.byte_slot[byte as usize];
        (slot != u16::MAX).then_some(slot as usize)
    }

    #[inline]
    pub fn get(&self, state: u32, byte: u8) -> Option<u32> {
        let column = self.column(byte)?;
        let target = match &self.transitions {
            ConfigMemoRows::Flat(rows) => rows[state as usize * self.width + column],
            ConfigMemoRows::Lazy(rows) => rows[state as usize].as_ref()?[column],
        };
        (target != Self::UNKNOWN).then_some(target)
    }

    #[inline]
    pub fn set(&mut self, state: u32, byte: u8, target: u32) {
        let Some(column) = self.column(byte) else { return; };
        match &mut self.transitions {
            ConfigMemoRows::Flat(rows) => rows[state as usize * self.width + column] = target,
            ConfigMemoRows::Lazy(rows) => {
                let row = rows[state as usize].get_or_insert_with(||
                    vec![Self::UNKNOWN; self.width].into_boxed_slice());
                row[column] = target;
            }
        }
    }
}

/// Source-pinned, unrestricted epsilon action shared by lexical observations.
/// Boundary pruning may supply a nonclosed set; the closed bit is only an
/// acceleration certificate, not part of its semantic identity. Every byte
/// step denotes closure(move(closure(S), byte)).
pub struct LexicalStateAction<'a> {
    tokenizer: &'a Tokenizer,
    memo: ConfigActionCache,
    closed: Vec<bool>,
    raw_start: Vec<u32>,
}

impl<'a> LexicalStateAction<'a> {
    pub fn new(tokenizer: &'a Tokenizer) -> Self {
        Self {
            tokenizer,
            memo: ConfigActionCache::new_lazy(0..=u8::MAX, |byte| byte as usize),
            closed: Vec::new(),
            raw_start: vec![ConfigActionCache::DEAD; tokenizer.num_states() as usize],
        }
    }

    pub fn states(&self, config: u32) -> &[u32] {
        if config == ConfigActionCache::DEAD { &[] }
        else { &self.memo.configs()[config as usize] }
    }

    fn publish(&mut self, id: u32, inserted: bool, closed: bool) -> u32 {
        if id == ConfigActionCache::DEAD { return id; }
        if inserted { self.closed.push(closed); }
        else { self.closed[id as usize] |= closed; }
        id
    }

    /// Arbitrary projected roots may need closure again before their first
    /// byte. Their membership is preserved exactly for endpoint observations.
    pub fn intern_boundary(&mut self, states: &[u32]) -> u32 {
        let mut states = TokenizerStateSet::from_slice(states);
        states.sort_unstable();
        states.dedup();
        self.intern_sorted(&states, false)
    }

    fn intern_sorted(&mut self, states: &[u32], closed: bool) -> u32 {
        let (id, inserted) = self.memo.intern_canonical_slice(states);
        self.publish(id, inserted, closed)
    }

    pub fn start(&mut self, state: u32) -> u32 {
        let cached = self.raw_start[state as usize];
        if cached != ConfigActionCache::DEAD { return cached; }
        let mut closure = self.tokenizer.execute_from_state_end_only(&[], state);
        closure.sort_unstable();
        closure.dedup();
        let config = self.intern_sorted(&closure, true);
        self.raw_start[state as usize] = config;
        config
    }

    pub fn step(&mut self, config: u32, byte: u8) -> u32 {
        if config == ConfigActionCache::DEAD { return config; }
        if let Some(target) = self.memo.get(config, byte) { return target; }
        let target = if self.closed[config as usize] {
            // Preserve the lazy raw-closure cache: large disjoint lexers must
            // not materialize every singleton closure for a small trie.
            let mut direct = SmallVec::<[u32; 8]>::new();
            for &state in self.states(config) {
                if let Some(target) = self.tokenizer.step(state, byte) {
                    direct.push(target);
                }
            }
            direct.sort_unstable();
            direct.dedup();
            let mut closure = TokenizerStateSet::new();
            for state in direct {
                let config = self.start(state);
                closure.extend_from_slice(self.states(config));
            }
            closure.sort_unstable();
            closure.dedup();
            self.intern_sorted(&closure, true)
        } else {
            let mut closure = self.tokenizer.step_all(self.states(config), byte);
            closure.sort_unstable();
            closure.dedup();
            self.intern_sorted(&closure, true)
        };
        self.memo.set(config, byte, target);
        target
    }
}

/// Reusable exact epsilon-image accumulator. The caller selects the
/// observation domain and whether the source is already closed. In
/// particular a PM boundary projection may require re-closing the source.
pub struct EpsilonImageScratch {
    marks: Vec<u32>,
    generation: u32,
    targets: Vec<u32>,
}

impl EpsilonImageScratch {
    pub fn new(state_count: usize) -> Self {
        Self { marks: vec![0; state_count], generation: 0, targets: Vec::new() }
    }

    pub fn step(
        &mut self,
        config: &[u32],
        byte: u8,
        source_closed: bool,
        closures: &super::dfa::SingletonEpsilonClosures,
        active: Option<&[bool]>,
        transition: impl FnMut(u32, u8) -> u32,
    ) -> &[u32] {
        self.step_union(config, &[], byte, source_closed, closures, active, transition)
    }

    /// Union the already-closed additional image, not its preimage. This
    /// factors repeated reset transitions without changing the recurrence.
    pub fn step_union(
        &mut self,
        config: &[u32],
        extra: &[u32],
        byte: u8,
        source_closed: bool,
        closures: &super::dfa::SingletonEpsilonClosures,
        active: Option<&[bool]>,
        mut transition: impl FnMut(u32, u8) -> u32,
    ) -> &[u32] {
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            self.marks.fill(0);
            self.generation = 1;
        }
        self.targets.clear();
        for &state in extra {
            if active.is_some_and(|a| !a[state as usize]) { continue; }
            let mark = &mut self.marks[state as usize];
            if *mark != self.generation {
                *mark = self.generation;
                self.targets.push(state);
            }
        }
        for &source in config {
            let sources = if source_closed {
                std::slice::from_ref(&source)
            } else {
                closures[source as usize].as_ref()
            };
            for &closed_source in sources {
                let target = transition(closed_source, byte);
                if target == u32::MAX { continue; }
                for &reachable in closures[target as usize].iter() {
                    if active.is_some_and(|a| !a[reachable as usize]) { continue; }
                    let mark = &mut self.marks[reachable as usize];
                    if *mark != self.generation {
                        *mark = self.generation;
                        self.targets.push(reachable);
                    }
                }
            }
        }
        self.targets.sort_unstable();
        &self.targets
    }
}
#[cfg(test)]
mod config_action_tests {
    use super::*;

    #[test]
    fn exact_byte_rows_match_physical_fill_for_nested_lexers() {
        use crate::automata::lexer::{ast::{bytes, choice, plus}, compile::{build_regex_monolithic, build_regex_partitioned}};
        let exprs = vec![bytes(b"abc"), plus(choice(vec![bytes(b"a"), bytes(b"bc")])), bytes(b"!ab")];
        let first = build_regex_monolithic(&exprs).into_tokenizer(3, None);
        let second = build_regex_partitioned(&exprs, &[0, 1, 2]).into_tokenizer(3, None);
        let (nested, _) = Tokenizer::disjoint_union_with_terminal_offsets(&[(&first, 0), (&second, 3)]);
        for tokenizer in [first, second, nested] {
            let rows = ByteRows::from_tokenizer(&tokenizer);
            assert_eq!(rows.num_states(), tokenizer.num_states() as usize);
            assert!(rows.width < 256);
            for state in 0..tokenizer.num_states() {
                let mut expected = [u32::MAX; 256];
                tokenizer.fill_transition_row(state, &mut expected);
                for (byte, target) in expected.into_iter().enumerate() {
                    assert_eq!(rows.trans(state as usize, byte), target);
                }
            }
        }
    }

    #[test]
    fn partition_refinement_matches_literal_projected_columns() {
        let rows = [
            [u32::MAX; 256],
            std::array::from_fn(|byte| (byte % 13) as u32),
            std::array::from_fn(|byte| if byte % 5 == 0 { u32::MAX } else { (byte % 7) as u32 }),
            std::array::from_fn(|byte| byte as u32),
        ];
        for count in 0..=rows.len() {
            let dense = ByteRows::dense(Arc::from(rows[..count].concat()));
            let all = [true; 256];
            let classes = dense.observed_classes(&all, |_, target| target);
            let compressed = ByteRows::new(dense.repack(&classes, false), classes);
            for relevant in [all, [false; 256], std::array::from_fn(|byte| byte % 3 == 0)] {
                let project = |state, target| if state == 3 || target == u32::MAX { u32::MAX } else { target % 3 };
                let actual = compressed.observed_classes(&relevant, project);
                column_count(&actual);
                for left in 0..256 {
                    if !relevant[left] {
                        assert_eq!(actual[left], 0);
                        continue;
                    }
                    for right in 0..256 {
                        if relevant[right] {
                            assert_eq!(actual[left] == actual[right],
                                (0..count).all(|state| project(state, rows[state][left]) == project(state, rows[state][right])));
                        }
                    }
                }
            }
        }
    }


    #[test]
    fn memo_rows_share_only_proven_observed_columns() {
        let mut cache = ConfigActionCache::new([b'a', b'b', b'c'],
            |byte| if byte == b'b' || byte == b'd' { b'a' as usize } else { byte as usize },
            Some(8));
        let (a, inserted) = cache.intern_canonical_slice(&[3]);
        assert!(inserted);
        assert_eq!(cache.intern(vec![3, 3]), (a, false));
        let (b, inserted) = cache.intern(vec![5, 1, 5]);
        assert!(inserted);
        assert_eq!(cache.configs()[b as usize].as_ref(), &[1, 5]);
        cache.set(a, b'a', b);
        assert_eq!(cache.get(a, b'b'), Some(b));
        assert_eq!(cache.get(a, b'c'), None);
        assert_eq!(cache.get(a, b'd'), None);
        cache.set(a, b'd', ConfigActionCache::DEAD);
        assert_eq!(cache.get(a, b'a'), Some(b));
        cache.set(b, b'c', ConfigActionCache::DEAD);
        assert_eq!(cache.get(b, b'c'), Some(ConfigActionCache::DEAD));
        assert_eq!(cache.intern(Vec::new()), (ConfigActionCache::DEAD, false));
    }
}
