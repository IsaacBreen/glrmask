//! Shared exact byte-action layouts for compiler observations.
use std::sync::Arc;
/// An exact byte action on the represented alphabet. Dense physical rows and
/// finite-vocabulary quotient rows differ only in their column map. The rows
/// always exist; derived transposes must never stand in for missing topology.
#[derive(Debug, Clone)]
pub struct ByteRows {
    pub values: Arc<[u32]>,
    pub byte_to_column: [u8; 256],
    pub width: usize,
}

pub fn column_representatives(classes: &[u8; 256]) -> Vec<u8> {
    let width = classes.iter().copied().max().unwrap() as usize + 1;
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

impl ByteRows {
    pub fn dense(values: Arc<[u32]>) -> Self {
        Self::new(values, std::array::from_fn(|byte| byte as u8))
    }

    pub fn new(values: Arc<[u32]>, byte_to_column: [u8; 256]) -> Self {
        let width = byte_to_column.iter().copied().max().unwrap() as usize + 1;
        assert_eq!(values.len() % width, 0, "invalid byte rows");
        Self { values, byte_to_column, width }
    }

    pub fn num_states(&self) -> usize {
        self.values.len() / self.width
    }

    #[inline]
    pub fn row(&self, state: usize) -> &[u32] {
        let base = state * self.width;
        &self.values[base..base + self.width]
    }

    #[inline]
    pub fn trans(&self, state: usize, byte: usize) -> u32 {
        self.values[state * self.width + self.byte_to_column[byte] as usize]
    }

    pub fn same_topology(&self, other: &Self) -> bool {
        self.byte_to_column == other.byte_to_column
            && (Arc::ptr_eq(&self.values, &other.values) || self.values == other.values)
    }

    /// Materialize an alternative exact layout, or borrow the existing row
    /// storage by Arc when the requested coordinate is already present.
    pub fn repack(&self, classes: &[u8; 256], transposed: bool) -> Arc<[u32]> {
        if !transposed && self.byte_to_column == *classes {
            return Arc::clone(&self.values);
        }
        let representatives = column_representatives(classes);
        let width = representatives.len();
        let states = self.num_states();
        let mut values = vec![u32::MAX; states * width];
        for state in 0..states {
            let row = self.row(state);
            for (class, &byte) in representatives.iter().enumerate() {
                let index = if transposed { class * states + state } else { state * width + class };
                values[index] = row[self.byte_to_column[byte as usize] as usize];
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
}

impl std::ops::Deref for ConfigInterner {
    type Target = [Box<[u32]>];
    fn deref(&self) -> &Self::Target { &self.configs }
}

impl ConfigInterner {
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

    pub fn intern(&mut self, states: Vec<u32>) -> u32 {
        let hash = Self::hash(&states);
        if let Some(id) = self.lookup(hash, &states) { return id; }
        self.insert(hash, states.into_boxed_slice())
    }

    pub fn intern_slice(&mut self, states: &[u32]) -> u32 {
        let hash = Self::hash(states);
        if let Some(id) = self.lookup(hash, states) { return id; }
        self.insert(hash, states.into())
    }

    pub fn into_configs(self) -> Vec<Box<[u32]>> { self.configs }
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
        mut transition: impl FnMut(u32, u8) -> u32,
    ) -> &[u32] {
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            self.marks.fill(0);
            self.generation = 1;
        }
        self.targets.clear();
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