use std::collections::{BTreeMap, BTreeSet};
use std::cell::{Cell, RefCell};
use std::hash::{Hash, Hasher};
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, OnceLock};

use range_set_blaze::{
    CheckSortedDisjoint, CheckSortedDisjointMap, RangeMapBlaze, RangeSetBlaze,
};
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHasher};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use super::nwa::Label;
use crate::ds::weight::{
    finalize_weight_map, finalize_weight_map_artifact_local, shared_rangeset,
    shared_rangeset_artifact_local, Weight,
};

#[derive(Debug)]
struct PackedDwaRowPool {
    labels: Vec<Vec<Label>>,
    targets: Vec<Vec<u32>>,
    weight_ids: Vec<Vec<u32>>,
    weights: Arc<[Weight]>,
}

#[derive(Debug)]
#[doc(hidden)]
pub struct PackedDwaTransitionRow {
    pool: Arc<PackedDwaRowPool>,
    label_id: usize,
    target_id: usize,
    weight_id: usize,
    materialized: OnceLock<BTreeMap<Label, (u32, Weight)>>,
}

impl PackedDwaTransitionRow {
    fn len(&self) -> usize {
        self.pool.labels[self.label_id].len()
    }

    fn materialized(&self) -> &BTreeMap<Label, (u32, Weight)> {
        self.materialized.get_or_init(|| {
            let labels = &self.pool.labels[self.label_id];
            let targets = &self.pool.targets[self.target_id];
            let weights = &self.pool.weight_ids[self.weight_id];
            labels
                .iter()
                .zip(targets)
                .zip(weights)
                .map(|((&label, &target), &weight)| {
                    (label, (target, self.pool.weights[weight as usize].clone()))
                })
                .collect()
        })
    }
}

pub struct DwaTransitionEntries<'a> {
    inner: DwaTransitionEntriesInner<'a>,
}

enum DwaTransitionEntriesInner<'a> {
    Tree(std::collections::btree_map::Iter<'a, Label, (u32, Weight)>),
    Packed {
        row: &'a PackedDwaTransitionRow,
        index: usize,
    },
}

impl<'a> Iterator for DwaTransitionEntries<'a> {
    type Item = (Label, u32, &'a Weight);

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.inner {
            DwaTransitionEntriesInner::Tree(entries) => entries
                .next()
                .map(|(&label, (target, weight))| (label, *target, weight)),
            DwaTransitionEntriesInner::Packed { row, index } => {
                let labels = &row.pool.labels[row.label_id];
                let position = *index;
                let &label = labels.get(position)?;
                let target = row.pool.targets[row.target_id][position];
                let weight_id = row.pool.weight_ids[row.weight_id][position] as usize;
                *index += 1;
                Some((label, target, &row.pool.weights[weight_id]))
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = match &self.inner {
            DwaTransitionEntriesInner::Tree(entries) => entries.len(),
            DwaTransitionEntriesInner::Packed { row, index } => row.len() - *index,
        };
        (len, Some(len))
    }
}

impl ExactSizeIterator for DwaTransitionEntries<'_> {}

#[derive(Debug, Clone)]
pub enum DwaTransitionMap {
    Owned(BTreeMap<Label, (u32, Weight)>),
    Shared(Arc<BTreeMap<Label, (u32, Weight)>>),
    Packed(Arc<PackedDwaTransitionRow>),
}

impl Default for DwaTransitionMap {
    fn default() -> Self {
        Self::Owned(BTreeMap::new())
    }
}

impl DwaTransitionMap {
    pub fn from_arc(transitions: Arc<BTreeMap<Label, (u32, Weight)>>) -> Self {
        Self::Shared(transitions)
    }

    pub fn ptr_key(&self) -> usize {
        match self {
            Self::Owned(transitions) => transitions as *const BTreeMap<_, _> as usize,
            Self::Shared(transitions) => Arc::as_ptr(transitions) as usize,
            Self::Packed(transitions) => Arc::as_ptr(transitions) as usize,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        match self {
            Self::Owned(transitions) => transitions.len(),
            Self::Shared(transitions) => transitions.len(),
            Self::Packed(transitions) => transitions.len(),
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn is_packed(&self) -> bool {
        matches!(self, Self::Packed(_))
    }

    pub fn entries(&self) -> DwaTransitionEntries<'_> {
        let inner = match self {
            Self::Owned(transitions) => DwaTransitionEntriesInner::Tree(transitions.iter()),
            Self::Shared(transitions) => DwaTransitionEntriesInner::Tree(transitions.iter()),
            Self::Packed(row) => DwaTransitionEntriesInner::Packed { row, index: 0 },
        };
        DwaTransitionEntries { inner }
    }

    #[inline]
    pub fn get_entry(&self, label: &Label) -> Option<(u32, &Weight)> {
        match self {
            Self::Owned(transitions) => transitions
                .get(label)
                .map(|(target, weight)| (*target, weight)),
            Self::Shared(transitions) => transitions
                .get(label)
                .map(|(target, weight)| (*target, weight)),
            Self::Packed(row) => {
                let labels = &row.pool.labels[row.label_id];
                let index = labels.binary_search(label).ok()?;
                let target = row.pool.targets[row.target_id][index];
                let weight_id = row.pool.weight_ids[row.weight_id][index] as usize;
                Some((target, &row.pool.weights[weight_id]))
            }
        }
    }
}

impl PartialEq for DwaTransitionMap {
    fn eq(&self, other: &Self) -> bool {
        if let (Self::Packed(left), Self::Packed(right)) = (self, other)
            && Arc::ptr_eq(left, right)
        {
            return true;
        }
        self.deref() == other.deref()
    }
}

impl Deref for DwaTransitionMap {
    type Target = BTreeMap<Label, (u32, Weight)>;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Owned(transitions) => transitions,
            Self::Shared(transitions) => transitions,
            Self::Packed(transitions) => transitions.materialized(),
        }
    }
}

impl DerefMut for DwaTransitionMap {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // `Shared` means the Arc identity is a canonical transition-row
        // identity.  Once a caller mutates the row that invariant no longer
        // holds, even if Arc::make_mut could give us unique storage.  Move or
        // clone it into `Owned` explicitly so serializers can continue to use
        // Shared pointer identity without missing equal rows after mutation.
        if matches!(self, Self::Shared(_) | Self::Packed(_)) {
            let current = std::mem::take(self);
            let owned = match current {
                Self::Shared(transitions) => Arc::try_unwrap(transitions)
                    .unwrap_or_else(|shared| shared.as_ref().clone()),
                Self::Packed(transitions) => transitions.materialized().clone(),
                Self::Owned(_) => unreachable!("checked shared representation above"),
            };
            *self = Self::Owned(owned);
        }
        match self {
            Self::Owned(transitions) => transitions,
            Self::Shared(_) | Self::Packed(_) => {
                unreachable!("shared row was converted to Owned")
            }
        }
    }
}

impl From<BTreeMap<Label, (u32, Weight)>> for DwaTransitionMap {
    fn from(transitions: BTreeMap<Label, (u32, Weight)>) -> Self {
        Self::Owned(transitions)
    }
}

impl FromIterator<(Label, (u32, Weight))> for DwaTransitionMap {
    fn from_iter<T: IntoIterator<Item = (Label, (u32, Weight))>>(iter: T) -> Self {
        BTreeMap::from_iter(iter).into()
    }
}

impl<'a> IntoIterator for &'a DwaTransitionMap {
    type Item = (&'a Label, &'a (u32, Weight));
    type IntoIter = std::collections::btree_map::Iter<'a, Label, (u32, Weight)>;

    fn into_iter(self) -> Self::IntoIter {
        self.deref().iter()
    }
}

impl<'a> IntoIterator for &'a mut DwaTransitionMap {
    type Item = (&'a Label, &'a mut (u32, Weight));
    type IntoIter = std::collections::btree_map::IterMut<'a, Label, (u32, Weight)>;

    fn into_iter(self) -> Self::IntoIter {
        self.deref_mut().iter_mut()
    }
}

#[derive(Debug, Clone, Default)]
pub struct DWAState {
    pub transitions: DwaTransitionMap,
    pub final_weight: Option<Weight>,
}

#[derive(Debug, Clone)]
pub struct DWA {
    states: Vec<DWAState>,
    start_state: u32,
    shared_transition_rows: bool,
    transition_count_cache: OnceLock<usize>,
    acyclic_cache: OnceLock<bool>,
}

#[derive(Debug, Clone, Copy)]
pub struct DwaStats {
    pub states: usize,
    pub transitions: usize,
    pub transition_pairs: usize,
    pub interned_ranges: usize,
}

/// Token-set identities observed cheaply while decoding the compact DWA.
/// Runtime finalization can consume this once instead of rescanning every
/// expanded transition solely to rebuild the same inventory.
#[derive(Default)]
pub struct PackedDwaTokenSetInventory {
    pub transition_sets: FxHashMap<usize, std::sync::Arc<RangeSetBlaze<u32>>>,
    pub final_sets: FxHashMap<usize, std::sync::Arc<RangeSetBlaze<u32>>>,
    pub transition_word_spans: FxHashMap<usize, u32>,
}

#[derive(Debug, Clone, Copy)]
struct PackedRuntimeTokenSetSpan {
    start: u32,
    len: u32,
    word_spans: u32,
}

#[derive(Debug)]
struct PackedRuntimeTokenSetChunk {
    ranges: Box<[[u32; 2]]>,
    spans: Box<[PackedRuntimeTokenSetSpan]>,
}

#[derive(Debug, Clone, Copy)]
struct PackedRuntimeTokenSetLocation {
    chunk: u32,
    local: u32,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct PackedRuntimeWeight {
    geometry: u32,
    token_ids_start: u32,
    full: u32,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct PackedRuntimeRow {
    labels: u32,
    targets: u32,
    weights: u32,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct PackedRuntimeState {
    row: u32,
    final_weight: u32,
}

#[derive(Debug)]
struct PackedRuntimeSeqPool<T> {
    values: Box<[T]>,
    spans: Box<[[u32; 2]]>,
}

impl<T> PackedRuntimeSeqPool<T> {
    #[inline]
    fn len(&self) -> usize {
        self.spans.len()
    }

    #[inline]
    fn get(&self, id: u32) -> Option<&[T]> {
        let &[start, len] = self.spans.get(id as usize)?;
        let start = start as usize;
        self.values.get(start..start + len as usize)
    }
}

/// Allocation-light, read-only view of the current packed parser DWA.
///
/// This is intentionally separate from the compiler-facing `DWA`: current
/// artifacts can execute directly from canonical packed pools, while a cold
/// composition/mutation path may still materialize an ordinary `DWA` later.
#[derive(Debug)]
pub struct PackedRuntimeDwa {
    start_state: u32,
    token_set_chunks: Box<[PackedRuntimeTokenSetChunk]>,
    token_set_locations: Box<[PackedRuntimeTokenSetLocation]>,
    // Freshly compiled constraints already own canonical interned token sets.
    // Keep those Arcs directly instead of immediately copying millions of
    // RangeSetBlaze ranges into a second flat slab. Loaded artifacts continue
    // to use the flat chunk/location representation above.
    materialized_token_sets: Option<Box<[Arc<RangeSetBlaze<u32>>]>>,
    materialized_token_word_spans: Option<Box<[u32]>>,
    // Freshly compiled large DWAs can build these while they already traverse
    // the canonical RangeSetBlaze token sets for runtime metadata.  This avoids
    // a second multi-million-range traversal on first save.  Loaded artifacts
    // do not need the cache because unchanged resave uses the whole-artifact
    // byte cache.
    fast_wire_token_chunks: Option<Box<[Box<[u8]>]>>,
    geometries: Box<[Vec<(u32, u32)>]>,
    weights: Box<[PackedRuntimeWeight]>,
    weight_token_ids: Box<[u32]>,
    label_pool: PackedRuntimeSeqPool<Label>,
    target_pool: PackedRuntimeSeqPool<u32>,
    weight_id_pool: PackedRuntimeSeqPool<u32>,
    rows: Box<[PackedRuntimeRow]>,
    states: Box<[PackedRuntimeState]>,
}

#[derive(Clone, Copy)]
pub struct PackedRuntimeTokenSetRef<'a> {
    id: u32,
    storage: PackedRuntimeTokenSetStorageRef<'a>,
    word_spans: u32,
}

#[derive(Clone, Copy)]
enum PackedRuntimeTokenSetStorageRef<'a> {
    Flat(&'a [[u32; 2]]),
    Materialized(&'a Arc<RangeSetBlaze<u32>>),
}

impl<'a> PackedRuntimeTokenSetRef<'a> {
    #[inline]
    pub fn id(self) -> u32 {
        self.id
    }

    #[inline]
    pub fn range_count(self) -> usize {
        match self.storage {
            PackedRuntimeTokenSetStorageRef::Flat(ranges) => ranges.len(),
            PackedRuntimeTokenSetStorageRef::Materialized(tokens) => tokens.ranges().len(),
        }
    }

    #[inline]
    pub fn word_spans(self) -> u32 {
        self.word_spans
    }

    #[inline]
    pub fn materialized_arc(self) -> Option<&'a Arc<RangeSetBlaze<u32>>> {
        match self.storage {
            PackedRuntimeTokenSetStorageRef::Materialized(tokens) => Some(tokens),
            PackedRuntimeTokenSetStorageRef::Flat(_) => None,
        }
    }

    #[inline]
    pub fn for_each_range(self, mut f: impl FnMut(u32, u32)) {
        match self.storage {
            PackedRuntimeTokenSetStorageRef::Flat(ranges) => {
                for &[start, end] in ranges {
                    f(start, end);
                }
            }
            PackedRuntimeTokenSetStorageRef::Materialized(tokens) => {
                for range in tokens.ranges() {
                    f(*range.start(), *range.end());
                }
            }
        }
    }
}

#[derive(Clone, Copy)]
pub struct PackedRuntimeWeightRef<'a> {
    dwa: &'a PackedRuntimeDwa,
    id: u32,
}

impl<'a> PackedRuntimeWeightRef<'a> {
    #[inline]
    pub fn id(self) -> u32 {
        self.id
    }

    #[inline]
    pub fn is_full(self) -> bool {
        self.dwa.weights[self.id as usize].full != 0
    }

    #[inline]
    pub fn is_empty(self) -> bool {
        let weight = self.dwa.weights[self.id as usize];
        weight.full == 0 && self.dwa.geometries[weight.geometry as usize].is_empty()
    }

    pub fn token_set_for_tsid(self, tsid: u32) -> Option<PackedRuntimeTokenSetRef<'a>> {
        let weight = self.dwa.weights[self.id as usize];
        if weight.full != 0 {
            return None;
        }
        let geometry = &self.dwa.geometries[weight.geometry as usize];
        let index = geometry
            .binary_search_by(|&(start, end)| {
                if tsid < start {
                    std::cmp::Ordering::Greater
                } else if tsid > end {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Equal
                }
            })
            .ok()?;
        let token_id = self.dwa.weight_token_ids[weight.token_ids_start as usize + index];
        self.dwa.token_set(token_id)
    }

    pub fn entries(
        self,
    ) -> impl Iterator<Item = ((u32, u32), PackedRuntimeTokenSetRef<'a>)> + 'a {
        let weight = self.dwa.weights[self.id as usize];
        let geometry: &'a [(u32, u32)] = if weight.full != 0 {
            &[]
        } else {
            &self.dwa.geometries[weight.geometry as usize]
        };
        let ids = &self.dwa.weight_token_ids
            [weight.token_ids_start as usize..weight.token_ids_start as usize + geometry.len()];
        geometry.iter().copied().zip(ids.iter().copied()).map(|(range, token_id)| {
            (
                range,
                self.dwa
                    .token_set(token_id)
                    .expect("validated packed runtime DWA token-set id"),
            )
        })
    }
}

impl PackedRuntimeDwa {

    fn fast_wire_len_for_chunks(&self, chunk_bodies: &[Box<[u8]>]) -> usize {
        let token_bytes = chunk_bodies.iter().map(|body| body.len()).sum::<usize>();
        let geometry_range_count = self.geometries.iter().map(Vec::len).sum::<usize>();
        12usize
            .saturating_add(token_bytes)
            .saturating_add(chunk_bodies.len().saturating_mul(4))
            .saturating_add(4)
            .saturating_add(self.geometries.len().saturating_mul(4))
            .saturating_add(geometry_range_count.saturating_mul(8))
            .saturating_add(4)
            .saturating_add(self.weights.len().saturating_mul(12))
            .saturating_add(4)
            .saturating_add(self.weight_token_ids.len().saturating_mul(4))
            .saturating_add(4)
            .saturating_add(self.label_pool.values.len().saturating_mul(4))
            .saturating_add(4)
            .saturating_add(self.label_pool.spans.len().saturating_mul(8))
            .saturating_add(4)
            .saturating_add(self.target_pool.values.len().saturating_mul(4))
            .saturating_add(4)
            .saturating_add(self.target_pool.spans.len().saturating_mul(8))
            .saturating_add(4)
            .saturating_add(self.weight_id_pool.values.len().saturating_mul(4))
            .saturating_add(4)
            .saturating_add(self.weight_id_pool.spans.len().saturating_mul(8))
            .saturating_add(4)
            .saturating_add(self.rows.len().saturating_mul(12))
            .saturating_add(4)
            .saturating_add(self.states.len().saturating_mul(8))
    }

    /// Exact DWF1 size when the token-set wire chunks have already been cached.
    /// Fresh compiler-produced packed DWAs satisfy this; loaded unchanged
    /// constraints return their cached whole artifact before reaching here.
    pub fn fast_wire_len(&self) -> Option<usize> {
        self.fast_wire_token_chunks
            .as_deref()
            .map(|chunks| self.fast_wire_len_for_chunks(chunks))
    }

    /// Runtime wire format: compress only the enormous token-range
    /// slab, while keeping the already-flat Weight/row/state arrays in a form
    /// that can be copied or viewed directly on load.  Token-set chunks are
    /// encoded independently so the expensive millions-of-ranges pass scales
    /// across cores without requiring lexicographic set sorting.
    pub fn fast_wire_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.append_fast_wire_bytes(&mut out);
        out
    }

    /// Append DWF1 directly to an existing artifact buffer. This is the same
    /// wire representation as [`Self::fast_wire_bytes`], but lets the outer
    /// constraint serializer avoid allocating and then copying a second
    /// multi-megabyte DWA buffer.
    pub fn append_fast_wire_bytes(&self, out: &mut Vec<u8>) {
        let profile = std::env::var_os("GLRMASK_PROFILE_DWA_FAST_WIRE").is_some();
        let total_started = profile.then(std::time::Instant::now);
        #[inline]
        fn put_u32(out: &mut Vec<u8>, value: u32) {
            out.extend_from_slice(&value.to_le_bytes());
        }
        #[inline]
        fn put_u32s(out: &mut Vec<u8>, values: &[u32]) {
            if cfg!(target_endian = "little") {
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        values.as_ptr().cast::<u8>(),
                        std::mem::size_of_val(values),
                    )
                };
                out.extend_from_slice(bytes);
            } else {
                for &value in values {
                    put_u32(out, value);
                }
            }
        }
        #[inline]
        fn put_i32s(out: &mut Vec<u8>, values: &[i32]) {
            if cfg!(target_endian = "little") {
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        values.as_ptr().cast::<u8>(),
                        std::mem::size_of_val(values),
                    )
                };
                out.extend_from_slice(bytes);
            } else {
                for &value in values {
                    out.extend_from_slice(&value.to_le_bytes());
                }
            }
        }
        #[inline]
        fn put_rows(out: &mut Vec<u8>, rows: &[PackedRuntimeRow]) {
            if cfg!(target_endian = "little") {
                debug_assert_eq!(std::mem::size_of::<PackedRuntimeRow>(), 12);
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        rows.as_ptr().cast::<u8>(),
                        std::mem::size_of_val(rows),
                    )
                };
                out.extend_from_slice(bytes);
            } else {
                for row in rows {
                    put_u32(out, row.labels);
                    put_u32(out, row.targets);
                    put_u32(out, row.weights);
                }
            }
        }
        #[inline]
        fn put_states(out: &mut Vec<u8>, states: &[PackedRuntimeState]) {
            if cfg!(target_endian = "little") {
                debug_assert_eq!(std::mem::size_of::<PackedRuntimeState>(), 8);
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        states.as_ptr().cast::<u8>(),
                        std::mem::size_of_val(states),
                    )
                };
                out.extend_from_slice(bytes);
            } else {
                for state in states {
                    put_u32(out, state.row);
                    put_u32(out, state.final_weight);
                }
            }
        }

        const TOKEN_SET_WIRE_CHUNK: usize = 1024;
        let token_set_count = self.token_set_count();
        let uncached_chunk_bodies;
        let chunk_bodies: &[Box<[u8]>] = if let Some(cached) = &self.fast_wire_token_chunks {
            cached
        } else {
            let chunk_ranges = (0..token_set_count)
                .step_by(TOKEN_SET_WIRE_CHUNK)
                .map(|start| (start, (start + TOKEN_SET_WIRE_CHUNK).min(token_set_count)))
                .collect::<Vec<_>>();
            let encode_chunk = |&(start_id, end_id): &(usize, usize)| {
                let mut body = Vec::<u8>::new();
                put_var_u32(&mut body, (end_id - start_id) as u32);
                for id in start_id..end_id {
                    let token_set = self
                        .token_set(id as u32)
                        .expect("packed runtime token-set id is valid");
                    put_var_u32(&mut body, token_set.range_count() as u32);
                    let mut previous_end_plus_one = 0u64;
                    token_set.for_each_range(|lo, hi| {
                        put_var_u64(&mut body, lo as u64 - previous_end_plus_one);
                        put_var_u32(&mut body, hi - lo);
                        previous_end_plus_one = hi as u64 + 1;
                    });
                }
                body.into_boxed_slice()
            };
            uncached_chunk_bodies = if chunk_ranges.len() >= 4
                && rayon::current_num_threads() > 1
            {
                chunk_ranges
                    .par_iter()
                    .map(encode_chunk)
                    .collect::<Vec<_>>()
            } else {
                chunk_ranges
                    .iter()
                    .map(encode_chunk)
                    .collect::<Vec<_>>()
            };
            &uncached_chunk_bodies
        };

        // Reserve the exact DWF1 payload. The old approximation implicitly
        // assumed all three sequence pools had the same span count. On large
        // parser DWAs it undershot by enough that the final row/state append
        // reallocated and copied the entire ~18 MB buffer.
        let wire_len = self.fast_wire_len_for_chunks(chunk_bodies);
        out.reserve(wire_len);
        out.extend_from_slice(b"DWF1");
        put_u32(out, self.start_state);
        put_u32(out, chunk_bodies.len() as u32);
        let token_started = profile.then(std::time::Instant::now);
        let token_frame_bytes = chunk_bodies
            .iter()
            .map(|body| 4usize.saturating_add(body.len()))
            .sum::<usize>();
        let token_frames_start = out.len();
        out.resize(token_frames_start + token_frame_bytes, 0);
        if chunk_bodies.len() >= 4 && rayon::current_num_threads() > 1 {
            let mut remaining = &mut out[token_frames_start..];
            let mut copies = Vec::with_capacity(chunk_bodies.len());
            for body in chunk_bodies {
                let frame_len = 4 + body.len();
                let (frame, rest) = remaining.split_at_mut(frame_len);
                copies.push((frame, body.as_ref()));
                remaining = rest;
            }
            copies.into_par_iter().for_each(|(frame, body)| {
                frame[..4].copy_from_slice(&(body.len() as u32).to_le_bytes());
                frame[4..].copy_from_slice(body);
            });
        } else {
            let mut pos = token_frames_start;
            for body in chunk_bodies {
                let frame_end = pos + 4 + body.len();
                out[pos..pos + 4].copy_from_slice(&(body.len() as u32).to_le_bytes());
                out[pos + 4..frame_end].copy_from_slice(body);
                pos = frame_end;
            }
        }
        let token_ms = token_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

        let geometry_started = profile.then(std::time::Instant::now);
        put_u32(out, self.geometries.len() as u32);
        for geometry in self.geometries.iter() {
            put_u32(out, geometry.len() as u32);
            for &(start, end) in geometry {
                put_u32(out, start);
                put_u32(out, end);
            }
        }
        put_u32(out, self.weights.len() as u32);
        for weight in self.weights.iter() {
            put_u32(out, weight.geometry);
            put_u32(out, weight.token_ids_start);
            put_u32(out, weight.full);
        }
        put_u32(out, self.weight_token_ids.len() as u32);
        put_u32s(out, &self.weight_token_ids);
        let geometry_ms = geometry_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

        let pools_started = profile.then(std::time::Instant::now);
        put_u32(out, self.label_pool.values.len() as u32);
        put_i32s(out, &self.label_pool.values);
        put_u32(out, self.label_pool.spans.len() as u32);
        for &[start, len] in self.label_pool.spans.iter() {
            put_u32(out, start);
            put_u32(out, len);
        }
        put_u32(out, self.target_pool.values.len() as u32);
        put_u32s(out, &self.target_pool.values);
        put_u32(out, self.target_pool.spans.len() as u32);
        for &[start, len] in self.target_pool.spans.iter() {
            put_u32(out, start);
            put_u32(out, len);
        }
        put_u32(out, self.weight_id_pool.values.len() as u32);
        put_u32s(out, &self.weight_id_pool.values);
        put_u32(out, self.weight_id_pool.spans.len() as u32);
        for &[start, len] in self.weight_id_pool.spans.iter() {
            put_u32(out, start);
            put_u32(out, len);
        }
        let pools_ms = pools_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

        let rows_started = profile.then(std::time::Instant::now);
        put_u32(out, self.rows.len() as u32);
        put_rows(out, &self.rows);
        put_u32(out, self.states.len() as u32);
        put_states(out, &self.states);
        let rows_ms = rows_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        if let Some(total_started) = total_started {
            eprintln!(
                "[glrmask/profile][packed_runtime_dwa_fast_wire_emit] token_ms={token_ms:.3} geometry_weight_ms={geometry_ms:.3} pools_ms={pools_ms:.3} rows_states_ms={rows_ms:.3} total_ms={:.3} bytes={}",
                total_started.elapsed().as_secs_f64() * 1000.0,
                out.len(),
            );
        }
    }

    pub fn from_fast_wire_bytes(input: &[u8]) -> Result<Self, String> {
        #[inline]
        fn take_fixed_u32(input: &[u8], pos: &mut usize) -> Result<u32, String> {
            let end = pos
                .checked_add(4)
                .ok_or_else(|| "overflowing fast DWA offset".to_owned())?;
            let bytes: [u8; 4] = input
                .get(*pos..end)
                .ok_or_else(|| "truncated fast DWA u32".to_owned())?
                .try_into()
                .expect("four-byte slice");
            *pos = end;
            Ok(u32::from_le_bytes(bytes))
        }
        fn take_u32_vec(input: &[u8], pos: &mut usize, len: usize) -> Result<Vec<u32>, String> {
            let byte_len = len
                .checked_mul(4)
                .ok_or_else(|| "overflowing fast DWA vector length".to_owned())?;
            let end = pos
                .checked_add(byte_len)
                .ok_or_else(|| "overflowing fast DWA vector offset".to_owned())?;
            let bytes = input
                .get(*pos..end)
                .ok_or_else(|| "truncated fast DWA u32 vector".to_owned())?;
            *pos = end;
            if cfg!(target_endian = "little") {
                let mut out = Vec::<u32>::with_capacity(len);
                unsafe {
                    out.set_len(len);
                    std::ptr::copy_nonoverlapping(
                        bytes.as_ptr(),
                        out.as_mut_ptr().cast::<u8>(),
                        byte_len,
                    );
                }
                Ok(out)
            } else {
                Ok(bytes
                    .chunks_exact(4)
                    .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
                    .collect())
            }
        }
        fn take_i32_vec(input: &[u8], pos: &mut usize, len: usize) -> Result<Vec<i32>, String> {
            let byte_len = len
                .checked_mul(4)
                .ok_or_else(|| "overflowing fast DWA i32 vector length".to_owned())?;
            let end = pos
                .checked_add(byte_len)
                .ok_or_else(|| "overflowing fast DWA i32 vector offset".to_owned())?;
            let bytes = input
                .get(*pos..end)
                .ok_or_else(|| "truncated fast DWA i32 vector".to_owned())?;
            *pos = end;
            if cfg!(target_endian = "little") {
                let mut out = Vec::<i32>::with_capacity(len);
                unsafe {
                    out.set_len(len);
                    std::ptr::copy_nonoverlapping(
                        bytes.as_ptr(),
                        out.as_mut_ptr().cast::<u8>(),
                        byte_len,
                    );
                }
                Ok(out)
            } else {
                Ok(bytes
                    .chunks_exact(4)
                    .map(|chunk| i32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
                    .collect())
            }
        }
        fn take_rows(
            input: &[u8],
            pos: &mut usize,
            len: usize,
        ) -> Result<Vec<PackedRuntimeRow>, String> {
            let byte_len = len
                .checked_mul(std::mem::size_of::<PackedRuntimeRow>())
                .ok_or_else(|| "overflowing fast DWA row vector length".to_owned())?;
            let end = pos
                .checked_add(byte_len)
                .ok_or_else(|| "overflowing fast DWA row vector offset".to_owned())?;
            let bytes = input
                .get(*pos..end)
                .ok_or_else(|| "truncated fast DWA row vector".to_owned())?;
            *pos = end;
            if cfg!(target_endian = "little") {
                debug_assert_eq!(std::mem::size_of::<PackedRuntimeRow>(), 12);
                let mut out = Vec::<PackedRuntimeRow>::with_capacity(len);
                unsafe {
                    out.set_len(len);
                    std::ptr::copy_nonoverlapping(
                        bytes.as_ptr(),
                        out.as_mut_ptr().cast::<u8>(),
                        byte_len,
                    );
                }
                Ok(out)
            } else {
                let mut local = 0usize;
                let mut out = Vec::with_capacity(len);
                for _ in 0..len {
                    out.push(PackedRuntimeRow {
                        labels: take_fixed_u32(bytes, &mut local)?,
                        targets: take_fixed_u32(bytes, &mut local)?,
                        weights: take_fixed_u32(bytes, &mut local)?,
                    });
                }
                Ok(out)
            }
        }
        fn take_weights(
            input: &[u8],
            pos: &mut usize,
            len: usize,
        ) -> Result<Vec<PackedRuntimeWeight>, String> {
            let byte_len = len
                .checked_mul(std::mem::size_of::<PackedRuntimeWeight>())
                .ok_or_else(|| "overflowing fast DWA Weight vector length".to_owned())?;
            let end = pos
                .checked_add(byte_len)
                .ok_or_else(|| "overflowing fast DWA Weight vector offset".to_owned())?;
            let bytes = input
                .get(*pos..end)
                .ok_or_else(|| "truncated fast DWA Weight vector".to_owned())?;
            *pos = end;
            if cfg!(target_endian = "little") {
                debug_assert_eq!(std::mem::size_of::<PackedRuntimeWeight>(), 12);
                let mut out = Vec::<PackedRuntimeWeight>::with_capacity(len);
                unsafe {
                    out.set_len(len);
                    std::ptr::copy_nonoverlapping(
                        bytes.as_ptr(),
                        out.as_mut_ptr().cast::<u8>(),
                        byte_len,
                    );
                }
                Ok(out)
            } else {
                let mut local = 0usize;
                let mut out = Vec::with_capacity(len);
                for _ in 0..len {
                    out.push(PackedRuntimeWeight {
                        geometry: take_fixed_u32(bytes, &mut local)?,
                        token_ids_start: take_fixed_u32(bytes, &mut local)?,
                        full: take_fixed_u32(bytes, &mut local)?,
                    });
                }
                Ok(out)
            }
        }
        fn take_u32_pairs(
            input: &[u8],
            pos: &mut usize,
            len: usize,
        ) -> Result<Vec<[u32; 2]>, String> {
            let byte_len = len
                .checked_mul(8)
                .ok_or_else(|| "overflowing fast DWA pair vector length".to_owned())?;
            let end = pos
                .checked_add(byte_len)
                .ok_or_else(|| "overflowing fast DWA pair vector offset".to_owned())?;
            let bytes = input
                .get(*pos..end)
                .ok_or_else(|| "truncated fast DWA pair vector".to_owned())?;
            *pos = end;
            if cfg!(target_endian = "little") {
                let mut out = Vec::<[u32; 2]>::with_capacity(len);
                unsafe {
                    out.set_len(len);
                    std::ptr::copy_nonoverlapping(
                        bytes.as_ptr(),
                        out.as_mut_ptr().cast::<u8>(),
                        byte_len,
                    );
                }
                Ok(out)
            } else {
                Ok(bytes
                    .chunks_exact(8)
                    .map(|chunk| {
                        [
                            u32::from_le_bytes(chunk[..4].try_into().expect("four-byte chunk")),
                            u32::from_le_bytes(chunk[4..].try_into().expect("four-byte chunk")),
                        ]
                    })
                    .collect())
            }
        }
        fn take_states(
            input: &[u8],
            pos: &mut usize,
            len: usize,
        ) -> Result<Vec<PackedRuntimeState>, String> {
            let byte_len = len
                .checked_mul(std::mem::size_of::<PackedRuntimeState>())
                .ok_or_else(|| "overflowing fast DWA state vector length".to_owned())?;
            let end = pos
                .checked_add(byte_len)
                .ok_or_else(|| "overflowing fast DWA state vector offset".to_owned())?;
            let bytes = input
                .get(*pos..end)
                .ok_or_else(|| "truncated fast DWA state vector".to_owned())?;
            *pos = end;
            if cfg!(target_endian = "little") {
                debug_assert_eq!(std::mem::size_of::<PackedRuntimeState>(), 8);
                let mut out = Vec::<PackedRuntimeState>::with_capacity(len);
                unsafe {
                    out.set_len(len);
                    std::ptr::copy_nonoverlapping(
                        bytes.as_ptr(),
                        out.as_mut_ptr().cast::<u8>(),
                        byte_len,
                    );
                }
                Ok(out)
            } else {
                let mut local = 0usize;
                let mut out = Vec::with_capacity(len);
                for _ in 0..len {
                    out.push(PackedRuntimeState {
                        row: take_fixed_u32(bytes, &mut local)?,
                        final_weight: take_fixed_u32(bytes, &mut local)?,
                    });
                }
                Ok(out)
            }
        }
        fn decode_token_chunk(body: &[u8]) -> Result<PackedRuntimeTokenSetChunk, String> {
            let mut pos = 0usize;
            let set_count = take_var_u32(body, &mut pos)? as usize;
            // Every range consumes at least one gap byte and one length byte;
            // reserve from that hard upper bound so large JS chunks do not
            // repeatedly reallocate/copy tens of thousands of decoded ranges.
            let mut ranges = Vec::<[u32; 2]>::with_capacity(body.len() / 2);
            let mut spans = Vec::<PackedRuntimeTokenSetSpan>::with_capacity(set_count);
            for _ in 0..set_count {
                let range_count = take_var_u32(body, &mut pos)? as usize;
                let start_index = u32::try_from(ranges.len())
                    .map_err(|_| "fast DWA token chunk exceeds u32 offsets".to_owned())?;
                let mut previous_end_plus_one = 0u64;
                let mut word_spans = 0u32;
                for _ in 0..range_count {
                    let gap = take_var_u64(body, &mut pos)?;
                    let lo64 = previous_end_plus_one
                        .checked_add(gap)
                        .ok_or_else(|| "overflowing fast DWA token range".to_owned())?;
                    let lo = u32::try_from(lo64)
                        .map_err(|_| "overflowing fast DWA token start".to_owned())?;
                    let len = take_var_u32(body, &mut pos)?;
                    let hi = lo
                        .checked_add(len)
                        .ok_or_else(|| "overflowing fast DWA token end".to_owned())?;
                    ranges.push([lo, hi]);
                    word_spans = word_spans.saturating_add(hi / 64 - lo / 64 + 1);
                    previous_end_plus_one = hi as u64 + 1;
                }
                spans.push(PackedRuntimeTokenSetSpan {
                    start: start_index,
                    len: range_count as u32,
                    word_spans,
                });
            }
            if pos != body.len() {
                return Err("trailing bytes in fast DWA token chunk".to_owned());
            }
            Ok(PackedRuntimeTokenSetChunk {
                ranges: ranges.into_boxed_slice(),
                spans: spans.into_boxed_slice(),
            })
        }

        let profile = std::env::var_os("GLRMASK_PROFILE_DWA_RUNTIME").is_some();
        let total_started = profile.then(std::time::Instant::now);
        if !input.starts_with(b"DWF1") {
            return Err("invalid fast runtime DWA header".to_owned());
        }
        let mut pos = 4usize;
        let start_state = take_fixed_u32(input, &mut pos)?;

        let scan_started = profile.then(std::time::Instant::now);
        let chunk_count = take_fixed_u32(input, &mut pos)? as usize;
        let mut chunk_bodies = Vec::<&[u8]>::with_capacity(chunk_count);
        for _ in 0..chunk_count {
            let len = take_fixed_u32(input, &mut pos)? as usize;
            let end = pos
                .checked_add(len)
                .ok_or_else(|| "overflowing fast DWA token-chunk length".to_owned())?;
            let body = input
                .get(pos..end)
                .ok_or_else(|| "truncated fast DWA token chunk".to_owned())?;
            chunk_bodies.push(body);
            pos = end;
        }
        let after_chunks = pos;
        let decode_tokens = || -> Result<(Vec<PackedRuntimeTokenSetChunk>, f64), String> {
            let started = profile.then(std::time::Instant::now);
            let chunks = if chunk_count >= 4 && rayon::current_num_threads() > 1 {
                chunk_bodies
                    .par_iter()
                    .map(|body| decode_token_chunk(body))
                    .collect::<Result<Vec<_>, _>>()?
            } else {
                chunk_bodies
                    .iter()
                    .map(|body| decode_token_chunk(body))
                    .collect::<Result<Vec<_>, _>>()?
            };
            let ms = started
                .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
            Ok((chunks, ms))
        };
        let decode_other = || -> Result<_, String> {
            let started = profile.then(std::time::Instant::now);
            let mut pos = after_chunks;
            let geometry_count = take_fixed_u32(input, &mut pos)? as usize;
            let mut geometries = Vec::<Vec<(u32, u32)>>::with_capacity(geometry_count);
            for _ in 0..geometry_count {
                let len = take_fixed_u32(input, &mut pos)? as usize;
                let mut geometry = Vec::with_capacity(len);
                for _ in 0..len {
                    geometry.push((
                        take_fixed_u32(input, &mut pos)?,
                        take_fixed_u32(input, &mut pos)?,
                    ));
                }
                geometries.push(geometry);
            }

            let weight_count = take_fixed_u32(input, &mut pos)? as usize;
            let weights = take_weights(input, &mut pos, weight_count)?;
            if weights.iter().any(|weight| {
                weight.full == 0 && weight.geometry as usize >= geometries.len()
            }) {
                return Err("invalid fast DWA Weight geometry".to_owned());
            }
            let weight_token_id_count = take_fixed_u32(input, &mut pos)? as usize;
            let weight_token_ids = take_u32_vec(input, &mut pos, weight_token_id_count)?;

            let label_value_count = take_fixed_u32(input, &mut pos)? as usize;
            let label_values = take_i32_vec(input, &mut pos, label_value_count)?;
            let label_span_count = take_fixed_u32(input, &mut pos)? as usize;
            let label_spans = take_u32_pairs(input, &mut pos, label_span_count)?;
            let target_value_count = take_fixed_u32(input, &mut pos)? as usize;
            let target_values = take_u32_vec(input, &mut pos, target_value_count)?;
            let target_span_count = take_fixed_u32(input, &mut pos)? as usize;
            let target_spans = take_u32_pairs(input, &mut pos, target_span_count)?;
            let weight_value_count = take_fixed_u32(input, &mut pos)? as usize;
            let weight_values = take_u32_vec(input, &mut pos, weight_value_count)?;
            let weight_span_count = take_fixed_u32(input, &mut pos)? as usize;
            let weight_spans = take_u32_pairs(input, &mut pos, weight_span_count)?;
            if label_span_count != target_span_count || label_span_count != weight_span_count {
                return Err("mismatched fast DWA row-pool spans".to_owned());
            }

            let row_count = take_fixed_u32(input, &mut pos)? as usize;
            let rows = take_rows(input, &mut pos, row_count)?;
            let state_count = take_fixed_u32(input, &mut pos)? as usize;
            if state_count != 0 && start_state as usize >= state_count {
                return Err("invalid fast DWA start state".to_owned());
            }
            let states = take_states(input, &mut pos, state_count)?;
            if states.iter().any(|state| {
                state.row as usize >= row_count
                    || (state.final_weight != u32::MAX
                        && state.final_weight as usize >= weight_count)
            }) {
                return Err("invalid fast DWA state row/final Weight".to_owned());
            }
            if pos != input.len() {
                return Err("trailing bytes in fast runtime DWA".to_owned());
            }
            let ms = started
                .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
            Ok((
                geometries,
                weights,
                weight_token_ids,
                label_values,
                label_spans,
                target_values,
                target_spans,
                weight_values,
                weight_spans,
                rows,
                states,
                ms,
            ))
        };
        let (token_result, other_result) = if rayon::current_num_threads() > 1 {
            rayon::join(decode_tokens, decode_other)
        } else {
            (decode_tokens(), decode_other())
        };
        let (token_set_chunks, token_ms) = token_result?;
        let (
            geometries,
            weights,
            weight_token_ids,
            label_values,
            label_spans,
            target_values,
            target_spans,
            weight_values,
            weight_spans,
            rows,
            states,
            other_ms,
        ) = other_result?;
        let token_set_count = token_set_chunks
            .iter()
            .map(|chunk| chunk.spans.len())
            .sum::<usize>();
        let mut token_set_locations = Vec::<PackedRuntimeTokenSetLocation>::with_capacity(token_set_count);
        for (chunk, decoded) in token_set_chunks.iter().enumerate() {
            for local in 0..decoded.spans.len() {
                token_set_locations.push(PackedRuntimeTokenSetLocation {
                    chunk: chunk as u32,
                    local: local as u32,
                });
            }
        }
        let scan_ms = scan_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

        if let Some(total_started) = total_started {
            eprintln!(
                "[glrmask/profile][packed_runtime_dwa_fast_load] token_ms={token_ms:.3} other_ms={other_ms:.3} scan_inclusive_ms={scan_ms:.3} total_ms={:.3} bytes={} states={} token_sets={} token_ranges={} weights={} rows={}",
                total_started.elapsed().as_secs_f64() * 1000.0,
                input.len(),
                states.len(),
                token_set_locations.len(),
                token_set_chunks.iter().map(|chunk| chunk.ranges.len()).sum::<usize>(),
                weights.len(),
                rows.len(),
            );
        }

        Ok(Self {
            start_state,
            token_set_chunks: token_set_chunks.into_boxed_slice(),
            token_set_locations: token_set_locations.into_boxed_slice(),
            materialized_token_sets: None,
            materialized_token_word_spans: None,
            fast_wire_token_chunks: None,
            geometries: geometries.into_boxed_slice(),
            weights: weights.into_boxed_slice(),
            weight_token_ids: weight_token_ids.into_boxed_slice(),
            label_pool: PackedRuntimeSeqPool {
                values: label_values.into_boxed_slice(),
                spans: label_spans.into_boxed_slice(),
            },
            target_pool: PackedRuntimeSeqPool {
                values: target_values.into_boxed_slice(),
                spans: target_spans.into_boxed_slice(),
            },
            weight_id_pool: PackedRuntimeSeqPool {
                values: weight_values.into_boxed_slice(),
                spans: weight_spans.into_boxed_slice(),
            },
            rows: rows.into_boxed_slice(),
            states: states.into_boxed_slice(),
        })
    }

    /// Build directly from a finalized DWA whose transition rows have already
    /// been canonicalized by identity.  This avoids the old temporary
    /// Vec<Vec<(label,target,weight-id)>> representation and retains canonical
    /// RangeSetBlaze token-set Arcs instead of copying every range immediately.
    fn from_shared_dwa_direct(dwa: &DWA) -> Result<Self, String> {
        let profile = std::env::var_os("GLRMASK_PROFILE_DWA_RUNTIME_BUILD").is_some();
        let total_started = profile.then(std::time::Instant::now);

        let row_classes_started = profile.then(std::time::Instant::now);
        let mut row_id_by_ptr = FxHashMap::<usize, u32>::default();
        row_id_by_ptr.reserve(dwa.states.len().min(32_768));
        let mut representative_states = Vec::<usize>::new();
        let mut state_row_ids = Vec::<u32>::with_capacity(dwa.states.len());
        for (state_index, state) in dwa.states.iter().enumerate() {
            let ptr = state.transitions.ptr_key();
            let row_id = if let Some(&row_id) = row_id_by_ptr.get(&ptr) {
                row_id
            } else {
                let row_id = representative_states.len() as u32;
                row_id_by_ptr.insert(ptr, row_id);
                representative_states.push(state_index);
                row_id
            };
            state_row_ids.push(row_id);
        }
        let row_classes_ms = row_classes_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

        let rows_started = profile.then(std::time::Instant::now);
        let mut local_weight_by_ptr = FxHashMap::<usize, u32>::default();
        local_weight_by_ptr.reserve(32_768);
        let mut weight_refs = Vec::<Weight>::new();
        let mut intern_weight = |weight: &Weight| -> u32 {
            let ptr = weight.ptr_key();
            *local_weight_by_ptr.entry(ptr).or_insert_with(|| {
                let id = weight_refs.len() as u32;
                weight_refs.push(weight.clone());
                id
            })
        };

        let total_row_entries = representative_states
            .iter()
            .map(|&state_index| dwa.states[state_index].transitions.len())
            .sum::<usize>();
        let mut label_values = Vec::<Label>::with_capacity(total_row_entries);
        let mut target_values = Vec::<u32>::with_capacity(total_row_entries);
        let mut weight_values = Vec::<u32>::with_capacity(total_row_entries);
        let mut label_spans = Vec::<[u32; 2]>::with_capacity(representative_states.len());
        let mut target_spans = Vec::<[u32; 2]>::with_capacity(representative_states.len());
        let mut weight_spans = Vec::<[u32; 2]>::with_capacity(representative_states.len());
        let mut rows = Vec::<PackedRuntimeRow>::with_capacity(representative_states.len());
        for (row_id, &state_index) in representative_states.iter().enumerate() {
            let row = &dwa.states[state_index].transitions;
            let start = u32::try_from(label_values.len())
                .map_err(|_| "packed runtime DWA row pool exceeds u32 offsets".to_owned())?;
            let len = u32::try_from(row.len())
                .map_err(|_| "packed runtime DWA row exceeds u32 entries".to_owned())?;
            for (label, target, weight) in row.entries() {
                label_values.push(label);
                target_values.push(target);
                weight_values.push(intern_weight(weight));
            }
            label_spans.push([start, len]);
            target_spans.push([start, len]);
            weight_spans.push([start, len]);
            let id = u32::try_from(row_id)
                .map_err(|_| "packed runtime DWA has too many rows".to_owned())?;
            rows.push(PackedRuntimeRow {
                labels: id,
                targets: id,
                weights: id,
            });
        }
        let rows_ms = rows_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

        let states_started = profile.then(std::time::Instant::now);
        let mut states = Vec::<PackedRuntimeState>::with_capacity(dwa.states.len());
        for (state, row) in dwa.states.iter().zip(state_row_ids) {
            let final_weight = state
                .final_weight
                .as_ref()
                .map(&mut intern_weight)
                .unwrap_or(u32::MAX);
            states.push(PackedRuntimeState { row, final_weight });
        }
        let states_ms = states_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

        let weights_started = profile.then(std::time::Instant::now);
        let token_table_capacity = (weight_refs.len().saturating_mul(4))
            .max(65_536)
            .next_power_of_two();
        let token_table_mask = token_table_capacity - 1;
        let mut token_keys = vec![0usize; token_table_capacity];
        let mut token_values = vec![0u32; token_table_capacity];
        let mut token_sets = Vec::<Arc<RangeSetBlaze<u32>>>::new();
        let mut geometries = vec![Vec::<(u32, u32)>::new()];
        let mut geometry_ids = FxHashMap::<Vec<(u32, u32)>, u32>::default();
        geometry_ids.insert(Vec::new(), 0);
        let mut weights = Vec::<PackedRuntimeWeight>::with_capacity(weight_refs.len());
        let mut weight_token_ids = Vec::<u32>::with_capacity(weight_refs.len().saturating_mul(2));
        for weight in &weight_refs {
            if weight.is_full() {
                weights.push(PackedRuntimeWeight {
                    geometry: 0,
                    token_ids_start: weight_token_ids.len() as u32,
                    full: 1,
                });
                continue;
            }
            let token_ids_start = u32::try_from(weight_token_ids.len())
                .map_err(|_| "packed runtime DWA weight token ids exceed u32 offsets".to_owned())?;
            let mut geometry = Vec::<(u32, u32)>::new();
            for (range, tokens) in weight.raw_range_values() {
                geometry.push((*range.start(), *range.end()));
                let ptr = Arc::as_ptr(tokens) as usize;
                let mut slot = ((ptr >> 4).wrapping_mul(0x9E37_79B9_7F4A_7C15usize))
                    & token_table_mask;
                let token_id = loop {
                    let key = token_keys[slot];
                    if key == ptr {
                        break token_values[slot];
                    }
                    if key == 0 {
                        let id = u32::try_from(token_sets.len())
                            .map_err(|_| "packed runtime DWA has too many token sets".to_owned())?;
                        token_keys[slot] = ptr;
                        token_values[slot] = id;
                        token_sets.push(Arc::clone(tokens));
                        break id;
                    }
                    slot = (slot + 1) & token_table_mask;
                };
                weight_token_ids.push(token_id);
            }
            let geometry_id = if let Some(&id) = geometry_ids.get(&geometry) {
                id
            } else {
                let id = u32::try_from(geometries.len())
                    .map_err(|_| "packed runtime DWA has too many weight geometries".to_owned())?;
                geometry_ids.insert(geometry.clone(), id);
                geometries.push(geometry);
                id
            };
            weights.push(PackedRuntimeWeight {
                geometry: geometry_id,
                token_ids_start,
                full: 0,
            });
        }
        let weight_pool_ms = weights_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

        let token_stats_started = profile.then(std::time::Instant::now);
        // DWF1 groups token sets into independently decodable chunks.  Build
        // those chunk bodies during the metadata traversal we already need for
        // compiled runtime masking. RangeSetBlaze exposes ranges_len() in O(1),
        // so each actual range is visited exactly once here.
        const TOKEN_SET_WIRE_CHUNK: usize = 1024;
        struct TokenChunkBuild {
            body: Box<[u8]>,
            word_spans: Vec<u32>,
            range_count: usize,
        }
        let chunk_ranges = (0..token_sets.len())
            .step_by(TOKEN_SET_WIRE_CHUNK)
            .map(|start| (start, (start + TOKEN_SET_WIRE_CHUNK).min(token_sets.len())))
            .collect::<Vec<_>>();
        let build_chunk = |&(start, end): &(usize, usize)| {
            let mut body = Vec::<u8>::new();
            put_var_u32(&mut body, (end - start) as u32);
            let mut spans = Vec::<u32>::with_capacity(end - start);
            let mut chunk_range_count = 0usize;
            for tokens in &token_sets[start..end] {
                let range_count = tokens.ranges_len();
                put_var_u32(&mut body, range_count as u32);
                chunk_range_count += range_count;
                let mut word_spans = 0u32;
                let mut previous_end_plus_one = 0u64;
                for token_range in tokens.ranges() {
                    let lo = *token_range.start();
                    let hi = *token_range.end();
                    word_spans = word_spans.saturating_add(hi / 64 - lo / 64 + 1);
                    put_var_u64(&mut body, lo as u64 - previous_end_plus_one);
                    put_var_u32(&mut body, hi - lo);
                    previous_end_plus_one = hi as u64 + 1;
                }
                spans.push(word_spans);
            }
            TokenChunkBuild {
                body: body.into_boxed_slice(),
                word_spans: spans,
                range_count: chunk_range_count,
            }
        };
        let token_chunks = if chunk_ranges.len() >= 4 && rayon::current_num_threads() > 1 {
            chunk_ranges.par_iter().map(build_chunk).collect::<Vec<_>>()
        } else {
            chunk_ranges.iter().map(build_chunk).collect::<Vec<_>>()
        };
        let token_range_count = token_chunks.iter().map(|chunk| chunk.range_count).sum::<usize>();
        let token_word_spans = token_chunks
            .iter()
            .flat_map(|chunk| chunk.word_spans.iter().copied())
            .collect::<Vec<_>>();
        let fast_wire_token_chunks = token_chunks
            .into_iter()
            .map(|chunk| chunk.body)
            .collect::<Vec<_>>();
        let token_stats_ms = token_stats_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

        if let Some(total_started) = total_started {
            eprintln!(
                "[glrmask/profile][packed_runtime_dwa_from_shared_dwa] row_classes_ms={row_classes_ms:.3} rows_ms={rows_ms:.3} states_ms={states_ms:.3} weight_pool_ms={weight_pool_ms:.3} token_stats_ms={token_stats_ms:.3} total_ms={:.3} token_sets={} token_ranges={} weight_entries={} weights={} geometries={} rows={} row_entries={} states={}",
                total_started.elapsed().as_secs_f64() * 1000.0,
                token_sets.len(),
                token_range_count,
                weight_token_ids.len(),
                weights.len(),
                geometries.len(),
                rows.len(),
                total_row_entries,
                states.len(),
            );
        }

        Ok(Self {
            start_state: dwa.start_state(),
            token_set_chunks: Box::new([]),
            token_set_locations: Box::new([]),
            materialized_token_sets: Some(token_sets.into_boxed_slice()),
            materialized_token_word_spans: Some(token_word_spans.into_boxed_slice()),
            fast_wire_token_chunks: Some(fast_wire_token_chunks.into_boxed_slice()),
            geometries: geometries.into_boxed_slice(),
            weights: weights.into_boxed_slice(),
            weight_token_ids: weight_token_ids.into_boxed_slice(),
            label_pool: PackedRuntimeSeqPool {
                values: label_values.into_boxed_slice(),
                spans: label_spans.into_boxed_slice(),
            },
            target_pool: PackedRuntimeSeqPool {
                values: target_values.into_boxed_slice(),
                spans: target_spans.into_boxed_slice(),
            },
            weight_id_pool: PackedRuntimeSeqPool {
                values: weight_values.into_boxed_slice(),
                spans: weight_spans.into_boxed_slice(),
            },
            rows: rows.into_boxed_slice(),
            states: states.into_boxed_slice(),
        })
    }

    /// Build the read-only runtime representation directly from a finalized
    /// compiler DWA.  Unlike `from_packed_bytes`, this does no wire-oriented
    /// canonical sorting, front coding, varint encoding, or reparsing.  It is
    /// intended to let compiler-created constraints use the same execution
    /// representation as loaded constraints without paying serialization work
    /// merely to construct it.
    pub fn from_dwa(dwa: &DWA) -> Result<Self, String> {
        if dwa.has_shared_transition_rows() {
            return Self::from_shared_dwa_direct(dwa);
        }
        let profile = std::env::var_os("GLRMASK_PROFILE_DWA_RUNTIME_BUILD").is_some();
        let total_started = profile.then(std::time::Instant::now);

        let pooled_started = profile.then(std::time::Instant::now);
        let (token_sets, weight_pool, transition_rows, state_rows, _) =
            dwa.packed_pooled_parts();
        let pooled_ms = pooled_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

        const TOKEN_SET_CHUNK: usize = 1024;
        let token_started = profile.then(std::time::Instant::now);
        let build_token_chunk = |sets: &[Arc<RangeSetBlaze<u32>>]| {
            let mut ranges = Vec::<[u32; 2]>::new();
            let mut spans = Vec::<PackedRuntimeTokenSetSpan>::with_capacity(sets.len());
            for tokens in sets {
                let start = u32::try_from(ranges.len())
                    .map_err(|_| "packed runtime token-set chunk exceeds u32 offsets".to_owned())?;
                let mut word_spans = 0u32;
                for range in tokens.ranges() {
                    let lo = *range.start();
                    let hi = *range.end();
                    ranges.push([lo, hi]);
                    word_spans = word_spans.saturating_add(hi / 64 - lo / 64 + 1);
                }
                let len = u32::try_from(ranges.len() - start as usize)
                    .map_err(|_| "packed runtime token set exceeds u32 ranges".to_owned())?;
                spans.push(PackedRuntimeTokenSetSpan {
                    start,
                    len,
                    word_spans,
                });
            }
            Ok::<_, String>(PackedRuntimeTokenSetChunk {
                ranges: ranges.into_boxed_slice(),
                spans: spans.into_boxed_slice(),
            })
        };
        let token_set_chunks = if token_sets.len() >= TOKEN_SET_CHUNK * 2
            && rayon::current_num_threads() > 1
        {
            token_sets
                .par_chunks(TOKEN_SET_CHUNK)
                .map(build_token_chunk)
                .collect::<Result<Vec<_>, _>>()?
        } else {
            token_sets
                .chunks(TOKEN_SET_CHUNK)
                .map(build_token_chunk)
                .collect::<Result<Vec<_>, _>>()?
        };
        let mut token_set_locations = Vec::with_capacity(token_sets.len());
        for (chunk, sets) in token_sets.chunks(TOKEN_SET_CHUNK).enumerate() {
            for local in 0..sets.len() {
                token_set_locations.push(PackedRuntimeTokenSetLocation {
                    chunk: chunk as u32,
                    local: local as u32,
                });
            }
        }
        let token_ms = token_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

        // Profile-only diagnostic: isolate RangeSetBlaze range iteration from
        // allocation/copying into the packed slabs. This deliberately runs
        // after the real token build so it cannot inflate token_ms.
        let (range_iter_ms, range_iter_count, range_iter_checksum) = if profile {
            let started = std::time::Instant::now();
            let scan_one = |tokens: &Arc<RangeSetBlaze<u32>>| {
                tokens.ranges().fold((0usize, 0u64), |(count, checksum), range| {
                    (
                        count + 1,
                        checksum
                            .wrapping_add(*range.start() as u64)
                            .wrapping_add((*range.end() as u64).rotate_left(17)),
                    )
                })
            };
            let (count, checksum) = if token_sets.len() >= TOKEN_SET_CHUNK * 2
                && rayon::current_num_threads() > 1
            {
                token_sets
                    .par_iter()
                    .map(scan_one)
                    .reduce(|| (0usize, 0u64), |a, b| (a.0 + b.0, a.1.wrapping_add(b.1)))
            } else {
                token_sets
                    .iter()
                    .map(scan_one)
                    .fold((0usize, 0u64), |a, b| (a.0 + b.0, a.1.wrapping_add(b.1)))
            };
            (started.elapsed().as_secs_f64() * 1000.0, count, checksum)
        } else {
            (0.0, 0usize, 0u64)
        };

        let weights_started = profile.then(std::time::Instant::now);
        let mut geometries = vec![Vec::<(u32, u32)>::new()];
        let mut geometry_ids = FxHashMap::<Vec<(u32, u32)>, u32>::default();
        geometry_ids.insert(Vec::new(), 0);
        let total_weight_entries = weight_pool
            .iter()
            .map(|weight| weight.entries.len())
            .sum::<usize>();
        let mut weights = Vec::<PackedRuntimeWeight>::with_capacity(weight_pool.len());
        let mut weight_token_ids = Vec::<u32>::with_capacity(total_weight_entries);
        for weight in &weight_pool {
            if weight.all {
                weights.push(PackedRuntimeWeight {
                    geometry: 0,
                    token_ids_start: weight_token_ids.len() as u32,
                    full: 1,
                });
                continue;
            }
            let geometry = weight
                .entries
                .iter()
                .map(|&(start, end, _)| (start, end))
                .collect::<Vec<_>>();
            let geometry_id = if let Some(&id) = geometry_ids.get(&geometry) {
                id
            } else {
                let id = u32::try_from(geometries.len())
                    .map_err(|_| "packed runtime DWA has too many weight geometries".to_owned())?;
                geometry_ids.insert(geometry.clone(), id);
                geometries.push(geometry);
                id
            };
            let token_ids_start = u32::try_from(weight_token_ids.len())
                .map_err(|_| "packed runtime DWA weight token ids exceed u32 offsets".to_owned())?;
            for &(_, _, token_set) in &weight.entries {
                if token_set as usize >= token_sets.len() {
                    return Err("invalid direct packed-runtime token-set id".to_owned());
                }
                weight_token_ids.push(token_set);
            }
            weights.push(PackedRuntimeWeight {
                geometry: geometry_id,
                token_ids_start,
                full: 0,
            });
        }
        let weights_ms = weights_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

        let rows_started = profile.then(std::time::Instant::now);
        let total_row_entries = transition_rows.iter().map(Vec::len).sum::<usize>();
        let mut label_values = Vec::<Label>::with_capacity(total_row_entries);
        let mut target_values = Vec::<u32>::with_capacity(total_row_entries);
        let mut weight_values = Vec::<u32>::with_capacity(total_row_entries);
        let mut label_spans = Vec::<[u32; 2]>::with_capacity(transition_rows.len());
        let mut target_spans = Vec::<[u32; 2]>::with_capacity(transition_rows.len());
        let mut weight_spans = Vec::<[u32; 2]>::with_capacity(transition_rows.len());
        let mut rows = Vec::<PackedRuntimeRow>::with_capacity(transition_rows.len());
        for (row_id, row) in transition_rows.iter().enumerate() {
            let start = u32::try_from(label_values.len())
                .map_err(|_| "packed runtime DWA row pool exceeds u32 offsets".to_owned())?;
            let len = u32::try_from(row.len())
                .map_err(|_| "packed runtime DWA row exceeds u32 entries".to_owned())?;
            for &(label, target, weight) in row {
                if weight as usize >= weights.len() {
                    return Err("invalid direct packed-runtime Weight id".to_owned());
                }
                label_values.push(label);
                target_values.push(target);
                weight_values.push(weight);
            }
            label_spans.push([start, len]);
            target_spans.push([start, len]);
            weight_spans.push([start, len]);
            let id = u32::try_from(row_id)
                .map_err(|_| "packed runtime DWA has too many rows".to_owned())?;
            rows.push(PackedRuntimeRow {
                labels: id,
                targets: id,
                weights: id,
            });
        }
        let rows_ms = rows_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

        let states_started = profile.then(std::time::Instant::now);
        let mut states = Vec::<PackedRuntimeState>::with_capacity(state_rows.len());
        for &(row, final_weight) in &state_rows {
            if row as usize >= rows.len() {
                return Err("invalid direct packed-runtime row id".to_owned());
            }
            let final_weight = final_weight.unwrap_or(u32::MAX);
            if final_weight != u32::MAX && final_weight as usize >= weights.len() {
                return Err("invalid direct packed-runtime final Weight id".to_owned());
            }
            states.push(PackedRuntimeState { row, final_weight });
        }
        let states_ms = states_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

        if let Some(total_started) = total_started {
            eprintln!(
                "[glrmask/profile][packed_runtime_dwa_from_dwa] pooled_ms={pooled_ms:.3} token_ms={token_ms:.3} weights_ms={weights_ms:.3} rows_ms={rows_ms:.3} states_ms={states_ms:.3} total_ms={:.3} token_sets={} token_ranges={} weights={} geometries={} rows={} row_entries={} states={}",
                total_started.elapsed().as_secs_f64() * 1000.0,
                token_sets.len(),
                token_set_chunks.iter().map(|chunk| chunk.ranges.len()).sum::<usize>(),
                weights.len(),
                geometries.len(),
                rows.len(),
                total_row_entries,
                states.len(),
            );
            eprintln!(
                "[glrmask/profile][packed_runtime_dwa_range_iter] ms={range_iter_ms:.3} ranges={range_iter_count} checksum={range_iter_checksum}",
            );
        }

        Ok(Self {
            start_state: dwa.start_state(),
            token_set_chunks: token_set_chunks.into_boxed_slice(),
            token_set_locations: token_set_locations.into_boxed_slice(),
            materialized_token_sets: None,
            materialized_token_word_spans: None,
            fast_wire_token_chunks: None,
            geometries: geometries.into_boxed_slice(),
            weights: weights.into_boxed_slice(),
            weight_token_ids: weight_token_ids.into_boxed_slice(),
            label_pool: PackedRuntimeSeqPool {
                values: label_values.into_boxed_slice(),
                spans: label_spans.into_boxed_slice(),
            },
            target_pool: PackedRuntimeSeqPool {
                values: target_values.into_boxed_slice(),
                spans: target_spans.into_boxed_slice(),
            },
            weight_id_pool: PackedRuntimeSeqPool {
                values: weight_values.into_boxed_slice(),
                spans: weight_spans.into_boxed_slice(),
            },
            rows: rows.into_boxed_slice(),
            states: states.into_boxed_slice(),
        })
    }

    pub fn from_packed_bytes(input: &[u8]) -> Result<Self, String> {
        let profile_runtime = std::env::var_os("GLRMASK_PROFILE_DWA_RUNTIME").is_some();
        let total_started = profile_runtime.then(std::time::Instant::now);
        let scan_started = profile_runtime.then(std::time::Instant::now);
        if !input.starts_with(b"DWP6") {
            return Err("invalid packed runtime DWA header".to_owned());
        }
        let mut pos = 4usize;
        let start_state = take_var_u32(input, &mut pos)?;

        let token_set_count = take_var_u32(input, &mut pos)? as usize;
        let token_set_chunk_count = take_var_u32(input, &mut pos)? as usize;
        let token_set_bodies = take_length_prefixed_slices(
            input,
            &mut pos,
            token_set_chunk_count,
            "token-set chunk",
        )?;

        let geometry_count = take_var_u32(input, &mut pos)? as usize;
        let geometry_bodies = take_length_prefixed_slices(
            input,
            &mut pos,
            geometry_count,
            "weight geometry",
        )?;

        let weight_count = take_var_u32(input, &mut pos)? as usize;
        let weight_bodies =
            take_length_prefixed_slices(input, &mut pos, weight_count, "weight")?;

        enum RowSource<'a> {
            Direct(Vec<&'a [u8]>),
            Components {
                labels: Vec<&'a [u8]>,
                targets: Vec<&'a [u8]>,
                weights: Vec<&'a [u8]>,
                rows: Vec<(u32, u32, u32)>,
            },
        }

        let row_mode = *input
            .get(pos)
            .ok_or_else(|| "truncated packed DWA row mode".to_owned())?;
        pos += 1;
        let row_source = match row_mode {
            0 => {
                let row_count = take_var_u32(input, &mut pos)? as usize;
                RowSource::Direct(take_length_prefixed_slices(
                    input,
                    &mut pos,
                    row_count,
                    "transition row",
                )?)
            }
            1 => {
                let label_count = take_var_u32(input, &mut pos)? as usize;
                let labels = take_length_prefixed_slices(
                    input,
                    &mut pos,
                    label_count,
                    "label sequence",
                )?;
                let target_count = take_var_u32(input, &mut pos)? as usize;
                let targets = take_length_prefixed_slices(
                    input,
                    &mut pos,
                    target_count,
                    "target sequence",
                )?;
                let weight_id_count = take_var_u32(input, &mut pos)? as usize;
                let weights = take_length_prefixed_slices(
                    input,
                    &mut pos,
                    weight_id_count,
                    "weight-id sequence",
                )?;
                let row_count = take_var_u32(input, &mut pos)? as usize;
                let mut rows = Vec::with_capacity(row_count);
                for _ in 0..row_count {
                    rows.push((
                        take_var_u32(input, &mut pos)?,
                        take_var_u32(input, &mut pos)?,
                        take_var_u32(input, &mut pos)?,
                    ));
                }
                RowSource::Components {
                    labels,
                    targets,
                    weights,
                    rows,
                }
            }
            _ => return Err("invalid packed DWA row mode".to_owned()),
        };

        let state_count = take_var_u32(input, &mut pos)? as usize;
        if start_state as usize >= state_count && state_count != 0 {
            return Err("invalid packed DWA start state".to_owned());
        }
        let row_count = match &row_source {
            RowSource::Direct(rows) => rows.len(),
            RowSource::Components { rows, .. } => rows.len(),
        };
        let mut states = Vec::with_capacity(state_count);
        for _ in 0..state_count {
            let row = take_var_u32(input, &mut pos)?;
            if row as usize >= row_count {
                return Err("invalid packed DWA transition-row index".to_owned());
            }
            let final_encoded = take_var_u32(input, &mut pos)?;
            let final_weight = if final_encoded == 0 {
                u32::MAX
            } else {
                let weight = final_encoded - 1;
                if weight as usize >= weight_count {
                    return Err("invalid packed DWA final-weight index".to_owned());
                }
                weight
            };
            states.push(PackedRuntimeState { row, final_weight });
        }
        if pos != input.len() {
            return Err("trailing bytes in packed runtime DWA".to_owned());
        }
        let scan_ms = scan_started.map_or(0.0, |s| s.elapsed().as_secs_f64() * 1000.0);

        let decode_token_sets = || -> Result<Vec<PackedRuntimeTokenSetChunk>, String> {
            let started = profile_runtime.then(std::time::Instant::now);
            if token_set_chunk_count >= 16 && rayon::current_num_threads() > 1 {
                let result = token_set_bodies
                    .par_iter()
                    .map(|body| decode_packed_runtime_token_set_chunk(body))
                    .collect();
                if let Some(started) = started {
                    eprintln!("[glrmask/profile][packed_runtime_dwa] token_sets_ms={:.3}", started.elapsed().as_secs_f64() * 1000.0);
                }
                result
            } else {
                let result = token_set_bodies
                    .iter()
                    .map(|body| decode_packed_runtime_token_set_chunk(body))
                    .collect();
                if let Some(started) = started {
                    eprintln!("[glrmask/profile][packed_runtime_dwa] token_sets_ms={:.3}", started.elapsed().as_secs_f64() * 1000.0);
                }
                result
            }
        };

        let decode_other = || -> Result<_, String> {
            let decode_geometry_weights = || -> Result<_, String> {
                let started = profile_runtime.then(std::time::Instant::now);
                let geometries: Vec<Vec<(u32, u32)>> =
                    if geometry_count >= 1024 && rayon::current_num_threads() > 1 {
                        geometry_bodies
                            .par_iter()
                            .map(|body| decode_packed_weight_geometry(body))
                            .collect::<Result<Vec<_>, _>>()?
                    } else {
                        geometry_bodies
                            .iter()
                            .map(|body| decode_packed_weight_geometry(body))
                            .collect::<Result<Vec<_>, _>>()?
                    };
                let (weights, weight_token_ids) = decode_packed_runtime_weights_flat(
                    &weight_bodies,
                    &geometries,
                    token_set_count,
                )?;
                if let Some(started) = started {
                    eprintln!("[glrmask/profile][packed_runtime_dwa] geometry_weights_ms={:.3}", started.elapsed().as_secs_f64() * 1000.0);
                }
                Ok((geometries, weights, weight_token_ids))
            };

            let decode_rows = || -> Result<_, String> {
                let started = profile_runtime.then(std::time::Instant::now);
                match row_source {
                    RowSource::Direct(ref bodies) => {
                        let result = decode_packed_runtime_direct_row_pools(bodies, weight_count);
                        if let Some(started) = started {
                            eprintln!("[glrmask/profile][packed_runtime_dwa] rows_ms={:.3}", started.elapsed().as_secs_f64() * 1000.0);
                        }
                        result
                    }
                    RowSource::Components {
                        ref labels,
                        ref targets,
                        ref weights,
                        ref rows,
                    } => {
                        let decode_labels = || decode_packed_runtime_label_pool(labels);
                        let decode_targets = || decode_packed_runtime_u32_pool(targets, "target");
                        let decode_weights = || decode_packed_runtime_u32_pool(weights, "weight-id");
                        let (label_pool, (target_pool, weight_pool)) = rayon::join(
                            decode_labels,
                            || rayon::join(decode_targets, decode_weights),
                        );
                        let label_pool = label_pool?;
                        let target_pool = target_pool?;
                        let weight_pool = weight_pool?;
                        let mut packed_rows = Vec::with_capacity(rows.len());
                        for &(labels, targets, weights) in rows {
                            if labels as usize >= label_pool.len()
                                || targets as usize >= target_pool.len()
                                || weights as usize >= weight_pool.len()
                            {
                                return Err("invalid packed DWA row-component index".to_owned());
                            }
                            let len = label_pool
                                .get(labels)
                                .expect("validated packed DWA label pool id")
                                .len();
                            if target_pool
                                .get(targets)
                                .expect("validated packed DWA target pool id")
                                .len()
                                != len
                                || weight_pool
                                    .get(weights)
                                    .expect("validated packed DWA weight pool id")
                                    .len()
                                    != len
                            {
                                return Err("mismatched packed DWA row-component lengths".to_owned());
                            }
                            if weight_pool
                                .get(weights)
                                .expect("validated packed DWA weight pool id")
                                .iter()
                                .any(|&weight| weight as usize >= weight_count)
                            {
                                return Err("invalid packed DWA weight index".to_owned());
                            }
                            packed_rows.push(PackedRuntimeRow {
                                labels,
                                targets,
                                weights,
                            });
                        }
                        let result = Ok((label_pool, target_pool, weight_pool, packed_rows));
                        if let Some(started) = started {
                            eprintln!("[glrmask/profile][packed_runtime_dwa] rows_ms={:.3}", started.elapsed().as_secs_f64() * 1000.0);
                        }
                        result
                    }
                }
            };

            let ((geometries, weights, weight_token_ids), (labels, targets, weight_ids, rows)) =
                if rayon::current_num_threads() > 1 {
                    let (left, right) = rayon::join(decode_geometry_weights, decode_rows);
                    (left?, right?)
                } else {
                    (decode_geometry_weights()?, decode_rows()?)
                };
            Ok((
                geometries,
                weights,
                weight_token_ids,
                labels,
                targets,
                weight_ids,
                rows,
            ))
        };

        let (token_set_chunks, other) = if rayon::current_num_threads() > 1 {
            let (token_sets, other) = rayon::join(decode_token_sets, decode_other);
            (token_sets?, other?)
        } else {
            (decode_token_sets()?, decode_other()?)
        };
        let (geometries, weights, weight_token_ids, labels, targets, weight_ids, rows) = other;

        let mut token_set_locations = Vec::with_capacity(token_set_count);
        for (chunk_index, chunk) in token_set_chunks.iter().enumerate() {
            for local in 0..chunk.spans.len() {
                token_set_locations.push(PackedRuntimeTokenSetLocation {
                    chunk: chunk_index as u32,
                    local: local as u32,
                });
            }
        }
        if token_set_locations.len() != token_set_count {
            return Err("invalid packed DWA token-set count".to_owned());
        }
        for index in 0..targets.len() {
            if targets
                .get(index as u32)
                .expect("packed runtime target pool span exists")
                .iter()
                .any(|&target| target as usize >= state_count)
            {
                return Err("invalid packed DWA target state".to_owned());
            }
        }

        let result = Self {
            start_state,
            token_set_chunks: token_set_chunks.into_boxed_slice(),
            token_set_locations: token_set_locations.into_boxed_slice(),
            materialized_token_sets: None,
            materialized_token_word_spans: None,
            fast_wire_token_chunks: None,
            geometries: geometries.into_boxed_slice(),
            weights: weights.into_boxed_slice(),
            weight_token_ids: weight_token_ids.into_boxed_slice(),
            label_pool: labels,
            target_pool: targets,
            weight_id_pool: weight_ids,
            rows: rows.into_boxed_slice(),
            states: states.into_boxed_slice(),
        };
        if let Some(total_started) = total_started {
            eprintln!(
                "[glrmask/profile][packed_runtime_dwa] scan_ms={scan_ms:.3} total_ms={:.3}",
                total_started.elapsed().as_secs_f64() * 1000.0,
            );
        }
        Ok(result)
    }

    #[inline]
    pub fn start_state(&self) -> u32 {
        self.start_state
    }

    #[inline]
    pub fn state_count(&self) -> usize {
        self.states.len()
    }

    #[inline]
    pub fn token_set_count(&self) -> usize {
        self.materialized_token_sets
            .as_ref()
            .map_or(self.token_set_locations.len(), |sets| sets.len())
    }

    #[inline]
    pub fn materialized_token_sets_with_word_spans(
        &self,
    ) -> Option<(&[Arc<RangeSetBlaze<u32>>], &[u32])> {
        Some((
            self.materialized_token_sets.as_deref()?,
            self.materialized_token_word_spans.as_deref()?,
        ))
    }

    #[inline]
    pub fn weight_count(&self) -> usize {
        self.weights.len()
    }

    #[inline]
    pub fn token_set(&self, id: u32) -> Option<PackedRuntimeTokenSetRef<'_>> {
        if let Some(sets) = &self.materialized_token_sets {
            let tokens = sets.get(id as usize)?;
            let word_spans = *self
                .materialized_token_word_spans
                .as_ref()?
                .get(id as usize)?;
            return Some(PackedRuntimeTokenSetRef {
                id,
                storage: PackedRuntimeTokenSetStorageRef::Materialized(tokens),
                word_spans,
            });
        }
        let location = *self.token_set_locations.get(id as usize)?;
        let chunk = self.token_set_chunks.get(location.chunk as usize)?;
        let span = *chunk.spans.get(location.local as usize)?;
        let start = span.start as usize;
        let end = start + span.len as usize;
        Some(PackedRuntimeTokenSetRef {
            id,
            storage: PackedRuntimeTokenSetStorageRef::Flat(&chunk.ranges[start..end]),
            word_spans: span.word_spans,
        })
    }

    #[inline]
    pub fn weight(&self, id: u32) -> Option<PackedRuntimeWeightRef<'_>> {
        ((id as usize) < self.weights.len()).then_some(PackedRuntimeWeightRef { dwa: self, id })
    }

    pub fn transition(&self, state: u32, label: Label) -> Option<(u32, PackedRuntimeWeightRef<'_>)> {
        let state = *self.states.get(state as usize)?;
        let row = *self.rows.get(state.row as usize)?;
        let labels = self.label_pool.get(row.labels)?;
        let index = labels.binary_search(&label).ok()?;
        let target = *self.target_pool.get(row.targets)?.get(index)?;
        let weight = *self.weight_id_pool.get(row.weights)?.get(index)?;
        Some((target, self.weight(weight)?))
    }

    pub fn final_weight(&self, state: u32) -> Option<PackedRuntimeWeightRef<'_>> {
        let state = *self.states.get(state as usize)?;
        (state.final_weight != u32::MAX)
            .then(|| self.weight(state.final_weight))
            .flatten()
    }

    #[inline]
    pub fn row_is_empty(&self, state: u32) -> bool {
        let Some(state) = self.states.get(state as usize) else {
            return true;
        };
        let Some(row) = self.rows.get(state.row as usize) else {
            return true;
        };
        self.label_pool
            .get(row.labels)
            .is_none_or(|labels| labels.is_empty())
    }
}

// --- Two-level weight-pool serde for DWA ---
// Level 1: Pool structurally equal RangeSetBlaze<u32> token sets.
// Level 2: Pool structurally equal Weight values, referencing token-set indices.
//
// Both levels retain a pointer cache as the hot path, but pointer identity is
// deliberately not the serialization identity.  Equivalent DWAs can be built
// with different Arc-sharing layouts (especially when independent support
// operations are evaluated in parallel); the artifact bytes must not depend
// on allocator/interner timing.

/// Serialized token set: Vec of [start, end] range pairs
type EncodedTokenSet = Vec<[u32; 2]>;

/// A single entry in a pooled weight: (tsid_start, tsid_end, token_set_pool_index)
#[derive(Serialize, Deserialize)]
struct WeightPoolEntry {
    all: bool,
    /// Entries: (tsid_range_start, tsid_range_end, token_set_pool_index)
    entries: Vec<(u32, u32, u32)>,
}

#[derive(Serialize, Deserialize)]
struct DWAStateSerde {
    /// transitions: (label, target_state, weight_pool_index)
    transitions: Vec<(Label, u32, u32)>,
    /// final_weight: Some(weight_pool_index) or None
    final_weight: Option<u32>,
}

#[derive(Serialize, Deserialize)]
struct DWASerde {
    /// Pool of unique token sets (level 1)
    token_set_pool: Vec<EncodedTokenSet>,
    /// Pool of unique weights referencing token_set_pool indices (level 2)
    weight_pool: Vec<WeightPoolEntry>,
    states: Vec<DWAStateSerde>,
    start_state: u32,
}

thread_local! {
    /// Constraint artifact v13+ uses a packed, uncompressed DWA wire format.
    /// Older artifact versions leave this disabled and continue to decode the
    /// historical serde representation byte-for-byte.
    static PACKED_DWA_SERDE: Cell<bool> = const { Cell::new(false) };
    /// v14+ sectioned constraint artifacts serialize parser_dwa in its own
    /// section. The Constraint core therefore carries only a one-byte DWA
    /// placeholder, allowing DWA and core decoding to proceed independently.
    static EXTERNAL_DWA_SERDE: Cell<bool> = const { Cell::new(false) };
    static PACKED_DWA_TOKEN_SET_INVENTORY: RefCell<Option<PackedDwaTokenSetInventory>> =
        const { RefCell::new(None) };
}

/// Enable or disable the packed DWA wire format on this thread, returning the
/// previous setting. Constraint serialization uses this as a version-scoped
/// switch so old artifact formats remain readable.
pub fn set_packed_serde(enabled: bool) -> bool {
    PACKED_DWA_SERDE.with(|mode| mode.replace(enabled))
}

pub fn set_external_serde(enabled: bool) -> bool {
    EXTERNAL_DWA_SERDE.with(|mode| mode.replace(enabled))
}

fn packed_serde_enabled() -> bool {
    PACKED_DWA_SERDE.with(Cell::get)
}


fn external_serde_enabled() -> bool {
    EXTERNAL_DWA_SERDE.with(Cell::get)
}

pub fn take_packed_decode_token_set_inventory() -> Option<PackedDwaTokenSetInventory> {
    PACKED_DWA_TOKEN_SET_INVENTORY.with(|slot| slot.borrow_mut().take())
}

pub fn install_packed_decode_token_set_inventory(inventory: PackedDwaTokenSetInventory) {
    PACKED_DWA_TOKEN_SET_INVENTORY.with(|slot| *slot.borrow_mut() = Some(inventory));
}

#[inline]
fn put_var_u32(out: &mut Vec<u8>, mut value: u32) {
    while value >= 0x80 {
        out.push((value as u8) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

#[inline]
fn put_var_u64(out: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        out.push((value as u8) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

#[inline]
fn put_var_i64(out: &mut Vec<u8>, value: i64) {
    let zigzag = ((value << 1) ^ (value >> 63)) as u64;
    put_var_u64(out, zigzag);
}

#[inline]
fn take_var_u32(input: &[u8], pos: &mut usize) -> Result<u32, String> {
    let start = *pos;
    let Some(&first) = input.get(start) else {
        return Err("truncated packed DWA varint".to_owned());
    };
    if first < 0x80 {
        *pos = start + 1;
        return Ok(first as u32);
    }
    let Some(&second) = input.get(start + 1) else {
        return Err("truncated packed DWA varint".to_owned());
    };
    let mut value = (first & 0x7f) as u32 | ((second & 0x7f) as u32) << 7;
    if second < 0x80 {
        *pos = start + 2;
        return Ok(value);
    }
    let mut cursor = start + 2;
    for shift in [14u32, 21, 28] {
        let byte = *input
            .get(cursor)
            .ok_or_else(|| "truncated packed DWA varint".to_owned())?;
        cursor += 1;
        if shift == 28 && byte > 0x0f {
            return Err("overflowing packed DWA u32 varint".to_owned());
        }
        value |= ((byte & 0x7f) as u32) << shift;
        if byte < 0x80 {
            *pos = cursor;
            return Ok(value);
        }
    }
    Err("overflowing packed DWA u32 varint".to_owned())
}

#[inline]
fn take_var_u64(input: &[u8], pos: &mut usize) -> Result<u64, String> {
    let start = *pos;
    let Some(&first) = input.get(start) else {
        return Err("truncated packed DWA varint".to_owned());
    };
    if first < 0x80 {
        *pos = start + 1;
        return Ok(first as u64);
    }
    let Some(&second) = input.get(start + 1) else {
        return Err("truncated packed DWA varint".to_owned());
    };
    let mut value = (first & 0x7f) as u64 | ((second & 0x7f) as u64) << 7;
    if second < 0x80 {
        *pos = start + 2;
        return Ok(value);
    }
    let mut cursor = start + 2;
    for index in 2..10 {
        let byte = *input
            .get(cursor)
            .ok_or_else(|| "truncated packed DWA varint".to_owned())?;
        cursor += 1;
        if index == 9 && byte > 1 {
            return Err("overflowing packed DWA u64 varint".to_owned());
        }
        value |= ((byte & 0x7f) as u64) << (index * 7);
        if byte < 0x80 {
            *pos = cursor;
            return Ok(value);
        }
    }
    Err("overflowing packed DWA u64 varint".to_owned())
}

#[inline]
fn take_var_i64(input: &[u8], pos: &mut usize) -> Result<i64, String> {
    let zigzag = take_var_u64(input, pos)?;
    Ok(((zigzag >> 1) as i64) ^ (-((zigzag & 1) as i64)))
}

fn take_length_prefixed_slices<'a>(
    input: &'a [u8],
    pos: &mut usize,
    count: usize,
    label: &str,
) -> Result<Vec<&'a [u8]>, String> {
    let mut slices = Vec::with_capacity(count);
    for _ in 0..count {
        let len = take_var_u32(input, pos)? as usize;
        let end = pos
            .checked_add(len)
            .ok_or_else(|| format!("overflowing packed DWA {label} length"))?;
        let body = input
            .get(*pos..end)
            .ok_or_else(|| format!("truncated packed DWA {label}"))?;
        slices.push(body);
        *pos = end;
    }
    Ok(slices)
}

#[inline]
fn var_u64_len(mut value: u64) -> usize {
    let mut len = 1usize;
    while value >= 0x80 {
        value >>= 7;
        len += 1;
    }
    len
}

#[inline]
fn zigzag_i64(value: i64) -> u64 {
    ((value << 1) ^ (value >> 63)) as u64
}

fn decode_packed_label_sequence(body: &[u8]) -> Result<Vec<Label>, String> {
    let mut pos = 0usize;
    let count = take_var_u32(body, &mut pos)? as usize;
    let mut labels = Vec::with_capacity(count);
    let mut previous = 0i64;
    for _ in 0..count {
        let delta = take_var_i64(body, &mut pos)?;
        let label64 = previous
            .checked_add(delta)
            .ok_or_else(|| "overflowing packed DWA label sequence".to_owned())?;
        labels.push(
            i32::try_from(label64)
                .map_err(|_| "overflowing packed DWA label sequence".to_owned())?,
        );
        previous = label64;
    }
    if pos != body.len() {
        return Err("trailing bytes in packed DWA label sequence".to_owned());
    }
    Ok(labels)
}

fn decode_packed_u32_sequence(body: &[u8], label: &str) -> Result<Vec<u32>, String> {
    let mut pos = 0usize;
    let mode = *body
        .get(pos)
        .ok_or_else(|| format!("truncated packed DWA {label} sequence mode"))?;
    pos += 1;
    let count = take_var_u32(body, &mut pos)? as usize;
    let mut values = Vec::with_capacity(count);
    match mode {
        0 => {
            for _ in 0..count {
                values.push(take_var_u32(body, &mut pos)?);
            }
        }
        1 => {
            let mut previous = 0i64;
            for _ in 0..count {
                let delta = take_var_i64(body, &mut pos)?;
                let value = previous
                    .checked_add(delta)
                    .ok_or_else(|| format!("overflowing packed DWA {label} sequence"))?;
                values.push(
                    u32::try_from(value)
                        .map_err(|_| format!("overflowing packed DWA {label} sequence"))?,
                );
                previous = value;
            }
        }
        _ => return Err(format!("invalid packed DWA {label} sequence mode")),
    }
    if pos != body.len() {
        return Err(format!("trailing bytes in packed DWA {label} sequence"));
    }
    Ok(values)
}

fn decode_packed_runtime_label_pool(
    bodies: &[&[u8]],
) -> Result<PackedRuntimeSeqPool<Label>, String> {
    let mut values = Vec::<Label>::new();
    let mut spans = Vec::<[u32; 2]>::with_capacity(bodies.len());
    for body in bodies {
        let mut pos = 0usize;
        let count = take_var_u32(body, &mut pos)? as usize;
        let start = u32::try_from(values.len())
            .map_err(|_| "packed runtime label pool exceeds u32".to_owned())?;
        values.reserve(count);
        let mut previous = 0i64;
        for _ in 0..count {
            let delta = take_var_i64(body, &mut pos)?;
            let label64 = previous
                .checked_add(delta)
                .ok_or_else(|| "overflowing packed DWA label sequence".to_owned())?;
            values.push(
                i32::try_from(label64)
                    .map_err(|_| "overflowing packed DWA label sequence".to_owned())?,
            );
            previous = label64;
        }
        if pos != body.len() {
            return Err("trailing bytes in packed DWA label sequence".to_owned());
        }
        spans.push([start, count as u32]);
    }
    Ok(PackedRuntimeSeqPool {
        values: values.into_boxed_slice(),
        spans: spans.into_boxed_slice(),
    })
}

fn decode_packed_runtime_u32_pool(
    bodies: &[&[u8]],
    label: &str,
) -> Result<PackedRuntimeSeqPool<u32>, String> {
    let mut values = Vec::<u32>::new();
    let mut spans = Vec::<[u32; 2]>::with_capacity(bodies.len());
    for body in bodies {
        let mut pos = 0usize;
        let mode = *body
            .get(pos)
            .ok_or_else(|| format!("truncated packed DWA {label} sequence mode"))?;
        pos += 1;
        let count = take_var_u32(body, &mut pos)? as usize;
        let start = u32::try_from(values.len())
            .map_err(|_| format!("packed runtime {label} pool exceeds u32"))?;
        values.reserve(count);
        match mode {
            0 => {
                for _ in 0..count {
                    values.push(take_var_u32(body, &mut pos)?);
                }
            }
            1 => {
                let mut previous = 0i64;
                for _ in 0..count {
                    let delta = take_var_i64(body, &mut pos)?;
                    let value = previous
                        .checked_add(delta)
                        .ok_or_else(|| format!("overflowing packed DWA {label} sequence"))?;
                    values.push(
                        u32::try_from(value)
                            .map_err(|_| format!("overflowing packed DWA {label} sequence"))?,
                    );
                    previous = value;
                }
            }
            _ => return Err(format!("invalid packed DWA {label} sequence mode")),
        }
        if pos != body.len() {
            return Err(format!("trailing bytes in packed DWA {label} sequence"));
        }
        spans.push([start, count as u32]);
    }
    Ok(PackedRuntimeSeqPool {
        values: values.into_boxed_slice(),
        spans: spans.into_boxed_slice(),
    })
}

fn decode_packed_runtime_weights_flat(
    bodies: &[&[u8]],
    geometries: &[Vec<(u32, u32)>],
    token_set_count: usize,
) -> Result<(Vec<PackedRuntimeWeight>, Vec<u32>), String> {
    let token_id_capacity = bodies
        .iter()
        .filter_map(|body| {
            let mut pos = 1usize;
            (body.first().copied() == Some(0))
                .then(|| take_var_u32(body, &mut pos).ok())
                .flatten()
                .and_then(|geometry| geometries.get(geometry as usize))
                .map(Vec::len)
        })
        .sum::<usize>();
    let mut weights = Vec::with_capacity(bodies.len());
    let mut token_ids = Vec::with_capacity(token_id_capacity);
    for body in bodies {
        let mut pos = 0usize;
        let tag = *body
            .get(pos)
            .ok_or_else(|| "truncated packed DWA weight tag".to_owned())?;
        pos += 1;
        if tag == 1 {
            if pos != body.len() {
                return Err("trailing bytes in packed DWA full weight".to_owned());
            }
            weights.push(PackedRuntimeWeight {
                geometry: 0,
                token_ids_start: token_ids.len() as u32,
                full: 1,
            });
            continue;
        }
        if tag != 0 {
            return Err("invalid packed DWA weight tag".to_owned());
        }
        let geometry = take_var_u32(body, &mut pos)?;
        let geometry_ranges = geometries
            .get(geometry as usize)
            .ok_or_else(|| "invalid packed DWA weight-geometry index".to_owned())?;
        let token_ids_start = u32::try_from(token_ids.len())
            .map_err(|_| "packed runtime Weight token ids exceed u32".to_owned())?;
        for _ in geometry_ranges {
            let token_id = take_var_u32(body, &mut pos)?;
            if token_id as usize >= token_set_count {
                return Err("invalid packed DWA token-set index".to_owned());
            }
            token_ids.push(token_id);
        }
        if pos != body.len() {
            return Err("trailing bytes in packed DWA weight body".to_owned());
        }
        weights.push(PackedRuntimeWeight {
            geometry,
            token_ids_start,
            full: 0,
        });
    }
    Ok((weights, token_ids))
}

fn decode_packed_runtime_direct_row_pools(
    bodies: &[&[u8]],
    weight_count: usize,
) -> Result<(
    PackedRuntimeSeqPool<Label>,
    PackedRuntimeSeqPool<u32>,
    PackedRuntimeSeqPool<u32>,
    Vec<PackedRuntimeRow>,
), String> {
    let mut label_values = Vec::<Label>::new();
    let mut target_values = Vec::<u32>::new();
    let mut weight_values = Vec::<u32>::new();
    let mut label_spans = Vec::with_capacity(bodies.len());
    let mut target_spans = Vec::with_capacity(bodies.len());
    let mut weight_spans = Vec::with_capacity(bodies.len());
    let mut rows = Vec::with_capacity(bodies.len());
    for (row_id, body) in bodies.iter().enumerate() {
        let mut pos = 0usize;
        let count = take_var_u32(body, &mut pos)? as usize;
        let label_start = label_values.len() as u32;
        let target_start = target_values.len() as u32;
        let weight_start = weight_values.len() as u32;
        label_values.reserve(count);
        target_values.reserve(count);
        weight_values.reserve(count);
        let mut previous_label = 0i64;
        for index in 0..count {
            let delta = take_var_i64(body, &mut pos)?;
            let label64 = previous_label
                .checked_add(delta)
                .ok_or_else(|| "overflowing packed DWA label".to_owned())?;
            let label = i32::try_from(label64)
                .map_err(|_| "overflowing packed DWA label".to_owned())?;
            if index != 0 && label64 <= previous_label {
                return Err("packed DWA transition labels are not strictly increasing".to_owned());
            }
            previous_label = label64;
            let target = take_var_u32(body, &mut pos)?;
            let weight = take_var_u32(body, &mut pos)?;
            if weight as usize >= weight_count {
                return Err("invalid packed DWA weight index".to_owned());
            }
            label_values.push(label);
            target_values.push(target);
            weight_values.push(weight);
        }
        if pos != body.len() {
            return Err("trailing bytes in packed DWA transition-row body".to_owned());
        }
        label_spans.push([label_start, count as u32]);
        target_spans.push([target_start, count as u32]);
        weight_spans.push([weight_start, count as u32]);
        let id = row_id as u32;
        rows.push(PackedRuntimeRow {
            labels: id,
            targets: id,
            weights: id,
        });
    }
    Ok((
        PackedRuntimeSeqPool {
            values: label_values.into_boxed_slice(),
            spans: label_spans.into_boxed_slice(),
        },
        PackedRuntimeSeqPool {
            values: target_values.into_boxed_slice(),
            spans: target_spans.into_boxed_slice(),
        },
        PackedRuntimeSeqPool {
            values: weight_values.into_boxed_slice(),
            spans: weight_spans.into_boxed_slice(),
        },
        rows,
    ))
}

fn decode_packed_token_set_chunk(
    body: &[u8],
) -> Result<Vec<(std::sync::Arc<RangeSetBlaze<u32>>, u32)>, String> {
    let mut pos = 0usize;
    let token_set_count = take_var_u32(body, &mut pos)? as usize;
    let mut out = Vec::with_capacity(token_set_count);
    let mut previous = EncodedTokenSet::new();
    let mut prefix_word_spans = Vec::<u32>::new();
    for _ in 0..token_set_count {
        let prefix_len = take_var_u32(body, &mut pos)? as usize;
        if prefix_len > previous.len() {
            return Err("invalid packed DWA token-set prefix length".to_owned());
        }
        let suffix_len = take_var_u32(body, &mut pos)? as usize;
        // Front coding means the next token set is literally a prefix of the
        // previous decoded range vector plus a suffix.  Reuse that allocation
        // in place instead of allocating a new Vec and copying the common
        // prefix for every set in the chunk.
        previous.truncate(prefix_len);
        prefix_word_spans.truncate(prefix_len);
        previous.reserve(suffix_len);
        prefix_word_spans.reserve(suffix_len);
        let mut word_spans = prefix_word_spans.last().copied().unwrap_or(0);
        let mut previous_end_plus_one = previous
            .last()
            .map_or(0u64, |range| range[1] as u64 + 1);
        for _ in 0..suffix_len {
            let gap = take_var_u64(body, &mut pos)?;
            let start64 = previous_end_plus_one
                .checked_add(gap)
                .ok_or_else(|| "overflowing packed token-set range".to_owned())?;
            let start = u32::try_from(start64)
                .map_err(|_| "overflowing packed token-set start".to_owned())?;
            let len = take_var_u32(body, &mut pos)?;
            let end = start
                .checked_add(len)
                .ok_or_else(|| "overflowing packed token-set end".to_owned())?;
            previous_end_plus_one = end as u64 + 1;
            previous.push([start, end]);
            word_spans = word_spans.saturating_add(end / 64 - start / 64 + 1);
            prefix_word_spans.push(word_spans);
        }
        let tokens = RangeSetBlaze::from_sorted_disjoint(CheckSortedDisjoint::new(
            previous.iter().map(|range| range[0]..=range[1]),
        ));
        out.push((shared_rangeset_artifact_local(tokens), word_spans));
    }
    if pos != body.len() {
        return Err("invalid packed DWA token-set chunk length".to_owned());
    }
    Ok(out)
}

fn decode_packed_runtime_token_set_chunk(
    body: &[u8],
) -> Result<PackedRuntimeTokenSetChunk, String> {
    let mut pos = 0usize;
    let token_set_count = take_var_u32(body, &mut pos)? as usize;
    let mut previous = EncodedTokenSet::new();
    let mut prefix_word_spans = Vec::<u32>::new();
    let mut ranges = Vec::<[u32; 2]>::new();
    let mut spans = Vec::<PackedRuntimeTokenSetSpan>::with_capacity(token_set_count);
    for _ in 0..token_set_count {
        let prefix_len = take_var_u32(body, &mut pos)? as usize;
        if prefix_len > previous.len() {
            return Err("invalid packed DWA token-set prefix length".to_owned());
        }
        let suffix_len = take_var_u32(body, &mut pos)? as usize;
        previous.truncate(prefix_len);
        prefix_word_spans.truncate(prefix_len);
        previous.reserve(suffix_len);
        prefix_word_spans.reserve(suffix_len);
        let mut word_spans = prefix_word_spans.last().copied().unwrap_or(0);
        let mut previous_end_plus_one = previous
            .last()
            .map_or(0u64, |range| range[1] as u64 + 1);
        for _ in 0..suffix_len {
            let gap = take_var_u64(body, &mut pos)?;
            let start64 = previous_end_plus_one
                .checked_add(gap)
                .ok_or_else(|| "overflowing packed token-set range".to_owned())?;
            let start = u32::try_from(start64)
                .map_err(|_| "overflowing packed token-set start".to_owned())?;
            let len = take_var_u32(body, &mut pos)?;
            let end = start
                .checked_add(len)
                .ok_or_else(|| "overflowing packed token-set end".to_owned())?;
            previous_end_plus_one = end as u64 + 1;
            previous.push([start, end]);
            word_spans = word_spans.saturating_add(end / 64 - start / 64 + 1);
            prefix_word_spans.push(word_spans);
        }
        let start = u32::try_from(ranges.len())
            .map_err(|_| "packed runtime token-set chunk exceeds u32 offsets".to_owned())?;
        let len = u32::try_from(previous.len())
            .map_err(|_| "packed runtime token set exceeds u32 ranges".to_owned())?;
        ranges.extend_from_slice(&previous);
        spans.push(PackedRuntimeTokenSetSpan {
            start,
            len,
            word_spans,
        });
    }
    if pos != body.len() {
        return Err("invalid packed DWA token-set chunk length".to_owned());
    }
    Ok(PackedRuntimeTokenSetChunk {
        ranges: ranges.into_boxed_slice(),
        spans: spans.into_boxed_slice(),
    })
}

fn decode_packed_runtime_weight(
    body: &[u8],
    geometries: &[Vec<(u32, u32)>],
    token_set_count: usize,
) -> Result<(u32, bool, Vec<u32>), String> {
    let mut pos = 0usize;
    let tag = *body
        .get(pos)
        .ok_or_else(|| "truncated packed DWA weight tag".to_owned())?;
    pos += 1;
    if tag == 1 {
        if pos != body.len() {
            return Err("trailing bytes in packed DWA full weight".to_owned());
        }
        return Ok((0, true, Vec::new()));
    }
    if tag != 0 {
        return Err("invalid packed DWA weight tag".to_owned());
    }
    let geometry = take_var_u32(body, &mut pos)?;
    let Some(geometry_ranges) = geometries.get(geometry as usize) else {
        return Err("invalid packed DWA weight-geometry index".to_owned());
    };
    let mut token_ids = Vec::with_capacity(geometry_ranges.len());
    for _ in geometry_ranges {
        let token_id = take_var_u32(body, &mut pos)?;
        if token_id as usize >= token_set_count {
            return Err("invalid packed DWA token-set index".to_owned());
        }
        token_ids.push(token_id);
    }
    if pos != body.len() {
        return Err("trailing bytes in packed DWA weight body".to_owned());
    }
    Ok((geometry, false, token_ids))
}

fn decode_packed_runtime_row(
    body: &[u8],
    weight_count: usize,
) -> Result<(Vec<Label>, Vec<u32>, Vec<u32>), String> {
    let mut pos = 0usize;
    let transition_count = take_var_u32(body, &mut pos)? as usize;
    let mut labels = Vec::with_capacity(transition_count);
    let mut targets = Vec::with_capacity(transition_count);
    let mut weights = Vec::with_capacity(transition_count);
    let mut previous_label = 0i64;
    for index in 0..transition_count {
        let delta = take_var_i64(body, &mut pos)?;
        let label64 = previous_label
            .checked_add(delta)
            .ok_or_else(|| "overflowing packed DWA label".to_owned())?;
        let label = i32::try_from(label64)
            .map_err(|_| "overflowing packed DWA label".to_owned())?;
        if index != 0 && label64 <= previous_label {
            return Err("packed DWA transition labels are not strictly increasing".to_owned());
        }
        previous_label = label64;
        let target = take_var_u32(body, &mut pos)?;
        let weight = take_var_u32(body, &mut pos)?;
        if weight as usize >= weight_count {
            return Err("invalid packed DWA weight index".to_owned());
        }
        labels.push(label);
        targets.push(target);
        weights.push(weight);
    }
    if pos != body.len() {
        return Err("trailing bytes in packed DWA transition-row body".to_owned());
    }
    Ok((labels, targets, weights))
}

fn decode_packed_weight_geometry(body: &[u8]) -> Result<Vec<(u32, u32)>, String> {
    let mut pos = 0usize;
    let entry_count = take_var_u32(body, &mut pos)? as usize;
    let mut ranges = Vec::with_capacity(entry_count);
    let mut previous_end_plus_one = 0u64;
    for _ in 0..entry_count {
        let gap = take_var_u64(body, &mut pos)?;
        let start64 = previous_end_plus_one
            .checked_add(gap)
            .ok_or_else(|| "overflowing packed weight geometry".to_owned())?;
        let start = u32::try_from(start64)
            .map_err(|_| "overflowing packed weight geometry start".to_owned())?;
        let len = take_var_u32(body, &mut pos)?;
        let end = start
            .checked_add(len)
            .ok_or_else(|| "overflowing packed weight geometry end".to_owned())?;
        ranges.push((start, end));
        previous_end_plus_one = end as u64 + 1;
    }
    if pos != body.len() {
        return Err("trailing bytes in packed weight geometry".to_owned());
    }
    Ok(ranges)
}

fn decode_packed_weight(
    body: &[u8],
    ts_pool: &[std::sync::Arc<RangeSetBlaze<u32>>],
    geometries: &[Vec<(u32, u32)>],
) -> Result<Weight, String> {
    let mut pos = 0usize;
    let tag = *body
        .get(pos)
        .ok_or_else(|| "truncated packed DWA weight tag".to_owned())?;
    pos += 1;
    if tag == 1 {
        if pos != body.len() {
            return Err("trailing bytes in packed DWA full weight".to_owned());
        }
        return Ok(Weight::all());
    }
    if tag != 0 {
        return Err("invalid packed DWA weight tag".to_owned());
    }
    let geometry_index = take_var_u32(body, &mut pos)? as usize;
    let geometry = geometries
        .get(geometry_index)
        .ok_or_else(|| "invalid packed DWA weight-geometry index".to_owned())?;
    if geometry.is_empty() {
        if pos != body.len() {
            return Err("trailing bytes in packed DWA empty weight".to_owned());
        }
        return Ok(Weight::empty());
    }
    // Most parser weights have only a small number of TSID ranges.  Keep just
    // their token-set indices inline while parsing, then stream the already
    // sorted shared geometry directly into RangeMapBlaze.  This avoids one
    // heap Vec of (RangeInclusive, Arc) entries per unique Weight.
    let mut token_set_ids = SmallVec::<[u32; 16]>::with_capacity(geometry.len());
    for _ in geometry {
        let token_set_idx = take_var_u32(body, &mut pos)?;
        if token_set_idx as usize >= ts_pool.len() {
            return Err("invalid packed DWA token-set index".to_owned());
        }
        token_set_ids.push(token_set_idx);
    }
    if pos != body.len() {
        return Err("trailing bytes in packed DWA weight body".to_owned());
    }
    let map = RangeMapBlaze::from_sorted_disjoint_map(CheckSortedDisjointMap::new(
        geometry
            .iter()
            .zip(token_set_ids.iter())
            .map(|(&(start, end), &token_set_idx)| {
                (start..=end, &ts_pool[token_set_idx as usize])
            }),
    ));
    Ok(finalize_weight_map_artifact_local(map))
}

fn decode_packed_transition_row(
    body: &[u8],
    w_pool: &[Weight],
    state_count: usize,
) -> Result<BTreeMap<Label, (u32, Weight)>, String> {
    let mut pos = 0usize;
    let transition_count = take_var_u32(body, &mut pos)? as usize;
    let mut entries = Vec::with_capacity(transition_count);
    let mut previous_label = 0i64;
    for index in 0..transition_count {
        let delta = take_var_i64(body, &mut pos)?;
        let label64 = previous_label
            .checked_add(delta)
            .ok_or_else(|| "overflowing packed DWA label".to_owned())?;
        let label = i32::try_from(label64)
            .map_err(|_| "overflowing packed DWA label".to_owned())?;
        if index != 0 && label64 <= previous_label {
            return Err("packed DWA transition labels are not strictly increasing".to_owned());
        }
        previous_label = label64;
        let target = take_var_u32(body, &mut pos)?;
        if target as usize >= state_count {
            return Err("invalid packed DWA target state".to_owned());
        }
        let weight_idx = take_var_u32(body, &mut pos)? as usize;
        let weight = w_pool
            .get(weight_idx)
            .cloned()
            .ok_or_else(|| "invalid packed DWA weight index".to_owned())?;
        entries.push((label, (target, weight)));
    }
    if pos != body.len() {
        return Err("trailing bytes in packed DWA transition-row body".to_owned());
    }
    Ok(entries.into_iter().collect())
}

fn hash_packed_transition_row(row: &[(Label, u32, u32)]) -> u64 {
    let mut hasher = FxHasher::default();
    row.hash(&mut hasher);
    hasher.finish()
}

impl DWA {
    /// Encode the compact parser-DWA section used by sectioned constraint
    /// artifacts, without an outer serde/bincode wrapper.
    pub fn artifact_packed_bytes(&self) -> Vec<u8> {
        self.to_packed_bytes()
    }

    /// Decode a compact parser-DWA section and return the token-set inventory
    /// observed during decode so a caller on another thread can hand it to
    /// runtime finalization without rescanning the expanded DWA.
    pub fn from_artifact_packed_bytes(
        input: &[u8],
    ) -> Result<(Self, Option<PackedDwaTokenSetInventory>), String> {
        let dwa = Self::from_packed_bytes(input)?;
        let inventory = take_packed_decode_token_set_inventory();
        Ok((dwa, inventory))
    }

    fn pooled_parts(&self) -> (Vec<EncodedTokenSet>, Vec<WeightPoolEntry>, Vec<DWAStateSerde>) {
        // These keys are already process-local interned pointers.  SipHash is
        // wasted work here: the pointer value itself is a perfectly adequate
        // input to FxHash, and exact identity is still checked by the map.
        let mut ts_ptr_to_idx: FxHashMap<usize, u32> = FxHashMap::default();
        ts_ptr_to_idx.reserve(32_768);
        // Arc pointer identity is the hot path, but structural identity is the
        // serialization identity. Independent parallel builders can produce
        // equal token sets with different Arc layouts.
        let mut ts_value_to_idx: FxHashMap<Arc<RangeSetBlaze<u32>>, u32> = FxHashMap::default();
        let mut token_set_pool: Vec<EncodedTokenSet> = Vec::new();
        let mut intern_token_set = |ts: &std::sync::Arc<RangeSetBlaze<u32>>| -> u32 {
            let ptr = std::sync::Arc::as_ptr(ts) as usize;
            if let Some(&idx) = ts_ptr_to_idx.get(&ptr) {
                return idx;
            }
            let idx = if let Some(&idx) = ts_value_to_idx.get(ts) {
                idx
            } else {
                let idx = token_set_pool.len() as u32;
                token_set_pool.push(ts.ranges().map(|r| [*r.start(), *r.end()]).collect());
                ts_value_to_idx.insert(std::sync::Arc::clone(ts), idx);
                idx
            };
            ts_ptr_to_idx.insert(ptr, idx);
            idx
        };

        let mut w_ptr_to_idx: FxHashMap<usize, u32> = FxHashMap::default();
        w_ptr_to_idx.reserve(32_768);
        // Level 2: same scheme for weights.  `Weight::Hash` is the cached full
        // structural hash and `Weight::Eq` is structural equality, so this
        // fallback is exact rather than probabilistic.
        let mut w_value_to_idx: FxHashMap<Weight, u32> = FxHashMap::default();
        let mut weight_pool: Vec<WeightPoolEntry> = Vec::new();
        let mut intern_weight = |w: &Weight| -> u32 {
            let ptr = w.ptr_key();
            if let Some(&idx) = w_ptr_to_idx.get(&ptr) {
                return idx;
            }
            let idx = if let Some(&idx) = w_value_to_idx.get(w) {
                idx
            } else {
                let idx = weight_pool.len() as u32;
                if w.is_full() {
                    weight_pool.push(WeightPoolEntry {
                        all: true,
                        entries: Vec::new(),
                    });
                } else {
                    let entries = w
                        .raw_range_values()
                        .map(|(range, tokens)| {
                            let ts_idx = intern_token_set(tokens);
                            (*range.start(), *range.end(), ts_idx)
                        })
                        .collect();
                    weight_pool.push(WeightPoolEntry {
                        all: false,
                        entries,
                    });
                }
                w_value_to_idx.insert(w.clone(), idx);
                idx
            };
            w_ptr_to_idx.insert(ptr, idx);
            idx
        };

        let states = self
            .states
            .iter()
            .map(|state| {
                let transitions = state
                    .transitions
                    .iter()
                    .map(|(&label, (target, weight))| (label, *target, intern_weight(weight)))
                    .collect();
                let final_weight = state.final_weight.as_ref().map(&mut intern_weight);
                DWAStateSerde {
                    transitions,
                    final_weight,
                }
            })
            .collect();
        (token_set_pool, weight_pool, states)
    }

    /// Build the pools needed by the compact artifact without first expanding
    /// every state into a second complete transition vector.  The compact wire
    /// format stores transition rows by identity anyway, so discover those row
    /// classes against the runtime DWA first and translate Weight pointers to
    /// pool indices only for the unique rows.
    fn packed_pooled_parts(
        &self,
    ) -> (
        Vec<Arc<RangeSetBlaze<u32>>>,
        Vec<WeightPoolEntry>,
        Vec<Vec<(Label, u32, u32)>>,
        Vec<(u32, Option<u32>)>,
        usize,
    ) {
        let profile = std::env::var_os("GLRMASK_PROFILE_DWA_SERIALIZATION").is_some();
        let total_started = profile.then(std::time::Instant::now);
        let phase_started = profile.then(std::time::Instant::now);
        let transition_count = self
            .states
            .iter()
            .map(|state| state.transitions.len())
            .sum::<usize>();

        let mut row_representatives = Vec::<usize>::new();
        let mut state_row_ids = Vec::<u32>::with_capacity(self.states.len());
        if self.shared_transition_rows {
            let mut row_id_by_ptr = FxHashMap::<usize, u32>::default();
            row_id_by_ptr.reserve(self.states.len().min(32_768));
            for (state_index, state) in self.states.iter().enumerate() {
                debug_assert!(matches!(
                    state.transitions,
                    DwaTransitionMap::Shared(_) | DwaTransitionMap::Packed(_)
                ));
                let ptr = state.transitions.ptr_key();
                let row_id = if let Some(&row_id) = row_id_by_ptr.get(&ptr) {
                    row_id
                } else {
                    let row_id = row_representatives.len() as u32;
                    row_representatives.push(state_index);
                    row_id_by_ptr.insert(ptr, row_id);
                    row_id
                };
                state_row_ids.push(row_id);
            }
        } else {
            let row_hash = |state: &DWAState| -> u64 {
            // Canonically shared rows can use pointer identity directly. Rows
            // that have been mutated are converted back to Owned by DerefMut,
            // so only those rows pay the structural hashing cost.
            if matches!(
                state.transitions,
                DwaTransitionMap::Shared(_) | DwaTransitionMap::Packed(_)
            ) {
                return state.transitions.ptr_key() as u64;
            }
            let mut hasher = FxHasher::default();
            state.transitions.len().hash(&mut hasher);
            for (label, (target, weight)) in state.transitions.iter() {
                label.hash(&mut hasher);
                target.hash(&mut hasher);
                weight.ptr_key().hash(&mut hasher);
            }
            hasher.finish()
            };
            let row_hashes = if self.states.len() >= 4_096 && rayon::current_num_threads() > 1 {
                self.states.par_iter().map(row_hash).collect::<Vec<_>>()
            } else {
                self.states.iter().map(row_hash).collect::<Vec<_>>()
            };
            let mut row_hash_to_indices = FxHashMap::<u64, Vec<u32>>::default();
            row_hash_to_indices.reserve(self.states.len().min(32_768));
            for (state_index, (&hash, state)) in row_hashes.iter().zip(&self.states).enumerate() {
                let bucket = row_hash_to_indices.entry(hash).or_default();
                let row_id = bucket
                    .iter()
                    .copied()
                    .find(|&candidate| {
                        let representative = row_representatives[candidate as usize];
                        self.states[representative].transitions == state.transitions
                    })
                    .unwrap_or_else(|| {
                        let row_id = row_representatives.len() as u32;
                        row_representatives.push(state_index);
                        bucket.push(row_id);
                        row_id
                    });
                state_row_ids.push(row_id);
            }
        }
        let row_classes_ms = phase_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let phase_started = profile.then(std::time::Instant::now);

        let mut weight_ptr_to_id: FxHashMap<usize, u32> = FxHashMap::default();
        // Small parser DWAs are common in ordinary schema constraints. A fixed
        // 32K reservation cost hundreds of microseconds even when the DWA had
        // only tens or hundreds of distinct weights.
        let weight_ref_count = transition_count.saturating_add(self.states.len());
        // Ordinary parser DWAs have far fewer distinct Weight objects than
        // transition/final references. Reserving for every reference touches
        // several times more hash-table memory than the whole packed wire.
        // Half the reference count avoids that fixed cost while still leaving
        // ample headroom for the observed ordinary-schema uniqueness ratio;
        // unusually high-entropy DWAs can grow normally.
        weight_ptr_to_id.reserve(weight_ref_count.div_ceil(2).clamp(64, 32_768));
        let mut weight_refs = Vec::<Weight>::new();
        let mut intern_weight = |w: &Weight| -> u32 {
            let ptr = w.ptr_key();
            *weight_ptr_to_id.entry(ptr).or_insert_with(|| {
                let idx = weight_refs.len() as u32;
                weight_refs.push(w.clone());
                idx
            })
        };

        let transition_rows_started = profile.then(std::time::Instant::now);
        let transition_rows = row_representatives
            .iter()
            .map(|&state_index| {
                self.states[state_index]
                    .transitions
                    .entries()
                    .map(|(label, target, weight)| (label, target, intern_weight(weight)))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let transition_rows_ms = transition_rows_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let state_rows_started = profile.then(std::time::Instant::now);
        let state_rows = self
            .states
            .iter()
            .zip(state_row_ids)
            .map(|(state, row)| (row, state.final_weight.as_ref().map(&mut intern_weight)))
            .collect::<Vec<_>>();
        let state_rows_ms = state_rows_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let weight_discovery_ms = phase_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        if profile {
            eprintln!(
                "[glrmask/profile][dwa_weight_discovery_split] transition_rows_ms={transition_rows_ms:.3} state_rows_ms={state_rows_ms:.3} transition_entries={} final_weights={}",
                transition_rows.iter().map(Vec::len).sum::<usize>(),
                state_rows.iter().filter(|(_, final_weight)| final_weight.is_some()).count(),
            );
        }
        let phase_started = profile.then(std::time::Instant::now);

        // Weight IDs above are assigned in stable first-encounter order. Walk
        // the unique Weight maps in parallel, then assign token-set IDs in that
        // same Weight/range order. This keeps the wire representation exactly
        // deterministic while avoiding a serial chase through tens of thousands
        // of scattered RangeMapBlaze allocations on first save after compile.
        struct RawWeightPoolEntry<'a> {
            all: bool,
            entries: Vec<(u32, u32, &'a Arc<RangeSetBlaze<u32>>)>,
        }
        fn materialize_weight(weight: &Weight) -> RawWeightPoolEntry<'_> {
            RawWeightPoolEntry {
                all: weight.is_full(),
                entries: if weight.is_full() {
                    Vec::new()
                } else {
                    weight
                        .raw_range_values()
                        .map(|(range, tokens)| (*range.start(), *range.end(), tokens))
                        .collect()
                },
            }
        }
        let raw_weight_pool = if weight_refs.len() >= 256 && rayon::current_num_threads() > 1 {
            weight_refs
                .par_iter()
                .map(materialize_weight)
                .collect::<Vec<_>>()
        } else {
            weight_refs
                .iter()
                .map(materialize_weight)
                .collect::<Vec<_>>()
        };
        let weight_materialize_ms = phase_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let phase_started = profile.then(std::time::Instant::now);

        let raw_token_set_refs = raw_weight_pool
            .iter()
            .map(|weight| weight.entries.len())
            .sum::<usize>();
        let mut ts_ptr_to_idx: FxHashMap<usize, u32> = FxHashMap::default();
        // Token sets are shared heavily across Weight ranges. Estimate from
        // the number of distinct weights instead of reserving for every range
        // reference; the map remains growable for unusual grammars.
        ts_ptr_to_idx.reserve(
            raw_token_set_refs
                .min(raw_weight_pool.len().saturating_mul(4).max(64))
                .min(32_768),
        );
        let mut token_set_pool: Vec<Arc<RangeSetBlaze<u32>>> = Vec::new();
        let mut weight_pool = Vec::<WeightPoolEntry>::with_capacity(raw_weight_pool.len());
        for raw in raw_weight_pool {
            if raw.all {
                weight_pool.push(WeightPoolEntry {
                    all: true,
                    entries: Vec::new(),
                });
                continue;
            }
            let entries = raw
                .entries
                .into_iter()
                .map(|(start, end, tokens)| {
                    let ptr = Arc::as_ptr(tokens) as usize;
                    let token_set = if let Some(&existing) = ts_ptr_to_idx.get(&ptr) {
                        existing
                    } else {
                        let idx = token_set_pool.len() as u32;
                        ts_ptr_to_idx.insert(ptr, idx);
                        token_set_pool.push(Arc::clone(tokens));
                        idx
                    };
                    (start, end, token_set)
                })
                .collect();
            weight_pool.push(WeightPoolEntry {
                all: false,
                entries,
            });
        }
        if let Some(total_started) = total_started {
            eprintln!(
                "[glrmask/profile][dwa_pooled_parts] row_classes_ms={row_classes_ms:.3} weight_discovery_ms={weight_discovery_ms:.3} weight_materialize_ms={weight_materialize_ms:.3} token_set_intern_ms={:.3} total_ms={:.3} unique_rows={} weights={} token_sets={}",
                phase_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
                total_started.elapsed().as_secs_f64() * 1000.0,
                transition_rows.len(),
                weight_pool.len(),
                token_set_pool.len(),
            );
        }

        (
            token_set_pool,
            weight_pool,
            transition_rows,
            state_rows,
            transition_count,
        )
    }

    fn to_packed_bytes(&self) -> Vec<u8> {
        let profile = std::env::var_os("GLRMASK_PROFILE_DWA_SERIALIZATION").is_some();
        let total_started = profile.then(std::time::Instant::now);
        let phase_started = profile.then(std::time::Instant::now);
        let (token_set_arcs, mut weight_pool, transition_rows, state_rows, transition_count) =
            self.packed_pooled_parts();
        let pooled_ms = phase_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let phase_started = profile.then(std::time::Instant::now);

        // Adjacent token sets are often almost identical. Sort them
        // lexicographically, remap weight references, and front-code them in
        // independently decodable chunks. Chunk boundaries preserve parallel
        // load while sacrificing very little prefix sharing.
        let sort_token_sets = token_set_arcs.len() >= 1_024;
        let token_set_pool = if !sort_token_sets {
            Vec::new()
        } else if rayon::current_num_threads() > 1 {
            token_set_arcs
                .par_iter()
                .map(|ts| ts.ranges().map(|r| [*r.start(), *r.end()]).collect::<EncodedTokenSet>())
                .collect::<Vec<_>>()
        } else {
            token_set_arcs
                .iter()
                .map(|ts| ts.ranges().map(|r| [*r.start(), *r.end()]).collect::<EncodedTokenSet>())
                .collect::<Vec<_>>()
        };
        let token_set_materialize_ms = phase_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let phase_started = profile.then(std::time::Instant::now);
        // Lexicographic ordering improves front-coding for large token-set
        // pools, but for ordinary schema DWAs the O(n log n) sort/remap cost is
        // a material fraction of the entire save. Preserve stable discovery
        // order below 1K sets; it is already locality-friendly because sets are
        // discovered by Weight/range traversal.
        let mut token_set_order = (0..token_set_pool.len()).collect::<Vec<_>>();
        if sort_token_sets {
            let compare_token_sets = |left: &usize, right: &usize| {
                token_set_pool[*left].cmp(&token_set_pool[*right])
            };
            if token_set_order.len() >= 4_096 && rayon::current_num_threads() > 1 {
                token_set_order.par_sort_unstable_by(compare_token_sets);
            } else {
                token_set_order.sort_unstable_by(compare_token_sets);
            }
            let mut old_to_new_token_set = vec![0u32; token_set_pool.len()];
            for (new_index, &old_index) in token_set_order.iter().enumerate() {
                old_to_new_token_set[old_index] = new_index as u32;
            }
            for weight in &mut weight_pool {
                for (_, _, token_set) in &mut weight.entries {
                    *token_set = old_to_new_token_set[*token_set as usize];
                }
            }
        }
        let mut token_set_slots = token_set_pool.into_iter().map(Some).collect::<Vec<_>>();
        let token_set_sort_remap_ms = phase_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let phase_started = profile.then(std::time::Instant::now);
        let token_set_pool = token_set_order
            .into_iter()
            .map(|old_index| {
                token_set_slots[old_index]
                    .take()
                    .expect("token-set permutation contains each index once")
            })
            .collect::<Vec<_>>();

        // Pool the outer TSID range geometry independently from the token-set
        // values. Parser weights often differ only in which token set is
        // attached to an otherwise identical TSID partition.
        let mut geometry_hash_to_indices = FxHashMap::<u64, Vec<u32>>::default();
        let mut weight_geometries = Vec::<Vec<(u32, u32)>>::new();
        let mut weight_geometry_ids = Vec::<u32>::with_capacity(weight_pool.len());
        for weight in &weight_pool {
            if weight.all {
                weight_geometry_ids.push(u32::MAX);
                continue;
            }
            let geometry = weight
                .entries
                .iter()
                .map(|&(start, end, _)| (start, end))
                .collect::<Vec<_>>();
            let mut hasher = FxHasher::default();
            geometry.hash(&mut hasher);
            let hash = hasher.finish();
            let mut existing = None;
            if let Some(candidates) = geometry_hash_to_indices.get(&hash) {
                for &candidate in candidates {
                    if weight_geometries[candidate as usize] == geometry {
                        existing = Some(candidate);
                        break;
                    }
                }
            }
            let geometry_id = if let Some(existing) = existing {
                existing
            } else {
                let index = weight_geometries.len() as u32;
                weight_geometries.push(geometry);
                geometry_hash_to_indices.entry(hash).or_default().push(index);
                index
            };
            weight_geometry_ids.push(geometry_id);
        }
        let geometry_ms = phase_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let phase_started = profile.then(std::time::Instant::now);

        // Row pooling was fused into packed_pooled_parts(), before Weight-id
        // translation, so duplicate state rows never become duplicate serde
        // transition vectors.

        let mut out = Vec::new();
        let mut body = Vec::new();
        // Format tag, so accidental cross-version use fails cleanly rather
        // than silently constructing nonsense.
        out.extend_from_slice(b"DWP6");
        put_var_u32(&mut out, self.start_state);

        let token_set_count = token_set_arcs.len();
        put_var_u32(&mut out, token_set_count as u32);
        const TOKEN_SET_CHUNK_SIZE: usize = 64;
        let token_set_chunk_count = token_set_count.div_ceil(TOKEN_SET_CHUNK_SIZE);
        let token_set_range_count = if sort_token_sets {
            token_set_pool.iter().map(Vec::len).sum::<usize>()
        } else {
            token_set_arcs.iter().map(|set| set.ranges().len()).sum::<usize>()
        };
        put_var_u32(&mut out, token_set_chunk_count as u32);
        let encode_token_set_chunk = |chunk: &[EncodedTokenSet]| {
            let mut encoded = Vec::new();
            put_var_u32(&mut encoded, chunk.len() as u32);
            let mut previous: &[[u32; 2]] = &[];
            for token_set in chunk {
                let prefix_len = previous
                    .iter()
                    .zip(token_set)
                    .take_while(|(left, right)| left == right)
                    .count();
                put_var_u32(&mut encoded, prefix_len as u32);
                put_var_u32(&mut encoded, (token_set.len() - prefix_len) as u32);
                let mut previous_end_plus_one = if prefix_len == 0 {
                    0u64
                } else {
                    previous[prefix_len - 1][1] as u64 + 1
                };
                for &[start, end] in &token_set[prefix_len..] {
                    let start64 = start as u64;
                    let gap = start64
                        .checked_sub(previous_end_plus_one)
                        .expect("token-set ranges are sorted and disjoint");
                    put_var_u64(&mut encoded, gap);
                    put_var_u32(&mut encoded, end - start);
                    previous_end_plus_one = end as u64 + 1;
                }
                previous = token_set;
            }
            encoded
        };
        let encode_token_set_arc_chunk = |chunk: &[Arc<RangeSetBlaze<u32>>]| {
            let mut encoded = Vec::new();
            put_var_u32(&mut encoded, chunk.len() as u32);
            // The direct Arc path used to walk each RangeSetBlaze three or
            // four times (prefix comparison, len, nth, then skip). Ordinary
            // schema token sets are small enough that two reusable flat range
            // buffers are much cheaper: traverse each set exactly once, reuse
            // the allocations for the whole chunk, and front-code the slices.
            let mut previous = EncodedTokenSet::new();
            let mut current = EncodedTokenSet::new();
            for token_set in chunk {
                current.clear();
                current.extend(
                    token_set
                        .ranges()
                        .map(|range| [*range.start(), *range.end()]),
                );
                let prefix_len = previous
                    .iter()
                    .zip(&current)
                    .take_while(|(left, right)| left == right)
                    .count();
                put_var_u32(&mut encoded, prefix_len as u32);
                put_var_u32(&mut encoded, (current.len() - prefix_len) as u32);
                let mut previous_end_plus_one = if prefix_len == 0 {
                    0u64
                } else {
                    previous[prefix_len - 1][1] as u64 + 1
                };
                for &[start, end] in &current[prefix_len..] {
                    let gap = (start as u64)
                        .checked_sub(previous_end_plus_one)
                        .expect("token-set ranges are sorted and disjoint");
                    put_var_u64(&mut encoded, gap);
                    put_var_u32(&mut encoded, end - start);
                    previous_end_plus_one = end as u64 + 1;
                }
                std::mem::swap(&mut previous, &mut current);
            }
            encoded
        };
        if !sort_token_sets {
            // The p99-ish ordinary schemas have only hundreds of token sets
            // but tens of thousands of ranges. At that point range encoding,
            // not scheduling, dominates. Encode independent front-coding
            // chunks in parallel; smaller p50/p90 pools stay serial.
            if token_set_range_count >= 16_384 && rayon::current_num_threads() > 1 {
                let encoded_chunks = token_set_arcs
                    .par_chunks(TOKEN_SET_CHUNK_SIZE)
                    .map(encode_token_set_arc_chunk)
                    .collect::<Vec<_>>();
                for encoded in encoded_chunks {
                    put_var_u32(&mut out, encoded.len() as u32);
                    out.extend_from_slice(&encoded);
                }
            } else {
                for chunk in token_set_arcs.chunks(TOKEN_SET_CHUNK_SIZE) {
                    body = encode_token_set_arc_chunk(chunk);
                    put_var_u32(&mut out, body.len() as u32);
                    out.extend_from_slice(&body);
                }
            }
        } else if token_set_range_count >= 1_000_000 && rayon::current_num_threads() > 1 {
            let encoded_chunks = token_set_pool
                .par_chunks(TOKEN_SET_CHUNK_SIZE)
                .map(encode_token_set_chunk)
                .collect::<Vec<_>>();
            for encoded in encoded_chunks {
                put_var_u32(&mut out, encoded.len() as u32);
                out.extend_from_slice(&encoded);
            }
        } else {
            for chunk in token_set_pool.chunks(TOKEN_SET_CHUNK_SIZE) {
                body = encode_token_set_chunk(chunk);
                put_var_u32(&mut out, body.len() as u32);
                out.extend_from_slice(&body);
            }
        }
        let token_sets_end = out.len();
        let token_set_encode_ms = phase_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let phase_started = profile.then(std::time::Instant::now);

        put_var_u32(&mut out, weight_geometries.len() as u32);
        for geometry in &weight_geometries {
            body.clear();
            put_var_u32(&mut body, geometry.len() as u32);
            let mut previous_end_plus_one = 0u64;
            for &(start, end) in geometry {
                let gap = (start as u64)
                    .checked_sub(previous_end_plus_one)
                    .expect("weight geometry is sorted and disjoint");
                put_var_u64(&mut body, gap);
                put_var_u32(&mut body, end - start);
                previous_end_plus_one = end as u64 + 1;
            }
            put_var_u32(&mut out, body.len() as u32);
            out.extend_from_slice(&body);
        }
        let weight_geometries_end = out.len();
        let geometry_encode_ms = phase_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let phase_started = profile.then(std::time::Instant::now);

        put_var_u32(&mut out, weight_pool.len() as u32);
        for (weight_index, weight) in weight_pool.iter().enumerate() {
            body.clear();
            if weight.all {
                body.push(1);
            } else {
                body.push(0);
                put_var_u32(&mut body, weight_geometry_ids[weight_index]);
                for &(_, _, token_set) in &weight.entries {
                    put_var_u32(&mut body, token_set);
                }
            }
            put_var_u32(&mut out, body.len() as u32);
            out.extend_from_slice(&body);
        }
        let weights_end = out.len();
        let weight_encode_ms = phase_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let phase_started = profile.then(std::time::Instant::now);

        let unique_row_transition_count = transition_rows.iter().map(Vec::len).sum::<usize>();
        let use_component_rows = unique_row_transition_count >= 100_000
            && std::env::var_os("GLRMASK_DWA_DIRECT_ARTIFACT_ROWS").is_none();
        out.push(u8::from(use_component_rows));
        if use_component_rows {
            let mut label_ids = FxHashMap::<u64, Vec<u32>>::default();
            let mut target_ids = FxHashMap::<u64, Vec<u32>>::default();
            let mut weight_ids = FxHashMap::<u64, Vec<u32>>::default();
            label_ids.reserve(512);
            target_ids.reserve(transition_rows.len());
            weight_ids.reserve(transition_rows.len());
            let mut label_pool = Vec::<Vec<Label>>::new();
            let mut target_pool = Vec::<Vec<u32>>::new();
            let mut weight_id_pool = Vec::<Vec<u32>>::new();
            let mut row_components = Vec::<(u32, u32, u32)>::with_capacity(transition_rows.len());

            for row in &transition_rows {
                let mut label_hasher = FxHasher::default();
                let mut target_hasher = FxHasher::default();
                let mut weight_hasher = FxHasher::default();
                row.len().hash(&mut label_hasher);
                row.len().hash(&mut target_hasher);
                row.len().hash(&mut weight_hasher);
                for &(label, target, weight) in row {
                    label.hash(&mut label_hasher);
                    target.hash(&mut target_hasher);
                    weight.hash(&mut weight_hasher);
                }

                let label_hash = label_hasher.finish();
                let label_id = label_ids
                    .get(&label_hash)
                    .and_then(|candidates| {
                        candidates.iter().copied().find(|&candidate| {
                            let existing = &label_pool[candidate as usize];
                            existing.len() == row.len()
                                && existing
                                    .iter()
                                    .zip(row)
                                    .all(|(&value, &(label, _, _))| value == label)
                        })
                    })
                    .unwrap_or_else(|| {
                        let id = label_pool.len() as u32;
                        label_pool.push(row.iter().map(|&(label, _, _)| label).collect());
                        label_ids.entry(label_hash).or_default().push(id);
                        id
                    });

                let target_hash = target_hasher.finish();
                let target_id = target_ids
                    .get(&target_hash)
                    .and_then(|candidates| {
                        candidates.iter().copied().find(|&candidate| {
                            let existing = &target_pool[candidate as usize];
                            existing.len() == row.len()
                                && existing
                                    .iter()
                                    .zip(row)
                                    .all(|(&value, &(_, target, _))| value == target)
                        })
                    })
                    .unwrap_or_else(|| {
                        let id = target_pool.len() as u32;
                        target_pool.push(row.iter().map(|&(_, target, _)| target).collect());
                        target_ids.entry(target_hash).or_default().push(id);
                        id
                    });

                let weight_hash = weight_hasher.finish();
                let weight_id = weight_ids
                    .get(&weight_hash)
                    .and_then(|candidates| {
                        candidates.iter().copied().find(|&candidate| {
                            let existing = &weight_id_pool[candidate as usize];
                            existing.len() == row.len()
                                && existing
                                    .iter()
                                    .zip(row)
                                    .all(|(&value, &(_, _, weight))| value == weight)
                        })
                    })
                    .unwrap_or_else(|| {
                        let id = weight_id_pool.len() as u32;
                        weight_id_pool.push(row.iter().map(|&(_, _, weight)| weight).collect());
                        weight_ids.entry(weight_hash).or_default().push(id);
                        id
                    });
                row_components.push((label_id, target_id, weight_id));
            }

            put_var_u32(&mut out, label_pool.len() as u32);
            for labels in &label_pool {
                body.clear();
                put_var_u32(&mut body, labels.len() as u32);
                let mut previous = 0i64;
                for &label in labels {
                    let label64 = label as i64;
                    put_var_i64(&mut body, label64 - previous);
                    previous = label64;
                }
                put_var_u32(&mut out, body.len() as u32);
                out.extend_from_slice(&body);
            }

            let mut encode_u32_pool = |pool: &[Vec<u32>]| {
                put_var_u32(&mut out, pool.len() as u32);
                if pool.len() >= 1_024 && rayon::current_num_threads() > 1 {
                    let encoded = pool
                        .par_iter()
                        .map(|sequence| {
                            let absolute_len = sequence
                                .iter()
                                .map(|&value| var_u64_len(value as u64))
                                .sum::<usize>();
                            let mut previous = 0i64;
                            let delta_len = sequence
                                .iter()
                                .map(|&value| {
                                    let value = value as i64;
                                    let len = var_u64_len(zigzag_i64(value - previous));
                                    previous = value;
                                    len
                                })
                                .sum::<usize>();
                            let use_delta = delta_len < absolute_len;
                            let mut encoded = Vec::with_capacity(
                                1 + var_u64_len(sequence.len() as u64)
                                    + absolute_len.min(delta_len),
                            );
                            encoded.push(u8::from(use_delta));
                            put_var_u32(&mut encoded, sequence.len() as u32);
                            if use_delta {
                                let mut previous = 0i64;
                                for &value in sequence {
                                    let value = value as i64;
                                    put_var_i64(&mut encoded, value - previous);
                                    previous = value;
                                }
                            } else {
                                for &value in sequence {
                                    put_var_u32(&mut encoded, value);
                                }
                            }
                            encoded
                        })
                        .collect::<Vec<_>>();
                    for encoded in encoded {
                        put_var_u32(&mut out, encoded.len() as u32);
                        out.extend_from_slice(&encoded);
                    }
                    return;
                }
                for sequence in pool {
                    body.clear();
                    let absolute_len = sequence
                        .iter()
                        .map(|&value| var_u64_len(value as u64))
                        .sum::<usize>();
                    let mut previous = 0i64;
                    let delta_len = sequence
                        .iter()
                        .map(|&value| {
                            let value = value as i64;
                            let len = var_u64_len(zigzag_i64(value - previous));
                            previous = value;
                            len
                        })
                        .sum::<usize>();
                    let use_delta = delta_len < absolute_len;
                    body.push(u8::from(use_delta));
                    put_var_u32(&mut body, sequence.len() as u32);
                    if use_delta {
                        let mut previous = 0i64;
                        for &value in sequence {
                            let value = value as i64;
                            put_var_i64(&mut body, value - previous);
                            previous = value;
                        }
                    } else {
                        for &value in sequence {
                            put_var_u32(&mut body, value);
                        }
                    }
                    put_var_u32(&mut out, body.len() as u32);
                    out.extend_from_slice(&body);
                }
            };
            encode_u32_pool(&target_pool);
            encode_u32_pool(&weight_id_pool);

            put_var_u32(&mut out, row_components.len() as u32);
            for &(labels, targets, weights) in &row_components {
                put_var_u32(&mut out, labels);
                put_var_u32(&mut out, targets);
                put_var_u32(&mut out, weights);
            }
        } else {
            put_var_u32(&mut out, transition_rows.len() as u32);
            for row in &transition_rows {
                body.clear();
                put_var_u32(&mut body, row.len() as u32);
                let mut previous_label = 0i64;
                for &(label, target, weight) in row {
                    let label64 = label as i64;
                    put_var_i64(&mut body, label64 - previous_label);
                    previous_label = label64;
                    put_var_u32(&mut body, target);
                    put_var_u32(&mut body, weight);
                }
                put_var_u32(&mut out, body.len() as u32);
                out.extend_from_slice(&body);
            }
        }
        let rows_end = out.len();
        let row_encode_ms = phase_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let phase_started = profile.then(std::time::Instant::now);

        put_var_u32(&mut out, state_rows.len() as u32);
        for &(row, final_weight) in &state_rows {
            put_var_u32(&mut out, row);
            put_var_u32(
                &mut out,
                final_weight.map_or(0, |weight| weight.saturating_add(1)),
            );
        }
        let state_encode_ms = phase_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        if profile {
            let weight_entries: usize = weight_pool.iter().map(|w| w.entries.len()).sum();
            let token_ranges = token_set_range_count;
            eprintln!(
                "[glrmask/profile][dwa_packed_phases] pooled_ms={pooled_ms:.3} token_set_materialize_ms={token_set_materialize_ms:.3} token_set_sort_remap_ms={token_set_sort_remap_ms:.3} geometry_ms={geometry_ms:.3} token_set_encode_ms={token_set_encode_ms:.3} geometry_encode_ms={geometry_encode_ms:.3} weight_encode_ms={weight_encode_ms:.3} row_encode_ms={row_encode_ms:.3} state_encode_ms={state_encode_ms:.3} total_ms={:.3}",
                total_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
            );
            eprintln!(
                "[glrmask/profile][dwa_packed] bytes={} token_set_bytes={} weight_geometry_bytes={} weight_bytes={} row_bytes={} state_ref_bytes={} states={} unique_rows={} transitions={} weights={} weight_geometries={} weight_entries={} token_sets={} token_ranges={}",
                out.len(),
                token_sets_end - 4,
                weight_geometries_end - token_sets_end,
                weights_end - weight_geometries_end,
                rows_end - weights_end,
                out.len() - rows_end,
                state_rows.len(),
                transition_rows.len(),
                transition_count,
                weight_pool.len(),
                weight_geometries.len(),
                weight_entries,
                token_set_count,
                token_ranges,
            );
        }
        out
    }

    fn from_packed_bytes(input: &[u8]) -> Result<Self, String> {
        PACKED_DWA_TOKEN_SET_INVENTORY.with(|slot| *slot.borrow_mut() = None);
        if !input.starts_with(b"DWP6") {
            return Err("invalid packed DWA header".to_owned());
        }
        let mut pos = 4usize;
        let start_state = take_var_u32(input, &mut pos)?;
        let profile = std::env::var_os("GLRMASK_PROFILE_DWA_SERIALIZATION").is_some();
        let token_sets_started = profile.then(std::time::Instant::now);

        let token_set_count = take_var_u32(input, &mut pos)? as usize;
        let token_set_chunk_count = take_var_u32(input, &mut pos)? as usize;
        let token_set_chunks = take_length_prefixed_slices(
            input,
            &mut pos,
            token_set_chunk_count,
            "token-set chunk",
        )?;
        let decoded_chunks: Vec<_> = if token_set_count >= 1024 && rayon::current_num_threads() > 1 {
            token_set_chunks
                .par_iter()
                .map(|body| decode_packed_token_set_chunk(body))
                .collect::<Result<Vec<_>, _>>()?
        } else {
            token_set_chunks
                .iter()
                .map(|body| decode_packed_token_set_chunk(body))
                .collect::<Result<Vec<_>, _>>()?
        };
        let decoded_token_sets = decoded_chunks.into_iter().flatten().collect::<Vec<_>>();
        let mut ts_pool = Vec::with_capacity(decoded_token_sets.len());
        let mut token_set_word_spans = Vec::with_capacity(decoded_token_sets.len());
        for (tokens, word_spans) in decoded_token_sets {
            ts_pool.push(tokens);
            token_set_word_spans.push(word_spans);
        }
        if ts_pool.len() != token_set_count {
            return Err("invalid packed DWA token-set count".to_owned());
        }
        let token_sets_ms = token_sets_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let geometries_started = profile.then(std::time::Instant::now);

        let geometry_count = take_var_u32(input, &mut pos)? as usize;
        let geometry_bodies =
            take_length_prefixed_slices(input, &mut pos, geometry_count, "weight geometry")?;
        let geometries: Vec<_> = if geometry_count >= 1024 && rayon::current_num_threads() > 1 {
            geometry_bodies
                .par_iter()
                .map(|body| decode_packed_weight_geometry(body))
                .collect::<Result<Vec<_>, _>>()?
        } else {
            geometry_bodies
                .iter()
                .map(|body| decode_packed_weight_geometry(body))
                .collect::<Result<Vec<_>, _>>()?
        };
        let geometries_ms = geometries_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let weights_started = profile.then(std::time::Instant::now);

        let weight_count = take_var_u32(input, &mut pos)? as usize;
        let weight_bodies =
            take_length_prefixed_slices(input, &mut pos, weight_count, "weight")?;
        let w_pool: Vec<_> = if weight_count >= 1024 && rayon::current_num_threads() > 1 {
            weight_bodies
                .par_iter()
                .map(|body| decode_packed_weight(body, &ts_pool, &geometries))
                .collect::<Result<Vec<_>, _>>()?
        } else {
            weight_bodies
                .iter()
                .map(|body| decode_packed_weight(body, &ts_pool, &geometries))
                .collect::<Result<Vec<_>, _>>()?
        };
        let w_pool = Arc::<[Weight]>::from(w_pool);
        let weights_ms = weights_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let rows_started = profile.then(std::time::Instant::now);
        let row_mode = *input
            .get(pos)
            .ok_or_else(|| "truncated packed DWA row mode".to_owned())?;
        pos += 1;
        let (transition_rows, state_count_pos, transition_weight_used) = match row_mode {
            0 => {
                let row_count = take_var_u32(input, &mut pos)? as usize;
                let row_bodies = take_length_prefixed_slices(
                    input,
                    &mut pos,
                    row_count,
                    "transition row",
                )?;
                let state_count_pos = pos;
                let state_count = take_var_u32(input, &mut pos)? as usize;
                let rows: Vec<_> =
                    if row_count >= 1024 && rayon::current_num_threads() > 1 {
                        row_bodies
                            .par_iter()
                            .map(|body| decode_packed_transition_row(body, &w_pool, state_count))
                            .collect::<Result<Vec<_>, _>>()?
                    } else {
                        row_bodies
                            .iter()
                            .map(|body| decode_packed_transition_row(body, &w_pool, state_count))
                            .collect::<Result<Vec<_>, _>>()?
                    };
                (
                    rows.into_iter()
                        .map(|row| DwaTransitionMap::Shared(Arc::new(row)))
                        .collect(),
                    state_count_pos,
                    None,
                )
            }
            1 => {
                let label_count = take_var_u32(input, &mut pos)? as usize;
                let label_bodies = take_length_prefixed_slices(
                    input,
                    &mut pos,
                    label_count,
                    "label sequence",
                )?;
                let label_pool: Vec<_> = label_bodies
                    .iter()
                    .map(|body| decode_packed_label_sequence(body))
                    .collect::<Result<Vec<_>, _>>()?;

                let target_count = take_var_u32(input, &mut pos)? as usize;
                let target_bodies = take_length_prefixed_slices(
                    input,
                    &mut pos,
                    target_count,
                    "target sequence",
                )?;
                let weight_id_count = take_var_u32(input, &mut pos)? as usize;
                let weight_id_bodies = take_length_prefixed_slices(
                    input,
                    &mut pos,
                    weight_id_count,
                    "weight-id sequence",
                )?;
                let decode_targets = || {
                    if target_count >= 1024 && rayon::current_num_threads() > 1 {
                        target_bodies
                            .par_iter()
                            .map(|body| decode_packed_u32_sequence(body, "target"))
                            .collect::<Result<Vec<_>, _>>()
                    } else {
                        target_bodies
                            .iter()
                            .map(|body| decode_packed_u32_sequence(body, "target"))
                            .collect::<Result<Vec<_>, _>>()
                    }
                };
                let decode_weight_ids = || {
                    if weight_id_count >= 1024 && rayon::current_num_threads() > 1 {
                        weight_id_bodies
                            .par_iter()
                            .map(|body| decode_packed_u32_sequence(body, "weight-id"))
                            .collect::<Result<Vec<_>, _>>()
                    } else {
                        weight_id_bodies
                            .iter()
                            .map(|body| decode_packed_u32_sequence(body, "weight-id"))
                            .collect::<Result<Vec<_>, _>>()
                    }
                };
                let (target_pool, weight_id_pool) = if rayon::current_num_threads() > 1 {
                    let (targets, weights) = rayon::join(decode_targets, decode_weight_ids);
                    (targets?, weights?)
                } else {
                    (decode_targets()?, decode_weight_ids()?)
                };
                let mut transition_weight_used = vec![false; w_pool.len()];
                for sequence in &weight_id_pool {
                    for &weight in sequence {
                        let Some(used) = transition_weight_used.get_mut(weight as usize) else {
                            return Err("invalid packed DWA weight index".to_owned());
                        };
                        *used = true;
                    }
                }

                let row_count = take_var_u32(input, &mut pos)? as usize;
                let mut row_components = Vec::with_capacity(row_count);
                for _ in 0..row_count {
                    let labels = take_var_u32(input, &mut pos)? as usize;
                    let targets = take_var_u32(input, &mut pos)? as usize;
                    let weights = take_var_u32(input, &mut pos)? as usize;
                    if labels >= label_pool.len()
                        || targets >= target_pool.len()
                        || weights >= weight_id_pool.len()
                    {
                        return Err("invalid packed DWA row-component index".to_owned());
                    }
                    row_components.push((labels, targets, weights));
                }
                let state_count_pos = pos;
                let state_count = take_var_u32(input, &mut pos)? as usize;
                for labels in &label_pool {
                    if labels.windows(2).any(|pair| pair[0] >= pair[1]) {
                        return Err(
                            "packed DWA transition labels are not strictly increasing".to_owned(),
                        );
                    }
                }
                for targets in &target_pool {
                    if targets.iter().any(|&target| target as usize >= state_count) {
                        return Err("invalid packed DWA target state".to_owned());
                    }
                }
                for &(label_id, target_id, weight_id) in &row_components {
                    let labels = &label_pool[label_id];
                    let targets = &target_pool[target_id];
                    let weights = &weight_id_pool[weight_id];
                    if labels.len() != targets.len() || labels.len() != weights.len() {
                        return Err("mismatched packed DWA row-component lengths".to_owned());
                    }
                }
                let pool = Arc::new(PackedDwaRowPool {
                    labels: label_pool,
                    targets: target_pool,
                    weight_ids: weight_id_pool,
                    weights: Arc::clone(&w_pool),
                });
                let rows: Vec<DwaTransitionMap> = row_components
                    .into_iter()
                    .map(|(label_id, target_id, weight_id)| {
                        DwaTransitionMap::Packed(Arc::new(PackedDwaTransitionRow {
                            pool: Arc::clone(&pool),
                            label_id,
                            target_id,
                            weight_id,
                            materialized: OnceLock::new(),
                        }))
                    })
                    .collect();
                (rows, state_count_pos, Some(transition_weight_used))
            }
            _ => return Err("invalid packed DWA row mode".to_owned()),
        };
        let rows_ms = rows_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);

        // Rewind to the state-count varint and decode compact row/final refs.
        pos = state_count_pos;
        let state_count = take_var_u32(input, &mut pos)? as usize;
        let state_refs_started = profile.then(std::time::Instant::now);
        let mut state_refs = Vec::with_capacity(state_count);
        let mut final_weight_used = vec![false; w_pool.len()];
        for _ in 0..state_count {
            let row = take_var_u32(input, &mut pos)? as usize;
            if row >= transition_rows.len() {
                return Err("invalid packed DWA transition-row index".to_owned());
            }
            let final_encoded = take_var_u32(input, &mut pos)?;
            let final_weight = if final_encoded == 0 {
                None
            } else {
                let index = (final_encoded - 1) as usize;
                if index >= w_pool.len() {
                    return Err("invalid packed DWA final-weight index".to_owned());
                }
                final_weight_used[index] = true;
                Some(index)
            };
            state_refs.push((row, final_weight));
        }
        if pos != input.len() {
            return Err("trailing bytes in packed DWA".to_owned());
        }
        let build_inventory = || {
            let transition_weight_used = transition_weight_used.as_deref()?;
            let mut transition_token_used = vec![false; ts_pool.len()];
            let mut final_token_used = vec![false; ts_pool.len()];
            // The Weight bodies were already fully validated while w_pool was
            // decoded.  Rescan only their compact token-set IDs here instead
            // of retaining per-Weight ID vectors or walking reconstructed
            // RangeMapBlaze values and hashing the same pointers repeatedly.
            for (weight_index, body) in weight_bodies.iter().enumerate() {
                let mark_transition = transition_weight_used[weight_index];
                let mark_final = final_weight_used[weight_index];
                if !mark_transition && !mark_final {
                    continue;
                }
                let mut body_pos = 0usize;
                let tag = body[body_pos];
                body_pos += 1;
                if tag == 1 {
                    continue;
                }
                debug_assert_eq!(tag, 0);
                let geometry_index = take_var_u32(body, &mut body_pos)
                    .expect("packed Weight body was validated during decode")
                    as usize;
                for _ in &geometries[geometry_index] {
                    let token_set = take_var_u32(body, &mut body_pos)
                        .expect("packed Weight body was validated during decode")
                        as usize;
                    if mark_transition {
                        transition_token_used[token_set] = true;
                    }
                    if mark_final {
                        final_token_used[token_set] = true;
                    }
                }
            }
            let transition_set_count = transition_token_used.iter().filter(|&&used| used).count();
            let final_set_count = final_token_used.iter().filter(|&&used| used).count();
            let mut transition_sets = FxHashMap::default();
            transition_sets.reserve(transition_set_count);
            let mut transition_word_spans = FxHashMap::default();
            transition_word_spans.reserve(transition_set_count);
            let mut final_sets = FxHashMap::default();
            final_sets.reserve(final_set_count);
            for (index, token_set) in ts_pool.iter().enumerate() {
                let key = Arc::as_ptr(token_set) as usize;
                if transition_token_used[index] {
                    transition_sets.insert(key, Arc::clone(token_set));
                    transition_word_spans.insert(key, token_set_word_spans[index]);
                }
                if final_token_used[index] {
                    final_sets.insert(key, Arc::clone(token_set));
                }
            }
            Some(PackedDwaTokenSetInventory {
                transition_sets,
                final_sets,
                transition_word_spans,
            })
        };
        let build_state = |&(row, final_weight): &(usize, Option<usize>)| DWAState {
            transitions: transition_rows[row].clone(),
            final_weight: final_weight.map(|index| w_pool[index].clone()),
        };
        let build_states = || {
            if state_count >= 1024 && rayon::current_num_threads() > 1 {
                state_refs.par_iter().map(build_state).collect()
            } else {
                state_refs.iter().map(build_state).collect()
            }
        };
        let (states, inventory): (Vec<_>, _) = if transition_weight_used.is_some()
            && state_count >= 1024
            && rayon::current_num_threads() > 1
        {
            rayon::join(build_states, build_inventory)
        } else {
            (build_states(), build_inventory())
        };
        if let Some(inventory) = inventory {
            PACKED_DWA_TOKEN_SET_INVENTORY.with(|slot| {
                *slot.borrow_mut() = Some(inventory);
            });
        }
        let state_refs_ms = state_refs_started
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        if start_state as usize >= states.len() && !states.is_empty() {
            return Err("invalid packed DWA start state".to_owned());
        }
        if profile {
            eprintln!(
                "[glrmask/profile][dwa_packed_load] bytes={} token_sets_ms={token_sets_ms:.3} geometries_ms={geometries_ms:.3} weights_ms={weights_ms:.3} rows_ms={rows_ms:.3} state_refs_ms={state_refs_ms:.3}",
                input.len(),
            );
        }
        Ok(Self {
            states,
            start_state,
            shared_transition_rows: true,
            transition_count_cache: OnceLock::new(),
            acyclic_cache: OnceLock::new(),
        })
    }
}

impl Serialize for DWA {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        if external_serde_enabled() {
            return 0u8.serialize(serializer);
        }
        if packed_serde_enabled() {
            return self.to_packed_bytes().serialize(serializer);
        }
        let (token_set_pool, weight_pool, states) = self.pooled_parts();

        let serde_repr = DWASerde {
            token_set_pool,
            weight_pool,
            states,
            start_state: self.start_state,
        };
        if std::env::var_os("GLRMASK_PROFILE_DWA_SERIALIZATION").is_some() {
            let transition_count: usize = serde_repr.states.iter().map(|s| s.transitions.len()).sum();
            let weight_entries: usize = serde_repr.weight_pool.iter().map(|w| w.entries.len()).sum();
            let token_ranges: usize = serde_repr.token_set_pool.iter().map(Vec::len).sum();
            eprintln!(
                "[glrmask/profile][dwa_serde] states={} transitions={} weights={} weight_entries={} token_sets={} token_ranges={}",
                serde_repr.states.len(),
                transition_count,
                serde_repr.weight_pool.len(),
                weight_entries,
                serde_repr.token_set_pool.len(),
                token_ranges,
            );
        }
        serde_repr.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for DWA {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        if external_serde_enabled() {
            let marker = u8::deserialize(deserializer)?;
            if marker != 0 {
                return Err(serde::de::Error::custom("invalid external DWA placeholder"));
            }
            return Ok(Self::new(0, 0));
        }
        if packed_serde_enabled() {
            let bytes = Vec::<u8>::deserialize(deserializer)?;
            return Self::from_packed_bytes(&bytes).map_err(serde::de::Error::custom);
        }
        let serde_repr = DWASerde::deserialize(deserializer)?;

        // Reconstruct token set pool (shared Arcs)
        let ts_pool: Vec<std::sync::Arc<RangeSetBlaze<u32>>> = serde_repr
            .token_set_pool
            .into_iter()
            .map(|encoded| {
                // Serialized token sets come directly from RangeSetBlaze::ranges(),
                // hence they are already sorted and disjoint.  Bypass the generic
                // normalization path when rebuilding millions of ranges.
                let rs = RangeSetBlaze::from_sorted_disjoint(CheckSortedDisjoint::new(
                    encoded.into_iter().map(|[s, e]| s..=e),
                ));
                shared_rangeset(rs)
            })
            .collect();

        // Reconstruct weight pool
        let w_pool: Vec<Weight> = serde_repr
            .weight_pool
            .into_iter()
            .map(|entry| {
                if entry.all {
                    return Weight::all();
                }
                if entry.entries.is_empty() {
                    return Weight::empty();
                }
                let ranges: Vec<_> = entry.entries.into_iter().map(|(start, end, ts_idx)| {
                    let tokens = ts_pool
                        .get(ts_idx as usize)
                        .cloned()
                        .unwrap_or_else(|| std::sync::Arc::new(RangeSetBlaze::new()));
                    (start..=end, tokens)
                }).collect();
                let map = RangeMapBlaze::from_sorted_disjoint_map(
                    CheckSortedDisjointMap::new(
                        ranges.iter().map(|(range, tokens)| (range.clone(), tokens)),
                    ),
                );
                finalize_weight_map(map)
            })
            .collect();

        // Reconstruct DWA states
        let states = serde_repr
            .states
            .into_iter()
            .map(|s| {
                let transitions = s
                    .transitions
                    .into_iter()
                    .map(|(label, target, weight_idx)| {
                        let weight = w_pool
                            .get(weight_idx as usize)
                            .cloned()
                            .unwrap_or_else(Weight::empty);
                        (label, (target, weight))
                    })
                    .collect::<BTreeMap<_, _>>()
                    .into();
                let final_weight = s.final_weight.map(|idx| {
                    w_pool
                        .get(idx as usize)
                        .cloned()
                        .unwrap_or_else(Weight::empty)
                });
                DWAState {
                    transitions,
                    final_weight,
                }
            })
            .collect();

        Ok(DWA {
            states,
            start_state: serde_repr.start_state,
            shared_transition_rows: false,
            transition_count_cache: OnceLock::new(),
            acyclic_cache: OnceLock::new(),
        })
    }
}

impl DWA {
    pub fn new(_num_tsids: u32, _max_token: u32) -> Self {
        Self {
            states: vec![DWAState::default()],
            start_state: 0,
            shared_transition_rows: false,
            transition_count_cache: OnceLock::new(),
            acyclic_cache: OnceLock::new(),
        }
    }

    #[inline]
    fn invalidate_graph_caches(&mut self) {
        let _ = self.transition_count_cache.take();
        let _ = self.acyclic_cache.take();
    }

    #[inline]
    pub fn states(&self) -> &[DWAState] {
        &self.states
    }

    #[inline]
    pub fn states_mut(&mut self) -> &mut Vec<DWAState> {
        self.invalidate_graph_caches();
        &mut self.states
    }

    #[inline]
    pub fn start_state(&self) -> u32 {
        self.start_state
    }

    pub fn from_parts(states: Vec<DWAState>, start_state: u32) -> Self {
        Self {
            states,
            start_state,
            shared_transition_rows: false,
            transition_count_cache: OnceLock::new(),
            acyclic_cache: OnceLock::new(),
        }
    }

    pub fn has_shared_transition_rows(&self) -> bool {
        self.shared_transition_rows
    }

    /// Canonicalize exact duplicate transition rows without changing state ids
    /// or graph topology.  Weight-coordinate remapping can legitimately turn
    /// previously shared rows back into `Owned` rows because `DerefMut` must
    /// break the Arc identity before mutating a Weight.  Once all remapping is
    /// finished, recovering row sharing is semantics-neutral and lets both the
    /// runtime fast-row builder and the artifact encoder operate once per
    /// unique row rather than once per state.
    pub fn share_exact_transition_rows_owned(mut self) -> Self {
        if self.shared_transition_rows || self.states.len() < 2 {
            return self;
        }

        let hash_row = |state: &DWAState| {
            let mut hasher = FxHasher::default();
            state.transitions.len().hash(&mut hasher);
            for (label, (target, weight)) in state.transitions.iter() {
                label.hash(&mut hasher);
                target.hash(&mut hasher);
                weight.ptr_key().hash(&mut hasher);
            }
            hasher.finish()
        };
        let hashes = if self.states.len() >= 4_096 && rayon::current_num_threads() > 1 {
            self.states.par_iter().map(hash_row).collect::<Vec<_>>()
        } else {
            self.states.iter().map(hash_row).collect::<Vec<_>>()
        };

        let mut buckets =
            FxHashMap::<u64, Vec<Arc<BTreeMap<Label, (u32, Weight)>>>>::default();
        buckets.reserve(self.states.len().min(32_768));
        for (state, hash) in self.states.iter_mut().zip(hashes) {
            let transitions = match std::mem::take(&mut state.transitions) {
                DwaTransitionMap::Owned(transitions) => transitions,
                DwaTransitionMap::Shared(transitions) => {
                    state.transitions = DwaTransitionMap::Shared(transitions);
                    continue;
                }
                DwaTransitionMap::Packed(transitions) => {
                    state.transitions = DwaTransitionMap::Packed(transitions);
                    continue;
                }
            };
            let bucket = buckets.entry(hash).or_default();
            let shared = bucket
                .iter()
                .find(|candidate| candidate.as_ref() == &transitions)
                .cloned()
                .unwrap_or_else(|| {
                    let shared = Arc::new(transitions);
                    bucket.push(Arc::clone(&shared));
                    shared
                });
            state.transitions = DwaTransitionMap::Shared(shared);
        }
        self.shared_transition_rows = true;
        self
    }

    /// Merge states whose complete local DWA representation is already
    /// identical: same final weight and the same label -> (target, weight)
    /// transition row. This is deliberately much cheaper than automaton
    /// minimization: it performs no weight algebra and no partition refinement.
    ///
    /// Replacing every reference to one exact duplicate by the other preserves
    /// the weighted language even for cyclic DWAs. A later invocation can expose
    /// additional duplicates after target ids have been remapped.
    pub fn merge_exact_duplicate_states_owned(self) -> Self {
        if self.states.len() < 2 {
            return self;
        }

        let hash_state = |state: &DWAState| {
            let mut hasher = FxHasher::default();
            // Compiler-produced Weights are globally interned, so pointer
            // identity is an extremely cheap structural fingerprint. Keep the
            // full DWAState equality check below as the authority.
            state
                .final_weight
                .as_ref()
                .map_or(0usize, Weight::ptr_key)
                .hash(&mut hasher);
            state.transitions.len().hash(&mut hasher);
            for (label, (target, weight)) in &state.transitions {
                label.hash(&mut hasher);
                target.hash(&mut hasher);
                weight.ptr_key().hash(&mut hasher);
            }
            hasher.finish()
        };
        let hashes = if self.states.len() >= 4_096 && rayon::current_num_threads() > 1 {
            self.states.par_iter().map(hash_state).collect::<Vec<_>>()
        } else {
            self.states.iter().map(hash_state).collect::<Vec<_>>()
        };

        let mut hash_buckets = FxHashMap::<u64, Vec<u32>>::default();
        let mut canonical_old = Vec::<u32>::with_capacity(self.states.len());
        let mut representatives = Vec::<u32>::new();

        for (state_id, (state, &hash)) in self.states.iter().zip(&hashes).enumerate() {
            let bucket = hash_buckets.entry(hash).or_default();
            let canonical = bucket
                .iter()
                .copied()
                .find(|&candidate| self.states[candidate as usize] == *state)
                .unwrap_or_else(|| {
                    let state_id = state_id as u32;
                    bucket.push(state_id);
                    representatives.push(state_id);
                    state_id
                });
            canonical_old.push(canonical);
        }

        if representatives.len() == self.states.len() {
            return self;
        }

        let mut old_to_new = vec![u32::MAX; self.states.len()];
        for (new_id, &old_id) in representatives.iter().enumerate() {
            old_to_new[old_id as usize] = new_id as u32;
        }
        for (old_id, &canonical) in canonical_old.iter().enumerate() {
            old_to_new[old_id] = old_to_new[canonical as usize];
        }

        let start_state = old_to_new[self.start_state as usize];
        let mut states = Vec::with_capacity(representatives.len());
        let mut row_buckets =
            FxHashMap::<u64, Vec<Arc<BTreeMap<Label, (u32, Weight)>>>>::default();
        for (old_id, mut state) in self.states.into_iter().enumerate() {
            if canonical_old[old_id] != old_id as u32 {
                continue;
            }

            // We already have to touch every surviving transition to remap its
            // target. Hash the remapped row in the same pass, then share exact
            // duplicate rows independently of final_weight. Large parser DWAs
            // commonly have far fewer unique outgoing rows than states.
            let mut row_hasher = FxHasher::default();
            state.transitions.len().hash(&mut row_hasher);
            for (label, (target, weight)) in state.transitions.iter_mut() {
                *target = old_to_new[*target as usize];
                label.hash(&mut row_hasher);
                target.hash(&mut row_hasher);
                weight.ptr_key().hash(&mut row_hasher);
            }
            let row_hash = row_hasher.finish();
            let transitions = match std::mem::take(&mut state.transitions) {
                DwaTransitionMap::Owned(transitions) => transitions,
                DwaTransitionMap::Shared(transitions) => Arc::try_unwrap(transitions)
                    .unwrap_or_else(|shared| shared.as_ref().clone()),
                DwaTransitionMap::Packed(transitions) => transitions.materialized().clone(),
            };
            let bucket = row_buckets.entry(row_hash).or_default();
            let shared = bucket
                .iter()
                .find(|candidate| candidate.as_ref() == &transitions)
                .cloned()
                .unwrap_or_else(|| {
                    let shared = Arc::new(transitions);
                    bucket.push(Arc::clone(&shared));
                    shared
                });
            state.transitions = DwaTransitionMap::Shared(shared);
            states.push(state);
        }

        Self {
            states,
            start_state,
            shared_transition_rows: true,
            transition_count_cache: OnceLock::new(),
            acyclic_cache: OnceLock::new(),
        }
    }

    pub fn set_start_state(&mut self, state: u32) {
        self.start_state = state;
    }

    pub fn add_state(&mut self) -> u32 {
        let id = self.states.len() as u32;
        self.states.push(DWAState::default());
        id
    }

    pub fn num_states(&self) -> u32 {
        self.states.len() as u32
    }

    pub fn num_transitions(&self) -> usize {
        *self.transition_count_cache.get_or_init(|| {
            self.states
                .iter()
                .map(|state| state.transitions.len())
                .sum()
        })
    }

    pub fn stats(&self) -> DwaStats {
        let mut transition_pairs = 0usize;
        let mut dsts = BTreeSet::new();
        for state in &self.states {
            dsts.clear();
            for (dst, _) in state.transitions.values() {
                dsts.insert(*dst);
            }
            transition_pairs += dsts.len();
        }

        let mut seen_weight_ptrs = BTreeSet::new();
        let mut seen_rangeset_ptrs = BTreeSet::new();
        let mut total_outer_ranges = 0usize;
        let mut total_inner_ranges = 0usize;

        let mut process_weight = |weight: &Weight| {
            let weight_ptr = weight.ptr_key();
            if seen_weight_ptrs.insert(weight_ptr) {
                total_outer_ranges += weight.raw_range_values().count();
            }
            for (_, tokens) in weight.raw_range_values() {
                let token_ptr = std::sync::Arc::as_ptr(tokens) as usize;
                if seen_rangeset_ptrs.insert(token_ptr) {
                    total_inner_ranges += tokens.ranges().count();
                }
            }
        };

        for state in &self.states {
            if let Some(final_weight) = &state.final_weight {
                process_weight(final_weight);
            }
            for (_, weight) in state.transitions.values() {
                process_weight(weight);
            }
        }

        DwaStats {
            states: self.states.len(),
            transitions: self.num_transitions(),
            transition_pairs,
            interned_ranges: total_outer_ranges + total_inner_ranges,
        }
    }

    pub fn set_final_weight(&mut self, state: u32, weight: Weight) {
        if let Some(entry) = self.states.get_mut(state as usize) {
            entry.final_weight = Some(weight);
        }
    }

    pub fn add_transition(&mut self, from: u32, label: Label, to: u32, weight: Weight) {
        self.invalidate_graph_caches();
        if let Some(entry) = self.states.get_mut(from as usize) {
            entry.transitions.insert(label, (to, weight));
        }
    }

    pub fn eval_word(&self, word: &[Label]) -> Weight {
        let mut state = self.start_state;
        let mut weight = Weight::all();
        for &label in word {
            let Some((next, edge_weight)) = self.states[state as usize].transitions.get(&label) else {
                return Weight::empty();
            };
            weight = weight.intersection(edge_weight);
            state = *next;
        }
        match self.states.get(state as usize).and_then(|state| state.final_weight.as_ref()) {
            Some(final_weight) => weight.intersection(final_weight),
            None => Weight::empty(),
        }
    }

    /// Clip all weights in the DWA so token sets contain only `0..=max_token`.
    pub fn clip_weights(&mut self, max_token: u32) {
        for state in &mut self.states {
            if let Some(fw) = &mut state.final_weight {
                fw.clip_tokens(max_token);
                if fw.is_empty() {
                    state.final_weight = None;
                }
            }
            for (_, (_, w)) in &mut state.transitions {
                w.clip_tokens(max_token);
            }
        }
    }

    pub fn labels(&self) -> Vec<Label> {
        self.states
            .iter()
            .flat_map(|state| state.transitions.keys().copied())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    pub fn is_acyclic(&self) -> bool {
        *self.acyclic_cache.get_or_init(|| self.compute_is_acyclic())
    }

    fn compute_is_acyclic(&self) -> bool {
        let num_states = self.states.len();
        let mut indegree = vec![0u32; num_states];
        for state in &self.states {
            for &(target, _) in state.transitions.values() {
                if let Some(degree) = indegree.get_mut(target as usize) {
                    *degree += 1;
                }
            }
        }
        let mut queue = std::collections::VecDeque::new();
        for (state, &degree) in indegree.iter().enumerate() {
            if degree == 0 {
                queue.push_back(state);
            }
        }
        let mut visited = 0usize;
        while let Some(state) = queue.pop_front() {
            visited += 1;
            for &(target, _) in self.states[state].transitions.values() {
                let Some(degree) = indegree.get_mut(target as usize) else {
                    continue;
                };
                *degree -= 1;
                if *degree == 0 {
                    queue.push_back(target as usize);
                }
            }
        }
        visited == num_states
    }

    /// Convert this DWA to an NWA representation.
    pub fn to_nwa(&self) -> super::nwa::NWA {
        use super::nwa::{NWA, NWAState};
        let mut nwa = NWA::from_parts(
            Vec::with_capacity(self.states.len()),
            vec![self.start_state],
        );
        for state in &self.states {
            let mut nwa_state = NWAState::default();
            nwa_state.final_weight = state.final_weight.clone();
            for (&label, (target, weight)) in &state.transitions {
                nwa_state
                    .transitions
                    .entry(label)
                    .or_default()
                    .push((*target, weight.clone()));
            }
            nwa.states_mut().push(nwa_state);
        }
        nwa
    }
}

fn fmt_dwa_states(
    dwa: &DWA,
    f: &mut std::fmt::Formatter<'_>,
    label_fn: &dyn Fn(Label) -> String,
    weight_fn: &dyn Fn(&Weight) -> String,
) -> std::fmt::Result {
    for (i, state) in dwa.states.iter().enumerate() {
        if state.transitions.is_empty() && state.final_weight.is_none() {
            continue;
        }

        let start_mark = if i as u32 == dwa.start_state { " [START]" } else { "" };
        writeln!(f, "  State {i}{start_mark}")?;

        if let Some(w) = &state.final_weight {
            writeln!(f, "    final: {}", weight_fn(w))?;
        }

        for (label, (tgt, w)) in &state.transitions {
            let lbl = label_fn(*label);
            writeln!(f, "    {lbl} → State {tgt}")?;
            writeln!(f, "      weight: {}", weight_fn(w))?;
        }
    }
    Ok(())
}

impl std::fmt::Display for DWA {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "DWA: {} states, start=State {}", self.states.len(), self.start_state)?;
        fmt_dwa_states(self, f, &|l| l.to_string(), &|w| format!("{w}"))
    }
}

impl PartialEq for DWA {
    fn eq(&self, other: &Self) -> bool {
        self.start_state == other.start_state && self.states == other.states
    }
}

impl PartialEq for DWAState {
    fn eq(&self, other: &Self) -> bool {
        self.transitions == other.transitions && self.final_weight == other.final_weight
    }
}

#[cfg(test)]
mod cache_tests {
    use super::*;
    use crate::automata::weighted_u32::equivalence::find_difference;

    #[test]
    fn mutating_shared_transition_row_converts_it_to_owned() {
        let mut row = BTreeMap::new();
        row.insert(7, (1, Weight::all()));
        let shared = Arc::new(row);
        let mut left = DwaTransitionMap::from_arc(Arc::clone(&shared));
        let right = DwaTransitionMap::from_arc(shared);

        left.insert(8, (2, Weight::all()));

        assert!(matches!(left, DwaTransitionMap::Owned(_)));
        assert!(matches!(right, DwaTransitionMap::Shared(_)));
        assert!(left.contains_key(&8));
        assert!(!right.contains_key(&8));
    }

    #[test]
    fn exact_duplicate_state_merge_preserves_weighted_language() {
        let mut dwa = DWA::new(1, 1);
        let left = dwa.add_state();
        let right = dwa.add_state();
        dwa.set_final_weight(left, Weight::all());
        dwa.set_final_weight(right, Weight::all());
        dwa.add_transition(0, 10, left, Weight::all());
        dwa.add_transition(0, 20, right, Weight::all());

        let reference = dwa.clone();
        let merged = dwa.merge_exact_duplicate_states_owned();
        assert_eq!(merged.num_states(), 2);
        assert_eq!(merged.num_transitions(), 2);
        assert_eq!(find_difference(&reference, &merged).unwrap(), None);
    }

    #[test]
    fn graph_property_caches_invalidate_on_transition_mutation() {
        let mut dwa = DWA::new(1, 1);
        let next = dwa.add_state();
        dwa.add_transition(0, 7, next, Weight::all());

        assert_eq!(dwa.num_transitions(), 1);
        assert!(dwa.is_acyclic());
        assert_eq!(dwa.transition_count_cache.get(), Some(&1));
        assert_eq!(dwa.acyclic_cache.get(), Some(&true));

        dwa.add_transition(next, 8, next, Weight::all());
        assert!(dwa.transition_count_cache.get().is_none());
        assert!(dwa.acyclic_cache.get().is_none());
        assert_eq!(dwa.num_transitions(), 2);
        assert!(!dwa.is_acyclic());
    }

    #[test]
    fn mutable_state_access_and_deserialization_reset_graph_caches() {
        let mut dwa = DWA::new(1, 1);
        let next = dwa.add_state();
        dwa.add_transition(0, 1, next, Weight::all());
        assert_eq!(dwa.num_transitions(), 1);
        assert!(dwa.is_acyclic());

        dwa.states_mut()[next as usize]
            .transitions
            .insert(2, (next, Weight::all()));
        assert!(dwa.transition_count_cache.get().is_none());
        assert!(dwa.acyclic_cache.get().is_none());
        assert_eq!(dwa.num_transitions(), 2);
        assert!(!dwa.is_acyclic());

        let decoded: DWA = bincode::deserialize(&bincode::serialize(&dwa).unwrap()).unwrap();
        assert!(decoded.transition_count_cache.get().is_none());
        assert!(decoded.acyclic_cache.get().is_none());
        assert_eq!(decoded.num_transitions(), 2);
        assert!(!decoded.is_acyclic());
    }

    #[test]
    fn serde_pools_structural_weights_and_token_sets_not_arc_identity() {
        use std::sync::Arc;

        fn weight_with_tokens(
            tsid: u32,
            tokens: Arc<RangeSetBlaze<u32>>,
        ) -> Weight {
            let mut map = RangeMapBlaze::new();
            map.extend_simple(std::iter::once((tsid..=tsid, tokens)));
            finalize_weight_map(map)
        }

        // Deliberately bypass the token-set interner so the two equal token
        // languages have different Arc identities.  `finalize_weight_map`
        // consequently also sees distinct token-body pointers, giving us the
        // allocation-layout case that used to leak into artifact bytes.
        let token_a = Arc::new(RangeSetBlaze::from_iter([3..=7, 11..=13]));
        let token_b = Arc::new(RangeSetBlaze::from_iter([3..=7, 11..=13]));
        assert!(!Arc::ptr_eq(&token_a, &token_b));

        let equal_a = weight_with_tokens(5, Arc::clone(&token_a));
        let equal_b = weight_with_tokens(5, Arc::clone(&token_b));
        assert_eq!(equal_a, equal_b);
        assert_ne!(equal_a.ptr_key(), equal_b.ptr_key());

        let distinct_a = weight_with_tokens(6, token_a);
        let distinct_b = weight_with_tokens(7, token_b);
        assert_ne!(distinct_a, distinct_b);

        let mut dwa = DWA::new(8, 13);
        let target = dwa.add_state();
        dwa.add_transition(0, 1, target, equal_a);
        dwa.add_transition(0, 2, target, equal_b);
        dwa.add_transition(0, 3, target, distinct_a);
        dwa.add_transition(0, 4, target, distinct_b);

        let bytes = bincode::serialize(&dwa).unwrap();
        let encoded: DWASerde = bincode::deserialize(&bytes).unwrap();
        // equal_a/equal_b share one structural weight pool entry, while the
        // two TSID-distinct weights remain separate.
        assert_eq!(encoded.weight_pool.len(), 3);
        // All three structural weights refer to the same token language even
        // though the source Arcs were intentionally different.
        assert_eq!(encoded.token_set_pool.len(), 1);

        let decoded: DWA = bincode::deserialize(&bytes).unwrap();
        assert_eq!(decoded, dwa);
    }
}
