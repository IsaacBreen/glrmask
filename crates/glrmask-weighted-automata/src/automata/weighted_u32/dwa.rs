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

#[derive(Debug, Clone, PartialEq)]
pub enum DwaTransitionMap {
    Owned(BTreeMap<Label, (u32, Weight)>),
    Shared(Arc<BTreeMap<Label, (u32, Weight)>>),
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
        }
    }
}

impl Deref for DwaTransitionMap {
    type Target = BTreeMap<Label, (u32, Weight)>;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Owned(transitions) => transitions,
            Self::Shared(transitions) => transitions,
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
        if matches!(self, Self::Shared(_)) {
            let current = std::mem::take(self);
            let owned = match current {
                Self::Shared(transitions) => Arc::try_unwrap(transitions)
                    .unwrap_or_else(|shared| shared.as_ref().clone()),
                Self::Owned(_) => unreachable!("checked Shared above"),
            };
            *self = Self::Owned(owned);
        }
        match self {
            Self::Owned(transitions) => transitions,
            Self::Shared(_) => unreachable!("Shared row was converted to Owned"),
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
    let mut value = 0u32;
    let mut shift = 0u32;
    for _ in 0..5 {
        let byte = *input
            .get(*pos)
            .ok_or_else(|| "truncated packed DWA varint".to_owned())?;
        *pos += 1;
        if shift == 28 && byte > 0x0f {
            return Err("overflowing packed DWA u32 varint".to_owned());
        }
        value |= ((byte & 0x7f) as u32) << shift;
        if byte & 0x80 == 0 {
            return Ok(value);
        }
        shift += 7;
    }
    Err("overflowing packed DWA u32 varint".to_owned())
}

#[inline]
fn take_var_u64(input: &[u8], pos: &mut usize) -> Result<u64, String> {
    let mut value = 0u64;
    let mut shift = 0u32;
    for index in 0..10 {
        let byte = *input
            .get(*pos)
            .ok_or_else(|| "truncated packed DWA varint".to_owned())?;
        *pos += 1;
        if index == 9 && byte > 1 {
            return Err("overflowing packed DWA u64 varint".to_owned());
        }
        value |= ((byte & 0x7f) as u64) << shift;
        if byte & 0x80 == 0 {
            return Ok(value);
        }
        shift += 7;
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
        let transition_count = self
            .states
            .iter()
            .map(|state| state.transitions.len())
            .sum::<usize>();

        let row_hash = |state: &DWAState| -> u64 {
            // Canonically shared rows can use pointer identity directly. Rows
            // that have been mutated are converted back to Owned by DerefMut,
            // so only those rows pay the structural hashing cost.
            if matches!(state.transitions, DwaTransitionMap::Shared(_)) {
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
        let mut row_representatives = Vec::<usize>::new();
        let mut state_row_ids = Vec::<u32>::with_capacity(self.states.len());
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

        let mut w_ptr_to_idx: FxHashMap<usize, u32> = FxHashMap::default();
        w_ptr_to_idx.reserve(32_768);
        let mut weight_refs = Vec::<Weight>::new();
        let mut intern_weight = |w: &Weight| -> u32 {
            let ptr = w.ptr_key();
            *w_ptr_to_idx.entry(ptr).or_insert_with(|| {
                let idx = weight_refs.len() as u32;
                weight_refs.push(w.clone());
                idx
            })
        };

        let transition_rows = row_representatives
            .iter()
            .map(|&state_index| {
                self.states[state_index]
                    .transitions
                    .iter()
                    .map(|(&label, (target, weight))| (label, *target, intern_weight(weight)))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let state_rows = self
            .states
            .iter()
            .zip(state_row_ids)
            .map(|(state, row)| (row, state.final_weight.as_ref().map(&mut intern_weight)))
            .collect::<Vec<_>>();

        // Weight IDs above are assigned in stable first-encounter order. Walk
        // the unique Weight maps in parallel, then assign token-set IDs in that
        // same Weight/range order. This keeps the wire representation exactly
        // deterministic while avoiding a serial chase through tens of thousands
        // of scattered RangeMapBlaze allocations on first save after compile.
        struct RawWeightPoolEntry {
            all: bool,
            entries: Vec<(u32, u32, Arc<RangeSetBlaze<u32>>)>,
        }
        let materialize_weight = |weight: &Weight| RawWeightPoolEntry {
            all: weight.is_full(),
            entries: if weight.is_full() {
                Vec::new()
            } else {
                weight
                    .raw_range_values()
                    .map(|(range, tokens)| {
                        (*range.start(), *range.end(), Arc::clone(tokens))
                    })
                    .collect()
            },
        };
        let raw_weight_pool = if weight_refs.len() >= 1_024 && rayon::current_num_threads() > 1 {
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

        let mut ts_ptr_to_idx: FxHashMap<usize, u32> = FxHashMap::default();
        ts_ptr_to_idx.reserve(32_768);
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
                    let ptr = Arc::as_ptr(&tokens) as usize;
                    let token_set = if let Some(&existing) = ts_ptr_to_idx.get(&ptr) {
                        existing
                    } else {
                        let idx = token_set_pool.len() as u32;
                        ts_ptr_to_idx.insert(ptr, idx);
                        token_set_pool.push(tokens);
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

        (
            token_set_pool,
            weight_pool,
            transition_rows,
            state_rows,
            transition_count,
        )
    }

    fn to_packed_bytes(&self) -> Vec<u8> {
        let (token_set_arcs, mut weight_pool, transition_rows, state_rows, transition_count) =
            self.packed_pooled_parts();

        // Adjacent token sets are often almost identical. Sort them
        // lexicographically, remap weight references, and front-code them in
        // independently decodable chunks. Chunk boundaries preserve parallel
        // load while sacrificing very little prefix sharing.
        let token_set_pool = if token_set_arcs.len() >= 1_024 && rayon::current_num_threads() > 1 {
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
        let mut token_set_order = (0..token_set_pool.len()).collect::<Vec<_>>();
        token_set_order.sort_unstable_by(|&left, &right| {
            token_set_pool[left].cmp(&token_set_pool[right])
        });
        let mut old_to_new_token_set = vec![0u32; token_set_pool.len()];
        for (new_index, &old_index) in token_set_order.iter().enumerate() {
            old_to_new_token_set[old_index] = new_index as u32;
        }
        for weight in &mut weight_pool {
            for (_, _, token_set) in &mut weight.entries {
                *token_set = old_to_new_token_set[*token_set as usize];
            }
        }
        let mut token_set_slots = token_set_pool
            .into_iter()
            .map(Some)
            .collect::<Vec<_>>();
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

        // Row pooling was fused into packed_pooled_parts(), before Weight-id
        // translation, so duplicate state rows never become duplicate serde
        // transition vectors.

        let mut out = Vec::new();
        let mut body = Vec::new();
        // Format tag, so accidental cross-version use fails cleanly rather
        // than silently constructing nonsense.
        out.extend_from_slice(b"DWP6");
        put_var_u32(&mut out, self.start_state);

        put_var_u32(&mut out, token_set_pool.len() as u32);
        const TOKEN_SET_CHUNK_SIZE: usize = 64;
        let token_set_chunk_count = token_set_pool.len().div_ceil(TOKEN_SET_CHUNK_SIZE);
        let token_set_range_count = token_set_pool.iter().map(Vec::len).sum::<usize>();
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
        if token_set_range_count >= 1_000_000 && rayon::current_num_threads() > 1 {
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

        let unique_row_transition_count = transition_rows.iter().map(Vec::len).sum::<usize>();
        let use_component_rows = unique_row_transition_count >= 100_000;
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

        put_var_u32(&mut out, state_rows.len() as u32);
        for &(row, final_weight) in &state_rows {
            put_var_u32(&mut out, row);
            put_var_u32(
                &mut out,
                final_weight.map_or(0, |weight| weight.saturating_add(1)),
            );
        }
        if std::env::var_os("GLRMASK_PROFILE_DWA_SERIALIZATION").is_some() {
            let weight_entries: usize = weight_pool.iter().map(|w| w.entries.len()).sum();
            let token_ranges: usize = token_set_pool.iter().map(Vec::len).sum();
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
                token_set_pool.len(),
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
                (rows, state_count_pos, None)
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
                let build_row = |&(label_id, target_id, weight_id): &(usize, usize, usize)| {
                    let labels = &label_pool[label_id];
                    let targets = &target_pool[target_id];
                    let weights = &weight_id_pool[weight_id];
                    if labels.len() != targets.len() || labels.len() != weights.len() {
                        return Err("mismatched packed DWA row-component lengths".to_owned());
                    }
                    if labels.windows(2).any(|pair| pair[0] >= pair[1]) {
                        return Err("packed DWA transition labels are not strictly increasing".to_owned());
                    }
                    let mut entries = Vec::with_capacity(labels.len());
                    for ((&label, &target), &weight_index) in
                        labels.iter().zip(targets).zip(weights)
                    {
                        if target as usize >= state_count {
                            return Err("invalid packed DWA target state".to_owned());
                        }
                        let weight = w_pool
                            .get(weight_index as usize)
                            .cloned()
                            .ok_or_else(|| "invalid packed DWA weight index".to_owned())?;
                        entries.push((label, (target, weight)));
                    }
                    Ok(entries.into_iter().collect::<BTreeMap<_, _>>())
                };
                let rows = if row_count >= 1024 && rayon::current_num_threads() > 1 {
                    row_components
                        .par_iter()
                        .map(build_row)
                        .collect::<Result<Vec<_>, _>>()?
                } else {
                    row_components
                        .iter()
                        .map(build_row)
                        .collect::<Result<Vec<_>, _>>()?
                };
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
        let transition_rows = transition_rows
            .into_iter()
            .map(Arc::new)
            .collect::<Vec<_>>();
        let build_state = |&(row, final_weight): &(usize, Option<usize>)| DWAState {
            transitions: DwaTransitionMap::from_arc(Arc::clone(&transition_rows[row])),
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
