use std::collections::{BTreeMap, HashMap};
use std::fmt::Debug;
use std::borrow::Borrow;
use std::collections::BTreeMap as StdMap;

use bimap::BiBTreeMap;
use json_convertible_derive::JSONConvertible;
use crate::datastructures::{
    hybrid_bitset::RangeSet,
};
use crate::datastructures::bitset::Bitset;
use crate::tokenizer::LLMTokenID;
use crate::json_serialization::{JSONConvertible, JSONNode};

// ---------------------------------------------------------------------------
// Basic aliases
// ---------------------------------------------------------------------------

pub type LLMTokenBV = RangeSet;
pub type TerminalBV = RangeSet;
pub type StateIDBV = RangeSet;

// ---------------------------------------------------------------------------
// Vocab structures
// ---------------------------------------------------------------------------

/// LLM vocabulary: maps byte sequences to token IDs.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, JSONConvertible)]
pub struct LLMVocab {
    pub llm_token_map: BTreeMap<Vec<u8>, LLMTokenID>,
    pub max_original_llm_token_id: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageVocab {
    pub original_to_internal: BTreeMap<usize, usize>,
    pub internal_to_original: BTreeMap<usize, LLMTokenBV>,
    pub internal_max_llm_token: usize,
    pub max_original_llm_token_id: usize,
    pub internal_to_original_sparse_matrix: Vec<Vec<(u16, u64)>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommitVocab {
    pub representatives: Vec<Vec<u8>>,
    pub original_to_representative: Vec<u32>,
}

impl CommitVocab {
    pub const INVALID_REPRESENTATIVE: u32 = u32::MAX;

    pub fn new(representatives: Vec<Vec<u8>>, original_to_representative: Vec<u32>) -> Self {
        Self { representatives, original_to_representative }
    }

    pub fn is_empty(&self) -> bool { self.representatives.is_empty() }

    pub fn representative_index(&self, token_id: LLMTokenID) -> Option<usize> {
        let rep_idx = *self.original_to_representative.get(token_id.0)?;
        if rep_idx == Self::INVALID_REPRESENTATIVE { None } else { Some(rep_idx as usize) }
    }

    pub fn token_bytes(&self, token_id: LLMTokenID) -> Option<&[u8]> {
        let idx = self.representative_index(token_id)?;
        self.representatives.get(idx).map(|bytes| bytes.as_slice())
    }
}

/// Intermediate JSON representation of StageVocab.
/// internal_to_original is stored as Vec<(usize, Vec<usize>)> for efficient serialization.
/// internal_to_original_sparse_matrix is skipped (rebuilt on load).
#[derive(Debug, Clone, JSONConvertible)]
struct StageVocabJSON {
    original_to_internal: BTreeMap<usize, usize>,
    internal_to_original: Vec<(usize, Vec<usize>)>,
    internal_max_llm_token: usize,
    max_original_llm_token_id: usize,
}

#[derive(Debug, Clone, JSONConvertible)]
struct CommitVocabJSON {
    representatives: Vec<Vec<u8>>,
    original_to_representative: Vec<u32>,
}

impl JSONConvertible for StageVocab {
    fn to_json(&self) -> JSONNode {
        let ito: Vec<(usize, Vec<usize>)> = self.internal_to_original
            .iter()
            .map(|(k, bv)| (*k, bv.iter_up_to(self.max_original_llm_token_id).collect()))
            .collect();
        
        let intermediate = StageVocabJSON {
            original_to_internal: self.original_to_internal.clone(),
            internal_to_original: ito,
            internal_max_llm_token: self.internal_max_llm_token,
            max_original_llm_token_id: self.max_original_llm_token_id,
        };
        intermediate.to_json()
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        let intermediate = StageVocabJSON::from_json(node)?;
        
        let internal_to_original: BTreeMap<usize, LLMTokenBV> = intermediate
            .internal_to_original
            .into_iter()
            .map(|(k, v)| (k, v.into_iter().collect()))
            .collect();
        
        let internal_to_original_sparse_matrix = Self::build_internal_to_original_sparse_matrix(
            &internal_to_original,
            intermediate.max_original_llm_token_id,
            intermediate.internal_max_llm_token,
        );

        Ok(StageVocab {
            original_to_internal: intermediate.original_to_internal,
            internal_to_original,
            internal_max_llm_token: intermediate.internal_max_llm_token,
            max_original_llm_token_id: intermediate.max_original_llm_token_id,
            internal_to_original_sparse_matrix,
        })
    }
}

impl JSONConvertible for CommitVocab {
    fn to_json(&self) -> JSONNode {
        let intermediate = CommitVocabJSON {
            representatives: self.representatives.clone(),
            original_to_representative: self.original_to_representative.clone(),
        };
        intermediate.to_json()
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        let intermediate = CommitVocabJSON::from_json(node)?;
        Ok(Self {
            representatives: intermediate.representatives,
            original_to_representative: intermediate.original_to_representative,
        })
    }
}

impl StageVocab {
    pub(crate) fn build_internal_to_original_sparse_matrix(
        internal_to_original: &BTreeMap<usize, LLMTokenBV>,
        max_original_llm_token_id: usize,
        internal_max_llm_token: usize,
    ) -> Vec<Vec<(u16, u64)>> {
        type Word = u64;
        const WORD_BITS: usize = 64;

        let num_internal_tokens = internal_max_llm_token + 1;
        let mut sparse_matrix: Vec<Vec<(u16, Word)>> = vec![Vec::new(); num_internal_tokens];

        for (internal_id, original_bv) in internal_to_original.iter() {
            if *internal_id >= num_internal_tokens {
                continue;
            }

            let mut temp_row = BTreeMap::<u16, Word>::new();
            for original_id in original_bv.iter_up_to(max_original_llm_token_id) {
                if original_id > max_original_llm_token_id {
                    continue;
                }
                let word_idx = (original_id / WORD_BITS) as u16;
                let bit_idx = original_id % WORD_BITS;
                *temp_row.entry(word_idx).or_insert(0) |= 1 << bit_idx;
            }
            if !temp_row.is_empty() {
                sparse_matrix[*internal_id] = temp_row.into_iter().collect();
            }
        }
        sparse_matrix
    }

    /// Convert an internal BV (using `self.vocab`) back to original IDs.
    pub fn internal_bv_to_original(&self, internal_bv: &LLMTokenBV) -> Bitset {
        if internal_bv.is_all() {
            let mut internal_bv_ones = RangeSet::ones(self.internal_max_llm_token + 1);
            return self.internal_bv_to_original(&internal_bv_ones);
        }

        type Word = u64;
        const WORD_BITS: usize = 64;

        let max_original_id = self.max_original_llm_token_id;
        let original_vocab_size_words = (max_original_id / WORD_BITS) + 1;
        let num_internal_tokens = self.internal_max_llm_token + 1;

        let mut result_bitset_words = vec![0 as Word; original_vocab_size_words];
        for internal_id in internal_bv.iter_up_to(self.internal_max_llm_token) {
            if internal_id >= num_internal_tokens {
                continue;
            }
            // It's possible for an internal ID to exist in the bitvector but not have a
            // corresponding entry in the sparse matrix if it corresponds to no original tokens.
            if let Some(sparse_row) = self.internal_to_original_sparse_matrix.get(internal_id) {
                for &(word_idx, word) in sparse_row {
                    result_bitset_words[word_idx as usize] |= word;
                }
            }
        }

        Bitset::from_words_vec(result_bitset_words)
    }

    /// Fill an i32 slice with the mask converted from internal to original token IDs.
    /// 
    /// This is a zero-allocation version that writes directly to the provided buffer.
    /// The output slice should have length `(max_original_llm_token_id + 32) / 32`.
    /// 
    /// This method zeros the buffer first, then ORs in the bits.
    #[inline]
    pub fn fill_internal_bv_to_original_i32(&self, internal_bv: &LLMTokenBV, out: &mut [i32]) {
        // Zero the output first
        out.fill(0);
        
        if internal_bv.is_all() {
            let internal_bv_ones = RangeSet::ones(self.internal_max_llm_token + 1);
            self.fill_internal_bv_to_original_i32_nozeroing(&internal_bv_ones, out);
            return;
        }
        
        self.fill_internal_bv_to_original_i32_nozeroing(internal_bv, out);
    }
    
    /// Fill an i32 slice without zeroing first (for internal use when buffer is already zeroed).
    /// 
    /// IMPORTANT: The output buffer must have correct size: `(max_original_llm_token_id + 32) / 32`.
    /// Using an incorrectly sized buffer can lead to incorrect results or panics in debug builds.
    #[inline]
    pub fn fill_internal_bv_to_original_i32_nozeroing(&self, internal_bv: &LLMTokenBV, out: &mut [i32]) {
        // Use unsafe transmute to view i32 slice as u64 slice for direct OR operations
        // This is safe because:
        // 1. out.len() is always even (computed as (max + 32) / 32)
        // 2. We're only doing OR operations, so endianness doesn't matter for correctness
        // 3. The memory layout of [i32; 2] is identical to u64 on little-endian systems
        // 
        // On big-endian systems the bit positions would be swapped within each u64,
        // but since we built the sparse matrix the same way, it's consistent.
        
        // SAFETY: We're transmuting a properly aligned i32 slice to u64 slice.
        // The slice length is always even.
        debug_assert!(out.len() % 2 == 0, "fill_internal_bv_to_original_i32_nozeroing: output buffer length must be even");
        
        let out_u64: &mut [u64] = unsafe {
            std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u64, out.len() / 2)
        };
        
        let sparse_matrix = &self.internal_to_original_sparse_matrix;
        
        for internal_id in internal_bv.iter_up_to(self.internal_max_llm_token) {
            // SAFETY: internal_id is bounded by internal_max_llm_token, which is < sparse_matrix.len()
            // The sparse_matrix was built with num_internal_tokens = internal_max_llm_token + 1
            let sparse_row = unsafe { sparse_matrix.get_unchecked(internal_id) };
            
            for &(word_idx, word) in sparse_row {
                // SAFETY: word_idx is bounded by (max_original_llm_token_id / 64), which is < out_u64.len()
                // The sparse matrix was built with correct bounds
                unsafe {
                    *out_u64.get_unchecked_mut(word_idx as usize) |= word;
                }
            }
        }
    }
    
    /// Fill an i32 slice with the mask via a raw pointer.
    /// 
    /// # Safety
    /// The caller must ensure that:
    /// - `ptr` points to at least `len` i32 values of valid, writable memory
    /// - The memory is properly aligned for i32
    /// - No other references to this memory exist during the call
    #[inline]
    pub unsafe fn fill_internal_bv_to_original_i32_ptr(&self, internal_bv: &LLMTokenBV, ptr: *mut i32, len: usize) {
        let out = std::slice::from_raw_parts_mut(ptr, len);
        self.fill_internal_bv_to_original_i32(internal_bv, out);
    }

    /// Returns the required buffer size in i32 elements for the mask.
    #[inline]
    pub fn mask_buffer_size_i32(&self) -> usize {
        (self.max_original_llm_token_id + 32) / 32
    }

    pub fn original_bv_to_internal(&self, original_bv: &LLMTokenBV) -> LLMTokenBV {
        let mut internal_bv = RangeSet::zeros();
        if original_bv.is_all() {
            for &internal_id in self.original_to_internal.values() {
                internal_bv.insert(internal_id);
            }
        } else {
            for i in original_bv.iter_up_to(self.max_original_llm_token_id) {
                if let Some(&internal_id) = self.original_to_internal.get(&i) {
                    internal_bv.insert(internal_id);
                }
            }
        }
        internal_bv
    }
}

// ---------------------------------------------------------------------------
// Deduplicating map for large values
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DedupValueMap<K, V>
where
    K: Ord + Clone + Eq,
    V: Clone + Eq + std::hash::Hash,
{
    pub key_to_id: BTreeMap<K, usize>,
    pub id_to_value: BTreeMap<usize, V>,
    pub value_to_id: HashMap<V, usize>,
    pub next_id: usize,
}

impl<K, V> Default for DedupValueMap<K, V>
where
    K: Ord + Clone + Eq,
    V: Clone + Eq + std::hash::Hash,
{
    fn default() -> Self {
        Self {
            key_to_id: BTreeMap::new(),
            id_to_value: BTreeMap::new(),
            value_to_id: HashMap::new(),
            next_id: 0,
        }
    }
}

impl<K, V> DedupValueMap<K, V>
where
    K: Ord + Clone + Eq,
    V: Clone + Eq + std::hash::Hash,
{
    pub fn new() -> Self { Self::default() }

    fn intern_value(&mut self, v: V) -> usize {
        if let Some(&id) = self.value_to_id.get(&v) { return id; }
        let id = self.next_id;
        self.next_id = self.next_id.checked_add(1).expect("DedupValueMap ID overflow");
        self.id_to_value.insert(id, v.clone());
        self.value_to_id.insert(v, id);
        id
    }

    pub fn len(&self) -> usize { self.key_to_id.len() }
    pub fn is_empty(&self) -> bool { self.key_to_id.is_empty() }

    pub fn contains_key<Q>(&self, k: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.key_to_id.contains_key(k)
    }

    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let id = self.intern_value(value);
        let old = self.key_to_id.insert(key, id);
        old.and_then(|old_id| self.id_to_value.get(&old_id).cloned())
    }

    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let id = self.key_to_id.get(key)?;
        self.id_to_value.get(id)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.key_to_id
            .iter()
            .map(|(k, id)| (k, self.id_to_value.get(id).expect("dangling id")))
    }
}

impl<K, V> JSONConvertible for DedupValueMap<K, V>
where
    K: Ord + Clone + Eq + JSONConvertible,
    V: Clone + Eq + std::hash::Hash + JSONConvertible,
{
    fn to_json(&self) -> JSONNode {
        let mut obj = StdMap::new();
        obj.insert("next_id".to_string(), self.next_id.to_json());
        let mut values_arr = Vec::new();
        for (id, v) in &self.id_to_value {
            values_arr.push(JSONNode::Array(vec![id.to_json(), v.to_json()]));
        }
        obj.insert("values".to_string(), JSONNode::Array(values_arr));
        let mut keys_arr = Vec::new();
        for (k, id) in &self.key_to_id {
            keys_arr.push(JSONNode::Array(vec![k.to_json(), id.to_json()]));
        }
        obj.insert("keys".to_string(), JSONNode::Array(keys_arr));
        JSONNode::Object(obj)
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        let mut obj = node.into_object()?;
        let next_id =
            usize::from_json(obj.remove("next_id").ok_or("DedupValueMap: missing 'next_id'")?)?;
        let values_arr = obj
            .remove("values")
            .ok_or("DedupValueMap: missing 'values'")?;
        let keys_arr = obj.remove("keys").ok_or("DedupValueMap: missing 'keys'")?;

        let mut id_to_value = BTreeMap::new();
        let mut value_to_id = HashMap::new();
        match values_arr {
            JSONNode::Array(a) => {
                for n in a {
                    let mut pair = match n {
                        JSONNode::Array(p) if p.len() == 2 => p,
                        _ => return Err("DedupValueMap: values entry must be [id, value]".to_string()),
                    };
                    let v_node = pair.pop().unwrap();
                    let id_node = pair.pop().unwrap();
                    let id = usize::from_json(id_node)?;
                    let v = V::from_json(v_node)?;
                    id_to_value.insert(id, v.clone());
                    value_to_id.insert(v, id);
                }
            }
            _ => return Err("DedupValueMap: 'values' must be an array".to_string()),
        }

        let mut key_to_id = BTreeMap::new();
        match keys_arr {
            JSONNode::Array(a) => {
                for n in a {
                    let mut pair = match n {
                        JSONNode::Array(p) if p.len() == 2 => p,
                        _ => return Err("DedupValueMap: keys entry must be [key, id]".to_string()),
                    };
                    let id_node = pair.pop().unwrap();
                    let key_node = pair.pop().unwrap();
                    let id = usize::from_json(id_node)?;
                    let k = K::from_json(key_node)?;
                    key_to_id.insert(k, id);
                }
            }
            _ => return Err("DedupValueMap: 'keys' must be an array".to_string()),
        }

        Ok(Self {
            key_to_id,
            id_to_value,
            value_to_id,
            next_id,
        })
    }
}
