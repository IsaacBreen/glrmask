use std::collections::{BTreeMap, HashMap};
use std::fmt::Debug;
use std::borrow::Borrow;
use std::collections::BTreeMap as StdMap;

use bimap::BiBTreeMap;
use crate::datastructures::{
    hybrid_bitset::HybridBitset,
    hybrid_l2_bitset::HybridL2Bitset,
};
use crate::datastructures::bitset::Bitset;
use crate::tokenizer::LLMTokenID;
use crate::json_serialization::{JSONConvertible, JSONNode};

// ---------------------------------------------------------------------------
// Basic aliases
// ---------------------------------------------------------------------------

pub type LLMTokenBV = HybridBitset;
pub type TerminalBV = HybridBitset;
pub type StateIDBV = HybridBitset;
/// A 2D bitset where L1 is tokenizer state and L2 is terminal ID.
pub type TerminalInfo = HybridL2Bitset;

// ---------------------------------------------------------------------------
// Vocab structures
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LLMVocab {
    pub llm_token_map: BiBTreeMap<Vec<u8>, LLMTokenID>,
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

impl JSONConvertible for LLMVocab {
    fn to_json(&self) -> JSONNode {
        let mut m = StdMap::new();
        m.insert(
            "llm_token_map".to_string(),
            self.llm_token_map.to_json(),
        );
        m.insert(
            "max_original_llm_token_id".to_string(),
            self.max_original_llm_token_id.to_json(),
        );
        JSONNode::Object(m)
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => {
                let llm_token_map = obj
                    .remove("llm_token_map")
                    .ok_or("LLMVocab: missing llm_token_map".to_string())
                    .and_then(|n| BiBTreeMap::<Vec<u8>, LLMTokenID>::from_json(n))?;
                let max_original_llm_token_id = obj
                    .remove("max_original_llm_token_id")
                    .ok_or("LLMVocab: missing max_original_llm_token_id".to_string())
                    .and_then(usize::from_json)?;
                Ok(LLMVocab {
                    llm_token_map,
                    max_original_llm_token_id,
                })
            }
            _ => Err("LLMVocab: expected object".to_string()),
        }
    }
}

impl JSONConvertible for StageVocab {
    fn to_json(&self) -> JSONNode {
        let mut m = StdMap::new();
        m.insert(
            "original_to_internal".to_string(),
            self.original_to_internal.to_json(),
        );
        let mut ito: Vec<(usize, Vec<usize>)> = Vec::new();
        for (k, bv) in &self.internal_to_original {
            ito.push((*k, bv.iter_up_to(self.max_original_llm_token_id).collect::<Vec<_>>()));
        }
        m.insert("internal_to_original".to_string(), ito.to_json());
        m.insert(
            "internal_max_llm_token".to_string(),
            self.internal_max_llm_token.to_json(),
        );
        m.insert(
            "max_original_llm_token_id".to_string(),
            self.max_original_llm_token_id.to_json(),
        );
        // SKIP: internal_to_original_sparse_matrix (rebuild on load)
        JSONNode::Object(m)
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => {
                let original_to_internal = obj
                    .remove("original_to_internal")
                    .ok_or("StageVocab: missing original_to_internal".to_string())
                    .and_then(BTreeMap::<usize, usize>::from_json)?;
                let internal_max_llm_token = obj
                    .remove("internal_max_llm_token")
                    .ok_or("StageVocab: missing internal_max_llm_token".to_string())
                    .and_then(usize::from_json)?;
                let max_original_llm_token_id = obj
                    .remove("max_original_llm_token_id")
                    .ok_or("StageVocab: missing max_original_llm_token_id".to_string())
                    .and_then(usize::from_json)?;
                let ito_vec: Vec<(usize, Vec<usize>)> = obj
                    .remove("internal_to_original")
                    .ok_or("StageVocab: missing internal_to_original".to_string())
                    .and_then(Vec::from_json)?;
                let internal_to_original: BTreeMap<usize, LLMTokenBV> = ito_vec
                    .into_iter()
                    .map(|(k, v)| (k, v.into_iter().collect()))
                    .collect();
                let internal_to_original_sparse_matrix =
            // Always rebuild sparse matrix from internal_to_original
            Self::build_internal_to_original_sparse_matrix(
                &internal_to_original,
                max_original_llm_token_id,
                internal_max_llm_token,
            );

                Ok(StageVocab {
                    original_to_internal,
                    internal_to_original,
                    internal_max_llm_token,
                    max_original_llm_token_id,
                    internal_to_original_sparse_matrix,
                })
            }
            _ => Err("StageVocab: expected object".to_string()),
        }
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
            let mut internal_bv_ones = HybridBitset::ones(self.internal_max_llm_token + 1);
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

    pub fn original_bv_to_internal(&self, original_bv: &LLMTokenBV) -> LLMTokenBV {
        let mut internal_bv = HybridBitset::zeros();
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
