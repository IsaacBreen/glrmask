use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock, RwLock};

use crate::glr::table::StateID;
use crate::constraint::{LLMTokenBV, PrecomputedNodeContents};
use crate::datastructures::trie::Trie as GenericTrie;

pub type PrecomputeNode2 = GenericTrie<(usize, Option<StateID>), LLMTokenBV, PrecomputedNodeContents>;

static PRECOMPUTED2_MAP: OnceLock<Arc<RwLock<HashMap<usize, Arc<Mutex<PrecomputeNode2>>>>>> = OnceLock::new();

fn ensure() -> &'static Arc<RwLock<HashMap<usize, Arc<Mutex<PrecomputeNode2>>>>> {
    PRECOMPUTED2_MAP.get_or_init(|| Arc::new(RwLock::new(HashMap::new())))
}

pub fn register(node_arc: &Arc<Mutex<PrecomputeNode2>>) -> usize {
    let id = Arc::as_ptr(node_arc) as usize;
    let map = ensure();
    {
        let map_read = map.read().unwrap();
        if map_read.contains_key(&id) {
            return id;
        }
    }
    {
        let mut map_write = map.write().unwrap();
        map_write.entry(id).or_insert(node_arc.clone());
    }
    id
}

pub fn resolve(id: usize) -> Option<Arc<Mutex<PrecomputeNode2>>> {
    let map = ensure();
    let map_read = map.read().unwrap();
    map_read.get(&id).cloned()
}

#[allow(dead_code)]
pub fn has(id: usize) -> bool {
    let map = ensure();
    let map_read = map.read().unwrap();
    map_read.contains_key(&id)
}
