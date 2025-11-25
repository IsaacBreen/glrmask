use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use crate::constraint::GrammarConstraint;
use crate::constraint_vocab::{DedupValueMap, LLMTokenBV, StageVocab};
use crate::datastructures::char_transitions::CharTransitions;
use crate::datastructures::compressed_state_set::DenseStateSet;
use crate::datastructures::hybrid_bitset::HybridBitset;
use crate::datastructures::u8set::U8Set;
use crate::finite_automata::{DFAState, GroupID, DFA, Regex};
use crate::json_serialization::{JSONConvertible, JSONNode};
use crate::precompute4::weighted_automata::{DWAState, DWAStates, SimpleBitset, StateID, Weight, DWA};

// ---------------------------------------------------------------------------
// Weight Pool
// ---------------------------------------------------------------------------

#[derive(Default)]
pub struct WeightPool {
    pub simple_bitsets: DedupValueMap<SimpleBitset, SimpleBitset>,
    pub hybrid_bitsets: DedupValueMap<HybridBitset, HybridBitset>,
    pub dfa_states: DedupValueMap<DFAState, DFAState>,
    // We can also pool Weights if they are frequently repeated and complex
    pub weights: DedupValueMap<Weight, Weight>,
}

impl WeightPool {
    pub fn intern_simple(&mut self, val: SimpleBitset) -> usize {
        // DedupValueMap returns the ID. If it exists, it returns the existing ID.
        // But wait, DedupValueMap::insert returns Option<V> (the old value).
        // We need a method to just get the ID.
        // DedupValueMap::intern_value is private.
        // Let's use insert and ignore the return, but we need the ID.
        // Actually, DedupValueMap doesn't expose `intern` directly in the snippet I saw.
        // It has `insert` which returns `Option<V>`.
        // And `get` which returns `Option<&V>`.
        // It has `key_to_id` map.
        
        // I should probably check if it exists first.
        if let Some(&id) = self.simple_bitsets.key_to_id.get(&val) {
            return id;
        }
        // If not, insert it.
        // But `insert` takes (Key, Value). Here Key and Value are the same.
        self.simple_bitsets.insert(val.clone(), val.clone());
        *self.simple_bitsets.key_to_id.get(&val).unwrap()
    }

    pub fn intern_hybrid(&mut self, val: HybridBitset) -> usize {
        if let Some(&id) = self.hybrid_bitsets.key_to_id.get(&val) {
            return id;
        }
        self.hybrid_bitsets.insert(val.clone(), val.clone());
        *self.hybrid_bitsets.key_to_id.get(&val).unwrap()
    }

    pub fn intern_dfa_state(&mut self, val: DFAState) -> usize {
        if let Some(&id) = self.dfa_states.key_to_id.get(&val) {
            return id;
        }
        self.dfa_states.insert(val.clone(), val.clone());
        *self.dfa_states.key_to_id.get(&val).unwrap()
    }
    
    pub fn intern_weight(&mut self, val: Weight) -> usize {
        if let Some(&id) = self.weights.key_to_id.get(&val) {
            return id;
        }
        self.weights.insert(val.clone(), val.clone());
        *self.weights.key_to_id.get(&val).unwrap()
    }
}

impl JSONConvertible for WeightPool {
    fn to_json(&self) -> JSONNode {
        let mut obj = BTreeMap::new();
        obj.insert("simple_bitsets".to_string(), self.simple_bitsets.to_json());
        obj.insert("hybrid_bitsets".to_string(), self.hybrid_bitsets.to_json());
        obj.insert("dfa_states".to_string(), self.dfa_states.to_json());
        obj.insert("weights".to_string(), self.weights.to_json());
        JSONNode::Object(obj)
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        let mut obj = node.into_object()?;
        Ok(WeightPool {
            simple_bitsets: DedupValueMap::from_json(obj.remove("simple_bitsets").ok_or("Missing simple_bitsets")?)?,
            hybrid_bitsets: DedupValueMap::from_json(obj.remove("hybrid_bitsets").ok_or("Missing hybrid_bitsets")?)?,
            dfa_states: DedupValueMap::from_json(obj.remove("dfa_states").ok_or("Missing dfa_states")?)?,
            weights: DedupValueMap::from_json(obj.remove("weights").ok_or("Missing weights")?)?,
        })
    }
}

// ---------------------------------------------------------------------------
// Pooled Structures
// ---------------------------------------------------------------------------

pub struct PooledDWAState {
    pub transitions: BTreeMap<crate::precompute4::weighted_automata::common::Label, StateID>,
    pub final_weight: Option<usize>, // Index into WeightPool::weights
    pub trans_weights: BTreeMap<crate::precompute4::weighted_automata::common::Label, usize>, // Index into WeightPool::weights
    pub state_weight: Option<usize>, // Index into WeightPool::weights
}

impl JSONConvertible for PooledDWAState {
    fn to_json(&self) -> JSONNode {
        let mut obj = BTreeMap::new();
        obj.insert("transitions".to_string(), self.transitions.to_json());
        obj.insert("final_weight".to_string(), self.final_weight.to_json());
        obj.insert("trans_weights".to_string(), self.trans_weights.to_json());
        obj.insert("state_weight".to_string(), self.state_weight.to_json());
        JSONNode::Object(obj)
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        let mut obj = node.into_object()?;
        Ok(PooledDWAState {
            transitions: BTreeMap::from_json(obj.remove("transitions").ok_or("Missing transitions")?)?,
            final_weight: Option::<usize>::from_json(obj.remove("final_weight").ok_or("Missing final_weight")?)?,
            trans_weights: BTreeMap::from_json(obj.remove("trans_weights").ok_or("Missing trans_weights")?)?,
            state_weight: Option::<usize>::from_json(obj.remove("state_weight").ok_or("Missing state_weight")?)?,
        })
    }
}

pub struct PooledDWA {
    pub states: Vec<PooledDWAState>,
    pub start_state: StateID,
}

impl JSONConvertible for PooledDWA {
    fn to_json(&self) -> JSONNode {
        let mut obj = BTreeMap::new();
        obj.insert("states".to_string(), self.states.to_json());
        obj.insert("start_state".to_string(), self.start_state.to_json());
        JSONNode::Object(obj)
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        let mut obj = node.into_object()?;
        Ok(PooledDWA {
            states: Vec::<PooledDWAState>::from_json(obj.remove("states").ok_or("Missing states")?)?,
            start_state: StateID::from_json(obj.remove("start_state").ok_or("Missing start_state")?)?,
        })
    }
}

pub struct PooledDFA {
    pub state_indices: Vec<usize>, // Index into WeightPool::dfa_states
    pub start_state: usize,
    pub non_greedy_finalizers: BTreeSet<GroupID>,
}

impl JSONConvertible for PooledDFA {
    fn to_json(&self) -> JSONNode {
        let mut obj = BTreeMap::new();
        obj.insert("state_indices".to_string(), self.state_indices.to_json());
        obj.insert("start_state".to_string(), self.start_state.to_json());
        obj.insert("non_greedy_finalizers".to_string(), self.non_greedy_finalizers.to_json());
        JSONNode::Object(obj)
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        let mut obj = node.into_object()?;
        Ok(PooledDFA {
            state_indices: Vec::<usize>::from_json(obj.remove("state_indices").ok_or("Missing state_indices")?)?,
            start_state: usize::from_json(obj.remove("start_state").ok_or("Missing start_state")?)?,
            non_greedy_finalizers: BTreeSet::from_json(obj.remove("non_greedy_finalizers").ok_or("Missing non_greedy_finalizers")?)?,
        })
    }
}

pub struct PooledGrammarConstraint {
    pub tokenizer_dfa: PooledDFA,
    pub dwa: PooledDWA,
    pub vocab: StageVocab,
    pub pool: WeightPool,
    // Non-pooled fields
    pub parser: crate::glr::parser::GLRParser,
    pub token_name_map: bimap::BiBTreeMap<crate::glr::grammar::Terminal, usize>,
    pub original_llm_vocab: Arc<crate::constraint_vocab::LLMVocab>,
    // Pooled possible_matches
    pub possible_matches: BTreeMap<crate::tokenizer::TokenizerStateID, BTreeMap<crate::types::TerminalID, usize>>, // Index into WeightPool::hybrid_bitsets
}

impl JSONConvertible for PooledGrammarConstraint {
    fn to_json(&self) -> JSONNode {
        let mut obj = BTreeMap::new();
        obj.insert("tokenizer_dfa".to_string(), self.tokenizer_dfa.to_json());
        obj.insert("dwa".to_string(), self.dwa.to_json());
        obj.insert("vocab".to_string(), self.vocab.to_json());
        obj.insert("pool".to_string(), self.pool.to_json());
        obj.insert("parser".to_string(), self.parser.to_json());
        obj.insert("token_name_map".to_string(), self.token_name_map.to_json());
        obj.insert("original_llm_vocab".to_string(), self.original_llm_vocab.to_json());
        obj.insert("possible_matches".to_string(), self.possible_matches.to_json());
        JSONNode::Object(obj)
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        let mut obj = node.into_object()?;
        Ok(PooledGrammarConstraint {
            tokenizer_dfa: PooledDFA::from_json(obj.remove("tokenizer_dfa").ok_or("Missing tokenizer_dfa")?)?,
            dwa: PooledDWA::from_json(obj.remove("dwa").ok_or("Missing dwa")?)?,
            vocab: StageVocab::from_json(obj.remove("vocab").ok_or("Missing vocab")?)?,
            pool: WeightPool::from_json(obj.remove("pool").ok_or("Missing pool")?)?,
            parser: crate::glr::parser::GLRParser::from_json(obj.remove("parser").ok_or("Missing parser")?)?,
            token_name_map: bimap::BiBTreeMap::from_json(obj.remove("token_name_map").ok_or("Missing token_name_map")?)?,
            original_llm_vocab: Arc::new(crate::constraint_vocab::LLMVocab::from_json(obj.remove("original_llm_vocab").ok_or("Missing original_llm_vocab")?)?),
            possible_matches: BTreeMap::from_json(obj.remove("possible_matches").ok_or("Missing possible_matches")?)?,
        })
    }
}

// ---------------------------------------------------------------------------
// Conversion Logic
// ---------------------------------------------------------------------------

impl PooledGrammarConstraint {
    pub fn from_constraint(gc: &GrammarConstraint) -> Self {
        let mut pool = WeightPool::default();

        // Pool DFA
        let dfa = &gc.tokenizer.dfa;
        let mut state_indices = Vec::with_capacity(dfa.states.len());
        for state in &dfa.states {
            state_indices.push(pool.intern_dfa_state(state.clone()));
        }
        let pooled_dfa = PooledDFA {
            state_indices,
            start_state: dfa.start_state,
            non_greedy_finalizers: dfa.non_greedy_finalizers.clone(),
        };

        // Pool DWA
        let dwa = &gc.precomputed4;
        let mut pooled_dwa_states = Vec::with_capacity(dwa.states.len());
        for state in &dwa.states.0 {
            let final_weight = state.final_weight.as_ref().map(|w| pool.intern_weight(w.clone()));
            let state_weight = state.state_weight.as_ref().map(|w| pool.intern_weight(w.clone()));
            let mut trans_weights = BTreeMap::new();
            for (k, v) in &state.trans_weights {
                trans_weights.insert(*k, pool.intern_weight(v.clone()));
            }
            
            pooled_dwa_states.push(PooledDWAState {
                transitions: state.transitions.clone(),
                final_weight,
                trans_weights,
                state_weight,
            });
        }
        let pooled_dwa = PooledDWA {
            states: pooled_dwa_states,
            start_state: dwa.body.start_state,
        };

        // Pool possible_matches
        let mut pooled_possible_matches = BTreeMap::new();
        for (state_id, inner_map) in &gc.possible_matches {
            let mut new_inner = BTreeMap::new();
            for (term_id, bv) in inner_map {
                new_inner.insert(*term_id, pool.intern_hybrid(bv.clone()));
            }
            pooled_possible_matches.insert(*state_id, new_inner);
        }

        PooledGrammarConstraint {
            tokenizer_dfa: pooled_dfa,
            dwa: pooled_dwa,
            vocab: gc.precompute4_vocab.clone(),
            pool,
            parser: gc.parser.clone(),
            token_name_map: gc.token_name_map.clone(),
            original_llm_vocab: gc.original_llm_vocab.clone(),
            possible_matches: pooled_possible_matches,
        }
    }

    pub fn to_constraint(self) -> GrammarConstraint {
        // Reconstruct DFA
        let mut dfa_states = Vec::with_capacity(self.tokenizer_dfa.state_indices.len());
        for &idx in &self.tokenizer_dfa.state_indices {
            let state = self.pool.dfa_states.id_to_value.get(&idx).expect("Invalid DFA state index").clone();
            dfa_states.push(state);
        }
        let dfa = DFA {
            states: dfa_states,
            start_state: self.tokenizer_dfa.start_state,
            non_greedy_finalizers: self.tokenizer_dfa.non_greedy_finalizers,
        };

        // Reconstruct DWA
        let mut dwa_states = Vec::with_capacity(self.dwa.states.len());
        for p_state in self.dwa.states {
            let final_weight = p_state.final_weight.map(|idx| self.pool.weights.id_to_value.get(&idx).expect("Invalid weight index").clone());
            let state_weight = p_state.state_weight.map(|idx| self.pool.weights.id_to_value.get(&idx).expect("Invalid weight index").clone());
            let mut trans_weights = BTreeMap::new();
            for (k, idx) in p_state.trans_weights {
                trans_weights.insert(k, self.pool.weights.id_to_value.get(&idx).expect("Invalid weight index").clone());
            }
            
            dwa_states.push(DWAState {
                transitions: p_state.transitions,
                final_weight,
                trans_weights,
                state_weight,
            });
        }
        let dwa = DWA {
            states: DWAStates(dwa_states),
            body: crate::precompute4::weighted_automata::dwa::DWABody { start_state: self.dwa.start_state },
        };

        // Reconstruct possible_matches
        let mut possible_matches = BTreeMap::new();
        for (state_id, inner_map) in self.possible_matches {
            let mut new_inner = BTreeMap::new();
            for (term_id, idx) in inner_map {
                let bv = self.pool.hybrid_bitsets.id_to_value.get(&idx).expect("Invalid hybrid bitset index").clone();
                new_inner.insert(term_id, bv);
            }
            possible_matches.insert(state_id, new_inner);
        }

        GrammarConstraint {
            tokenizer: Regex { dfa },
            parser: self.parser,
            precomputed4: dwa,
            original_llm_vocab: self.original_llm_vocab,
            token_name_map: self.token_name_map,
            possible_matches,
            precompute4_vocab: self.vocab,
        }
    }
}
