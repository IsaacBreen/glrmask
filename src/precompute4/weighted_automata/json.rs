// src/precompute4/weighted_automata/json.rs

#![allow(dead_code)]
#![allow(clippy::needless_borrow)]

use super::bitset::SimpleBitset;
use super::common::{Label, StateID, Weight};
use super::dwa::{DWABody, DWAState, DWAStates, DWA, DWABuildError};
use super::nwa::{NWABody, NWAState, NWAStates, NWA};
use crate::json_serialization::{JSONConvertible, JSONNode};
use range_set_blaze::RangeSetBlaze;
use std::collections::{BTreeMap, HashMap};
use std::iter::FromIterator;
use crate::json_serialization::JSONNode::Array;

impl JSONConvertible for SimpleBitset {
    fn to_json(&self) -> JSONNode {
        let ranges_vec: Vec<Vec<usize>> = self.rsb.ranges().map(|ri| vec![*ri.start(), *ri.end()]).collect();
        ranges_vec.to_json()
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        let ranges_vec: Vec<Vec<usize>> = Vec::from_json(node)?;
        let mut ranges = Vec::new();
        for mut v in ranges_vec {
            if v.len() != 2 { return Err(format!("Expected 2-element array, got {:?}", v)); }
            let end = v.pop().unwrap(); let start = v.pop().unwrap();
            ranges.push(start..=end);
        }
        Ok(SimpleBitset::from_rsb(RangeSetBlaze::from_iter(ranges)))
    }
}

impl JSONConvertible for NWAState {
    fn to_json(&self) -> JSONNode {
        let mut obj = BTreeMap::new();
        obj.insert("final_weight".to_string(), self.final_weight.to_json());
        obj.insert("epsilons".to_string(), self.epsilons.to_json());
        obj.insert("transitions".to_string(), self.transitions.to_json());
        JSONNode::Object(obj)
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        let mut obj = node.into_object()?;
        Ok(NWAState {
            final_weight: Option::<Weight>::from_json(obj.remove("final_weight").ok_or("Missing final_weight")?)?,
            epsilons: Vec::<(StateID, Weight)>::from_json(obj.remove("epsilons").ok_or("Missing epsilons")?)?,
            transitions: BTreeMap::<Label, Vec<(StateID, Weight)>>::from_json(obj.remove("transitions").ok_or("Missing transitions")?)?,
        })
    }
}

impl JSONConvertible for NWA {
    fn to_json(&self) -> JSONNode {
        let mut obj = BTreeMap::new();
        obj.insert("states".to_string(), self.states.0.to_json());
        obj.insert("start_states".to_string(), self.body.start_states.to_json());
        JSONNode::Object(obj)
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        let mut obj = node.into_object()?;
        Ok(NWA {
            states: NWAStates(Vec::<NWAState>::from_json(obj.remove("states").ok_or("Missing states")?)?),
            body: NWABody { start_states: Vec::<StateID>::from_json(obj.remove("start_states").ok_or("Missing start_states")?)? },
        })
    }
}

impl JSONConvertible for DWAState {
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
        Ok(DWAState {
            transitions: BTreeMap::<Label, StateID>::from_json(obj.remove("transitions").ok_or("Missing transitions")?)?,
            final_weight: Option::<Weight>::from_json(obj.remove("final_weight").ok_or("Missing final_weight")?)?,
            trans_weights: BTreeMap::<Label, Weight>::from_json(obj.remove("trans_weights").ok_or("Missing trans_weights")?)?,
            state_weight: Option::<Weight>::from_json(obj.remove("state_weight").ok_or("Missing state_weight")?)?,
        })
    }
}

impl JSONConvertible for DWA {
    fn to_json(&self) -> JSONNode {
        // Local pool for weights to compact serialization
        let mut weight_pool: Vec<Weight> = Vec::new();
        let mut weight_map: HashMap<Weight, usize> = HashMap::new();

        let mut intern_weight = |w: &Weight| -> usize {
            if let Some(&id) = weight_map.get(w) {
                id
            } else {
                let id = weight_pool.len();
                weight_pool.push(w.clone());
                weight_map.insert(w.clone(), id);
                id
            }
        };

        let mut states_json = Vec::with_capacity(self.states.len());
        for state in &self.states.0 {
            // Group transitions by (dest, weight_id)
            let mut groups: BTreeMap<(StateID, Option<usize>), Vec<Label>> = BTreeMap::new();
            for (label, dest) in &state.transitions {
                let w = state.trans_weights.get(label);
                let w_id = w.map(|x| intern_weight(x));
                groups.entry((*dest, w_id)).or_default().push(*label);
            }

            // Serialize groups: [dest, weight_id_opt, [labels...]]
            let mut trans_json = Vec::new();
            for ((dest, w_id), labels) in groups {
                let w_val = w_id.map(|n| n.to_json()).unwrap_or(JSONNode::Null);
                trans_json.push(JSONNode::Array(vec![
                    dest.to_json(),
                    w_val,
                    labels.to_json()
                ]));
            }

            let mut s_obj = BTreeMap::new();
            s_obj.insert("t".to_string(), JSONNode::Array(trans_json));
            if let Some(w) = &state.final_weight {
                let w_id = intern_weight(w);
                s_obj.insert("f".to_string(), w_id.to_json());
            }
            if let Some(w) = &state.state_weight {
                let w_id = intern_weight(w);
                s_obj.insert("s".to_string(), w_id.to_json());
            }
            states_json.push(JSONNode::Object(s_obj));
        }

        let mut obj = BTreeMap::new();
        obj.insert("states".to_string(), JSONNode::Array(states_json));
        obj.insert("start_state".to_string(), self.body.start_state.to_json());
        obj.insert("weight_pool".to_string(), weight_pool.to_json());
        JSONNode::Object(obj)
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        let mut obj = node.into_object()?;

        let pool_node = obj.remove("weight_pool").ok_or("Missing weight_pool")?;
        let weight_pool = Vec::<Weight>::from_json(pool_node)?;
        let get_weight = |n: JSONNode| -> Result<Weight, String> {
            let idx = usize::from_json(n)?;
            weight_pool.get(idx).cloned().ok_or_else(|| "Weight index out of bounds".to_string())
        };

        let states_node = obj.remove("states").ok_or("Missing states")?;
        let states_arr = states_node.into_array()?;
        let mut states = Vec::with_capacity(states_arr.len());

        for s_node in states_arr {
            let mut s_obj = s_node.into_object()?;
            let final_weight = if let Some(n) = s_obj.remove("f") { Some(get_weight(n)?) } else { None };
            let state_weight = if let Some(n) = s_obj.remove("s") { Some(get_weight(n)?) } else { None };

            let mut transitions = BTreeMap::new();
            let mut trans_weights = BTreeMap::new();

            if let Some(t_node) = s_obj.remove("t") {
                let t_arr = t_node.into_array()?;
                for group_node in t_arr {
                    let mut group = group_node.into_array()?;
                    if group.len() != 3 { return Err("Invalid transition group format".to_string()); }
                    let labels_node = group.pop().unwrap();
                    let w_node = group.pop().unwrap();
                    let dest_node = group.pop().unwrap();

                    let dest = StateID::from_json(dest_node)?;
                    let weight = match w_node {
                        JSONNode::Null => None,
                        n => Some(get_weight(n)?),
                    };

                    let labels = Vec::<Label>::from_json(labels_node)?;
                    for label in labels {
                        transitions.insert(label, dest);
                        if let Some(w) = &weight {
                            trans_weights.insert(label, w.clone());
                        }
                    }
                }
            }
            states.push(DWAState { transitions, final_weight, trans_weights, state_weight });
        }

        Ok(DWA {
            states: DWAStates(states),
            body: DWABody { start_state: StateID::from_json(obj.remove("start_state").ok_or("Missing start_state")?)? },
        })
    }
}
