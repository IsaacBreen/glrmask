// src/precompute4/weighted_automata/json.rs

#![allow(dead_code)]
#![allow(clippy::needless_borrow)]

use super::rangeset::RangeSet;
use super::common::{Label, StateID, Weight};
use super::dwa::{DWABody, DWAState, DWAStates, DWA, DWABuildError};
use super::nwa::{NWABody, NWAState, NWAStates, NWA};
use crate::json_serialization::{JSONConvertible, JSONNode};
use crate::json_serialization::JSONNode::Array;
use range_set_blaze::RangeSetBlaze;
use std::collections::{BTreeMap, HashMap};
use std::iter::FromIterator;

impl JSONConvertible for RangeSet {
    fn to_json(&self) -> JSONNode {
        let ranges_vec: Vec<Vec<usize>> = self
            .rsb
            .ranges()
            .map(|ri| vec![*ri.start(), *ri.end()])
            .collect();
        ranges_vec.to_json()
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        let ranges_vec: Vec<Vec<usize>> = Vec::from_json(node)?;
        let mut ranges = Vec::new();
        for mut v in ranges_vec {
            if v.len() != 2 {
                return Err(format!("Expected 2-element array, got {:?}", v));
            }
            let end = v.pop().unwrap();
            let start = v.pop().unwrap();
            ranges.push(start..=end);
        }
        Ok(RangeSet::from_rsb(RangeSetBlaze::from_iter(ranges)))
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
            final_weight: Option::<Weight>::from_json(
                obj.remove("final_weight").ok_or("Missing final_weight")?,
            )?,
            epsilons: Vec::<(StateID, Weight)>::from_json(
                obj.remove("epsilons").ok_or("Missing epsilons")?,
            )?,
            transitions: BTreeMap::<Label, Vec<(StateID, Weight)>>::from_json(
                obj.remove("transitions").ok_or("Missing transitions")?,
            )?,
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
            states: NWAStates(Vec::<NWAState>::from_json(
                obj.remove("states").ok_or("Missing states")?,
            )?),
            body: NWABody {
                start_states: Vec::<StateID>::from_json(
                    obj.remove("start_states").ok_or("Missing start_states")?,
                )?,
            },
        })
    }
}

impl JSONConvertible for DWAState {
    fn to_json(&self) -> JSONNode {
        let mut obj = BTreeMap::new();
        obj.insert("transitions".to_string(), self.transitions.to_json());
        obj.insert("final_weight".to_string(), self.final_weight.to_json());
        obj.insert("trans_weights".to_string(), self.trans_weights.to_json());
        JSONNode::Object(obj)
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        let mut obj = node.into_object()?;
        Ok(DWAState {
            transitions: BTreeMap::<Label, StateID>::from_json(
                obj.remove("transitions").ok_or("Missing transitions")?,
            )?,
            final_weight: Option::<Weight>::from_json(
                obj.remove("final_weight").ok_or("Missing final_weight")?,
            )?,
            trans_weights: BTreeMap::<Label, Weight>::from_json(
                obj.remove("trans_weights").ok_or("Missing trans_weights")?,
            )?,
            state_weight: None,
        })
    }
}

impl JSONConvertible for DWA {
    fn to_json(&self) -> JSONNode {
        // Local pool for weights to compact serialization.
        // Each distinct Weight appears once in `weight_pool` and is
        // referenced by index from states/transitions.
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

        // States are serialized as arrays to reduce key overhead while
        // keeping the structure reasonably interpretable:
        //
        //   "states": [
        //     [transitions],
        //     [final_weight_idx, transitions],
        //     ...
        //   ]
        //
        // `transitions` is an array of groups:
        //
        //   [dest_state, weight_idx_or_null, [label1, label2, ...]]
        //
        // meaning: for each label in the label-list, there is a transition
        // `label -> dest_state` with the given weight (or default weight if null).
        let mut states_json = Vec::with_capacity(self.states.len());

        for state in &self.states.0 {
            // Group transitions by (dest, weight_id) to compress runs of
            // transitions that share the same target and weight.
            let mut groups: BTreeMap<(StateID, Option<usize>), Vec<Label>> = BTreeMap::new();
            for (label, dest) in &state.transitions {
                let w_id = state.trans_weights.get(label).map(|w| intern_weight(w));
                groups.entry((*dest, w_id)).or_default().push(*label);
            }

            // Serialize transition groups: [dest_state, weight_idx_or_null, [labels...]]
            let mut trans_json = Vec::with_capacity(groups.len());
            for ((dest, w_id), labels) in groups {
                let w_node = match w_id {
                    Some(id) => id.to_json(),
                    None => JSONNode::Null,
                };
                trans_json.push(Array(vec![
                    dest.to_json(),
                    w_node,
                    labels.to_json(),
                ]));
            }

            let fw_idx_opt = state.final_weight.as_ref().map(|w| intern_weight(w));
            let trans_node = Array(trans_json);

            // Use the shortest array form that still encodes all information.
            let state_node = match fw_idx_opt {
                None => Array(vec![trans_node]),
                Some(fw_idx) => Array(vec![fw_idx.to_json(), trans_node]),
            };

            states_json.push(state_node);
        }

        let mut obj = BTreeMap::new();
        obj.insert("states".to_string(), Array(states_json));
        obj.insert("start_state".to_string(), self.body.start_state.to_json());
        obj.insert("weight_pool".to_string(), weight_pool.to_json());
        JSONNode::Object(obj)
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        let mut obj = node.into_object()?;

        let pool_node = obj
            .remove("weight_pool")
            .ok_or("Missing weight_pool")?;
        let weight_pool = Vec::<Weight>::from_json(pool_node)?;

        let get_weight = |n: JSONNode| -> Result<Weight, String> {
            let idx = usize::from_json(n)?;
            weight_pool
                .get(idx)
                .cloned()
                .ok_or_else(|| format!("Weight index {} out of bounds", idx))
        };

        let states_node = obj
            .remove("states")
            .ok_or("Missing states")?;
        let states_arr = states_node.into_array()?;
        let mut states = Vec::with_capacity(states_arr.len());

        for (state_idx, s_node) in states_arr.into_iter().enumerate() {
            let mut parts = s_node
                .into_array()
                .map_err(|e| format!("Expected array for state {}: {}", state_idx, e))?;
            if parts.is_empty() {
                return Err(format!("State {} array is empty", state_idx));
            }

            // Last element is always the transitions array.
            let trans_node = parts.pop().unwrap();
            let trans_arr = trans_node
                .into_array()
                .map_err(|e| format!("Invalid transitions array at state {}: {}", state_idx, e))?;

            let final_weight = match parts.len() {
                0 => None,
                _ => {
                    let fw_node = parts.pop().unwrap();
                    match fw_node {
                        JSONNode::Null => None,
                        n => Some(get_weight(n)?),
                    }
                }
            };

            let mut transitions = BTreeMap::new();
            let mut trans_weights = BTreeMap::new();

            for (group_idx, group_node) in trans_arr.into_iter().enumerate() {
                let mut group = group_node
                    .into_array()
                    .map_err(|e| {
                        format!(
                            "Invalid transition group at state {}, group {}: {}",
                            state_idx, group_idx, e
                        )
                    })?;
                if group.len() != 3 {
                    return Err(format!(
                        "Invalid transition group format at state {}, group {}: expected 3 elements, got {}",
                        state_idx,
                        group_idx,
                        group.len()
                    ));
                }

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
                    if transitions.insert(label, dest).is_some() {
                        return Err(format!(
                            "Duplicate transition on label {} in state {}",
                            label, state_idx
                        ));
                    }
                    if let Some(w) = &weight {
                        trans_weights.insert(label, w.clone());
                    }
                }
            }

            states.push(DWAState {
                transitions,
                final_weight,
                trans_weights,
                state_weight: None,
            });
        }

        Ok(DWA {
            states: DWAStates(states),
            body: DWABody {
                start_state: StateID::from_json(
                    obj.remove("start_state").ok_or("Missing start_state")?,
                )?,
            },
        })
    }
}