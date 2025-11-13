// src/precompute4/weighted_automata/json.rs

#![allow(dead_code)]
#![allow(clippy::needless_borrow)]

use super::bitset::SimpleBitset;
use super::common::{StateID, Weight};
use super::dwa::{DWABody, DWAState, DWAStates, DWA};
use crate::json_serialization::{JSONConvertible, JSONNode};
use range_set_blaze::RangeSetBlaze;
use std::collections::BTreeMap;
use std::iter::FromIterator;

impl JSONConvertible for SimpleBitset {
    fn to_json(&self) -> JSONNode {
        let ranges_vec: Vec<Vec<usize>> = self.rsb.ranges().map(|ri| vec![*ri.start(), *ri.end()]).collect();
        ranges_vec.to_json()
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        let ranges_vec: Vec<Vec<usize>> = Vec::from_json(node)?;
        let mut ranges = Vec::new();
        for mut v in ranges_vec {
            if v.len() != 2 {
                return Err(format!("Expected 2-element array for SimpleBitset range, got {:?}", v));
            }
            let end = v.pop().unwrap();
            let start = v.pop().unwrap();
            ranges.push(start..=end);
        }
        Ok(SimpleBitset::from_rsb(RangeSetBlaze::from_iter(ranges)))
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
        let transitions =
            BTreeMap::<i16, StateID>::from_json(obj.remove("transitions").ok_or("Missing 'transitions' field")?)?;
        let final_weight =
            Option::<Weight>::from_json(obj.remove("final_weight").ok_or("Missing 'final_weight' field")?)?;
        let trans_weights = BTreeMap::<i16, Weight>::from_json(
            obj.remove("trans_weights").ok_or("Missing 'trans_weights' field")?,
        )?;
        let state_weight =
            Option::<Weight>::from_json(obj.remove("state_weight").ok_or("Missing 'state_weight' field")?)?;
        Ok(DWAState { transitions, final_weight, trans_weights, state_weight })
    }
}

impl JSONConvertible for DWA {
    fn to_json(&self) -> JSONNode {
        let mut obj = BTreeMap::new();
        obj.insert("states".to_string(), self.states.0.to_json());
        obj.insert("start_state".to_string(), self.body.start_state.to_json());
        JSONNode::Object(obj)
    }
    fn from_json(node: JSONNode) -> Result<Self, String> {
        let mut obj = node.into_object()?;
        let states = Vec::<DWAState>::from_json(obj.remove("states").ok_or("Missing 'states' field")?)?;
        let start_state = StateID::from_json(obj.remove("start_state").ok_or("Missing 'start_state' field")?)?;
        Ok(DWA { states: DWAStates(states), body: DWABody { start_state } })
    }
}
