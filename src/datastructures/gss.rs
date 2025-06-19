use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::fmt::{Debug, Write};
use std::hash::{Hash, Hasher};
use std::cmp::Ordering;
use std::collections::hash_map::DefaultHasher;
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Sub, SubAssign};
use bimap::BiBTreeMap;
use deterministic_hash::DeterministicHasher;
use std::any::{Any, TypeId};
use profiler_macro::{time_it, timeit};

use crate::glr::parser::ParseStateEdgeContent;
use crate::constraint::{LLMTokenBV, TerminalBV};
use crate::datastructures::gss::acc_mod::Acc;
use crate::datastructures::hybrid_bitset::HybridBitset;
use crate::glr::grammar::Terminal;
use crate::tokenizer::{LLMTokenID, TokenizerStateID};
use crate::types::TerminalID;

// Type aliases for cleaner signatures, now concrete
pub type MaxDepth = usize;
type NodeMap = BTreeMap<MaxDepth, BTreeMap<ParseStateEdgeContent, Arc<GSSNode>>>;
type NodeCache = HashMap<NodeMap, Arc<GSSNode>>;
type NodeSet = BTreeSet<(Arc<GSSNode>, ParseStateEdgeContent)>;

pub type LLMTokenInfo = Option<LLMTokenBV>;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TerminalInfoValue {
    pub union: TerminalBV,
    pub intersection: TerminalBV,
}

impl TerminalInfoValue {
    pub fn new(union: TerminalBV, intersection: TerminalBV) -> Self {
        Self { union, intersection }
    }

    fn zeros() -> Self {
        Self {
            union: TerminalBV::zeros(),
            intersection: TerminalBV::zeros(),
        }
    }

    pub fn identity_for_union_or_intersection() -> Self {
        Self {
            union: TerminalBV::zeros(),
            intersection: TerminalBV::max_ones(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.union.is_empty()
    }

    pub fn contains(&self, index: usize) -> bool {
        self.union.contains(index)
    }

    pub fn insert(&mut self, index: usize) {
        self.union.insert(index);
    }

    pub fn is_disjoint(&self, other: &HybridBitset) -> bool {
        self.union.is_disjoint(&other)
    }
}

impl BitAnd<&TerminalBV> for &TerminalInfoValue {
    type Output = TerminalInfoValue;
    fn bitand(self, rhs: &TerminalBV) -> Self::Output {
        TerminalInfoValue {
            union: &self.union & rhs,
            intersection: &self.intersection & rhs,
        }
    }
}

impl BitAndAssign<&TerminalBV> for TerminalInfoValue {
    fn bitand_assign(&mut self, rhs: &TerminalBV) {
        self.union &= rhs;
        self.intersection &= rhs;
    }
}

impl BitOr<&TerminalBV> for &TerminalInfoValue {
    type Output = TerminalInfoValue;
    fn bitor(self, rhs: &TerminalBV) -> Self::Output {
        TerminalInfoValue {
            union: &self.union | rhs,
            intersection: &self.intersection & rhs,
        }
    }
}

impl BitOrAssign<&TerminalBV> for TerminalInfoValue {
    fn bitor_assign(&mut self, rhs: &TerminalBV) {
        self.union |= rhs;
        self.intersection &= rhs;
    }
}

impl BitXor<&TerminalBV> for &TerminalInfoValue {
    type Output = TerminalInfoValue;
    fn bitxor(self, rhs: &TerminalBV) -> Self::Output {
        TerminalInfoValue {
            union: &self.union | rhs,
            intersection: &self.intersection | rhs,
        }
    }
}

impl BitXorAssign<&TerminalBV> for TerminalInfoValue {
    fn bitxor_assign(&mut self, rhs: &TerminalBV) {
        self.union |= rhs;
        self.intersection |= rhs;
    }
}

impl Sub<&TerminalBV> for &TerminalInfoValue {
    type Output = TerminalInfoValue;
    fn sub(self, rhs: &TerminalBV) -> Self::Output {
        TerminalInfoValue {
            union: &self.union - rhs,
            intersection: &self.intersection - rhs,
        }
    }
}

impl SubAssign<&TerminalBV> for TerminalInfoValue {
    fn sub_assign(&mut self, rhs: &TerminalBV) {
        self.union -= rhs;
        self.intersection -= rhs;
    }
}

impl BitAnd<&TerminalInfoValue> for &TerminalInfoValue {
    type Output = TerminalInfoValue;
    fn bitand(self, rhs: &TerminalInfoValue) -> Self::Output {
        TerminalInfoValue {
            union: &self.union & &rhs.union,
            intersection: &self.intersection & &rhs.intersection,
        }
    }
}

impl BitAndAssign<&TerminalInfoValue> for TerminalInfoValue {
    fn bitand_assign(&mut self, rhs: &TerminalInfoValue) {
        self.union &= &rhs.union;
        self.intersection &= &rhs.intersection;
    }
}

impl BitOr<&TerminalInfoValue> for &TerminalInfoValue {
    type Output = TerminalInfoValue;
    fn bitor(self, rhs: &TerminalInfoValue) -> Self::Output {
        TerminalInfoValue {
            union: &self.union | &rhs.union,
            intersection: &self.intersection & &rhs.intersection,
        }
    }
}

impl BitOrAssign<&TerminalInfoValue> for TerminalInfoValue {
    fn bitor_assign(&mut self, rhs: &TerminalInfoValue) {
        self.union |= &rhs.union;
        self.intersection &= &rhs.intersection;
    }
}

impl BitXor<&TerminalInfoValue> for &TerminalInfoValue {
    type Output = TerminalInfoValue;
    fn bitxor(self, rhs: &TerminalInfoValue) -> Self::Output {
        TerminalInfoValue {
            union: &self.union | &rhs.union,
            intersection: &self.intersection | &rhs.intersection,
        }
    }
}

impl BitXorAssign<&TerminalInfoValue> for TerminalInfoValue {
    fn bitxor_assign(&mut self, rhs: &TerminalInfoValue) {
        self.union |= &rhs.union;
        self.intersection |= &rhs.intersection;
    }
}

pub type TerminalInfo = BTreeMap<TokenizerStateID, TerminalInfoValue>;

pub trait PathAccumulator<Other=Self>: Sized + Clone + Debug + Eq + PartialEq + Ord + PartialOrd + Hash {
    fn union_assign(&mut self, other: Other);
    fn intersect_assign(&mut self, right: Other); // Renamed from pop_assign
    fn union(mut self, other: Other) -> Self {
        self.union_assign(other);
        self
    }
    fn intersect(mut self, right: Other) -> Self { // Renamed from pop
        self.intersect_assign(right);
        self
    }
    fn intersect_has_effect(&self, right: &Other) -> bool;
}

impl PathAccumulator for () {
    fn union_assign(&mut self, _other: Self) { }
    fn intersect_assign(&mut self, _right: Self) { } // Renamed from pop_assign
    fn intersect_has_effect(&self, _right: &Self) -> bool { false }
}

impl PathAccumulator for TerminalInfoValue {
    fn union_assign(&mut self, other: Self) {
        self.union |= &other.union;
        self.intersection &= &other.intersection;
    }

    fn intersect_assign(&mut self, right: Self) {
        self.union &= &right.union;
        self.union |= &right.intersection;
        self.intersection |= &right.intersection;
    }

    // fn intersect_has_effect(&self, right: &Self) -> bool {
    //     let new_union = &self.union & &right.union;
    //     let new_intersection = &self.intersection & &right.intersection;
    //     new_union != self.union || new_intersection != self.intersection
    // }
}

impl PathAccumulator for Option<LLMTokenBV> {
    // #[time_it]
    fn union_assign(&mut self, other: Self) {
        match (self.as_mut(), other) {
            (Some(self_bv), Some(other_bv)) => {
                if self_bv.inner() == other_bv.inner() {
                    return;
                }
                if self_bv.is_empty() {
                    *self_bv = other_bv;
                    return;
                }
                if other_bv.is_empty() {
                    return;
                }
                if false {
                    // let BIG_RANGE_LEN = 1;
                    // if other_bv.inner().ranges_len() > BIG_RANGE_LEN && self_bv.inner().ranges_len() > BIG_RANGE_LEN {
                    //     println!("WARNING: union_assign: self_bv.inner().ranges_len() > BIG_RANGE_LEN && other_bv.inner().ranges_len() > BIG_RANGE_LEN, self_bv.inner().ranges_len(): {}, other_bv.inner().ranges_len(): {}", self_bv.inner().ranges_len(), other_bv.inner().ranges_len());
                    //     println!("self_bv: {:?}", &self_bv);
                    //     println!("other_bv: {:?}", &other_bv);
                    // }

                    // Count number of 'holes' - gaps between ranges of size 1
                    let BIG_HOLE_LEN = 20;
                    let mut self_holes = 0;
                    let mut right_holes = 0;
                    let mut self_holes_pos = Vec::new();
                    let mut right_holes_pos = Vec::new();
                    let mut ranges = self_bv.inner().ranges();
                    let mut prev_range_end;
                    if let Some(start_range) = ranges.next() {
                        prev_range_end = *start_range.end();
                        for range in ranges {
                            let gap = range.start() - prev_range_end;
                            if gap == 2 {
                                self_holes += 1;
                                self_holes_pos.push(range.start() - 1);
                            }
                            prev_range_end = *range.end();
                        }
                    }
                    let mut ranges = other_bv.inner().ranges();
                    let mut prev_range_end;
                    if let Some(start_range) = ranges.next() {
                        prev_range_end = *start_range.end();
                        for range in ranges {
                            let gap = range.start() - prev_range_end;
                            if gap == 2 {
                                right_holes += 1;
                                right_holes_pos.push(range.start() - 1);
                            }
                            prev_range_end = *range.end();
                        }
                    }
                    // let min_hole_pos = 2560;
                    // let max_hole_pos = 4343;
                    let min_hole_pos = 0;
                    let max_hole_pos = 1000000;
                    let is_eligible = self_holes_pos.iter().any(|&pos| min_hole_pos < pos && pos < max_hole_pos) || right_holes_pos.iter().any(|&pos| min_hole_pos < pos && pos < max_hole_pos);
                    if (self_holes > BIG_HOLE_LEN || right_holes > BIG_HOLE_LEN) && is_eligible {
                        eprintln!("WARNING: union_assign: self_holes > BIG_HOLE_LEN || right_holes > BIG_HOLE_LEN, self_holes: {}, right_holes: {}", self_holes, right_holes);
                        eprintln!("self_bv: {:?}", &self_bv);
                        eprintln!("other_bv: {:?}", &other_bv);
                        eprintln!("self_holes_pos: {:?}", &self_holes_pos);
                        eprintln!("right_holes_pos: {:?}", &right_holes_pos);
                        // panic!("union_assign: self_holes > BIG_HOLE_LEN && right_holes > BIG_HOLE_LEN");
                    }
                    //
                    // let time_str = format!("union_assign: self_bv.inner().ranges_len(): {}, other_bv.inner().ranges_len(): {}", self_bv.inner().ranges_len(), other_bv.inner().ranges_len());

                    // fn round_down_to_power_of_10(x: usize) -> usize {
                    //     if x == 0 {
                    //         return 0;
                    //     }
                    //
                    //     let mut power = 1;
                    //     while power * 10 <= x {
                    //         power *= 10;
                    //     }
                    //     power
                    // }
                    // let self_bv_len = round_down_to_power_of_10(self_bv.inner().ranges_len());
                    // let other_bv_len = round_down_to_power_of_10(other_bv.inner().ranges_len());
                    // let overlap_len = round_down_to_power_of_10((&*self_bv & &other_bv).inner().ranges_len());
                    // let difference_len = round_down_to_power_of_10(((&*self_bv | &other_bv) - (&*self_bv & &other_bv)).inner().ranges_len());
                    // let time_str = format!("union_assign: self_bv.inner().ranges_len(): {}, other_bv.inner().ranges_len(): {}, overlap_len: {}, difference_len: {}",
                    //     self_bv_len, other_bv_len, overlap_len, difference_len
                    // );
                }
                // timeit!(time_str,
                    *self_bv |= other_bv
                // );
                // An empty bitset resulting from a union is still Some(empty_bv), not None.
            }
            (None, Some(other_bv)) => {
                *self = Some(LLMTokenBV::max_ones());
            }
            (Some(_), None) => {
                *self = Some(LLMTokenBV::max_ones());
            }
            (None, None) => {
                // self remains None
            }
        }
    }

    // #[time_it]
    fn intersect_assign(&mut self, right: Self) {
        match (self.as_mut(), right) {
            (Some(self_bv), Some(right_bv)) => {
                if self_bv.inner() == right_bv.inner() {
                    return;
                }
                if self_bv.is_empty() {
                    return;
                }
                if right_bv.is_empty() {
                    *self_bv = right_bv;
                    return;
                }
                // let BIG_RANGE_LEN = 1;
                // if right_bv.inner().ranges_len() > BIG_RANGE_LEN && self_bv.inner().ranges_len() > BIG_RANGE_LEN {
                //     println!("WARNING: intersection_assign: self_bv.inner().ranges_len() > BIG_RANGE_LEN && right_bv.inner().ranges_len() > BIG_RANGE_LEN, self_bv.inner().ranges_len(): {}, right_bv.inner().ranges_len(): {}", self_bv.inner().ranges_len(), right_bv.inner().ranges_len());
                //     println!("self_bv: {:?}", &self_bv);
                //     println!("right_bv: {:?}", &right_bv);
                // }

                // Count number of 'holes' - gaps between ranges of size 1
                if false {
                    let BIG_HOLE_LEN = 10;
                    let mut self_holes = 0;
                    let mut right_holes = 0;
                    let mut self_holes_pos = Vec::new();
                    let mut right_holes_pos = Vec::new();
                    let mut ranges = self_bv.inner().ranges();
                    let mut prev_range_end;
                    if let Some(start_range) = ranges.next() {
                        prev_range_end = *start_range.end();
                        for range in ranges {
                            let gap = range.start() - prev_range_end;
                            if gap == 2 {
                                self_holes += 1;
                                self_holes_pos.push(range.start() - 1);
                            }
                            prev_range_end = *range.end();
                        }
                    }
                    let mut ranges = right_bv.inner().ranges();
                    let mut prev_range_end;
                    if let Some(start_range) = ranges.next() {
                        prev_range_end = *start_range.end();
                        for range in ranges {
                            let gap = range.start() - prev_range_end;
                            if gap == 2 {
                                right_holes += 1;
                                right_holes_pos.push(range.start() - 1);
                            }
                            prev_range_end = *range.end();
                        }
                    }
                    let min_hole_pos = 2560;
                    let max_hole_pos = 4343;
                    let is_eligible = self_holes_pos.iter().any(|&pos| min_hole_pos < pos && pos < max_hole_pos) || right_holes_pos.iter().any(|&pos| min_hole_pos < pos && pos < max_hole_pos);
                    if (self_holes > BIG_HOLE_LEN || right_holes > BIG_HOLE_LEN) && is_eligible {
                        eprintln!("WARNING: intersection_assign: self_holes > BIG_HOLE_LEN || right_holes > BIG_HOLE_LEN, self_holes: {}, right_holes: {}", self_holes, right_holes);
                        eprintln!("self_bv: {:?}", &self_bv);
                        eprintln!("right_bv: {:?}", &right_bv);
                        eprintln!("self_holes_pos: {:?}", &self_holes_pos);
                        eprintln!("right_holes_pos: {:?}", &right_holes_pos);
                        // panic!("intersection_assign: self_holes > BIG_HOLE_LEN && right_holes > BIG_HOLE_LEN");
                    }
                    //
                    // // let time_str = format!("intersection_assign: self_bv.inner().ranges_len(): {}, right_bv.inner().ranges_len(): {}", self_bv.inner().ranges_len(), right_bv.inner().ranges_len());
                    //
                    // // fn round_down_to_power_of_10(x: usize) -> usize {
                    // //     if x == 0 {
                    // //         return 0;
                    // //     }
                    // //
                    // //     let mut power = 1;
                    // //     while power * 10 <= x {
                    // //         power *= 10;
                    // //     }
                    // //     power
                    // // }
                    // // let self_bv_len = round_down_to_power_of_10(self_bv.inner().ranges_len());
                    // // let right_bv_len = round_down_to_power_of_10(right_bv.inner().ranges_len());
                    // // let overlap_len = round_down_to_power_of_10((&*self_bv & &right_bv).inner().ranges_len());
                    // // let difference_len = round_down_to_power_of_10(((&*self_bv | &right_bv) - (&*self_bv & &right_bv)).inner().ranges_len());
                    // // let time_str = format!("intersection_assign: self_bv.inner().ranges_len(): {}, right_bv.inner().ranges_len(): {}, overlap_len: {}, difference_len: {}",
                    // //     self_bv_len, right_bv_len, overlap_len, difference_len
                    // // );
                }
                // // timeit!(time_str,
                *self_bv &= right_bv
                // );
            }
            (None, Some(right_bv)) => {
                *self = Some(right_bv);
            }
            (Some(_), None) => {}
            (None, None) => {}
        }
    }

    fn intersect_has_effect(&self, right: &Self) -> bool {
        // self.clone().intersect(right.clone()) != *self
        match (self, right) {
            (Some(self_bv), Some(right_bv)) => {
                self_bv.is_subset(right_bv)
            }
            (None, Some(right_bv)) => {
                true
            }
            (Some(_), None) => {
                false
            }
            (None, None) => {
                false
            }
        }
    }
}

fn compute_max_depth(predecessors: &NodeMap) -> MaxDepth {
    predecessors.keys().next_back().map_or(0, |max_pred_depth| max_pred_depth + 1)
    // 0
}

fn compute_hash_key(predecessors: &NodeMap) -> u64 {
    let mut hasher = DeterministicHasher::new(DefaultHasher::new());
    for (depth, preds_for_depth) in predecessors {
        depth.hash(&mut hasher);
        for (edge_val, pred_arc) in preds_for_depth {
            edge_val.hash(&mut hasher);
            pred_arc.hash_key_cache.hash(&mut hasher);
        }
    }
    hasher.finish()
}

#[time_it]
pub fn disallowed_terminals_intersect_assign(left: &mut TerminalInfo, right: TerminalInfo) {
    let mut all_keys = BTreeSet::new();
    all_keys.extend(left.keys());
    all_keys.extend(right.keys());
    for tokenizer_state_id in all_keys {
        // An absent key means "no terminals disallowed" -> zeros()
        let left_value = left.get(&tokenizer_state_id).cloned().unwrap_or_else(TerminalInfoValue::identity_for_union_or_intersection);
        let right_value = right.get(&tokenizer_state_id).cloned().unwrap_or_else(TerminalInfoValue::identity_for_union_or_intersection);
        let intersection = &left_value & &right_value;
        if !intersection.is_empty() {
            left.insert(tokenizer_state_id, intersection);
        } else {
            left.remove(&tokenizer_state_id);
        }
    }
}

#[time_it]
pub fn disallowed_terminals_union_assign(left: &mut TerminalInfo, right: TerminalInfo) {
    let mut all_keys = BTreeSet::new();
    all_keys.extend(left.keys());
    all_keys.extend(right.keys());
    for tokenizer_state_id in all_keys {
        // An absent key means "no terminals disallowed" -> zeros()
        let left_value = left.get(&tokenizer_state_id).cloned().unwrap_or_else(TerminalInfoValue::identity_for_union_or_intersection);
        let right_value = right.get(&tokenizer_state_id).cloned().unwrap_or_else(TerminalInfoValue::identity_for_union_or_intersection);
        let union = &left_value | &right_value;
        if !union.is_empty() {
            left.insert(tokenizer_state_id, union);
        } else {
            left.remove(&tokenizer_state_id);
        }
    }
}

#[time_it]
pub fn disallowed_terminals_add(left: &mut TerminalInfo, right: BTreeMap<TokenizerStateID, TerminalBV>) {
    let mut all_keys = BTreeSet::new();
    all_keys.extend(left.keys());
    all_keys.extend(right.keys());
    for tokenizer_state_id in all_keys {
        // An absent key means "no terminals disallowed" -> zeros()
        let mut left_value = left.get(&tokenizer_state_id).cloned().unwrap_or_else(TerminalInfoValue::zeros);
        let right_value = right.get(&tokenizer_state_id).cloned().unwrap_or_else(TerminalBV::zeros);
        // Proper union
        left_value.union |= &right_value;
        left_value.intersection |= &right_value;
        left.insert(tokenizer_state_id, left_value);
    }
}

#[derive(Clone, Copy)]
pub struct GSSPeek<'a> {
    pub(crate) parent_node: &'a GSSNode,
    edge_value: &'a ParseStateEdgeContent,
    pub(crate) predecessor_node: &'a Arc<GSSNode>,
}

impl<'a> GSSPeek<'a> {
    pub fn edge_value(&self) -> &'a ParseStateEdgeContent {
        self.edge_value
    }

    pub fn predecessor(&self) -> &'a Arc<GSSNode> {
        self.predecessor_node
    }

    /// Returns a GSS node representing the stack for this specific peeked path.
    /// This is equivalent to popping 0 elements.
    pub fn to_node(&self) -> GSSNode {
        GSSNode::new_with_single_predecessor(
            self.predecessor_node.clone(),
            self.edge_value.clone(),
            self.parent_node.acc.clone(),
        )
    }

    pub fn to_arc_node(&self) -> Arc<GSSNode> {
        Arc::new(self.to_node())
    }

    /// Pops `n` elements from the stack represented by this peek.
    /// `n=0` is equivalent to `to_node()`.
    /// `n=1` returns the predecessor node with an updated accumulator.
    /// `n>1` pops `n-1` from the predecessor.
    /// The accumulator of the returned node is correctly adjusted for the path.
    #[time_it]
    pub fn popn(&self, n: usize) -> Arc<GSSNode> {
        if n == 0 {
            return self.to_arc_node();
        }

        // For n >= 1, the result is based on the predecessor.
        // First, calculate the accumulator for the path to the predecessor.
        let path_acc = self.parent_node.acc.clone().intersect(self.predecessor_node.acc.clone());
        let pred_with_path_acc = Arc::new(self.predecessor_node.as_ref().clone().with_acc(path_acc));

        if n == 1 {
            pred_with_path_acc
        } else { // n > 1
            Arc::new(pred_with_path_acc.popn(n - 1))
        }
    }
}

pub mod acc_mod {
    use std::collections::{BTreeMap, BTreeSet};
    use profiler_macro::time_it;
    use crate::constraint::{LLMTokenBV, TerminalBV};
    use crate::datastructures::gss::{disallowed_terminals_union_assign, disallowed_terminals_intersect_assign, LLMTokenInfo, PathAccumulator, TerminalInfo, TerminalInfoValue};
    use crate::glr::grammar::Symbol::Terminal;
    use crate::tokenizer::TokenizerStateID;
    use crate::types::TerminalID;

    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct Acc {
        acc: LLMTokenInfo,
        // A map from tokenizer state ID to a set of terminals.
        // This LLM token info is valid for a given tokenizer state in the map if any terminals it *does* match against the **next** input string
        // are in the set of terminals under that tokenizer state in the map.
        // If this LLM token info is not valid for any tokenizer state, it is not valid at all (dead).
        // TODO: What about when a tokenizer state *can't* match the disallowed terminal? Shouldn't be necessary to have an entry for it right?
        //  But then we need an all-ones entry here, otherwise there's no tokenizer states in the map and it's considered 'not valid'.
        //  Perhaps we should...
        disallowed_terminals: TerminalInfo,
    }

    impl Acc {
        pub fn new(acc: LLMTokenInfo, disallowed_terminals: TerminalInfo) -> Self {
            Self { acc, disallowed_terminals }
        }

        pub fn new_fresh() -> Self {
            Self { acc: None, disallowed_terminals: BTreeMap::new() }
        }

        pub fn new_for_merging() -> Self {
            Self { acc: None, disallowed_terminals: BTreeMap::new() }
        }

        pub fn acc(&self) -> &LLMTokenInfo {
            &self.acc
        }

        pub fn acc_mut(&mut self) -> &mut LLMTokenInfo {
            &mut self.acc
        }

        pub fn disallowed_terminals(&self) -> &TerminalInfo {
            &self.disallowed_terminals
        }

        pub fn disallowed_terminals_mut(&mut self) -> &mut TerminalInfo {
            &mut self.disallowed_terminals
        }

        pub fn is_default(&self) -> bool {
            self.acc.is_none() && self.disallowed_terminals.is_empty()
        }

        pub fn is_dead(&self) -> bool {
            if let Some(acc) = &self.acc {
                if acc.is_empty() {
                    return true;
                }
            }
            false
        }

        pub fn is_alive(&self) -> bool {
            !self.is_dead()
        }
    }

    impl PathAccumulator for Acc {
        #[time_it]
        fn union_assign(&mut self, other: Self) {
            self.acc.union_assign(other.acc);
            disallowed_terminals_union_assign(&mut self.disallowed_terminals, other.disallowed_terminals);
        }
        #[time_it]
        fn intersect_assign(&mut self, right: Self) {
            self.acc.intersect_assign(right.acc);
            disallowed_terminals_intersect_assign(&mut self.disallowed_terminals, right.disallowed_terminals);
        }
        #[time_it]
        fn intersect_has_effect(&self, right: &Self) -> bool {
            self.acc.intersect_has_effect(&right.acc)
        }
    }
}

#[derive(Debug, Clone)]
pub struct GSSNode {
    acc: acc_mod::Acc,
    predecessors: NodeMap,
    hash_key_cache: u64,
    max_depth: MaxDepth,
}

#[derive(Clone)]
pub struct PathsIter<'a> { // No longer generic
    queue: VecDeque<(&'a GSSNode, Vec<ParseStateEdgeContent>)>,
}

impl<'a> Iterator for PathsIter<'a> { // No longer generic
    type Item = Vec<ParseStateEdgeContent>;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some((current_node, mut path_suffix)) = self.queue.pop_front() {
            if current_node.predecessors.is_empty() {
                path_suffix.reverse();
                return Some(path_suffix);
            }

            for pred_arc in current_node.predecessors.values().flat_map(|m| m.values()) {
                let mut new_path = path_suffix.clone();
                // This is not quite right, we need the edge value.
                // The logic of this iterator might need rethinking if edge values are important.
                // For now, let's assume we can get it.
                // The original code was: for (edge_val, pred_arc) in &current_node.predecessors
                // Let's fix this.
            }
            for (_, preds_for_depth) in &current_node.predecessors {
                for (edge_val, pred_arc) in preds_for_depth {
                    let mut new_path = path_suffix.clone();
                    new_path.push(edge_val.clone());
                    self.queue.push_back((pred_arc.as_ref(), new_path));
                }
            }
        }
        None
    }
}

fn process_predecessors(
    incoming: &NodeSet
) -> NodeMap {
    let mut grouped_by_depth: BTreeMap<MaxDepth, BTreeMap<ParseStateEdgeContent, Vec<Arc<GSSNode>>>> = BTreeMap::new();

    for (pred_arc, edge_val) in incoming {
        let depth = pred_arc.max_depth;
        grouped_by_depth
            .entry(depth)
            .or_default()
            .entry(edge_val.clone())
            .or_default()
            .push(pred_arc.clone());
    }

    let mut result: NodeMap = BTreeMap::new();
    for (depth, grouped_by_edge) in grouped_by_depth {
        let mut result_for_depth = BTreeMap::new();
        for (edge_val, pred_arcs) in grouped_by_edge {
            if pred_arcs.is_empty() { continue; }

            let mut iter = pred_arcs.into_iter();
            let first = iter.next().unwrap();

            if iter.len() == 0 {
                result_for_depth.insert(edge_val, first);
            } else {
                let mut merged_node_data = (*first).clone();
                for other_arc in iter {
                    merged_node_data.merge(&other_arc);
                }
                result_for_depth.insert(edge_val, Arc::new(merged_node_data));
            }
        }
        if !result_for_depth.is_empty() {
            result.insert(depth, result_for_depth);
        }
    }
    result
}

// Basic node creation and manipulation
impl GSSNode {
    pub fn new(acc: Acc) -> Self {
        let predecessors = NodeMap::new();
        let hash_key_cache = compute_hash_key(&predecessors);
        let max_depth = compute_max_depth(&predecessors);
        Self { acc, predecessors, hash_key_cache, max_depth }
    }
    
    // Private constructor used by simplification and other internal methods
    fn new_with_map(acc: Acc, predecessors: NodeMap) -> Self {
        let hash_key_cache = compute_hash_key(&predecessors);
        let max_depth = compute_max_depth(&predecessors);
        Self { acc, predecessors, hash_key_cache, max_depth }
    }

    // Helper to create a GSSNode with a single predecessor, used by push.
    fn new_with_single_predecessor(predecessor_arc: Arc<GSSNode>, edge_value: ParseStateEdgeContent, acc: Acc) -> Self {
        let mut predecessors_map = NodeMap::new();
        let mut inner_map = BTreeMap::new();
        inner_map.insert(edge_value, predecessor_arc.clone());
        predecessors_map.insert(predecessor_arc.max_depth, inner_map);
        Self::new_with_map(acc, predecessors_map)
    }

    fn predecessors(&self) -> &NodeMap {
        &self.predecessors
    }

    pub fn num_predecessors(&self) -> usize {
        self.predecessors.values().map(|inner_map| inner_map.len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.predecessors.is_empty()
    }

    pub fn acc_acc(&self) -> &LLMTokenInfo {
        &self.acc.acc()
    }

    pub fn acc_acc_mut(&mut self) -> &mut LLMTokenInfo {
        self.acc.acc_mut()
    }

    pub fn acc2(&self) -> &Acc {
        &self.acc
    }

    pub fn acc_mut2(&mut self) -> &mut Acc {
        &mut self.acc
    }

    // Helper to clone the node and set a new accumulator. Used internally.
    fn with_acc(mut self, acc: Acc) -> Self {
        self.acc = acc;
        self.hash_key_cache = compute_hash_key(&self.predecessors); // Recalculate hash if acc changes meaning
        self
    }
}


// Core manipulation methods
impl GSSNode {
    // Push now takes the acc for the new node
    pub fn push_with_acc(self, edge_value: ParseStateEdgeContent, acc_for_new_node: Acc) -> Self {
        Self::new_with_single_predecessor(Arc::new(self), edge_value, acc_for_new_node)
    }

    #[time_it]
    pub fn pop(&self) -> Self {
        // let mut result_acc = Acc::new_for_merging();
        let mut result_accs = Vec::new();
        let mut result_predecessors = NodeMap::new();

        for pred_arc in self.predecessors.values().flat_map(|m| m.values()) {
            // The acc of the path *through* self to pred_arc is self.acc intersected with pred_arc.acc
            let path_acc = self.acc.clone().intersect(pred_arc.acc.clone());
            if path_acc.is_dead() {
                continue;
            }
            result_accs.push(path_acc.clone());

            // Merge predecessors of pred_arc into result_predecessors
            // Each merged predecessor needs its acc updated based on path_acc
            for (inner_depth, inner_preds_for_depth) in &pred_arc.predecessors {
                let result_preds_for_depth = result_predecessors.entry(*inner_depth).or_default();
                for (inner_edge, inner_pred_arc) in inner_preds_for_depth {
                    let mut new_inner_pred_node_data = (**inner_pred_arc).clone();
                    new_inner_pred_node_data.acc = path_acc.clone().intersect(inner_pred_arc.acc.clone());
                    if new_inner_pred_node_data.acc.is_dead() {
                        continue;
                    }

                    match result_preds_for_depth.entry(inner_edge.clone()) {
                        std::collections::btree_map::Entry::Vacant(entry) => {
                            entry.insert(Arc::new(new_inner_pred_node_data));
                        }
                        std::collections::btree_map::Entry::Occupied(mut entry) => {
                            Arc::make_mut(entry.get_mut()).merge(&Arc::new(new_inner_pred_node_data));
                        }
                    }
                }
            }
        }
        let result_acc = if result_accs.is_empty() {
            Acc::new_for_merging()
        } else {
            let mut acc = result_accs.remove(0);
            for acc_to_merge in result_accs {
                acc.union_assign(acc_to_merge);
            }
            acc
        };
        Self::new_with_map(result_acc, result_predecessors)
    }


    // #[time_it("GSSNode::popn")]
    pub fn popn(&self, n: usize) -> Self {
        if n == 0 {
            self.clone()
        } else {
            self.pop().popn(n - 1)
        }
    }

    #[time_it]
    pub fn pop_iter(&self) -> Vec<(ParseStateEdgeContent, Arc<Self>)> {
        self.predecessors.values().flat_map(|m| m.iter()).filter_map(|(edge_val, pred_arc)| {
            let mut pred_arc = pred_arc.clone();
            // The acc for the path ending at pred_arc (after popping self)
            // is self.acc intersected with pred_arc's original acc.
            let path_acc = self.acc.clone().intersect(pred_arc.acc.clone());
            if path_acc.is_alive() {
                pred_arc = Arc::new(pred_arc.as_ref().clone().with_acc(path_acc));
                Some((edge_val.clone(), pred_arc))
            } else {
                None
            }
        }).collect()
    }

    pub fn peek_iter(&self) -> impl Iterator<Item = GSSPeek<'_>> {
        self.predecessors.values().flat_map(|m| m.iter()).map(|(edge_val, pred_arc)| {
            GSSPeek {
                parent_node: self,
                edge_value: edge_val,
                predecessor_node: pred_arc,
            }
        })
    }

    // #[time_it("GSSNode::merge")]
    pub fn merge(&mut self, other: &Self) {
        if self == other { return; }

        if other.predecessors.is_empty() {
            return;
        }
        if self.predecessors.is_empty() {
            *self = other.clone();
            return;
        }
        self.acc.union_assign(other.acc.clone());

        for (other_depth, other_preds_for_depth) in &other.predecessors {
            let self_preds_for_depth = self.predecessors.entry(*other_depth).or_default();
            for (edge_val, other_pred_arc) in other_preds_for_depth {
                match self_preds_for_depth.entry(edge_val.clone()) {
                    std::collections::btree_map::Entry::Vacant(entry) => {
                        entry.insert(other_pred_arc.clone());
                    }
                    std::collections::btree_map::Entry::Occupied(mut entry) => {
                        Arc::make_mut(entry.get_mut()).merge(other_pred_arc);
                    }
                }
            }
        }
        self.hash_key_cache = compute_hash_key(&self.predecessors);
        self.max_depth = compute_max_depth(&self.predecessors);
    }

    pub fn merged(mut self, other: Self) -> Self {
        self.merge(&other);
        self
    }

    pub fn iter_paths(&self) -> PathsIter<'_> {
        let mut queue = VecDeque::new();
        queue.push_back((self, Vec::new()));
        PathsIter { queue }
    }

    pub fn flatten(&self) -> Vec<Vec<(ParseStateEdgeContent, LLMTokenInfo)>> {
        let mut results = Vec::new();
        let mut queue = VecDeque::new();
        queue.push_back((self, Vec::new()));

        while let Some((node, mut path)) = queue.pop_front() {
            if node.predecessors.is_empty() {
                path.reverse();
                results.push(path);
            } else {
                for (_, preds_for_depth) in &node.predecessors {
                    for (edge_val, pred_arc) in preds_for_depth {
                        let mut new_path = path.clone();
                        new_path.push((edge_val.clone(), node.acc.acc().clone()));
                        queue.push_back((pred_arc.as_ref(), new_path));
                    }
                }
            }
        }
        results
    }

    pub fn flatten_bulk(nodes: &[Arc<Self>]) -> Vec<Vec<(ParseStateEdgeContent, LLMTokenInfo)>> {
        nodes.iter().flat_map(|node| node.flatten()).collect()
    }

    // map method is complex with non-generic GSSNode. If needed, it would be specific.
    // For now, let's assume it's not immediately required for this refactoring.
}

// Trait implementations
impl Hash for GSSNode {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.hash_key_cache.hash(state);
        self.acc.hash(state); // Accumulator should be part of the hash for equality
    }
}

impl PartialEq for GSSNode {
    // // #[time_it("GSSNode::eq")]
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self, other) || (
            self.hash_key_cache == other.hash_key_cache && // Structural hash
            self.acc == other.acc && // Accumulator equality
            self.predecessors == other.predecessors // Deep predecessor equality
        )
    }
}

impl Eq for GSSNode {}

impl PartialOrd for GSSNode {
    // // #[time_it("GSSNode::partial_cmp")]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if std::ptr::eq(self, other) { return Some(Ordering::Equal); }
        // Order by hash_key_cache, then acc, then predecessors
        self.hash_key_cache.partial_cmp(&other.hash_key_cache)
            .and_then(|ord| if ord == Ordering::Equal { self.acc.partial_cmp(&other.acc) } else { Some(ord) })
            .and_then(|ord| if ord == Ordering::Equal { self.predecessors.partial_cmp(&other.predecessors) } else { Some(ord) })
    }
}

impl Ord for GSSNode {
    // // #[time_it]("GSSNode::cmp")]
    fn cmp(&self, other: &Self) -> Ordering {
        if std::ptr::eq(self, other) { return Ordering::Equal; }
        self.hash_key_cache.cmp(&other.hash_key_cache)
            .then_with(|| self.acc.cmp(&other.acc))
            .then_with(|| self.predecessors.cmp(&other.predecessors))
    }
}

impl Drop for GSSNode {
    fn drop(&mut self) {
        // Custom drop logic to break cycles if Arcs are used internally in a complex way.
        // Standard Arc drop should handle most cases unless there are self-referential Arcs
        // not managed by the main GSS structure (which shouldn't be the case here).
        // The current predecessor map uses Arc, so standard drop is likely sufficient.
        // The previous custom drop logic was to manually traverse and break cycles
        // if Arc::try_unwrap could be used. This is complex and error-prone.
        // Relying on Arc's standard drop is safer unless specific cycle issues are proven.
    }
}

// Simplified trait for GSS operations
pub trait GSSTrait { // No longer generic
    fn push_with_acc(&self, edge_value: ParseStateEdgeContent, acc_for_new_node: Acc) -> GSSNode;
    fn push_with_existing_acc(&self, edge_value: ParseStateEdgeContent) -> GSSNode {
        let acc_for_new_node = self.acc2().clone();
        self.push_with_acc(edge_value, acc_for_new_node)
    }
    // push_to is removed as it's complex with private acc_mut and less idiomatic with Arc.
    fn pop(&self) -> GSSNode;
    fn popn(&self, n: usize) -> GSSNode;
    fn acc2(&self) -> &Acc;
}

impl GSSTrait for GSSNode {
    fn push_with_acc(&self, edge_value: ParseStateEdgeContent, acc_for_new_node: Acc) -> GSSNode {
        GSSNode::push_with_acc(self.clone(), edge_value, acc_for_new_node)
    }

    fn pop(&self) -> GSSNode {
        GSSNode::pop(self)
    }

    fn popn(&self, n: usize) -> GSSNode {
        GSSNode::popn(self, n)
    }

    fn acc2(&self) -> &Acc {
        GSSNode::acc2(self)
    }
}

impl GSSTrait for Arc<GSSNode> {
    fn push_with_acc(&self, edge_value: ParseStateEdgeContent, acc_for_new_node: Acc) -> GSSNode {
        GSSNode::new_with_single_predecessor(self.clone(), edge_value, acc_for_new_node)
    }

    fn pop(&self) -> GSSNode {
        self.as_ref().pop()
    }

    fn popn(&self, n: usize) -> GSSNode {
        self.as_ref().popn(n)
    }

    fn acc2(&self) -> &Acc {
        self.as_ref().acc2()
    }
}

// Removed GSSTrait for Option<Arc<GSSNode>> and Option<GSSNode> for brevity,
// can be added back if specific use cases require them.

// Pruning and Transformation
fn prune_and_transform_recursive(
    node_arc: &Arc<GSSNode>,
    closure: &impl Fn(&Acc) -> Option<(Acc, bool)>,
    memo: &mut HashMap<*const GSSNode, Option<Arc<GSSNode>>>,
) -> Option<Arc<GSSNode>> {
    let node_ptr = Arc::as_ptr(node_arc);
    if let Some(cached_result) = memo.get(&node_ptr) {
        return cached_result.clone();
    }

    match closure(&node_arc.acc2()) {
        None => { // Prune this node
            memo.insert(node_ptr, None);
            None
        }
        Some((new_acc, continue_recursion)) => {
            let new_predecessors_set = if continue_recursion {
                node_arc.predecessors.values().flat_map(|m| m.iter())
                    .filter_map(|(edge_val, pred_arc_val)| { // Renamed pred_arc
                        prune_and_transform_recursive(pred_arc_val, closure, memo)
                            .map(|new_pred_arc| (new_pred_arc, edge_val.clone())) // Renamed new_pred
                    })
                    .collect::<NodeSet>() // Explicit type for collect
            } else { // Don't recurse, keep existing predecessors but point to original Arcs
                node_arc.predecessors.values().flat_map(|m| m.iter())
                    .map(|(edge_val, pred_arc_val)| (pred_arc_val.clone(), edge_val.clone())) // Renamed pred_arc
                    .collect::<NodeSet>() // Explicit type for collect
            };

            // Create a new node with the transformed accumulator and new predecessors
            // GSSNode::new_with_predecessors computes its own acc by union. We want new_acc.
            let new_node_predecessors_map = process_predecessors(&new_predecessors_set);
            let transformed_node = GSSNode::new_with_map(new_acc, new_node_predecessors_map);
            
            let result_arc = Arc::new(transformed_node);
            memo.insert(node_ptr, Some(result_arc.clone()));

            Some(result_arc)
        }
    }
}

#[time_it]
pub fn intersect_llm_tokens_and_prune_arc(
    root_arc: &mut Arc<GSSNode>, 
    tokens_to_intersect: &LLMTokenBV,
    memo: &mut HashMap<*const GSSNode, Option<Arc<GSSNode>>>,
) {
    let closure = |current_acc: &Acc| -> Option<(Acc, bool)> {
        let mut new_acc = current_acc.clone();
        if let Some(bv) = new_acc.acc_mut() {
            *bv &= tokens_to_intersect;
        } else {
            new_acc = Acc::new(Some(tokens_to_intersect.clone()), current_acc.disallowed_terminals().clone());
        }
        if new_acc.is_alive() {
            // let continue_recursion = &new_acc != current_acc;
            let continue_recursion = false;
            Some((new_acc, continue_recursion))
        } else {
            None // Prune this node
        }
    };

    if let Some(new_root) = prune_and_transform_recursive(root_arc, &closure, memo) {
        *root_arc = new_root;
    } else {
        // The entire GSS was pruned, set root_arc to an empty GSSNode
        *root_arc = Arc::new(GSSNode::new(root_arc.acc2().clone()));
    }
}

#[time_it]
pub fn subtract_llm_tokens_and_prune_arc(
    root_arc: &mut Arc<GSSNode>,
    llm_tokens: &LLMTokenBV,
    memo: &mut HashMap<*const GSSNode, Option<Arc<GSSNode>>>,
) {
    let closure = |current_acc: &Acc| -> Option<(Acc, bool)> {
        let mut new_acc = current_acc.clone();
        if let Some(bv) = new_acc.acc_mut() {
            *bv -= llm_tokens;
        } else {
            new_acc = Acc::new(Some(LLMTokenBV::max_ones() - llm_tokens.clone()), current_acc.disallowed_terminals().clone());
        }
        if new_acc.is_alive() {
            // let continue_recursion = &new_acc != current_acc;
            let continue_recursion = false;
            Some((new_acc, continue_recursion))
        } else {
            None // Prune this node
        }
    };
    if let Some(new_root) = prune_and_transform_recursive(root_arc, &closure, memo) {
        *root_arc = new_root;
    } else {
        // The entire GSS was pruned, set root_arc to an empty GSSNode
        *root_arc = Arc::new(GSSNode::new(root_arc.acc2().clone()));
    }
}

pub fn reset_llm_tokens(
    root_arc: &mut Arc<GSSNode>,
    memo: &mut HashMap<*const GSSNode, Option<Arc<GSSNode>>>,
) {
    let closure = |current_acc: &Acc| -> Option<(Acc, bool)> {
        let continue_recursion = !current_acc.is_default();
        Some((Acc::new(None, current_acc.disallowed_terminals().clone()), continue_recursion)) // Keep node, continue recursion
    };
    if let Some(new_root) = prune_and_transform_recursive(root_arc, &closure, memo) {
        *root_arc = new_root;
    } else {
        // The entire GSS was pruned, set root_arc to an empty GSSNode
        *root_arc = Arc::new(GSSNode::new(root_arc.acc2().clone()));
    }
}

pub fn disallow_terminals_and_prune_arc(
    root_arc: &mut Arc<GSSNode>,
    disallowed_terminals: &BTreeMap<TokenizerStateID, TerminalBV>,
    memo: &mut HashMap<*const GSSNode, Option<Arc<GSSNode>>>,
) {
    // TODO: shouldn't be necessary to do this...
    let closure = |current_acc: &Acc| -> Option<(Acc, bool)> {
        let mut new_acc = current_acc.clone();
        disallowed_terminals_add(new_acc.disallowed_terminals_mut(), disallowed_terminals.clone());
        if new_acc.is_alive() {
            // let continue_recursion = *current_acc != new_acc;
            // let continue_recursion = !current_acc.disallowed_terminals().is_empty();
            // let continue_recursion = true;
            let continue_recursion = false;
            Some((new_acc, continue_recursion))
        } else {
            None // Prune this node
        }
    };
    if let Some(new_root) = prune_and_transform_recursive(root_arc, &closure, memo) {
        *root_arc = new_root;
    } else {
        // The entire GSS was pruned, set root_arc to an empty GSSNode
        *root_arc = Arc::new(GSSNode::new(root_arc.acc2().clone()));
    }
}

pub fn prune_disallowed_terminals(
    root_arc: &mut Arc<GSSNode>, 
    terminals_map: &BTreeMap<TokenizerStateID, TerminalBV>,
    memo: &mut HashMap<*const GSSNode, Option<Arc<GSSNode>>>,
) {
    // terminals_map: For each TokenizerStateID, a TerminalBV of terminals that are disallowed.
    let closure = |current_acc: &Acc| -> Option<(Acc, bool)> {
        let mut continue_recursion = false;
        let mut new_acc = current_acc.clone();
        for (gss_state_id, gss_disallowed_bv) in new_acc.disallowed_terminals_mut() {
            if let Some(actual_bv_for_state) = terminals_map.get(gss_state_id) {
                // If any terminal disallowed by GSS is also matched by current segment, prune.
                // This means (gss_disallowed_bv AND actual_bv_for_state) must be empty.
                if !gss_disallowed_bv.intersection.is_disjoint(actual_bv_for_state) {
                    return None;
                }
                if !gss_disallowed_bv.union.is_disjoint(actual_bv_for_state) {
                    continue_recursion = true;
                    *gss_disallowed_bv -= actual_bv_for_state;
                }
            }
        }
        Some((new_acc, continue_recursion))
    };

    if let Some(new_root) = prune_and_transform_recursive(root_arc, &closure, memo) {
        *root_arc = new_root;
    } else {
        *root_arc = Arc::new(GSSNode::new(root_arc.acc2().clone()));
    }
}

pub fn map_allowed_terminals_tokenizer_states(
    root_arc: &mut Arc<GSSNode>,
    map: &BTreeMap<TokenizerStateID, TokenizerStateID>,
    memo: &mut HashMap<*const GSSNode, Option<Arc<GSSNode>>>,
) {
    let closure = |current_acc: &Acc| -> Option<(Acc, bool)> {
        let mut new_disallowed_terminals = BTreeMap::new();
        let mut changed = false;

        for (old_id, bv) in current_acc.disallowed_terminals() {
            if let Some(new_id) = map.get(old_id) {
                *new_disallowed_terminals.entry(*new_id).or_insert_with(TerminalInfoValue::zeros) ^= bv;
                if new_disallowed_terminals.get(new_id) != Some(bv) || old_id != new_id { // Basic change check
                    changed = true;
                }
            } else {
                changed = true; // A state was removed, which is a change.
            }
        }
        new_disallowed_terminals.retain(|_, bv| !bv.is_empty()); // Remove empty entries
        if !changed && current_acc.disallowed_terminals().len() == new_disallowed_terminals.len() { // No structural change
             // No change in content or structure of allowed_terminals
        } else {
            changed = true;
        }

        let new_acc = Acc::new(current_acc.acc().clone(), new_disallowed_terminals);
        let continue_recursion = changed || !current_acc.disallowed_terminals().is_empty(); // Recurse if there was something to map or a change occurred.
        Some((new_acc, continue_recursion))
    };
    if let Some(new_root) = prune_and_transform_recursive(root_arc, &closure, memo) {
        *root_arc = new_root;
    } else {
        *root_arc = Arc::new(GSSNode::new(root_arc.acc2().clone()));
    }
}

pub fn find_longest_path(
    root_node: &GSSNode
) -> Option<Vec<(ParseStateEdgeContent, Arc<GSSNode>)>> {
    if root_node.predecessors.is_empty() {
        return None;
    }

    fn find_longest_recursive(
        node_arc: &Arc<GSSNode>,
        memo: &mut HashMap<*const GSSNode, Vec<(ParseStateEdgeContent, Arc<GSSNode>)>>,
        visited: &mut HashSet<*const GSSNode>,
    ) -> Vec<(ParseStateEdgeContent, Arc<GSSNode>)> {
        let node_ptr = Arc::as_ptr(node_arc);

        if let Some(cached) = memo.get(&node_ptr) {
            return cached.clone();
        }
        if !visited.insert(node_ptr) { // Cycle detected
            return Vec::new();
        }

        if node_arc.predecessors.is_empty() { // Base case: leaf node in recursion
            visited.remove(&node_ptr);
            memo.insert(node_ptr, Vec::new());
            return Vec::new();
        }

        let mut longest = Vec::new();
        for (_, preds_for_depth) in &node_arc.predecessors {
            for (edge_val, pred_arc_val) in preds_for_depth { // Renamed pred_arc
                let mut path = find_longest_recursive(pred_arc_val, memo, visited);
                path.push((edge_val.clone(), node_arc.clone())); // Path stores (edge, child_node_it_points_to)
                if path.len() > longest.len() {
                    longest = path;
                }
            }
        }

        memo.insert(node_ptr, longest.clone());
        visited.remove(&node_ptr);
        longest
    }

    let mut memo = HashMap::new();
    let mut longest_overall_path = Vec::new(); // Initialize with an empty path

    // The root_node itself is the start of paths, its predecessors are the first step.
    // The path should be from a leaf up to the direct children of root_node.
    for (_, preds_for_depth) in root_node.predecessors() {
        for (edge_val, pred_arc) in preds_for_depth {
            let mut visited_for_this_branch = HashSet::new();
             // Path from a leaf up to pred_arc
            let mut path_to_pred = find_longest_recursive(pred_arc, &mut memo, &mut visited_for_this_branch);
            path_to_pred.push((edge_val.clone(), Arc::new(root_node.clone()))); // Add the step from pred_arc to root_node

            if path_to_pred.len() > longest_overall_path.len() {
                longest_overall_path = path_to_pred;
            }
        }
    }
    if longest_overall_path.is_empty() { None } else { Some(longest_overall_path) }
}

impl GSSNode {
    pub fn prune_and_transform_recursive(
        &mut self,
        closure: &impl Fn(&Acc) -> Option<(Acc, bool)>,
        memo: &mut HashMap<*const GSSNode, Option<Arc<GSSNode>>>,
    ) {
        let node_arc = Arc::new(self.clone());
        if let Some(new_node_arc) = prune_and_transform_recursive(&node_arc, closure, memo) {
            *self = new_node_arc.as_ref().clone();
        } else {
            *self = GSSNode::new(self.acc2().clone());
        }
    }

    pub fn intersect_llm_tokens_and_prune_arc(
        &mut self,
        llm_tokens: &LLMTokenBV,
    ) {
        let mut node_arc = Arc::new(self.clone());
        let mut memo = HashMap::new();
        intersect_llm_tokens_and_prune_arc(&mut node_arc, &llm_tokens, &mut memo);
        *self = node_arc.as_ref().clone();
    }

    pub fn subtract_llm_tokens_and_prune_arc(
        &mut self,
        llm_tokens: &LLMTokenBV,
    ) {
        let mut node_arc = Arc::new(self.clone());
        let mut memo = HashMap::new();
        subtract_llm_tokens_and_prune_arc(&mut node_arc, &llm_tokens, &mut memo);
        *self = node_arc.as_ref().clone();
    }

    pub fn reset_llm_tokens(&mut self) {
        let mut node_arc = Arc::new(self.clone());
        let mut memo = HashMap::new();
        reset_llm_tokens(&mut node_arc, &mut memo);
        *self = node_arc.as_ref().clone();
    }

    pub fn disallow_terminals_and_prune_arc(
        &mut self,
        disallowed_terminals: &BTreeMap<TokenizerStateID, TerminalBV>,
    ) {
        let mut node_arc = Arc::new(self.clone());
        let mut memo = HashMap::new();
        disallow_terminals_and_prune_arc(&mut node_arc, &disallowed_terminals, &mut memo);
        *self = node_arc.as_ref().clone();
    }

    pub fn prune_disallowed_terminals(
        &mut self, 
        terminals_map: &BTreeMap<TokenizerStateID, TerminalBV>,
    ) {
        let mut node_arc = Arc::new(self.clone());
        let mut memo = HashMap::new();
        prune_disallowed_terminals(&mut node_arc, terminals_map, &mut memo);
        *self = node_arc.as_ref().clone();
    }

    pub fn map_allowed_terminals_tokenizer_states(
        &mut self, 
        map: &BTreeMap<TokenizerStateID, TokenizerStateID>,
    ) {
        let mut node_arc = Arc::new(self.clone());
        let mut memo = HashMap::new();
        map_allowed_terminals_tokenizer_states(&mut node_arc, map, &mut memo);
        *self = node_arc.as_ref().clone();
    }

    pub fn find_longest_path(&self) -> Option<Vec<(ParseStateEdgeContent, Arc<GSSNode>)>> {
        find_longest_path(&self)
    }
}

#[derive(Debug, Clone, Default)]
pub struct GSSStats {
    pub num_roots: usize,
    pub unique_nodes: usize,
    pub structurally_unique_nodes: usize,
    pub structural_redundancy: f64,
    pub max_depth: usize,
    pub average_depth: f64,
    pub merge_points: usize,
    pub max_predecessors_with_values: usize,
    pub average_predecessors_with_values: f64,
}

fn get_structural_id(
    node: &GSSNode,
    memo: &mut HashMap<*const GSSNode, usize>,
    structural_cache: &mut BTreeMap<BTreeMap<MaxDepth, BTreeMap<ParseStateEdgeContent, usize>>, usize>,
) -> usize {
    let node_ptr = node as *const GSSNode;
    if let Some(id) = memo.get(&node_ptr) {
        return *id;
    }

    let mut pred_structural_ids = BTreeMap::new();
    for (depth, preds_for_depth) in &node.predecessors {
        let mut inner_map = BTreeMap::new();
        for (edge_val, pred_arc) in preds_for_depth {
            let pred_id = get_structural_id(pred_arc.as_ref(), memo, structural_cache);
            inner_map.insert(edge_val.clone(), pred_id);
        }
        pred_structural_ids.insert(*depth, inner_map);
    }

    let next_id = structural_cache.len();
    let id = *structural_cache.entry(pred_structural_ids).or_insert(next_id);
    
    memo.insert(node_ptr, id);
    id
}

pub fn gather_gss_stats(roots: &[&GSSNode]) -> GSSStats { // Takes slice of references
    let mut stats = GSSStats::default();
    stats.num_roots = roots.len();

    let mut visited_pointers = HashSet::new(); // To track unique nodes by pointer
    let mut processed_pointers = HashSet::new(); // For BFS traversal
    let mut queue = VecDeque::new();
    let mut total_depth = 0u64;
    let mut total_preds = 0u64;

    for root_node_ref in roots { // Renamed root to root_node_ref
        let node_ptr = *root_node_ref as *const GSSNode;
        if visited_pointers.insert(node_ptr) { // Check against visited_pointers for uniqueness
            queue.push_back((*root_node_ref, 0)); // Push the reference and depth
        }
    }
    stats.unique_nodes = visited_pointers.len(); // Initial unique nodes are the unique roots

    // Reset visited_pointers for BFS traversal if we want to count all reachable nodes
    // Or, ensure the queue only gets truly unique items.
    // The current logic for unique_nodes might be off if roots share children.
    // Let's refine:
    visited_pointers.clear(); // Clear for BFS count
    stats.unique_nodes = 0; // Reset unique_nodes for BFS count

    let mut bfs_queue = VecDeque::new();
    for root_node_ref in roots {
        let node_ptr = *root_node_ref as *const GSSNode;
        if !processed_pointers.contains(&node_ptr) { // Ensure each root starts BFS once
             bfs_queue.push_back((*root_node_ref, 0));
             processed_pointers.insert(node_ptr); // Mark as added to queue
        }
    }
    processed_pointers.clear(); // Clear for actual processing check

    while let Some((node, depth)) = bfs_queue.pop_front() {
        let node_ptr = node as *const GSSNode;
        if !visited_pointers.insert(node_ptr) { // If already visited and processed by BFS
            continue;
        }

        stats.unique_nodes += 1;
        stats.max_depth = stats.max_depth.max(depth);
        total_depth += depth as u64;

        let num_preds = node.num_predecessors();
        stats.max_predecessors_with_values = stats.max_predecessors_with_values.max(num_preds);
        total_preds += num_preds as u64;

        let unique_pred_arcs: HashSet<_> = node.predecessors.values().flat_map(|m| m.values())
            .map(|arc_val| Arc::as_ptr(arc_val)) // Renamed arc
            .collect();
        if unique_pred_arcs.len() > 1 && num_preds > 1 { // A merge point has multiple distinct predecessor nodes
            stats.merge_points += 1;
        }

        for pred_arc_val in node.predecessors.values().flat_map(|m| m.values()) { // Renamed pred_arc
            let pred_ptr = pred_arc_val.as_ref() as *const GSSNode;
             // Add to queue if not yet added for BFS processing from any path
            if !processed_pointers.contains(&pred_ptr) {
                bfs_queue.push_back((pred_arc_val.as_ref(), depth + 1));
                processed_pointers.insert(pred_ptr);
            }
        }
    }


    if stats.unique_nodes > 0 {
        stats.average_depth = total_depth as f64 / stats.unique_nodes as f64;
        stats.average_predecessors_with_values = total_preds as f64 / stats.unique_nodes as f64;
    }

    // --- Calculate structural uniqueness ---
    let mut structural_memo = HashMap::new();
    let mut structural_cache = BTreeMap::new();
    for root_node_ref in roots {
        get_structural_id(*root_node_ref, &mut structural_memo, &mut structural_cache);
    }
    stats.structurally_unique_nodes = structural_cache.len();
    if stats.unique_nodes > 0 {
        stats.structural_redundancy = 1.0 - (stats.structurally_unique_nodes as f64 / stats.unique_nodes as f64);
    }
    stats
}

/// Formats the accumulator for concise display.
fn format_acc(
    acc: &acc_mod::Acc,
    terminal_map: &BiBTreeMap<Terminal, TerminalID>,
    original_internal_bimap: Option<&BiBTreeMap<usize, usize>>,
    llm_token_map: Option<&BiBTreeMap<Vec<u8>, LLMTokenID>>,
) -> String {
    let llm_info = match acc.acc() {
        None => "LLM(Any)".to_string(),
        Some(bv) if bv.is_empty() => "LLM(None)".to_string(),
        Some(bv) => {
            if let (Some(bimap), Some(token_map)) = (original_internal_bimap, llm_token_map) {
                let mut token_samples = Vec::new();
                const MAX_SAMPLES: usize = 3;
                for internal_id in bv.iter().take(MAX_SAMPLES) {
                    if let Some(original_id) = bimap.get_by_right(&internal_id) {
                        if let Some(token_bytes) = token_map.get_by_right(&LLMTokenID(*original_id)) {
                            token_samples.push(format!("{:?}", String::from_utf8_lossy(token_bytes)));
                        }
                    }
                }
                let samples_str = token_samples.join(", ");
                let total_tokens = bv.len();
                if total_tokens > MAX_SAMPLES {
                    format!("LLM({} tokens: [{}, ...])", total_tokens, samples_str)
                } else {
                    format!("LLM({} tokens: [{}])", total_tokens, samples_str)
                }
            } else {
                format!("LLM({:?})", bv)
            }
        }
    };

    let disallowed_info = if acc.disallowed_terminals().is_empty() {
        "".to_string()
    } else {
        let mut parts = Vec::new();
        for (tokenizer_state_id, terminal_info_value) in acc.disallowed_terminals() {
            if terminal_info_value.union.is_empty() && terminal_info_value.intersection.is_empty() {
                continue;
            }

            let format_names = |bv: &TerminalBV| -> String {
                if bv.is_empty() {
                    return "[]".to_string();
                }
                let mut terminal_names = Vec::new();
                for terminal_id_val in bv.iter() {
                    let terminal_id = TerminalID(terminal_id_val);
                    if let Some(terminal) = terminal_map.get_by_right(&terminal_id) {
                        terminal_names.push(terminal.0.clone());
                    } else {
                        terminal_names.push(format!("<ID:{}>", terminal_id.0));
                    }
                }
                
                const MAX_NAMES_TO_SHOW: usize = 3;
                let names_str = if terminal_names.len() > MAX_NAMES_TO_SHOW {
                    format!("{}...", terminal_names.iter().take(MAX_NAMES_TO_SHOW).cloned().collect::<Vec<_>>().join(", "))
                } else {
                    terminal_names.join(", ")
                };
                format!("[{}]", names_str)
            };

            let mut part_str = String::new();
            if !terminal_info_value.union.is_empty() {
                part_str.push_str(&format!("U: {}", format_names(&terminal_info_value.union)));
                if !part_str.is_empty() {
                    part_str.push_str(", ");
                }
                part_str.push_str(&format!("I: {}", format_names(&terminal_info_value.intersection)));
            } else if !terminal_info_value.intersection.is_empty() {
                part_str.push_str(&format!("I: {}", format_names(&terminal_info_value.intersection)));
            }

            if !part_str.is_empty() {
                parts.push(format!("State {}: {}", tokenizer_state_id.0, part_str));
            }
        }
        if parts.is_empty() {
            "".to_string()
        } else {
            format!(", Disallowed({})", parts.join("; "))
        }
    };

    format!("({}{})", llm_info, disallowed_info)
}

pub fn print_gss_forest(
    roots: &[Arc<GSSNode>],
    labels: Option<&[String]>,
    max_nodes: usize,
    terminal_map: &BiBTreeMap<Terminal, TerminalID>,
    original_internal_bimap: Option<&BiBTreeMap<usize, usize>>,
    llm_token_map: Option<&BiBTreeMap<Vec<u8>, LLMTokenID>>,
) -> String {
    // The recursive helper function. It's responsible for printing the children (predecessors) of a given node.
    fn print_predecessors_recursive(
        node_arc: &Arc<GSSNode>,
        node_ids: &mut HashMap<*const GSSNode, usize>,
        visited_nodes: &mut HashSet<*const GSSNode>, // Nodes whose children have been printed
        prefix: &str,
        node_count: &mut usize,
        max_nodes: usize,
        output: &mut String,
        terminal_map: &BiBTreeMap<Terminal, TerminalID>,
        original_internal_bimap: Option<&BiBTreeMap<usize, usize>>,
        llm_token_map: Option<&BiBTreeMap<Vec<u8>, LLMTokenID>>,
    ) -> Result<(), std::fmt::Error> {
        let node_ptr = Arc::as_ptr(node_arc);
        if visited_nodes.contains(&node_ptr) {
            return Ok(()); // Already printed this node's children
        }
        visited_nodes.insert(node_ptr);

        let predecessors: Vec<_> = node_arc.predecessors()
            .values()
            .flat_map(|m| m.iter())
            .collect();

        let num_preds = predecessors.len();
        for (i, (edge_val, pred_arc)) in predecessors.iter().enumerate() {
            if *node_count >= max_nodes {
                writeln!(output, "{}... (Truncated)", prefix)?;
                return Ok(());
            }

            let is_last = i == num_preds - 1;
            let connector = if is_last { "└──" } else { "├──" };
            let new_prefix = format!("{}  {}", prefix, if is_last { "  " } else { "│ " });

            let pred_ptr = Arc::as_ptr(pred_arc);
            let node_ids_len = node_ids.len();
            let pred_id = *node_ids.entry(pred_ptr).or_insert(node_ids_len);

            // If the predecessor's children have already been printed, just show a reference.
            if visited_nodes.contains(&pred_ptr) {
                writeln!(
                    output,
                    "{}{} Edge {:?} -> Node {} (ref)",
                    prefix,
                    connector,
                    edge_val.state_id,
                    pred_id,
                )?;
            } else {
                // Otherwise, print full info and recurse.
                let acc_child = format_acc(pred_arc.acc2(), terminal_map, original_internal_bimap, llm_token_map);
                let child_depth = pred_arc.max_depth;
                writeln!(
                    output,
                    "{}{} Edge {:?} -> Node {} (depth {}) {}",
                    prefix,
                    connector,
                    edge_val.state_id,
                    pred_id,
                    child_depth,
                    acc_child,
                )?;
                *node_count += 1;

                print_predecessors_recursive(
                    pred_arc,
                    node_ids,
                    visited_nodes,
                    &new_prefix,
                    node_count,
                    max_nodes,
                    output,
                    terminal_map,
                    original_internal_bimap,
                    llm_token_map,
                )?;
            }
        }
        Ok(())
    }

    let mut node_ids = HashMap::new();
    let mut visited_nodes = HashSet::new();
    let mut count = 0;
    let mut out_str = String::new();

    if let Some(labels) = labels {
        if roots.len() != labels.len() {
            assert_eq!(roots.len(), labels.len(), "Number of roots and labels must match for print_gss_forest");
        }
    }

    if roots.is_empty() { return "GSS Forest: (No roots)".to_string(); }
    writeln!(&mut out_str, "GSS Forest (Max Nodes: {}):", max_nodes).unwrap();

    for (i, root_arc) in roots.iter().enumerate() {
        if count >= max_nodes {
            writeln!(&mut out_str, "... (Truncated)").unwrap();
            break;
        }

        let root_ptr = Arc::as_ptr(root_arc);
        let node_ids_len = node_ids.len();
        let root_id = *node_ids.entry(root_ptr).or_insert(node_ids_len);
        
        let acc_str = format_acc(root_arc.acc2(), terminal_map, original_internal_bimap, llm_token_map);
        
        let root_label = if let Some(labels) = labels {
            labels[i].clone()
        } else {
            format!("Root {}", i)
        };

        if visited_nodes.contains(&root_ptr) {
            writeln!(&mut out_str, "{}: Node {} (ref)", root_label, root_id).unwrap();
        } else {
            writeln!(&mut out_str, "{}: Node {} (depth {}) {}", root_label, root_id, root_arc.max_depth, acc_str).unwrap();
            count += 1;
            if print_predecessors_recursive(root_arc, &mut node_ids, &mut visited_nodes, "  ", &mut count, max_nodes, &mut out_str, terminal_map, original_internal_bimap, llm_token_map).is_err() {
                return "Error writing GSS structure".to_string();
            }
        }
    }

    out_str
}


// Simplification methods
// This is the main simplification routine. It uses a cache for structural sharing.
fn simplify_node_recursive(
    node_arc: &Arc<GSSNode>,
    memo: &mut HashMap<*const GSSNode, Arc<GSSNode>>, // Memoizes input Arc raw pointer to simplified Arc
    cache: &mut NodeCache, // Cache for structural sharing: NodeMap -> Arc<GSSNode>
) -> Arc<GSSNode> {
    let node_ptr = Arc::as_ptr(node_arc);
    if let Some(simplified_arc) = memo.get(&node_ptr) { // Renamed simplified
        return simplified_arc.clone();
    }

    // Recursively simplify predecessors
    let simplified_predecessors_set: NodeSet = node_arc.predecessors.values().flat_map(|m| m.iter())
        .map(|(edge_val, pred_arc_val)| { // Renamed pred_arc
            let simplified_pred_arc = simplify_node_recursive(pred_arc_val, memo, cache); // Renamed simplified_pred
            (simplified_pred_arc, edge_val.clone())
        })
        .collect();
    
    let simplified_predecessors_map = process_predecessors(&simplified_predecessors_set);

    // Get a structurally canonical Arc from the cache, or create and insert it.
    // The acc of this cached_structural_node is the union of its predecessors' accs.
    let cached_structural_node = cache.entry(simplified_predecessors_map.clone())
        .or_insert_with(|| {
            let unioned_acc = if simplified_predecessors_map.is_empty() {
                Acc::new_for_merging()
            } else {
                let mut iter = simplified_predecessors_map.values().flat_map(|m| m.values());
                let mut acc = iter.next().unwrap().acc2().clone();
                for p_arc in iter { // Renamed p
                    acc.union_assign(p_arc.acc2().clone());
                }
                acc
            };
            Arc::new(GSSNode::new_with_map(unioned_acc, simplified_predecessors_map))
        });

    // The final simplified node has the structure of cached_structural_node,
    // but its accumulator is the one from the original node_arc.
    let mut final_node_data = (**cached_structural_node).clone(); // Clone GSSNode data
    final_node_data.acc = node_arc.acc.clone(); // Set the specific acc from original node
    // Recompute hash key for final_node_data as its acc might differ from cached_structural_node's acc
    final_node_data.hash_key_cache = compute_hash_key(&final_node_data.predecessors);


    let result_arc = Arc::new(final_node_data);
    memo.insert(node_ptr, result_arc.clone());
    result_arc
}


impl GSSNode {
    pub fn simplify(&mut self) {
        // Create a temporary Arc to self to use with simplify_node_recursive
        // This requires `self` to be cloneable and then update `self` with the result.
        let temp_arc = Arc::new(self.clone());
        let mut memo = HashMap::new();
        let mut cache = NodeCache::new(); // Cache for structural sharing
        let simplified_arc = simplify_node_recursive(&temp_arc, &mut memo, &mut cache);
        
        // Update self with the simplified version's data
        // This is safe because simplify_node_recursive returns a potentially new Arc.
        // We take ownership of the data from the simplified Arc.
        if Arc::ptr_eq(&temp_arc, &simplified_arc) {
            // No change, or already canonical.
            // However, predecessors might have changed, so self might need update.
            // The most robust way is to replace self's content.
        }
        // Replace self's content with the (potentially) new simplified content
        let new_data = Arc::try_unwrap(simplified_arc).unwrap_or_else(|arc| (*arc).clone());
        *self = new_data;

    }

    // simplify_recursive is effectively what simplify_node_recursive does.
    // pub fn simplify_recursive(
    //     this_arc: &mut Arc<Self>,
    //     memo: &mut HashMap<*const Self, Arc<Self>>,
    //     cache: &mut NodeCache,
    // ) {
    //     *this_arc = simplify_node_recursive(this_arc, memo, cache);
    // }

    pub fn simplify_together(nodes: &mut [&mut Arc<Self>]) {
        let mut memo = HashMap::new(); // Memoization for input node pointers
        let mut cache = NodeCache::new(); // Cache for structural sharing of predecessor maps
        for node_arc_ref_mut in nodes { // Renamed node_arc
            // We need to pass a reference to the Arc to simplify_node_recursive
            // and then update the Arc in the slice.
            let current_arc = (*node_arc_ref_mut).clone(); // Clone the Arc to pass by value/ref
            let simplified_arc = simplify_node_recursive(&current_arc, &mut memo, &mut cache);
            **node_arc_ref_mut = simplified_arc; // Update the Arc in the slice
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::constraint::LLMTokenBV;
    use super::*;
    use crate::glr::parser::ParseStateEdgeContent;
    use crate::glr::table::StateID;

    // MockPathAccumulator is now LLMTokenInfo, use that directly or a simplified version if needed for tests.
    // For simplicity, let's use LLMTokenInfo with basic active/intersection sets.

    type TestGSSNode = GSSNode; // GSSNode is now concrete

    fn mock_llm_token_info(active_val: usize, intersection_val: usize) -> Acc {
        let mut active = LLMTokenBV::zeros();
        active.insert(active_val);
        Acc::new(Some(active), Default::default())
    }
    
    fn mock_edge(id: usize) -> ParseStateEdgeContent {
        ParseStateEdgeContent { state_id: StateID(id) }
    }


    #[test]
    fn test_gss_simplification_basic() {
        let acc_base = mock_llm_token_info(0,0);
        let acc_other = mock_llm_token_info(1,1);

        // Node N4 (leaf)
        let n4_v1 = Arc::new(TestGSSNode::new(acc_base.clone())); // depth 0
        let n4_v2 = Arc::new(TestGSSNode::new(acc_other.clone())); // depth 0

        // D1: ... -> 40 -> N4(acc_base)
        let d1_orig = Arc::new(TestGSSNode::new_with_single_predecessor(
            n4_v1.clone(), mock_edge(40), acc_base.clone()
        )); // depth 1

        // D2: ... -> 40 -> N4(acc_other)
         let d2_orig = Arc::new(TestGSSNode::new_with_single_predecessor(
            n4_v2.clone(), mock_edge(40), acc_other.clone()
        )); // depth 1

        // C1: ... -> 30 -> D1
        let c1_orig = Arc::new(TestGSSNode::new_with_single_predecessor(
            d1_orig.clone(), mock_edge(30), acc_base.clone()
        )); // depth 2

        // B1: ... -> 20 -> C1
        let b1_orig = Arc::new(TestGSSNode::new_with_single_predecessor(
            c1_orig.clone(), mock_edge(20), acc_base.clone()
        )); // depth 3
        
        // A1: (root)
        // preds: B1 (via edge 10, depth 3), D2 (via edge 10, depth 1)
        // Since B1 and D2 have different depths, they should NOT be merged by simplification,
        // even though they are reached via the same edge value.
        
        let mut a1_preds_set = NodeSet::new();
        a1_preds_set.insert((b1_orig.clone(), mock_edge(10)));
        a1_preds_set.insert((d2_orig.clone(), mock_edge(10)));
        
        let acc_a1 = acc_base.clone().union(acc_other.clone());
        // process_predecessors will create a NodeMap with two depth entries
        let a1_preds_map = process_predecessors(&a1_preds_set);
        let a1_orig = Arc::new(TestGSSNode::new_with_map(acc_a1.clone(), a1_preds_map));


        let mut roots_to_simplify = vec![a1_orig.clone()];
        let mut refs_to_simplify: Vec<&mut Arc<TestGSSNode>> = roots_to_simplify.iter_mut().collect();
        TestGSSNode::simplify_together(&mut refs_to_simplify);
        
        let s_a1 = refs_to_simplify[0].clone();

        // --- Verification ---
        // A1 should have two predecessor maps because its predecessors have different depths.
        assert_eq!(s_a1.predecessors.len(), 2, "A1 should have 2 predecessor maps for different depths");
        
        // Accumulator of A1 should remain as it was.
        assert_eq!(s_a1.acc2(), &acc_a1, "A1 accumulator mismatch");

        // Check predecessor from D2 (depth 1)
        let preds_at_depth_1 = s_a1.predecessors.get(&1).expect("No predecessors at depth 1");
        assert_eq!(preds_at_depth_1.len(), 1, "Should be 1 predecessor at depth 1");
        let s_d2 = preds_at_depth_1.get(&mock_edge(10)).expect("Edge 10 not found for depth 1 pred");
        assert_eq!(s_d2.acc2(), &acc_other, "Simplified D2 accumulator mismatch");
        assert_eq!(s_d2.max_depth, 1, "Simplified D2 depth mismatch");

        // Check predecessor from B1 (depth 3)
        let preds_at_depth_3 = s_a1.predecessors.get(&3).expect("No predecessors at depth 3");
        assert_eq!(preds_at_depth_3.len(), 1, "Should be 1 predecessor at depth 3");
        let s_b1 = preds_at_depth_3.get(&mock_edge(10)).expect("Edge 10 not found for depth 3 pred");
        assert_eq!(s_b1.acc2(), &acc_base, "Simplified B1 accumulator mismatch");
        assert_eq!(s_b1.max_depth, 3, "Simplified B1 depth mismatch");

        // Verify the structure of the unmerged paths
        // Path from s_b1
        let s_c1 = s_b1.predecessors.get(&2).unwrap().get(&mock_edge(20)).unwrap();
        assert_eq!(s_c1.acc2(), &acc_base);
        assert_eq!(s_c1.max_depth, 2);
        let s_d1 = s_c1.predecessors.get(&1).unwrap().get(&mock_edge(30)).unwrap();
        assert_eq!(s_d1.acc2(), &acc_base);
        assert_eq!(s_d1.max_depth, 1);
        let s_n4_from_d1 = s_d1.predecessors.get(&0).unwrap().get(&mock_edge(40)).unwrap();
        assert_eq!(s_n4_from_d1.acc2(), &acc_base);
        assert!(s_n4_from_d1.predecessors.is_empty());

        // Path from s_d2
        let s_n4_from_d2 = s_d2.predecessors.get(&0).unwrap().get(&mock_edge(40)).unwrap();
        assert_eq!(s_n4_from_d2.acc2(), &acc_other);
        assert!(s_n4_from_d2.predecessors.is_empty());

        // The two N4 nodes should be different because their accumulators are different.
        assert!(!Arc::ptr_eq(s_n4_from_d1, s_n4_from_d2));

        // Count total unique nodes in the simplified graph starting from s_a1
        let mut all_nodes = HashSet::new();
        fn collect_all_nodes(node: &Arc<TestGSSNode>, set: &mut HashSet<*const TestGSSNode>) {
            if set.insert(Arc::as_ptr(node)) {
                for pred_map in node.predecessors.values() {
                    for pred_arc in pred_map.values() {
                        collect_all_nodes(pred_arc, set);
                    }
                }
            }
        }
        collect_all_nodes(&s_a1, &mut all_nodes);
        // Expected nodes: A1, B1, C1, D1, N4_v1, D2, N4_v2
        // Total = 7 nodes
        assert_eq!(all_nodes.len(), 7, "Incorrect number of unique nodes in simplified graph. Actual: {:?}", all_nodes.len());
    }
}
