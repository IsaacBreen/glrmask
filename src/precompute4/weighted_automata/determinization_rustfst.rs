// src/precompute4/weighted_automata/determinization_rustfst.rs

#![allow(dead_code)]
#![allow(clippy::needless_borrow)]

use super::common::{Label, StateID, Weight};
use super::dwa::DWA;
use super::nwa::NWA;
use crate::precompute4::weighted_automata::NWAStateID;
use anyhow::Result;
use nom::IResult;
use once_cell::sync::Lazy;
use range_set_blaze::RangeSetBlaze;
use rustfst::algorithms::determinize::{determinize_with_config, DeterminizeConfig, DeterminizeType};
use rustfst::algorithms::rm_epsilon::rm_epsilon;
use rustfst::fst_properties::FstProperties;
use rustfst::prelude::{Tr, EPS_LABEL, StateId, VectorFst, MutableFst, CoreFst, ExpandedFst};
use rustfst::semirings::{
    DivideType, ReverseBack, SemiringProperties, SerializableSemiring, WeaklyDivisibleSemiring, WeightQuantize,
};
use rustfst::{NomCustomError, Semiring, Trs};
use std::borrow::Borrow;
use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::sync::{Arc, Mutex};


fn _label_to_fst_label(label: Label) -> u32 {
    // (((label as isize) - (Label::MIN as isize)) + 1) as u32
    (label as Label - Label::MIN as Label) as u32 + 1
}
fn _fst_label_to_label(label: u32) -> Label {
    // (label as isize + Label::MIN as isize - 1) as Label
    ((label - 1) as Label + Label::MIN as Label) as Label
}
fn fst_label_to_label(label: u32) -> Label {
    assert_ne!(label, 0);
    let result = _fst_label_to_label(label);
    let remapped = _label_to_fst_label(result);
    assert!(label == remapped, "label: {}, result: {}, remapped: {}", label, result, remapped);
    result
}
fn label_to_fst_label(label: Label) -> u32 {
    let result = _label_to_fst_label(label);
    assert_ne!(result, 0);
    let remapped = _fst_label_to_label(result);
    assert!(label == remapped, "label: {}, result: {}, remapped: {}", label, result, remapped);
    result
}

static WEIGHT_INTERNER: Lazy<Mutex<HashSet<Arc<Weight>>>> =
    Lazy::new(|| Mutex::new(HashSet::new()));

fn intern_weight(weight: Weight) -> Arc<Weight> {
    let mut interner = WEIGHT_INTERNER.lock().unwrap();
    if let Some(w) = interner.get(&weight) {
        return w.clone();
    }
    let arc_weight = Arc::new(weight);
    interner.insert(arc_weight.clone());
    arc_weight
}

/// Semiring over bitset weights: plus = union, times = intersection.
#[derive(Clone, Debug, PartialOrd, Default, Eq, Hash)]
pub struct BitsetWeight(pub Arc<Weight>);

impl PartialEq for BitsetWeight {
    fn eq(&self, other: &Self) -> bool { Arc::ptr_eq(&self.0, &other.0) || *self.0 == *other.0 }
}

impl Semiring for BitsetWeight {
    type Type = Weight;
    type ReverseWeight = BitsetWeight;

    fn zero() -> Self { BitsetWeight(intern_weight(Weight::zeros())) }
    fn one() -> Self { BitsetWeight(intern_weight(Weight::all())) }
    fn new(value: Self::Type) -> Self { BitsetWeight(intern_weight(value)) }

    fn plus_assign<P: Borrow<Self>>(&mut self, rhs: P) -> Result<()> {
        let new_weight = &*self.0 | &*rhs.borrow().0;
        self.0 = intern_weight(new_weight);
        Ok(())
    }

    fn times_assign<P: Borrow<Self>>(&mut self, rhs: P) -> Result<()> {
        let new_weight = &*self.0 & &*rhs.borrow().0;
        self.0 = intern_weight(new_weight);
        Ok(())
    }

    fn approx_equal<P: Borrow<Self>>(&self, rhs: P, _delta: f32) -> bool { *self.0 == *rhs.borrow().0 }
    fn value(&self) -> &Self::Type { &self.0 }
    fn take_value(self) -> Self::Type { Arc::try_unwrap(self.0).unwrap_or_else(|arc| (*arc).clone()) }
    fn set_value(&mut self, value: Self::Type) { self.0 = intern_weight(value); }
    fn reverse(&self) -> Result<Self::ReverseWeight> { Ok(self.clone()) }

    fn properties() -> SemiringProperties {
        SemiringProperties::LEFT_SEMIRING
            | SemiringProperties::RIGHT_SEMIRING
            | SemiringProperties::COMMUTATIVE
            | SemiringProperties::IDEMPOTENT
            | SemiringProperties::PATH
    }
}

impl ReverseBack<BitsetWeight> for BitsetWeight {
    fn reverse_back(&self) -> Result<BitsetWeight> { Ok(self.clone()) }
}

impl WeaklyDivisibleSemiring for BitsetWeight {
    fn divide_assign(&mut self, rhs: &Self, _divide_type: DivideType) -> Result<()> {
        let new_weight = &*self.0 | &!&*rhs.0;
        self.0 = intern_weight(new_weight);
        Ok(())
    }
}

impl WeightQuantize for BitsetWeight {
    fn quantize_assign(&mut self, _delta: f32) -> Result<()> { Ok(()) }
}

impl SerializableSemiring for BitsetWeight {
    fn weight_type() -> String { "bitset".to_string() }

    fn parse_binary(i: &[u8]) -> IResult<&[u8], Self, NomCustomError<&[u8]>> {
        use nom::number::complete::le_u64;

        let (mut i, num_ranges) = le_u64(i)?;
        let mut ranges = Vec::with_capacity(num_ranges as usize);
        for _ in 0..num_ranges {
            let (next_i, start) = le_u64(i)?;
            let (next_i, end) = le_u64(next_i)?;
            ranges.push(start as usize..=end as usize);
            i = next_i;
        }
        let rsb = RangeSetBlaze::from_iter(ranges);
        Ok((i, BitsetWeight(intern_weight(Weight::from_rsb(rsb)))))
    }

    fn write_binary<F: Write>(&self, file: &mut F) -> Result<()> {
        let ranges: Vec<_> = self.0.rsb.ranges().collect();
        file.write_all(&(ranges.len() as u64).to_le_bytes())?;
        for range in ranges {
            file.write_all(&(*range.start() as u64).to_le_bytes())?;
            file.write_all(&(*range.end() as u64).to_le_bytes())?;
        }
        Ok(())
    }

    fn parse_text(i: &str) -> IResult<&str, Self> {
        use nom::combinator::map_res;
        map_res(nom::combinator::rest, |s: &str| -> Result<BitsetWeight, _> {
            serde_json::from_str::<Weight>(s)
                .map(|w| BitsetWeight(intern_weight(w)))
                .map_err(|e| e.to_string())
        })(i)
    }
}

impl std::fmt::Display for BitsetWeight {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", serde_json::to_string(self.0.as_ref()).unwrap_or_else(|_| "err".to_string()))
    }
}

pub fn nwa_to_vector_fst(nwa: &NWA) -> VectorFst<BitsetWeight> {
    let mut fst = VectorFst::<BitsetWeight>::new();
    let mut state_map = HashMap::<NWAStateID, StateId>::new();

    for i in 0..nwa.states.len() {
        let s = fst.add_state();
        state_map.insert(i, s);
    }

    if !nwa.states.0.is_empty() {
        fst.set_start(state_map[&nwa.body.start_state]).unwrap();
    }

    for (i, nwa_state) in nwa.states.0.iter().enumerate() {
        let fst_state_id = state_map[&i];

        if let Some(w) = &nwa_state.final_weight {
            if !w.is_empty() {
                fst.set_final(fst_state_id, BitsetWeight::new(w.clone())).unwrap();
            }
        }

        for (label, targets) in &nwa_state.transitions {
            for (target, weight) in targets {
                if !weight.is_empty() {
                    fst.add_tr(
                        fst_state_id,
                        Tr::new(
                            label_to_fst_label(*label),
                            label_to_fst_label(*label),
                            BitsetWeight::new(weight.clone()),
                            state_map[target],
                        ),
                    )
                    .unwrap();
                }
            }
        }

        for (target, weight) in &nwa_state.epsilons {
            if !weight.is_empty() {
                fst.add_tr(fst_state_id, Tr::new(EPS_LABEL, EPS_LABEL, BitsetWeight::new(weight.clone()), state_map[target]))
                    .unwrap();
            }
        }
    }

    crate::debug!(5, "NWA to FST conversion done:\n{}", fst);
    fst
}

pub fn vector_fst_to_dwa(fst: &VectorFst<BitsetWeight>) -> DWA {
    let fst_start = match fst.start() {
        Some(s) => s,
        None => return DWA::new(),
    };

    let mut dwa = DWA::new();
    dwa.states.0.clear();
    let mut state_map = HashMap::<StateId, StateID>::new();

    for i in 0..fst.num_states() {
        let s = dwa.add_state();
        state_map.insert(i as StateId, s);
    }
    dwa.body.start_state = state_map[&fst_start];

    for i in 0..fst.num_states() {
        let fst_state_id = i as StateId;
        if !state_map.contains_key(&fst_state_id) {
            continue;
        }
        let dwa_state_id = state_map[&fst_state_id];

        if let Some(w) = fst.final_weight(fst_state_id).unwrap() {
            if !w.0.is_empty() {
                dwa.set_final_weight(dwa_state_id, w.value().clone()).unwrap();
            }
        }

        for tr in fst.get_trs(fst_state_id).unwrap().trs() {
            if !tr.weight.0.is_empty() {
                if !state_map.contains_key(&tr.nextstate) {
                    continue;
                }
                let res = dwa.add_transition(
                    dwa_state_id,
                    fst_label_to_label(tr.ilabel),
                    state_map[&tr.nextstate],
                    tr.weight.value().clone(),
                );
                if let Err(e) = res {
                    panic!(
                        "Error converting VectorFst to DWA: transition already exists. This indicates non-determinism. Error: {:?}",
                        e
                    );
                }
            }
        }
    }

    dwa
}

pub fn vector_fst_to_nwa(fst: &VectorFst<BitsetWeight>) -> NWA {
    if fst.num_states() == 0 {
        return NWA::new();
    }

    let mut nwa = NWA::new();
    nwa.states.0.clear();
    let mut state_map = HashMap::<StateId, NWAStateID>::new();

    for i in 0..fst.num_states() {
        let s = nwa.states.add_state();
        state_map.insert(i as StateId, s);
    }

    if let Some(fst_start) = fst.start() {
        nwa.body.start_state = state_map[&fst_start];
    } else {
        nwa.body.start_state = 0;
    }

    for i in 0..fst.num_states() {
        let fst_state_id = i as StateId;
        let nwa_state_id = state_map[&fst_state_id];

        if let Some(w) = fst.final_weight(fst_state_id).unwrap() {
            if !w.0.is_empty() {
                nwa.states[nwa_state_id].final_weight = Some(w.value().clone());
            }
        }

        for tr in fst.get_trs(fst_state_id).unwrap().trs() {
            if !tr.weight.0.is_empty() {
                let target_nwa_id = state_map[&tr.nextstate];
                let weight = tr.weight.value().clone();

                if tr.ilabel == EPS_LABEL {
                    nwa.states.add_epsilon(nwa_state_id, target_nwa_id, weight);
                } else {
                    let label = fst_label_to_label(tr.ilabel);
                    nwa.states.add_transition(nwa_state_id, label, target_nwa_id, weight).unwrap();
                }
            }
        }
    }

    nwa
}

pub fn determinize_nwa_to_dwa(nwa: &NWA) -> DWA {
    let mut fst = nwa_to_vector_fst(nwa);
    fst.compute_and_update_properties_all().unwrap();
    assert!(fst.properties().contains(FstProperties::ACCEPTOR), "FST should be an acceptor before determinization");

    rm_epsilon(&mut fst).unwrap();

    let det_config = DeterminizeConfig::default().with_det_type(DeterminizeType::DeterminizeFunctional);
    let det_fst: VectorFst<BitsetWeight> = determinize_with_config(&fst, det_config).unwrap();

    vector_fst_to_dwa(&det_fst)
}

impl DWA {
    pub fn to_rustfst(&self) -> VectorFst<BitsetWeight> {
        nwa_to_vector_fst(&NWA::from_dwa(self))
    }

    pub fn from_rustfst(fst: &VectorFst<BitsetWeight>) -> DWA {
        vector_fst_to_dwa(fst)
    }
}

impl NWA {
    pub fn to_rustfst(&self) -> VectorFst<BitsetWeight> {
        nwa_to_vector_fst(self)
    }

    pub fn from_rustfst(fst: &VectorFst<BitsetWeight>) -> NWA {
        vector_fst_to_nwa(fst)
    }
}