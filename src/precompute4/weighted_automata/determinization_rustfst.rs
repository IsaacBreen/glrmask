// src/precompute4/weighted_automata/determinization_rustfst.rs

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
use rustfst::prelude::{CoreFst, ExpandedFst, MutableFst, StateId, Tr, Trs, VectorFst, EPS_LABEL};
use rustfst::semirings::{DivideType, ReverseBack, SemiringProperties, SerializableSemiring, WeaklyDivisibleSemiring, WeightQuantize};
use rustfst::{NomCustomError, Semiring};
use std::borrow::Borrow;
use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::sync::{Arc, Mutex};

#[inline]
fn label_to_fst(l: Label) -> u32 { ((l as isize - Label::MIN as isize + 1) as u32) }
#[inline]
fn fst_to_label(l: u32) -> Label { ((l as isize + Label::MIN as isize - 1) as Label) }

static W_INTERN: Lazy<Mutex<HashSet<Arc<Weight>>>> = Lazy::new(|| Mutex::new(HashSet::new()));
fn intern(w: Weight) -> Arc<Weight> {
    let mut s = W_INTERN.lock().unwrap();
    if let Some(existing) = s.get(&w) { existing.clone() } else { let a = Arc::new(w); s.insert(a.clone()); a }
}

#[derive(Clone, Debug, PartialOrd, Default, Eq, Hash)]
pub struct BitsetWeight(pub Arc<Weight>);
impl PartialEq for BitsetWeight { fn eq(&self, o: &Self) -> bool { Arc::ptr_eq(&self.0, &o.0) || *self.0 == *o.0 } }
impl Semiring for BitsetWeight {
    type Type = Weight; type ReverseWeight = BitsetWeight;
    fn zero() -> Self { Self(intern(Weight::zeros())) }
    fn one() -> Self { Self(intern(Weight::all())) }
    fn new(v: Self::Type) -> Self { Self(intern(v)) }
    fn plus_assign<P: Borrow<Self>>(&mut self, rhs: P) -> Result<()> { self.0 = intern(&*self.0 | &*rhs.borrow().0); Ok(()) }
    fn times_assign<P: Borrow<Self>>(&mut self, rhs: P) -> Result<()> { self.0 = intern(&*self.0 & &*rhs.borrow().0); Ok(()) }
    fn approx_equal<P: Borrow<Self>>(&self, rhs: P, _: f32) -> bool { *self.0 == *rhs.borrow().0 }
    fn value(&self) -> &Weight { &self.0 }
    fn take_value(self) -> Weight { Arc::try_unwrap(self.0).unwrap_or_else(|a| (*a).clone()) }
    fn set_value(&mut self, v: Weight) { self.0 = intern(v); }
    fn reverse(&self) -> Result<Self> { Ok(self.clone()) }
    fn properties() -> SemiringProperties { SemiringProperties::LEFT_SEMIRING | SemiringProperties::RIGHT_SEMIRING | SemiringProperties::COMMUTATIVE | SemiringProperties::IDEMPOTENT | SemiringProperties::PATH }
}
impl ReverseBack<BitsetWeight> for BitsetWeight { fn reverse_back(&self) -> Result<Self> { Ok(self.clone()) } }
impl WeaklyDivisibleSemiring for BitsetWeight { fn divide_assign(&mut self, rhs: &Self, _: DivideType) -> Result<()> { self.0 = intern(&*self.0 | &!&*rhs.0); Ok(()) } }
impl WeightQuantize for BitsetWeight { fn quantize_assign(&mut self, _: f32) -> Result<()> { Ok(()) } }
impl SerializableSemiring for BitsetWeight {
    fn weight_type() -> String { "bitset".to_string() }
    fn parse_binary(i: &[u8]) -> IResult<&[u8], Self, NomCustomError<&[u8]>> {
        let (mut i, num) = nom::number::complete::le_u64(i)?;
        let mut ranges = Vec::with_capacity(num as usize);
        for _ in 0..num {
            let (ni, s) = nom::number::complete::le_u64(i)?;
            let (ni, e) = nom::number::complete::le_u64(ni)?;
            ranges.push(s as usize..=e as usize);
            i = ni;
        }
        Ok((i, Self(intern(Weight::from_rsb(RangeSetBlaze::from_iter(ranges))))))
    }
    fn write_binary<F: Write>(&self, f: &mut F) -> Result<()> {
        let ranges: Vec<_> = self.0.rsb.ranges().collect();
        f.write_all(&(ranges.len() as u64).to_le_bytes())?;
        for r in ranges { f.write_all(&(*r.start() as u64).to_le_bytes())?; f.write_all(&(*r.end() as u64).to_le_bytes())?; }
        Ok(())
    }
    fn parse_text(i: &str) -> IResult<&str, Self> {
        nom::combinator::map_res(nom::combinator::rest, |s: &str| serde_json::from_str::<Weight>(s).map(|w| Self(intern(w))).map_err(|e| e.to_string()))(i)
    }
}
impl std::fmt::Display for BitsetWeight { fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{}", serde_json::to_string(&self.0).unwrap()) } }

pub fn nwa_to_vector_fst(nwa: &NWA) -> VectorFst<BitsetWeight> {
    let mut fst = VectorFst::new();
    let map: HashMap<NWAStateID, StateId> = (0..nwa.states.len()).map(|i| (i, fst.add_state())).collect();
    if !nwa.body.start_states.is_empty() {
        if nwa.body.start_states.len() == 1 { fst.set_start(map[&nwa.body.start_states[0]]).unwrap(); }
        else {
            let start = fst.add_state();
            fst.set_start(start).unwrap();
            for &s in &nwa.body.start_states { fst.add_tr(start, Tr::new(EPS_LABEL, EPS_LABEL, BitsetWeight::one(), map[&s])).unwrap(); }
        }
    }
    for (i, st) in nwa.states.0.iter().enumerate() {
        let fid = map[&i];
        if let Some(w) = &st.final_weight { if !w.is_empty() { fst.set_final(fid, BitsetWeight::new(w.clone())).unwrap(); } }
        for (l, ts) in &st.transitions { for (t, w) in ts { if !w.is_empty() { fst.add_tr(fid, Tr::new(label_to_fst(*l), label_to_fst(*l), BitsetWeight::new(w.clone()), map[t])).unwrap(); } } }
        for (t, w) in &st.epsilons { if !w.is_empty() { fst.add_tr(fid, Tr::new(EPS_LABEL, EPS_LABEL, BitsetWeight::new(w.clone()), map[t])).unwrap(); } }
    }
    fst
}

pub fn vector_fst_to_dwa(fst: &VectorFst<BitsetWeight>) -> DWA {
    let mut dwa = DWA::new();
    let fst_start = if let Some(s) = fst.start() { s } else { return dwa; };
    dwa.states.0.clear();
    let map: HashMap<StateId, StateID> = (0..fst.num_states()).map(|i| (i as StateId, dwa.add_state())).collect();
    dwa.body.start_state = map[&fst_start];
    for i in 0..fst.num_states() {
        let fid = i as StateId;
        if !map.contains_key(&fid) { continue; }
        let did = map[&fid];
        if let Some(w) = fst.final_weight(fid).unwrap() { if !w.0.is_empty() { dwa.set_final_weight(did, w.value().clone()).unwrap(); } }
        for tr in fst.get_trs(fid).unwrap().trs() {
            if !tr.weight.0.is_empty() && map.contains_key(&tr.nextstate) {
                let _ = dwa.add_transition(did, fst_to_label(tr.ilabel), map[&tr.nextstate], tr.weight.value().clone());
            }
        }
    }
    dwa
}

pub fn vector_fst_to_nwa(fst: &VectorFst<BitsetWeight>) -> NWA {
    let mut nwa = NWA::new();
    nwa.states.0.clear();
    let map: HashMap<StateId, NWAStateID> = (0..fst.num_states()).map(|i| (i as StateId, nwa.states.add_state())).collect();
    nwa.body.start_states = if let Some(s) = fst.start() { vec![map[&s]] } else { vec![] };
    for i in 0..fst.num_states() {
        let fid = i as StateId;
        let nid = map[&fid];
        if let Some(w) = fst.final_weight(fid).unwrap() { if !w.0.is_empty() { nwa.states[nid].final_weight = Some(w.value().clone()); } }
        for tr in fst.get_trs(fid).unwrap().trs() {
            if !tr.weight.0.is_empty() {
                let tid = map[&tr.nextstate];
                if tr.ilabel == EPS_LABEL { nwa.states.add_epsilon(nid, tid, tr.weight.value().clone()); }
                else { nwa.states.add_transition(nid, fst_to_label(tr.ilabel), tid, tr.weight.value().clone()).unwrap(); }
            }
        }
    }
    nwa
}

pub fn determinize_nwa_to_dwa(nwa: &NWA) -> DWA {
    let mut fst = nwa_to_vector_fst(nwa);
    fst.compute_and_update_properties_all().unwrap();
    rm_epsilon(&mut fst).unwrap();
    let det: VectorFst<BitsetWeight> = determinize_with_config(&fst, DeterminizeConfig::default().with_det_type(DeterminizeType::DeterminizeFunctional)).unwrap();
    vector_fst_to_dwa(&det)
}

impl DWA {
    pub fn to_rustfst(&self) -> VectorFst<BitsetWeight> { nwa_to_vector_fst(&NWA::from_dwa(self)) }
    pub fn from_rustfst(fst: &VectorFst<BitsetWeight>) -> DWA { vector_fst_to_dwa(fst) }
}
impl NWA {
    pub fn determinize_to_dwa_with_rustfst(&self) -> DWA { determinize_nwa_to_dwa(self) }
    pub fn to_rustfst(&self) -> VectorFst<BitsetWeight> { nwa_to_vector_fst(self) }
    pub fn from_rustfst(fst: &VectorFst<BitsetWeight>) -> NWA { vector_fst_to_nwa(fst) }
}
