// src/precompute4/weighted_automata/determinization_rustfst.rs

#![allow(dead_code)]
#![allow(clippy::needless_borrow)]

use super::common::{Label, StateID, Weight};
use super::dwa::DWA;
use super::nwa::NWA;
use crate::dwa_i32::NWAStateID;
use anyhow::Result;
use nom::IResult;
use profiler_macro::time_it;
use range_set_blaze::RangeSetBlaze;
use rustfst::algorithms::determinize::{determinize_with_config, DeterminizeConfig, DeterminizeType};
use rustfst::algorithms::rm_epsilon::rm_epsilon;
use rustfst::prelude::{CoreFst, ExpandedFst, MutableFst, StateId, Tr, Trs, VectorFst, EPS_LABEL};
use rustfst::semirings::{
    DivideType, ReverseBack, SemiringProperties, SerializableSemiring, WeaklyDivisibleSemiring, WeightQuantize,
};
use rustfst::{NomCustomError, Semiring};
use std::borrow::Borrow;
use std::collections::HashMap;
use std::io::Write;
use std::ops::{BitAndAssign, BitOrAssign};
use std::time::Instant;

#[inline]
fn _label_to_fst_label(label: Label) -> u32 {
    (label as isize - Label::MIN as isize + 1) as u32
}

#[inline]
fn _fst_label_to_label(label: u32) -> Label {
    (label as isize + Label::MIN as isize - 1) as Label
}

#[inline]
fn fst_label_to_label(label: u32) -> Label {
    assert_ne!(label, 0);
    let result = _fst_label_to_label(label);
    let remapped = _label_to_fst_label(result);
    assert!(label == remapped, "label: {}, result: {}, remapped: {}", label, result, remapped);
    result
}

#[inline]
fn label_to_fst_label(label: Label) -> u32 {
    let result = _label_to_fst_label(label);
    assert_ne!(result, 0);
    let remapped = _fst_label_to_label(result);
    assert!(label == remapped, "label: {}, result: {}, remapped: {}", label, result, remapped);
    result
}

impl Semiring for Weight {
    type Type = Weight;
    type ReverseWeight = Weight;

    fn zero() -> Self {
        Weight::zeros()
    }

    fn one() -> Self {
        Weight::all()
    }

    fn new(value: Self::Type) -> Self {
        value
    }

    fn plus_assign<P: Borrow<Self>>(&mut self, rhs: P) -> Result<()> {
        self.bitor_assign(rhs.borrow());
        Ok(())
    }

    fn times_assign<P: Borrow<Self>>(&mut self, rhs: P) -> Result<()> {
        self.bitand_assign(rhs.borrow());
        Ok(())
    }

    fn approx_equal<P: Borrow<Self>>(&self, rhs: P, _delta: f32) -> bool {
        *self == *rhs.borrow()
    }

    fn value(&self) -> &Self::Type {
        self
    }

    fn take_value(self) -> Self::Type {
        self
    }

    fn set_value(&mut self, value: Self::Type) {
        *self = value;
    }

    fn reverse(&self) -> Result<Self::ReverseWeight> {
        Ok(self.clone())
    }

    fn properties() -> SemiringProperties {
        SemiringProperties::LEFT_SEMIRING
            | SemiringProperties::RIGHT_SEMIRING
            | SemiringProperties::COMMUTATIVE
            | SemiringProperties::IDEMPOTENT
            | SemiringProperties::PATH
    }
}

impl ReverseBack<Weight> for Weight {
    fn reverse_back(&self) -> Result<Weight> {
        Ok(self.clone())
    }
}

impl WeaklyDivisibleSemiring for Weight {
    fn divide_assign(&mut self, rhs: &Self, _divide_type: DivideType) -> Result<()> {
        let new_weight = if *self == *rhs {
            Weight::all()
        } else if rhs.is_empty() {
            Weight::all()
        } else if rhs.is_all_fast() {
            self.clone()
        } else if self.is_all_fast() {
            Weight::all()
        } else if self.is_empty() {
            rhs.complement()
        } else {
            self.divide(rhs)
        };

        *self = new_weight;
        Ok(())
    }
}

impl WeightQuantize for Weight {
    fn quantize_assign(&mut self, _delta: f32) -> Result<()> {
        Ok(())
    }
}

impl SerializableSemiring for Weight {
    fn weight_type() -> String {
        "bitset".to_string()
    }

    #[time_it("Weight::parse_binary")]
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
        Ok((i, Weight::from_rsb(rsb)))
    }

    #[time_it("Weight::write_binary")]
    fn write_binary<F: Write>(&self, file: &mut F) -> Result<()> {
        let ranges: Vec<_> = self.to_rsb().ranges().collect();
        file.write_all(&(ranges.len() as u64).to_le_bytes())?;
        for range in ranges {
            file.write_all(&(*range.start() as u64).to_le_bytes())?;
            file.write_all(&(*range.end() as u64).to_le_bytes())?;
        }
        Ok(())
    }

    #[time_it("Weight::parse_text")]
    fn parse_text(i: &str) -> IResult<&str, Self> {
        use nom::combinator::map_res;
        map_res(nom::combinator::rest, |s: &str| -> Result<Weight, _> {
            serde_json::from_str::<Weight>(s).map_err(|e| e.to_string())
        })(i)
    }
}

pub fn nwa_to_vector_fst(nwa: &NWA) -> VectorFst<Weight> {
    let total_start = std::time::Instant::now();
    let mut fst = VectorFst::<Weight>::new();
    let mut state_map = HashMap::<NWAStateID, StateId>::new();
    let add_state_start = std::time::Instant::now();
    for i in 0..nwa.states.len() {
        let s = fst.add_state();
        state_map.insert(i, s);
    }
    let add_state_time = add_state_start.elapsed();

    let mut start_time = std::time::Duration::ZERO;
    let mut start_eps_count = 0usize;

    if !nwa.body.start_states.is_empty() {
        if nwa.body.start_states.len() == 1 {
            let start_set = std::time::Instant::now();
            fst.set_start(state_map[&nwa.body.start_states[0]]).unwrap();
            start_time += start_set.elapsed();
        } else {
            let start_set = std::time::Instant::now();
            let super_start = fst.add_state();
            fst.set_start(super_start).unwrap();
            for &s_idx in &nwa.body.start_states {
                if let Some(&target) = state_map.get(&s_idx) {
                    let add_start = std::time::Instant::now();
                    fst.add_tr(super_start, Tr::new(EPS_LABEL, EPS_LABEL, Weight::one(), target)).unwrap();
                    start_time += add_start.elapsed();
                    start_eps_count += 1;
                }
            }
            start_time += start_set.elapsed();
        }
    }

    let mut final_clone_time = std::time::Duration::ZERO;
    let mut final_set_time = std::time::Duration::ZERO;
    let mut final_count = 0usize;
    let mut trans_clone_time = std::time::Duration::ZERO;
    let mut trans_add_time = std::time::Duration::ZERO;
    let mut trans_count = 0usize;
    let mut eps_clone_time = std::time::Duration::ZERO;
    let mut eps_add_time = std::time::Duration::ZERO;
    let mut eps_count = 0usize;

    for (i, nwa_state) in nwa.states.0.iter().enumerate() {
        let fst_state_id = state_map[&i];

        if let Some(w) = &nwa_state.final_weight {
            if !w.is_empty() {
                let clone_start = std::time::Instant::now();
                let w_clone = w.clone();
                final_clone_time += clone_start.elapsed();
                let set_start = std::time::Instant::now();
                final_count += 1;
                fst.set_final(fst_state_id, Weight::new(w_clone)).unwrap();
                final_set_time += set_start.elapsed();
            }
        }

        for (label, targets) in &nwa_state.transitions {
            for (target, weight) in targets {
                if !weight.is_empty() {
                    let clone_start = std::time::Instant::now();
                    let w_clone = weight.clone();
                    trans_clone_time += clone_start.elapsed();
                    let add_start = std::time::Instant::now();
                    trans_count += 1;
                    fst.add_tr(
                        fst_state_id,
                        Tr::new(
                            label_to_fst_label(*label),
                            label_to_fst_label(*label),
                            Weight::new(w_clone),
                            state_map[target],
                        ),
                    )
                    .unwrap();
                    trans_add_time += add_start.elapsed();
                }
            }
        }

        for (target, weight) in &nwa_state.epsilons {
            if !weight.is_empty() {
                let clone_start = std::time::Instant::now();
                let w_clone = weight.clone();
                eps_clone_time += clone_start.elapsed();
                let add_start = std::time::Instant::now();
                eps_count += 1;
                fst.add_tr(fst_state_id, Tr::new(EPS_LABEL, EPS_LABEL, Weight::new(w_clone), state_map[target]))
                    .unwrap();
                eps_add_time += add_start.elapsed();
            }
        }
    }
    let total_time = total_start.elapsed();
    crate::debug!(5, "nwa_to_vector_fst breakdown: add_state={:?}, set_start={:?}, final_clone={:?}, final_set={:?}, trans_clone={:?}, trans_add={:?}, eps_clone={:?}, eps_add={:?}, total={:?}, counts: finals={}, trans={}, eps={}, start_eps={}",
        add_state_time,
        start_time,
        final_clone_time,
        final_set_time,
        trans_clone_time,
        trans_add_time,
        eps_clone_time,
        eps_add_time,
        total_time,
        final_count,
        trans_count,
        eps_count,
        start_eps_count,
    );
    fst
}

pub fn vector_fst_to_dwa(fst: &VectorFst<Weight>) -> DWA {
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
            if !w.is_empty() {
                dwa.set_final_weight(dwa_state_id, w.clone()).unwrap();
            }
        }

        for tr in fst.get_trs(fst_state_id).unwrap().trs() {
            if !tr.weight.is_empty() {
                if !state_map.contains_key(&tr.nextstate) {
                    continue;
                }
                let res = dwa.add_transition(
                    dwa_state_id,
                    fst_label_to_label(tr.ilabel),
                    state_map[&tr.nextstate],
                    tr.weight.clone(),
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

pub fn vector_fst_to_nwa(fst: &VectorFst<Weight>) -> NWA {
    let total_start = std::time::Instant::now();
    if fst.num_states() == 0 {
        return NWA::new_empty();
    }

    let mut nwa = NWA::new();
    nwa.states.0.clear();
    let mut state_map: Vec<NWAStateID> = Vec::with_capacity(fst.num_states());

    let add_state_start = std::time::Instant::now();
    for _ in 0..fst.num_states() {
        let s = nwa.states.add_state();
        state_map.push(s);
    }
    let add_state_time = add_state_start.elapsed();

    let mut start_time = std::time::Duration::ZERO;
    let mut final_clone_time = std::time::Duration::ZERO;
    let mut final_set_time = std::time::Duration::ZERO;
    let mut final_count = 0usize;
    let mut trans_clone_time = std::time::Duration::ZERO;
    let mut trans_add_time = std::time::Duration::ZERO;
    let mut trans_count = 0usize;
    let mut eps_clone_time = std::time::Duration::ZERO;
    let mut eps_add_time = std::time::Duration::ZERO;
    let mut eps_count = 0usize;

    if let Some(fst_start) = fst.start() {
        let start_set = std::time::Instant::now();
        nwa.body.start_states = vec![state_map[fst_start as usize]];
        start_time += start_set.elapsed();
    } else {
        let start_set = std::time::Instant::now();
        nwa.body.start_states.clear();
        start_time += start_set.elapsed();
    }

    for i in 0..fst.num_states() {
        let fst_state_id = i as StateId;
        let nwa_state_id = state_map[fst_state_id as usize];

        if let Some(w) = fst.final_weight(fst_state_id).unwrap() {
            if !w.is_empty() {
                let clone_start = std::time::Instant::now();
                let w_clone = w.clone();
                final_clone_time += clone_start.elapsed();
                let set_start = std::time::Instant::now();
                final_count += 1;
                nwa.states[nwa_state_id].final_weight = Some(w_clone);
                final_set_time += set_start.elapsed();
            }
        }

        for tr in fst.get_trs(fst_state_id).unwrap().trs() {
            if !tr.weight.is_empty() {
                let target_nwa_id = state_map[tr.nextstate as usize];
                let clone_start = std::time::Instant::now();
                let weight = tr.weight.clone();
                let clone_time = clone_start.elapsed();

                if tr.ilabel == EPS_LABEL {
                    eps_clone_time += clone_time;
                    let add_start = std::time::Instant::now();
                    nwa.states.add_epsilon(nwa_state_id, target_nwa_id, weight);
                    eps_add_time += add_start.elapsed();
                    eps_count += 1;
                } else {
                    let label = fst_label_to_label(tr.ilabel);
                    trans_clone_time += clone_time;
                    let add_start = std::time::Instant::now();
                    nwa.states.add_transition(nwa_state_id, label, target_nwa_id, weight).unwrap();
                    trans_add_time += add_start.elapsed();
                    trans_count += 1;
                }
            }
        }
    }

    // Attempt to reduce "super-start" state if it looks artificial.
    // nwa_to_vector_fst creates a super-start at the highest index if multiple start states exist.
    let cleanup_start = std::time::Instant::now();
    if nwa.body.start_states.len() == 1 {
        let candidate = nwa.body.start_states[0];
        let last_idx = nwa.states.len().saturating_sub(1);

        // The super-start is always the last state added
        if candidate == last_idx && candidate > 0 {
            let is_candidate_prop = {
                let st = &nwa.states[candidate];
                st.final_weight.as_ref().map_or(true, |w| w.is_empty())
                    && st.transitions.is_empty()
                    && !st.epsilons.is_empty()
                    && st.epsilons.iter().all(|(_, w)| w.is_all_fast())
            };

            if is_candidate_prop {
                // Verify no incoming edges point to this candidate
                let has_incoming = nwa.states.0.iter().enumerate().any(|(i, s)| {
                    if i == candidate {
                        // Check for self-loops (which super-start shouldn't have)
                        return s.epsilons.iter().any(|(t, _)| *t == candidate);
                    }
                    // Check labeled transitions
                    for targets in s.transitions.values() {
                        for (t, _) in targets {
                            if *t == candidate {
                                return true;
                            }
                        }
                    }
                    // Check epsilon transitions
                    for (t, _) in &s.epsilons {
                        if *t == candidate {
                            return true;
                        }
                    }
                    false
                });

                if !has_incoming {
                    // Inline the super-start: replace it with its targets
                    let new_starts: Vec<NWAStateID> = nwa.states[candidate].epsilons.iter().map(|(t, _)| *t).collect();
                    nwa.body.start_states = new_starts;
                    nwa.states.0.pop(); // Remove the last state
                }
            }
        }
    }
    let cleanup_time = cleanup_start.elapsed();

    let total_time = total_start.elapsed();
    crate::debug!(5, "vector_fst_to_nwa breakdown: add_state={:?}, set_start={:?}, final_clone={:?}, final_set={:?}, trans_clone={:?}, trans_add={:?}, eps_clone={:?}, eps_add={:?}, cleanup={:?}, total={:?}, counts: finals={}, trans={}, eps={}",
        add_state_time,
        start_time,
        final_clone_time,
        final_set_time,
        trans_clone_time,
        trans_add_time,
        eps_clone_time,
        eps_add_time,
        cleanup_time,
        total_time,
        final_count,
        trans_count,
        eps_count,
    );

    nwa
}

pub fn determinize_nwa_to_dwa(nwa: &NWA) -> DWA {
    if let Some(dwa) = try_direct_dwa_from_deterministic_nwa(nwa) {
        crate::debug!(5, "Determinization: fast-path deterministic NWA -> DWA");
        return dwa;
    }

    let has_eps = nwa.body.start_states.len() > 1 || nwa.states.0.iter().any(|s| !s.epsilons.is_empty());
    let mut fst = nwa_to_vector_fst(nwa);

    let skip_rm_epsilon = std::env::var("RUSTFST_SKIP_RM_EPSILON")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    if has_eps {
        if skip_rm_epsilon {
            crate::debug!(5, "rustfst: RUSTFST_SKIP_RM_EPSILON set but epsilons present; keeping rm_epsilon for correctness");
        }
        rm_epsilon(&mut fst).unwrap();
    } else if skip_rm_epsilon {
        crate::debug!(5, "rustfst: skipping rm_epsilon (no epsilons)");
    }

    let det_type = std::env::var("RUSTFST_DETERMINIZE_TYPE")
        .ok()
        .and_then(|v| match v.to_ascii_lowercase().as_str() {
            "functional" => Some(DeterminizeType::DeterminizeFunctional),
            "nonfunctional" | "non-functional" => Some(DeterminizeType::DeterminizeNonFunctional),
            "disambiguate" => Some(DeterminizeType::DeterminizeDisambiguate),
            _ => None,
        })
        .unwrap_or(DeterminizeType::DeterminizeFunctional);
    let det_delta = std::env::var("RUSTFST_DETERMINIZE_DELTA")
        .ok()
        .and_then(|v| v.parse::<f32>().ok());
    let mut det_config = DeterminizeConfig::default().with_det_type(det_type);
    if let Some(delta) = det_delta {
        det_config = det_config.with_delta(delta);
    }
    let det_fst: VectorFst<Weight> = determinize_with_config(&fst, det_config).unwrap();

    vector_fst_to_dwa(&det_fst)
}

fn try_direct_dwa_from_deterministic_nwa(nwa: &NWA) -> Option<DWA> {
    if nwa.body.start_states.len() != 1 {
        return None;
    }
    if nwa.states.0.iter().any(|s| !s.epsilons.is_empty()) {
        return None;
    }

    for state in &nwa.states.0 {
        for targets in state.transitions.values() {
            if targets.len() > 1 {
                return None;
            }
        }
    }

    let mut dwa = DWA::new();
    dwa.states.0.clear();
    let mut state_map = Vec::with_capacity(nwa.states.len());
    for _ in 0..nwa.states.len() {
        state_map.push(dwa.add_state());
    }

    dwa.body.start_state = state_map[nwa.body.start_states[0]];

    for (i, state) in nwa.states.0.iter().enumerate() {
        let dwa_state = state_map[i];
        if let Some(w) = &state.final_weight {
            if !w.is_empty() {
                let _ = dwa.set_final_weight(dwa_state, w.clone());
            }
        }
        for (label, targets) in &state.transitions {
            if let Some((target, weight)) = targets.first() {
                if weight.is_empty() {
                    continue;
                }
                let _ = dwa.add_transition(dwa_state, *label, state_map[*target], weight.clone());
            }
        }
    }

    Some(dwa)
}

impl DWA {
    pub fn to_rustfst(&self) -> VectorFst<Weight> {
        nwa_to_vector_fst(&NWA::from_dwa(self))
    }

    pub fn from_rustfst(fst: &VectorFst<Weight>) -> DWA {
        vector_fst_to_dwa(fst)
    }
}

impl NWA {
    pub fn determinize_to_dwa_with_rustfst(&self) -> DWA {
        determinize_nwa_to_dwa(self)
    }

    pub fn to_rustfst(&self) -> VectorFst<Weight> {
        nwa_to_vector_fst(self)
    }

    pub fn from_rustfst(fst: &VectorFst<Weight>) -> NWA {
        vector_fst_to_nwa(fst)
    }
    
    /// Remove epsilon transitions by converting to rustfst and back.
    /// This canonicalizes the NWA structure for more efficient determinization.
    pub fn remove_epsilons(&self) -> NWA {
        let mut fst = nwa_to_vector_fst(self);
        fst.compute_and_update_properties_all().unwrap();
        rm_epsilon(&mut fst).unwrap();
        vector_fst_to_nwa(&fst)
    }
}
