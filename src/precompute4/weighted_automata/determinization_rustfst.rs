// src/precompute4/weighted_automata/determinization_rustfst.rs

use super::common::{StateID, Weight};
use super::dwa::{DWA, DWABuildError};
use super::nwa::NWA;
use crate::precompute4::weighted_automata::bitset::SimpleBitset;
use crate::precompute4::weighted_automata::{NWAStateID};
use anyhow::Result;
use nom::IResult;
use rustfst::NomCustomError;
use rustfst::prelude::*;
use rustfst::semirings::{
    DivideType, ReverseBack, SemiringProperties, SerializableSemiring, WeaklyDivisibleSemiring,
    WeightQuantize,
};
use serde::{Deserialize, Serialize};
use std::borrow::Borrow;
use std::collections::HashMap;
use std::io::Write;
use rustfst::algorithms::determinize::{determinize_with_config, DeterminizeConfig, DeterminizeType};
use rustfst::algorithms::rm_epsilon::rm_epsilon;
use range_set_blaze::RangeSetBlaze;

#[derive(Clone, Debug, PartialEq, PartialOrd, Default, Eq, Hash, Serialize, Deserialize)]
pub struct BitsetWeight(pub Weight);

impl Semiring for BitsetWeight {
    type Type = Weight;
    type ReverseWeight = BitsetWeight;

    fn zero() -> Self {
        BitsetWeight(Weight::zeros())
    }
    fn one() -> Self {
        BitsetWeight(Weight::all())
    }
    fn new(value: Self::Type) -> Self {
        BitsetWeight(value)
    }

    fn plus_assign<P: Borrow<Self>>(&mut self, rhs: P) -> Result<()> {
        self.0 |= &rhs.borrow().0;
        Ok(())
    }

    fn times_assign<P: Borrow<Self>>(&mut self, rhs: P) -> Result<()> {
        self.0 &= &rhs.borrow().0;
        Ok(())
    }

    fn approx_equal<P: Borrow<Self>>(&self, rhs: P, _delta: f32) -> bool {
        self.0 == rhs.borrow().0
    }

    fn value(&self) -> &Self::Type {
        &self.0
    }
    fn take_value(self) -> Self::Type {
        self.0
    }
    fn set_value(&mut self, value: Self::Type) {
        self.0 = value;
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

impl ReverseBack<BitsetWeight> for BitsetWeight {
    fn reverse_back(&self) -> Result<BitsetWeight> {
        Ok(self.clone())
    }
}

impl WeaklyDivisibleSemiring for BitsetWeight {
    fn divide_assign(&mut self, _rhs: &Self, _divide_type: DivideType) -> Result<()> {
        // For a boolean algebra (with OR as plus and AND as times), division a/b is a | !b.
        // This is because we need `(a/b) & b = a` when `a` is a "sub-weight" of `b` (i.e. a subset).
        // `(a | !b) & b = (a & b) | (!b & b) = a & b`.
        // Since `a` is a subset of `b` in this context, `a & b = a`.
        self.0 |= &!&_rhs.0;
        Ok(())
    }
}

impl WeightQuantize for BitsetWeight {
    fn quantize_assign(&mut self, _delta: f32) -> Result<()> {
        Ok(())
    }
}

impl SerializableSemiring for BitsetWeight {
    fn weight_type() -> String {
        "bitset".to_string()
    }

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
        Ok((i, BitsetWeight(Weight::from_rsb(rsb))))
    }

    fn write_binary<F: Write>(&self, file: &mut F) -> Result<()> {
        let ranges: Vec<_> = self.0.rsb.ranges().collect();
        file.write_all(&(ranges.len() as u64).to_le_bytes())?;
        for range in ranges {
            file.write_all(&((*range.start() as u64).to_le_bytes()))?;
            file.write_all(&((*range.end() as u64).to_le_bytes()))?;
        }
        Ok(())
    }

    fn parse_text(i: &str) -> IResult<&str, Self> {
        use nom::combinator::map_res;
        map_res(nom::combinator::rest, |s: &str| -> Result<BitsetWeight, _> {
            serde_json::from_str::<Weight>(s)
                .map(BitsetWeight)
                .map_err(|e| e.to_string())
        })(i)
    }
}

impl std::fmt::Display for BitsetWeight {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            serde_json::to_string(&self.0).unwrap_or_else(|_| "err".to_string())
        )
    }
}

fn nwa_to_vector_fst(nwa: &NWA) -> VectorFst<BitsetWeight> {
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
                fst.set_final(fst_state_id, BitsetWeight(w.clone()))
                    .unwrap();
            }
        }

        for (label, targets) in &nwa_state.transitions {
            for (target, weight) in targets {
                if !weight.is_empty() {
                    fst.add_tr(
                        fst_state_id,
                        Tr::new(
                            (*label as u16 as u64) + 1,
                            (*label as u16 as u64) + 1,
                            BitsetWeight(weight.clone()),
                            state_map[target],
                        ),
                    )
                    .unwrap();
                }
            }
        }

        for (target, weight) in &nwa_state.epsilons {
            if !weight.is_empty() {
                fst.add_tr(
                    fst_state_id,
                    Tr::new(
                        EPS_LABEL,
                        EPS_LABEL,
                        BitsetWeight(weight.clone()),
                        state_map[target],
                    ),
                )
                .unwrap();
            }
        }
    }
    crate::debug!(5, "NWA to FST conversion done:\n{}", fst);
    fst
}

fn vector_fst_to_dwa(fst: &VectorFst<BitsetWeight>) -> DWA {
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
                dwa.set_final_weight(dwa_state_id, w.0).unwrap();
            }
        }

        for tr in fst.get_trs(fst_state_id).unwrap().trs() {
            if !tr.weight.0.is_empty() {
                if !state_map.contains_key(&tr.nextstate) {
                    continue;
                }
                let res = dwa.add_transition(
                    dwa_state_id,
                    ((tr.ilabel - 1) as u16) as i16,
                    state_map[&tr.nextstate],
                    tr.weight.0.clone(),
                );
                if let Err(e) = res {
                    // This should not happen if the input FST is deterministic.
                    panic!("Error converting VectorFst to DWA: transition already exists. This indicates non-determinism. Error: {:?}", e);
                }
            }
        }
    }
    dwa
}

pub fn determinize_nwa_to_dwa(nwa: &NWA) -> DWA {
    let mut fst = nwa_to_vector_fst(nwa);
    rm_epsilon(&mut fst).unwrap();
    let det_config =
        DeterminizeConfig::default().with_det_type(DeterminizeType::DeterminizeNonFunctional);
    let det_fst: VectorFst<BitsetWeight> = determinize_with_config(&fst, det_config).unwrap();
    crate::debug!(5, "Determinized FST:\n{}", det_fst);
    vector_fst_to_dwa(&det_fst)
}
