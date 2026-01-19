// src/precompute4/weighted_automata/determinization_rustfst.rs

#![allow(dead_code)]
#![allow(clippy::needless_borrow)]

use super::common::{Label, StateID, Weight};
use super::dwa::DWA;
use super::nwa::NWA;
use crate::dwa_i32::NWAStateID;
use anyhow::Result;
use lru::LruCache;
use nom::IResult;
use once_cell::sync::Lazy;
use range_set_blaze::RangeSetBlaze;
use rustfst::algorithms::determinize::{determinize_with_config, DeterminizeConfig, DeterminizeType};
use rustfst::algorithms::rm_epsilon::rm_epsilon;
use rustfst::fst_properties::FstProperties;
use rustfst::prelude::{CoreFst, ExpandedFst, MutableFst, StateId, Tr, Trs, VectorFst, EPS_LABEL};
use rustfst::semirings::{
    DivideType, ReverseBack, SemiringProperties, SerializableSemiring, WeaklyDivisibleSemiring, WeightQuantize,
};
use rustfst::{NomCustomError, Semiring};
use profiler_macro::time_it;
use std::borrow::Borrow;
use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
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

static WEIGHT_INTERNER: Lazy<Mutex<HashSet<Arc<Weight>>>> = Lazy::new(|| Mutex::new(HashSet::new()));

const WEIGHT_DIVIDE_CACHE_CAPACITY: usize = 100_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct DivideKey {
    a: usize,
    b: usize,
}

static WEIGHT_DIVIDE_CACHE: Lazy<Mutex<LruCache<DivideKey, Arc<Weight>>>> = Lazy::new(|| {
    Mutex::new(LruCache::new(
        NonZeroUsize::new(WEIGHT_DIVIDE_CACHE_CAPACITY).unwrap(),
    ))
});

static RUSTFST_WEIGHT_COUNT_ZERO: AtomicU64 = AtomicU64::new(0);
static RUSTFST_WEIGHT_TIME_ZERO: AtomicU64 = AtomicU64::new(0);
static RUSTFST_WEIGHT_COUNT_ONE: AtomicU64 = AtomicU64::new(0);
static RUSTFST_WEIGHT_TIME_ONE: AtomicU64 = AtomicU64::new(0);
static RUSTFST_WEIGHT_COUNT_NEW: AtomicU64 = AtomicU64::new(0);
static RUSTFST_WEIGHT_TIME_NEW: AtomicU64 = AtomicU64::new(0);
static RUSTFST_WEIGHT_COUNT_PROPERTIES: AtomicU64 = AtomicU64::new(0);
static RUSTFST_WEIGHT_TIME_PROPERTIES: AtomicU64 = AtomicU64::new(0);
static RUSTFST_WEIGHT_COUNT_PLUS: AtomicU64 = AtomicU64::new(0);
static RUSTFST_WEIGHT_TIME_PLUS: AtomicU64 = AtomicU64::new(0);
static RUSTFST_WEIGHT_COUNT_TIMES: AtomicU64 = AtomicU64::new(0);
static RUSTFST_WEIGHT_TIME_TIMES: AtomicU64 = AtomicU64::new(0);
static RUSTFST_WEIGHT_COUNT_DIVIDE: AtomicU64 = AtomicU64::new(0);
static RUSTFST_WEIGHT_TIME_DIVIDE: AtomicU64 = AtomicU64::new(0);
static RUSTFST_WEIGHT_COUNT_APPROX_EQUAL: AtomicU64 = AtomicU64::new(0);
static RUSTFST_WEIGHT_TIME_APPROX_EQUAL: AtomicU64 = AtomicU64::new(0);
static RUSTFST_WEIGHT_COUNT_VALUE: AtomicU64 = AtomicU64::new(0);
static RUSTFST_WEIGHT_TIME_VALUE: AtomicU64 = AtomicU64::new(0);
static RUSTFST_WEIGHT_COUNT_TAKE_VALUE: AtomicU64 = AtomicU64::new(0);
static RUSTFST_WEIGHT_TIME_TAKE_VALUE: AtomicU64 = AtomicU64::new(0);
static RUSTFST_WEIGHT_COUNT_SET_VALUE: AtomicU64 = AtomicU64::new(0);
static RUSTFST_WEIGHT_TIME_SET_VALUE: AtomicU64 = AtomicU64::new(0);
static RUSTFST_WEIGHT_COUNT_REVERSE: AtomicU64 = AtomicU64::new(0);
static RUSTFST_WEIGHT_TIME_REVERSE: AtomicU64 = AtomicU64::new(0);
static RUSTFST_WEIGHT_COUNT_REVERSE_BACK: AtomicU64 = AtomicU64::new(0);
static RUSTFST_WEIGHT_TIME_REVERSE_BACK: AtomicU64 = AtomicU64::new(0);
static RUSTFST_WEIGHT_COUNT_QUANTIZE: AtomicU64 = AtomicU64::new(0);
static RUSTFST_WEIGHT_TIME_QUANTIZE: AtomicU64 = AtomicU64::new(0);
static RUSTFST_WEIGHT_COUNT_WEIGHT_TYPE: AtomicU64 = AtomicU64::new(0);
static RUSTFST_WEIGHT_TIME_WEIGHT_TYPE: AtomicU64 = AtomicU64::new(0);
static RUSTFST_WEIGHT_COUNT_PARSE_BINARY: AtomicU64 = AtomicU64::new(0);
static RUSTFST_WEIGHT_TIME_PARSE_BINARY: AtomicU64 = AtomicU64::new(0);
static RUSTFST_WEIGHT_COUNT_WRITE_BINARY: AtomicU64 = AtomicU64::new(0);
static RUSTFST_WEIGHT_TIME_WRITE_BINARY: AtomicU64 = AtomicU64::new(0);

fn rustfst_weight_profile_enabled() -> bool {
    static ENABLED: Lazy<bool> = Lazy::new(|| {
        let macro_level = std::env::var("MACRO_DEBUG_LEVEL")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(0);
        std::env::var("PROFILE_RUSTFST_WEIGHT_OPS")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
            || macro_level >= 5
    });
    *ENABLED
}

pub fn reset_rustfst_weight_profile() {
    RUSTFST_WEIGHT_COUNT_ZERO.store(0, Ordering::Relaxed);
    RUSTFST_WEIGHT_TIME_ZERO.store(0, Ordering::Relaxed);
    RUSTFST_WEIGHT_COUNT_ONE.store(0, Ordering::Relaxed);
    RUSTFST_WEIGHT_TIME_ONE.store(0, Ordering::Relaxed);
    RUSTFST_WEIGHT_COUNT_NEW.store(0, Ordering::Relaxed);
    RUSTFST_WEIGHT_TIME_NEW.store(0, Ordering::Relaxed);
    RUSTFST_WEIGHT_COUNT_PROPERTIES.store(0, Ordering::Relaxed);
    RUSTFST_WEIGHT_TIME_PROPERTIES.store(0, Ordering::Relaxed);
    RUSTFST_WEIGHT_COUNT_PLUS.store(0, Ordering::Relaxed);
    RUSTFST_WEIGHT_TIME_PLUS.store(0, Ordering::Relaxed);
    RUSTFST_WEIGHT_COUNT_TIMES.store(0, Ordering::Relaxed);
    RUSTFST_WEIGHT_TIME_TIMES.store(0, Ordering::Relaxed);
    RUSTFST_WEIGHT_COUNT_DIVIDE.store(0, Ordering::Relaxed);
    RUSTFST_WEIGHT_TIME_DIVIDE.store(0, Ordering::Relaxed);
    RUSTFST_WEIGHT_COUNT_APPROX_EQUAL.store(0, Ordering::Relaxed);
    RUSTFST_WEIGHT_TIME_APPROX_EQUAL.store(0, Ordering::Relaxed);
    RUSTFST_WEIGHT_COUNT_VALUE.store(0, Ordering::Relaxed);
    RUSTFST_WEIGHT_TIME_VALUE.store(0, Ordering::Relaxed);
    RUSTFST_WEIGHT_COUNT_TAKE_VALUE.store(0, Ordering::Relaxed);
    RUSTFST_WEIGHT_TIME_TAKE_VALUE.store(0, Ordering::Relaxed);
    RUSTFST_WEIGHT_COUNT_SET_VALUE.store(0, Ordering::Relaxed);
    RUSTFST_WEIGHT_TIME_SET_VALUE.store(0, Ordering::Relaxed);
    RUSTFST_WEIGHT_COUNT_REVERSE.store(0, Ordering::Relaxed);
    RUSTFST_WEIGHT_TIME_REVERSE.store(0, Ordering::Relaxed);
    RUSTFST_WEIGHT_COUNT_REVERSE_BACK.store(0, Ordering::Relaxed);
    RUSTFST_WEIGHT_TIME_REVERSE_BACK.store(0, Ordering::Relaxed);
    RUSTFST_WEIGHT_COUNT_QUANTIZE.store(0, Ordering::Relaxed);
    RUSTFST_WEIGHT_TIME_QUANTIZE.store(0, Ordering::Relaxed);
    RUSTFST_WEIGHT_COUNT_WEIGHT_TYPE.store(0, Ordering::Relaxed);
    RUSTFST_WEIGHT_TIME_WEIGHT_TYPE.store(0, Ordering::Relaxed);
    RUSTFST_WEIGHT_COUNT_PARSE_BINARY.store(0, Ordering::Relaxed);
    RUSTFST_WEIGHT_TIME_PARSE_BINARY.store(0, Ordering::Relaxed);
    RUSTFST_WEIGHT_COUNT_WRITE_BINARY.store(0, Ordering::Relaxed);
    RUSTFST_WEIGHT_TIME_WRITE_BINARY.store(0, Ordering::Relaxed);
}

pub fn print_rustfst_weight_profile(label: &str) {
    let count_plus = RUSTFST_WEIGHT_COUNT_PLUS.load(Ordering::Relaxed);
    let count_times = RUSTFST_WEIGHT_COUNT_TIMES.load(Ordering::Relaxed);
    let count_divide = RUSTFST_WEIGHT_COUNT_DIVIDE.load(Ordering::Relaxed);
    let count_zero = RUSTFST_WEIGHT_COUNT_ZERO.load(Ordering::Relaxed);
    let count_one = RUSTFST_WEIGHT_COUNT_ONE.load(Ordering::Relaxed);
    let count_new = RUSTFST_WEIGHT_COUNT_NEW.load(Ordering::Relaxed);
    let count_properties = RUSTFST_WEIGHT_COUNT_PROPERTIES.load(Ordering::Relaxed);
    let count_approx = RUSTFST_WEIGHT_COUNT_APPROX_EQUAL.load(Ordering::Relaxed);
    let count_value = RUSTFST_WEIGHT_COUNT_VALUE.load(Ordering::Relaxed);
    let count_take_value = RUSTFST_WEIGHT_COUNT_TAKE_VALUE.load(Ordering::Relaxed);
    let count_set_value = RUSTFST_WEIGHT_COUNT_SET_VALUE.load(Ordering::Relaxed);
    let count_reverse = RUSTFST_WEIGHT_COUNT_REVERSE.load(Ordering::Relaxed);
    let count_reverse_back = RUSTFST_WEIGHT_COUNT_REVERSE_BACK.load(Ordering::Relaxed);
    let count_quantize = RUSTFST_WEIGHT_COUNT_QUANTIZE.load(Ordering::Relaxed);
    let count_weight_type = RUSTFST_WEIGHT_COUNT_WEIGHT_TYPE.load(Ordering::Relaxed);
    let count_parse_binary = RUSTFST_WEIGHT_COUNT_PARSE_BINARY.load(Ordering::Relaxed);
    let count_write_binary = RUSTFST_WEIGHT_COUNT_WRITE_BINARY.load(Ordering::Relaxed);

    if count_plus
        + count_times
        + count_divide
        + count_zero
        + count_one
        + count_new
        + count_properties
        + count_approx
        + count_value
        + count_take_value
        + count_set_value
        + count_reverse
        + count_reverse_back
        + count_quantize
        + count_weight_type
        + count_parse_binary
        + count_write_binary
        == 0
    {
        return;
    }

    let avg_us = |time_us: u64, count: u64| -> f64 {
        if count == 0 { 0.0 } else { time_us as f64 / count as f64 }
    };

    println!("RUSTFST_WEIGHT_PROF [{}]:", label);
    println!(
        "  plus_assign:   {:9} ops, {:9} us (avg {:.2} us)",
        count_plus,
        RUSTFST_WEIGHT_TIME_PLUS.load(Ordering::Relaxed),
        avg_us(RUSTFST_WEIGHT_TIME_PLUS.load(Ordering::Relaxed), count_plus)
    );
    println!(
        "  times_assign:  {:9} ops, {:9} us (avg {:.2} us)",
        count_times,
        RUSTFST_WEIGHT_TIME_TIMES.load(Ordering::Relaxed),
        avg_us(RUSTFST_WEIGHT_TIME_TIMES.load(Ordering::Relaxed), count_times)
    );
    println!(
        "  divide_assign: {:9} ops, {:9} us (avg {:.2} us)",
        count_divide,
        RUSTFST_WEIGHT_TIME_DIVIDE.load(Ordering::Relaxed),
        avg_us(RUSTFST_WEIGHT_TIME_DIVIDE.load(Ordering::Relaxed), count_divide)
    );
    println!(
        "  zero:          {:9} ops, {:9} us (avg {:.2} us)",
        count_zero,
        RUSTFST_WEIGHT_TIME_ZERO.load(Ordering::Relaxed),
        avg_us(RUSTFST_WEIGHT_TIME_ZERO.load(Ordering::Relaxed), count_zero)
    );
    println!(
        "  one:           {:9} ops, {:9} us (avg {:.2} us)",
        count_one,
        RUSTFST_WEIGHT_TIME_ONE.load(Ordering::Relaxed),
        avg_us(RUSTFST_WEIGHT_TIME_ONE.load(Ordering::Relaxed), count_one)
    );
    println!(
        "  new:           {:9} ops, {:9} us (avg {:.2} us)",
        count_new,
        RUSTFST_WEIGHT_TIME_NEW.load(Ordering::Relaxed),
        avg_us(RUSTFST_WEIGHT_TIME_NEW.load(Ordering::Relaxed), count_new)
    );
    println!(
        "  properties:    {:9} ops, {:9} us (avg {:.2} us)",
        count_properties,
        RUSTFST_WEIGHT_TIME_PROPERTIES.load(Ordering::Relaxed),
        avg_us(RUSTFST_WEIGHT_TIME_PROPERTIES.load(Ordering::Relaxed), count_properties)
    );
    println!(
        "  approx_equal:  {:9} ops, {:9} us (avg {:.2} us)",
        count_approx,
        RUSTFST_WEIGHT_TIME_APPROX_EQUAL.load(Ordering::Relaxed),
        avg_us(RUSTFST_WEIGHT_TIME_APPROX_EQUAL.load(Ordering::Relaxed), count_approx)
    );
    println!(
        "  value:         {:9} ops, {:9} us (avg {:.2} us)",
        count_value,
        RUSTFST_WEIGHT_TIME_VALUE.load(Ordering::Relaxed),
        avg_us(RUSTFST_WEIGHT_TIME_VALUE.load(Ordering::Relaxed), count_value)
    );
    println!(
        "  take_value:    {:9} ops, {:9} us (avg {:.2} us)",
        count_take_value,
        RUSTFST_WEIGHT_TIME_TAKE_VALUE.load(Ordering::Relaxed),
        avg_us(RUSTFST_WEIGHT_TIME_TAKE_VALUE.load(Ordering::Relaxed), count_take_value)
    );
    println!(
        "  set_value:     {:9} ops, {:9} us (avg {:.2} us)",
        count_set_value,
        RUSTFST_WEIGHT_TIME_SET_VALUE.load(Ordering::Relaxed),
        avg_us(RUSTFST_WEIGHT_TIME_SET_VALUE.load(Ordering::Relaxed), count_set_value)
    );
    println!(
        "  reverse:       {:9} ops, {:9} us (avg {:.2} us)",
        count_reverse,
        RUSTFST_WEIGHT_TIME_REVERSE.load(Ordering::Relaxed),
        avg_us(RUSTFST_WEIGHT_TIME_REVERSE.load(Ordering::Relaxed), count_reverse)
    );
    println!(
        "  reverse_back:  {:9} ops, {:9} us (avg {:.2} us)",
        count_reverse_back,
        RUSTFST_WEIGHT_TIME_REVERSE_BACK.load(Ordering::Relaxed),
        avg_us(RUSTFST_WEIGHT_TIME_REVERSE_BACK.load(Ordering::Relaxed), count_reverse_back)
    );
    println!(
        "  quantize:      {:9} ops, {:9} us (avg {:.2} us)",
        count_quantize,
        RUSTFST_WEIGHT_TIME_QUANTIZE.load(Ordering::Relaxed),
        avg_us(RUSTFST_WEIGHT_TIME_QUANTIZE.load(Ordering::Relaxed), count_quantize)
    );
    println!(
        "  weight_type:   {:9} ops, {:9} us (avg {:.2} us)",
        count_weight_type,
        RUSTFST_WEIGHT_TIME_WEIGHT_TYPE.load(Ordering::Relaxed),
        avg_us(RUSTFST_WEIGHT_TIME_WEIGHT_TYPE.load(Ordering::Relaxed), count_weight_type)
    );
    println!(
        "  parse_binary:  {:9} ops, {:9} us (avg {:.2} us)",
        count_parse_binary,
        RUSTFST_WEIGHT_TIME_PARSE_BINARY.load(Ordering::Relaxed),
        avg_us(RUSTFST_WEIGHT_TIME_PARSE_BINARY.load(Ordering::Relaxed), count_parse_binary)
    );
    println!(
        "  write_binary:  {:9} ops, {:9} us (avg {:.2} us)",
        count_write_binary,
        RUSTFST_WEIGHT_TIME_WRITE_BINARY.load(Ordering::Relaxed),
        avg_us(RUSTFST_WEIGHT_TIME_WRITE_BINARY.load(Ordering::Relaxed), count_write_binary)
    );
}

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
#[derive(Clone, Debug, PartialOrd, Default, Eq)]
pub struct BitsetWeight(pub Arc<Weight>);

impl PartialEq for BitsetWeight {
    fn eq(&self, other: &Self) -> bool { Arc::ptr_eq(&self.0, &other.0) || *self.0 == *other.0 }
}

impl std::hash::Hash for BitsetWeight {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Hash the underlying weight value for consistency with PartialEq
        self.0.hash(state);
    }
}

impl Semiring for BitsetWeight {
    type Type = Weight;
    type ReverseWeight = BitsetWeight;

    fn zero() -> Self {
        let prof = rustfst_weight_profile_enabled();
        let start = if prof { Some(Instant::now()) } else { None };
        let res = BitsetWeight(intern_weight(Weight::zeros()));
        if let Some(start) = start {
            RUSTFST_WEIGHT_COUNT_ZERO.fetch_add(1, Ordering::Relaxed);
            RUSTFST_WEIGHT_TIME_ZERO.fetch_add(start.elapsed().as_micros() as u64, Ordering::Relaxed);
        }
        res
    }

    fn one() -> Self {
        let prof = rustfst_weight_profile_enabled();
        let start = if prof { Some(Instant::now()) } else { None };
        let res = BitsetWeight(intern_weight(Weight::all()));
        if let Some(start) = start {
            RUSTFST_WEIGHT_COUNT_ONE.fetch_add(1, Ordering::Relaxed);
            RUSTFST_WEIGHT_TIME_ONE.fetch_add(start.elapsed().as_micros() as u64, Ordering::Relaxed);
        }
        res
    }

    fn new(value: Self::Type) -> Self {
        let prof = rustfst_weight_profile_enabled();
        let start = if prof { Some(Instant::now()) } else { None };
        let res = BitsetWeight(intern_weight(value));
        if let Some(start) = start {
            RUSTFST_WEIGHT_COUNT_NEW.fetch_add(1, Ordering::Relaxed);
            RUSTFST_WEIGHT_TIME_NEW.fetch_add(start.elapsed().as_micros() as u64, Ordering::Relaxed);
        }
        res
    }

    #[time_it("BitsetWeight::plus_assign")]
    fn plus_assign<P: Borrow<Self>>(&mut self, rhs: P) -> Result<()> {
        let prof = rustfst_weight_profile_enabled();
        let start = if prof { Some(Instant::now()) } else { None };
        let new_weight = &*self.0 | &*rhs.borrow().0;
        self.0 = intern_weight(new_weight);
        if let Some(start) = start {
            RUSTFST_WEIGHT_COUNT_PLUS.fetch_add(1, Ordering::Relaxed);
            RUSTFST_WEIGHT_TIME_PLUS.fetch_add(start.elapsed().as_micros() as u64, Ordering::Relaxed);
        }
        Ok(())
    }

    #[time_it("BitsetWeight::times_assign")]
    fn times_assign<P: Borrow<Self>>(&mut self, rhs: P) -> Result<()> {
        let prof = rustfst_weight_profile_enabled();
        let start = if prof { Some(Instant::now()) } else { None };
        let new_weight = &*self.0 & &*rhs.borrow().0;
        self.0 = intern_weight(new_weight);
        if let Some(start) = start {
            RUSTFST_WEIGHT_COUNT_TIMES.fetch_add(1, Ordering::Relaxed);
            RUSTFST_WEIGHT_TIME_TIMES.fetch_add(start.elapsed().as_micros() as u64, Ordering::Relaxed);
        }
        Ok(())
    }

    fn approx_equal<P: Borrow<Self>>(&self, rhs: P, _delta: f32) -> bool {
        let prof = rustfst_weight_profile_enabled();
        let start = if prof { Some(Instant::now()) } else { None };
        let res = *self.0 == *rhs.borrow().0;
        if let Some(start) = start {
            RUSTFST_WEIGHT_COUNT_APPROX_EQUAL.fetch_add(1, Ordering::Relaxed);
            RUSTFST_WEIGHT_TIME_APPROX_EQUAL.fetch_add(start.elapsed().as_micros() as u64, Ordering::Relaxed);
        }
        res
    }

    fn value(&self) -> &Self::Type {
        let prof = rustfst_weight_profile_enabled();
        let start = if prof { Some(Instant::now()) } else { None };
        let res: &Self::Type = &self.0;
        if let Some(start) = start {
            RUSTFST_WEIGHT_COUNT_VALUE.fetch_add(1, Ordering::Relaxed);
            RUSTFST_WEIGHT_TIME_VALUE.fetch_add(start.elapsed().as_micros() as u64, Ordering::Relaxed);
        }
        res
    }

    fn take_value(self) -> Self::Type {
        let prof = rustfst_weight_profile_enabled();
        let start = if prof { Some(Instant::now()) } else { None };
        let res = Arc::try_unwrap(self.0).unwrap_or_else(|arc| (*arc).clone());
        if let Some(start) = start {
            RUSTFST_WEIGHT_COUNT_TAKE_VALUE.fetch_add(1, Ordering::Relaxed);
            RUSTFST_WEIGHT_TIME_TAKE_VALUE.fetch_add(start.elapsed().as_micros() as u64, Ordering::Relaxed);
        }
        res
    }

    fn set_value(&mut self, value: Self::Type) {
        let prof = rustfst_weight_profile_enabled();
        let start = if prof { Some(Instant::now()) } else { None };
        self.0 = intern_weight(value);
        if let Some(start) = start {
            RUSTFST_WEIGHT_COUNT_SET_VALUE.fetch_add(1, Ordering::Relaxed);
            RUSTFST_WEIGHT_TIME_SET_VALUE.fetch_add(start.elapsed().as_micros() as u64, Ordering::Relaxed);
        }
    }

    fn reverse(&self) -> Result<Self::ReverseWeight> {
        let prof = rustfst_weight_profile_enabled();
        let start = if prof { Some(Instant::now()) } else { None };
        let res = Ok(self.clone());
        if let Some(start) = start {
            RUSTFST_WEIGHT_COUNT_REVERSE.fetch_add(1, Ordering::Relaxed);
            RUSTFST_WEIGHT_TIME_REVERSE.fetch_add(start.elapsed().as_micros() as u64, Ordering::Relaxed);
        }
        res
    }

    fn properties() -> SemiringProperties {
        let prof = rustfst_weight_profile_enabled();
        let start = if prof { Some(Instant::now()) } else { None };
        let props = SemiringProperties::LEFT_SEMIRING
            | SemiringProperties::RIGHT_SEMIRING
            | SemiringProperties::COMMUTATIVE
            | SemiringProperties::IDEMPOTENT
            | SemiringProperties::PATH;
        if let Some(start) = start {
            RUSTFST_WEIGHT_COUNT_PROPERTIES.fetch_add(1, Ordering::Relaxed);
            RUSTFST_WEIGHT_TIME_PROPERTIES.fetch_add(start.elapsed().as_micros() as u64, Ordering::Relaxed);
        }
        props
    }
}

impl ReverseBack<BitsetWeight> for BitsetWeight {
    fn reverse_back(&self) -> Result<BitsetWeight> {
        let prof = rustfst_weight_profile_enabled();
        let start = if prof { Some(Instant::now()) } else { None };
        let res = Ok(self.clone());
        if let Some(start) = start {
            RUSTFST_WEIGHT_COUNT_REVERSE_BACK.fetch_add(1, Ordering::Relaxed);
            RUSTFST_WEIGHT_TIME_REVERSE_BACK.fetch_add(start.elapsed().as_micros() as u64, Ordering::Relaxed);
        }
        res
    }
}

impl WeaklyDivisibleSemiring for BitsetWeight {
    #[time_it("BitsetWeight::divide_assign")]
    fn divide_assign(&mut self, rhs: &Self, _divide_type: DivideType) -> Result<()> {
        let prof = rustfst_weight_profile_enabled();
        let start = if prof { Some(Instant::now()) } else { None };
        let lhs = Arc::clone(&self.0);
        let rhs = Arc::clone(&rhs.0);

        let new_arc = if Arc::ptr_eq(&lhs, &rhs) {
            intern_weight(Weight::all())
        } else if rhs.is_empty() {
            intern_weight(Weight::all())
        } else if rhs.is_all_fast() {
            lhs
        } else if lhs.is_all_fast() {
            intern_weight(Weight::all())
        } else if lhs.is_empty() {
            intern_weight(rhs.complement())
        } else {
            let key = DivideKey {
                a: Arc::as_ptr(&lhs) as usize,
                b: Arc::as_ptr(&rhs) as usize,
            };
            if let Some(hit) = {
                let mut cache = WEIGHT_DIVIDE_CACHE.lock().unwrap();
                cache.get(&key).cloned()
            } {
                hit
            } else {
                let new_weight = lhs.divide(&rhs);
                let arc = intern_weight(new_weight);
                let mut cache = WEIGHT_DIVIDE_CACHE.lock().unwrap();
                cache.put(key, arc.clone());
                arc
            }
        };

        self.0 = new_arc;
        if let Some(start) = start {
            RUSTFST_WEIGHT_COUNT_DIVIDE.fetch_add(1, Ordering::Relaxed);
            RUSTFST_WEIGHT_TIME_DIVIDE.fetch_add(start.elapsed().as_micros() as u64, Ordering::Relaxed);
        }
        Ok(())
    }
}

impl WeightQuantize for BitsetWeight {
    fn quantize_assign(&mut self, _delta: f32) -> Result<()> {
        let prof = rustfst_weight_profile_enabled();
        let start = if prof { Some(Instant::now()) } else { None };
        let res = Ok(());
        if let Some(start) = start {
            RUSTFST_WEIGHT_COUNT_QUANTIZE.fetch_add(1, Ordering::Relaxed);
            RUSTFST_WEIGHT_TIME_QUANTIZE.fetch_add(start.elapsed().as_micros() as u64, Ordering::Relaxed);
        }
        res
    }
}

impl SerializableSemiring for BitsetWeight {
    fn weight_type() -> String {
        let prof = rustfst_weight_profile_enabled();
        let start = if prof { Some(Instant::now()) } else { None };
        let res = "bitset".to_string();
        if let Some(start) = start {
            RUSTFST_WEIGHT_COUNT_WEIGHT_TYPE.fetch_add(1, Ordering::Relaxed);
            RUSTFST_WEIGHT_TIME_WEIGHT_TYPE.fetch_add(start.elapsed().as_micros() as u64, Ordering::Relaxed);
        }
        res
    }

    fn parse_binary(i: &[u8]) -> IResult<&[u8], Self, NomCustomError<&[u8]>> {
        let prof = rustfst_weight_profile_enabled();
        let start = if prof { Some(Instant::now()) } else { None };
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
        let res = Ok((i, BitsetWeight(intern_weight(Weight::from_rsb(rsb)))));
        if let Some(start) = start {
            RUSTFST_WEIGHT_COUNT_PARSE_BINARY.fetch_add(1, Ordering::Relaxed);
            RUSTFST_WEIGHT_TIME_PARSE_BINARY.fetch_add(start.elapsed().as_micros() as u64, Ordering::Relaxed);
        }
        res
    }

    fn write_binary<F: Write>(&self, file: &mut F) -> Result<()> {
        let prof = rustfst_weight_profile_enabled();
        let start = if prof { Some(Instant::now()) } else { None };
        let ranges: Vec<_> = self.0.to_rsb().ranges().collect();
        file.write_all(&(ranges.len() as u64).to_le_bytes())?;
        for range in ranges {
            file.write_all(&(*range.start() as u64).to_le_bytes())?;
            file.write_all(&(*range.end() as u64).to_le_bytes())?;
        }
        let res = Ok(());
        if let Some(start) = start {
            RUSTFST_WEIGHT_COUNT_WRITE_BINARY.fetch_add(1, Ordering::Relaxed);
            RUSTFST_WEIGHT_TIME_WRITE_BINARY.fetch_add(start.elapsed().as_micros() as u64, Ordering::Relaxed);
        }
        res
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
    let total_start = std::time::Instant::now();
    let mut fst = VectorFst::<BitsetWeight>::new();
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
                    fst.add_tr(super_start, Tr::new(EPS_LABEL, EPS_LABEL, BitsetWeight::one(), target)).unwrap();
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
                fst.set_final(fst_state_id, BitsetWeight::new(w_clone)).unwrap();
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
                            BitsetWeight::new(w_clone),
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
                fst.add_tr(fst_state_id, Tr::new(EPS_LABEL, EPS_LABEL, BitsetWeight::new(w_clone), state_map[target]))
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
    let total_start = std::time::Instant::now();
    if fst.num_states() == 0 {
        return NWA::new_empty();
    }

    let mut nwa = NWA::new();
    nwa.states.0.clear();
    let mut state_map = HashMap::<StateId, NWAStateID>::new();

    let add_state_start = std::time::Instant::now();
    for i in 0..fst.num_states() {
        let s = nwa.states.add_state();
        state_map.insert(i as StateId, s);
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
        nwa.body.start_states = vec![state_map[&fst_start]];
        start_time += start_set.elapsed();
    } else {
        let start_set = std::time::Instant::now();
        nwa.body.start_states.clear();
        start_time += start_set.elapsed();
    }

    for i in 0..fst.num_states() {
        let fst_state_id = i as StateId;
        let nwa_state_id = state_map[&fst_state_id];

        if let Some(w) = fst.final_weight(fst_state_id).unwrap() {
            if !w.0.is_empty() {
                let clone_start = std::time::Instant::now();
                let w_clone = w.value().clone();
                final_clone_time += clone_start.elapsed();
                let set_start = std::time::Instant::now();
                final_count += 1;
                nwa.states[nwa_state_id].final_weight = Some(w_clone);
                final_set_time += set_start.elapsed();
            }
        }

        for tr in fst.get_trs(fst_state_id).unwrap().trs() {
            if !tr.weight.0.is_empty() {
                let target_nwa_id = state_map[&tr.nextstate];
                let clone_start = std::time::Instant::now();
                let weight = tr.weight.value().clone();
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
    pub fn determinize_to_dwa_with_rustfst(&self) -> DWA {
        determinize_nwa_to_dwa(self)
    }

    pub fn to_rustfst(&self) -> VectorFst<BitsetWeight> {
        nwa_to_vector_fst(self)
    }

    pub fn from_rustfst(fst: &VectorFst<BitsetWeight>) -> NWA {
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
