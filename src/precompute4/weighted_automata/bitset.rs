// src/precompute4/weighted_automata/bitset.rs
//
// Deterministic, canonical SimpleBitset implementation used as the Weight in our semiring.
// We model sets of nonnegative integers with support for the "top" (ALL) element, i.e. a
// cofinite set. This gives:
//   - Finite sets: mode = Finite, 'set' holds included elements.
//   - Cofinite sets ("ALL minus a finite set"): mode = Cofinite, 'set' holds excluded elements.
// Operations (&, |, -) follow standard set logic, with distributivity and idempotence,
// matching the intended semiring (intersection = ∧, union = ∨).
//
// Key properties:
//   - canonical_string() provides a stable, deterministic textual representation.
//     Equal weights always serialize to the same string.
//   - Display is defined in terms of canonical_string().
//
// Semiring identity elements:
//   - zeros() is the empty set (absorbing for ∧).
//   - all() is the universal set ("ALL"), neutral for ∧ and absorbing for ∨.
//
// Notes:
//   - The universe is the set of all nonnegative integers. Cofinite mode lets us represent "ALL".
//   - This is sufficient for all operations used by the rest of the code: &, |, -, is_empty, clone, etc.

#![allow(dead_code)]
#![allow(clippy::needless_borrow)]

use std::collections::BTreeSet;
use std::fmt::{Display, Formatter};
use std::ops::{BitAnd, BitAndAssign, BitOr, Sub, SubAssign};

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
enum Mode {
    Finite,   // 'set' = included elements
    Cofinite, // 'set' = excluded elements from ALL
}

#[derive(Clone, Debug, Eq, PartialEq, Hash, Default)]
pub struct SimpleBitset {
    mode: Mode,
    set: BTreeSet<usize>,
}

impl SimpleBitset {
    // Constructors

    // Empty set (absorbing for ∧)
    pub fn zeros() -> Self {
        SimpleBitset { mode: Mode::Finite, set: BTreeSet::new() }
    }

    // Universal set ("ALL", neutral for ∧, absorbing for ∨)
    pub fn all() -> Self {
        SimpleBitset { mode: Mode::Cofinite, set: BTreeSet::new() }
    }

    // Singleton {idx}
    pub fn from_item(idx: usize) -> Self {
        let mut s = BTreeSet::new();
        s.insert(idx);
        SimpleBitset { mode: Mode::Finite, set: s }
    }

    // Predicates / utilities

    // Strict emptiness (only the finite empty set is empty)
    pub fn is_empty(&self) -> bool {
        matches!(self.mode, Mode::Finite) && self.set.is_empty()
    }

    // Canonical, deterministic serialization.
    // - Finite:  "F[1,2,3]"
    // - Cofinite with no exclusions (ALL): "ALL"
    // - Cofinite with exclusions {1,3}:    "ALL\\{1,3}"
    pub fn canonical_string(&self) -> String {
        match self.mode {
            Mode::Finite => {
                if self.set.is_empty() {
                    "F[]".to_string()
                } else {
                    let elems: Vec<String> = self.set.iter().map(|x| x.to_string()).collect();
                    format!("F[{}]", elems.join(","))
                }
            }
            Mode::Cofinite => {
                if self.set.is_empty() {
                    "ALL".to_string()
                } else {
                    let excl: Vec<String> = self.set.iter().map(|x| x.to_string()).collect();
                    format!("ALL\\{{{}}}", excl.join(","))
                }
            }
        }
    }

    // Set complement (relative to ALL)
    fn complement(&self) -> Self {
        match self.mode {
            Mode::Finite => SimpleBitset { mode: Mode::Cofinite, set: self.set.clone() },
            Mode::Cofinite => SimpleBitset { mode: Mode::Finite, set: self.set.clone() },
        }
    }

    // Intersection core (by reference)
    fn and_ref(a: &Self, b: &Self) -> Self {
        match (&a.mode, &b.mode) {
            (Mode::Finite, Mode::Finite) => {
                let inter = a.set.intersection(&b.set).cloned().collect::<BTreeSet<_>>();
                SimpleBitset { mode: Mode::Finite, set: inter }
            }
            (Mode::Finite, Mode::Cofinite) => {
                // A ∩ (ALL \ B) = A \ B
                let mut res = a.set.clone();
                for x in b.set.iter() {
                    res.remove(x);
                }
                SimpleBitset { mode: Mode::Finite, set: res }
            }
            (Mode::Cofinite, Mode::Finite) => {
                // (ALL \ A) ∩ B = B \ A
                let mut res = b.set.clone();
                for x in a.set.iter() {
                    res.remove(x);
                }
                SimpleBitset { mode: Mode::Finite, set: res }
            }
            (Mode::Cofinite, Mode::Cofinite) => {
                // (ALL \ A) ∩ (ALL \ B) = ALL \ (A ∪ B)
                let mut excl = a.set.clone();
                excl.extend(b.set.iter().cloned());
                SimpleBitset { mode: Mode::Cofinite, set: excl }
            }
        }
    }

    // Union core (by reference)
    fn or_ref(a: &Self, b: &Self) -> Self {
        match (&a.mode, &b.mode) {
            (Mode::Finite, Mode::Finite) => {
                let mut uni = a.set.clone();
                uni.extend(b.set.iter().cloned());
                SimpleBitset { mode: Mode::Finite, set: uni }
            }
            (Mode::Finite, Mode::Cofinite) => {
                // A ∪ (ALL \ B) = ALL \ (B \ A)
                let mut excl = b.set.clone();
                // remove elements that are in A (since they are included by A already)
                for x in a.set.iter() {
                    excl.remove(x);
                }
                SimpleBitset { mode: Mode::Cofinite, set: excl }
            }
            (Mode::Cofinite, Mode::Finite) => {
                // (ALL \ A) ∪ B = ALL \ (A \ B)
                let mut excl = a.set.clone();
                // remove elements that are in B (since union includes them)
                for x in b.set.iter() {
                    excl.remove(x);
                }
                SimpleBitset { mode: Mode::Cofinite, set: excl }
            }
            (Mode::Cofinite, Mode::Cofinite) => {
                // (ALL \ A) ∪ (ALL \ B) = ALL \ (A ∩ B)
                let inter = a.set.intersection(&b.set).cloned().collect::<BTreeSet<_>>();
                SimpleBitset { mode: Mode::Cofinite, set: inter }
            }
        }
    }

    // Difference core (by reference): A \ B = A ∩ complement(B)
    fn diff_ref(a: &Self, b: &Self) -> Self {
        let cb = b.complement();
        Self::and_ref(a, &cb)
    }
}

// Display uses canonical_string (stable)
impl Display for SimpleBitset {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.canonical_string())
    }
}

// Operators

impl BitAnd for SimpleBitset {
    type Output = SimpleBitset;
    fn bitand(self, rhs: Self) -> Self::Output {
        SimpleBitset::and_ref(&self, &rhs)
    }
}

impl<'a> BitAnd<&'a SimpleBitset> for SimpleBitset {
    type Output = SimpleBitset;
    fn bitand(self, rhs: &'a SimpleBitset) -> Self::Output {
        SimpleBitset::and_ref(&self, rhs)
    }
}

impl<'a> BitAnd<SimpleBitset> for &'a SimpleBitset {
    type Output = SimpleBitset;
    fn bitand(self, rhs: SimpleBitset) -> Self::Output {
        SimpleBitset::and_ref(self, &rhs)
    }
}

impl<'a, 'b> BitAnd<&'b SimpleBitset> for &'a SimpleBitset {
    type Output = SimpleBitset;
    fn bitand(self, rhs: &'b SimpleBitset) -> Self::Output {
        SimpleBitset::and_ref(self, rhs)
    }
}

impl BitAndAssign for SimpleBitset {
    fn bitand_assign(&mut self, rhs: Self) {
        let res = SimpleBitset::and_ref(self, &rhs);
        *self = res;
    }
}

impl<'a> BitAndAssign<&'a SimpleBitset> for SimpleBitset {
    fn bitand_assign(&mut self, rhs: &'a SimpleBitset) {
        let res = SimpleBitset::and_ref(self, rhs);
        *self = res;
    }
}

impl BitOr for SimpleBitset {
    type Output = SimpleBitset;
    fn bitor(self, rhs: Self) -> Self::Output {
        SimpleBitset::or_ref(&self, &rhs)
    }
}

impl<'a> BitOr<&'a SimpleBitset> for SimpleBitset {
    type Output = SimpleBitset;
    fn bitor(self, rhs: &'a SimpleBitset) -> Self::Output {
        SimpleBitset::or_ref(&self, rhs)
    }
}

impl<'a> BitOr<SimpleBitset> for &'a SimpleBitset {
    type Output = SimpleBitset;
    fn bitor(self, rhs: SimpleBitset) -> Self::Output {
        SimpleBitset::or_ref(self, &rhs)
    }
}

impl<'a, 'b> BitOr<&'b SimpleBitset> for &'a SimpleBitset {
    type Output = SimpleBitset;
    fn bitor(self, rhs: &'b SimpleBitset) -> Self::Output {
        SimpleBitset::or_ref(self, rhs)
    }
}

impl Sub for SimpleBitset {
    type Output = SimpleBitset;
    fn sub(self, rhs: Self) -> Self::Output {
        SimpleBitset::diff_ref(&self, &rhs)
    }
}

impl<'a> Sub<&'a SimpleBitset> for SimpleBitset {
    type Output = SimpleBitset;
    fn sub(self, rhs: &'a SimpleBitset) -> Self::Output {
        SimpleBitset::diff_ref(&self, rhs)
    }
}

impl<'a> SubAssign<&'a SimpleBitset> for SimpleBitset {
    fn sub_assign(&mut self, rhs: &'a SimpleBitset) {
        let res = SimpleBitset::diff_ref(self, rhs);
        *self = res;
    }
}
