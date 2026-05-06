use serde::{Deserialize, Serialize};

use crate::grammar::flat::NonterminalID;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StackShift {
    pub pop: u32,
    pub pushes: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct StackShiftGuard {
    pub pop: u32,
    pub states: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct GuardedStackShift {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub guards: Vec<StackShiftGuard>,
    pub pop: u32,
    pub pushes: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Action {
    Shift(u32, bool),
    StackShifts(Vec<StackShift>),
    GuardedStackShifts(Vec<GuardedStackShift>),
    Reduce(NonterminalID, u32),
    Split {
        shift: Option<(u32, bool)>,
        reduces: Vec<(NonterminalID, u32)>,
        accept: bool,
    },
    Accept,
}

impl Action {
    #[inline]
    pub fn shift_target(&self) -> Option<u32> {
        match self {
            Action::Shift(t, _) => Some(*t),
            Action::Split { shift: Some((t, _)), .. } => Some(*t),
            Action::StackShifts(shifts)
                if shifts.len() == 1 && shifts[0].pushes.len() == 1 && shifts[0].pop <= 1 =>
            {
                Some(shifts[0].pushes[0])
            }
            Action::GuardedStackShifts(_) => None,
            _ => None,
        }
    }

    #[inline]
    pub fn shift_is_replace(&self) -> bool {
        match self {
            Action::Shift(_, r) => *r,
            Action::Split { shift: Some((_, r)), .. } => *r,
            Action::StackShifts(shifts) if shifts.len() == 1 => {
                shifts[0].pop == 1 && shifts[0].pushes.len() == 1
            }
            Action::GuardedStackShifts(_) => false,
            _ => false,
        }
    }

    #[inline]
    pub fn for_each_stack_shift(&self, mut f: impl FnMut(u32, &[u32])) {
        match self {
            Action::Shift(target, false) => f(0, std::slice::from_ref(target)),
            Action::Shift(target, true) => f(1, std::slice::from_ref(target)),
            Action::StackShifts(shifts) => {
                for shift in shifts {
                    f(shift.pop, &shift.pushes);
                }
            }
            Action::GuardedStackShifts(_) => {}
            Action::Split { shift: Some((target, false)), .. } => {
                f(0, std::slice::from_ref(target));
            }
            Action::Split { shift: Some((target, true)), .. } => {
                f(1, std::slice::from_ref(target));
            }
            _ => {}
        }
    }

    #[inline]
    pub fn for_each_reduce(&self, mut f: impl FnMut(NonterminalID, u32)) {
        match self {
            Action::Reduce(nt, len) => f(*nt, *len),
            Action::Split { reduces, .. } => {
                for &(nt, len) in reduces {
                    f(nt, len);
                }
            }
            _ => {}
        }
    }

    #[inline]
    pub fn reduce_count(&self) -> usize {
        match self {
            Action::Reduce(..) => 1,
            Action::Split { reduces, .. } => reduces.len(),
            _ => 0,
        }
    }
}