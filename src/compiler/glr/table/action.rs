use serde::{Deserialize, Serialize};
use std::sync::Arc;

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
    #[serde(default)]
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
    /// Alternative pop-one/push-one replacements, stored compactly as targets.
    ReplaceShifts(Arc<[u32]>),
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
            Action::ReplaceShifts(targets) if targets.len() == 1 => Some(targets[0]),
            Action::GuardedStackShifts(_) | Action::ReplaceShifts(_) => None,
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
            Action::ReplaceShifts(targets) => targets.len() == 1,
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
            Action::ReplaceShifts(targets) => {
                for target in targets.iter() {
                    f(1, std::slice::from_ref(target));
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

#[cfg(test)]
mod tests {
    use super::{Action, GuardedStackShift, StackShift, StackShiftGuard};
    use std::sync::Arc;

    #[test]
    fn guarded_stack_shifts_bincode_roundtrip_preserves_empty_guards() {
        let action = Action::GuardedStackShifts(vec![
            GuardedStackShift {
                guards: Vec::new(),
                pop: 0,
                pushes: vec![1],
            },
            GuardedStackShift {
                guards: vec![StackShiftGuard {
                    pop: 1,
                    states: vec![2],
                }],
                pop: 1,
                pushes: vec![3],
            },
        ]);

        let bytes = bincode::serialize(&action).expect("serialization should succeed");
        let decoded: Action = bincode::deserialize(&bytes).expect("deserialization should succeed");

        assert_eq!(decoded, action);
    }

    #[test]
    fn action_bincode_discriminants_remain_stable() {
        let actions = [
            Action::Shift(1, true),
            Action::StackShifts(vec![StackShift {
                pop: 1,
                pushes: vec![2],
            }]),
            Action::GuardedStackShifts(Vec::new()),
            Action::Reduce(3, 1),
            Action::Split {
                shift: None,
                reduces: Vec::new(),
                accept: true,
            },
            Action::Accept,
            Action::ReplaceShifts(Arc::from([1, 3, 5, 8])),
        ];

        for (discriminant, action) in actions.into_iter().enumerate() {
            let bytes = bincode::serialize(&action).expect("serialization should succeed");
            assert_eq!(&bytes[..4], &(discriminant as u32).to_le_bytes());
            let decoded: Action =
                bincode::deserialize(&bytes).expect("deserialization should succeed");
            assert_eq!(decoded, action);
        }

        let targets = vec![1u32, 3, 5, 8];
        let mut legacy_wire = 6u32.to_le_bytes().to_vec();
        legacy_wire.extend(
            bincode::serialize(&targets).expect("legacy Vec target payload should serialize"),
        );
        let shared = Action::ReplaceShifts(Arc::from(targets.into_boxed_slice()));
        assert_eq!(
            bincode::serialize(&shared).expect("shared target payload should serialize"),
            legacy_wire,
            "Arc<[u32]> must retain the historical Vec<u32> bincode wire shape",
        );
        assert_eq!(
            bincode::deserialize::<Action>(&legacy_wire)
                .expect("historical ReplaceShifts payload should still deserialize"),
            shared,
        );
    }
}
