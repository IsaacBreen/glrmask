//! Exact runtime representation for a zero-minimum bounded repetition whose
//! body consumes exactly one byte.
//!
//! The physical tokenizer contains only its reset state. Reached positive
//! repetition counts are encoded arithmetically in a disjoint logical-state
//! interval, so storage remains constant regardless of `max` or input length.

use crate::ds::bitset::BitSet;
use crate::ds::u8set::U8Set;
use crate::grammar::flat::TerminalID;

/// Dynamic NFA configurations reserve the high bit of a raw tokenizer state.
/// Virtual tokenizer ids must therefore remain below it.
const VIRTUAL_STATE_LIMIT: u32 = 1 << 31;

/// Whether an arithmetic unit-repeat runtime with `max` positive residuals
/// fits beside `physical_state_count` materialized tokenizer states. Dynamic
/// compilation uses this before mutating the tokenizer so an oversized unit
/// lane can be routed to the exact repeat-product runtime instead.
pub(super) fn virtual_unit_repeat_state_ids_fit(
    max: usize,
    physical_state_count: u32,
) -> bool {
    let Ok(virtual_count) = u32::try_from(max) else {
        return false;
    };
    physical_state_count > 0
        && physical_state_count < VIRTUAL_STATE_LIMIT
        && physical_state_count
            .checked_add(virtual_count)
            .is_some_and(|end| end <= VIRTUAL_STATE_LIMIT)
}

#[derive(Debug)]
pub(super) struct VirtualZeroMinUnitRepeatRuntime {
    body: U8Set,
    max: usize,
    physical_state_count: u32,
    root_state: u32,
    terminal: TerminalID,
    accepting: BitSet,
    live: BitSet,
    dead: BitSet,
    accepting_list: Box<[TerminalID]>,
}

/// Exact finite-token projection for the arithmetic repeat coordinate.
///
/// The full runtime state id is the consumed-byte count (`0` is the drained
/// reset state).  The mask tokenizer keeps that reset state, one positive
/// interior state, and an exact upper-bound tail.  This object is deliberately
/// independent of any runtime interner: projection is closed-form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[doc(hidden)]
pub struct VirtualZeroMinUnitRepeatMaskProjection {
    full_max: u32,
    horizon: u32,
    full_physical_state_count: u32,
    mask_positive_state_count: u32,
}

impl VirtualZeroMinUnitRepeatMaskProjection {
    pub(super) fn new(
        full_max: usize,
        horizon: usize,
        full_physical_state_count: u32,
    ) -> Option<Self> {
        let full_max = u32::try_from(full_max).ok()?;
        let horizon = u32::try_from(horizon).ok()?;
        // If the exact bound itself is already within the finite vocabulary
        // horizon, retain it verbatim. Otherwise reserve state 0 for the
        // drained reset, state 1 for the translation-invariant positive
        // interior, and K+1 exact distances to the upper bound.
        let mask_positive_state_count = if full_max <= horizon.saturating_add(1) {
            full_max
        } else {
            horizon.checked_add(2)?
        };
        let mask_state_count = full_physical_state_count
            .checked_add(mask_positive_state_count)?;
        (mask_state_count <= VIRTUAL_STATE_LIMIT).then_some(Self {
            full_max,
            horizon,
            full_physical_state_count,
            mask_positive_state_count,
        })
    }

    #[inline]
    pub fn project(self, full_state: u32) -> Option<u32> {
        if full_state < self.full_physical_state_count {
            return Some(full_state);
        }
        let consumed = full_state
            .checked_sub(self.full_physical_state_count)?
            .checked_add(1)?;
        if consumed > self.full_max {
            return None;
        }
        if self.full_max == self.mask_positive_state_count {
            return self
                .full_physical_state_count
                .checked_add(consumed.saturating_sub(1));
        }
        let remaining = self.full_max - consumed;
        Some(if remaining > self.horizon {
            self.full_physical_state_count
        } else {
            self.full_physical_state_count
                .checked_add(self.mask_positive_state_count - 1 - remaining)?
        })
    }

    #[inline]
    pub fn mask_state_count(self) -> u32 {
        self.full_physical_state_count + self.mask_positive_state_count
    }

    pub fn multiplicities(self) -> Vec<usize> {
        let mut counts = vec![1usize; self.full_physical_state_count as usize];
        counts.resize(self.mask_state_count() as usize, 0);
        if self.full_max == self.mask_positive_state_count {
            counts[self.full_physical_state_count as usize..].fill(1);
            return counts;
        }
        // Positive full states with more than K bytes remaining form exactly
        // the one interior class. The exact K..0 tail follows it.
        counts[self.full_physical_state_count as usize] = self
            .full_max
            .saturating_sub(self.horizon)
            .saturating_sub(1) as usize;
        for count in &mut counts[self.full_physical_state_count as usize + 1..] {
            *count = 1;
        }
        counts
    }

    pub fn unique_full_states(self) -> Vec<u32> {
        let mut unique = vec![u32::MAX; self.mask_state_count() as usize];
        for (state, slot) in unique[..self.full_physical_state_count as usize]
            .iter_mut()
            .enumerate()
        {
            *slot = state as u32;
        }
        if self.full_max == self.mask_positive_state_count {
            for (state, slot) in unique[self.full_physical_state_count as usize..]
                .iter_mut()
                .enumerate()
            {
                *slot = self.full_physical_state_count + state as u32;
            }
            return unique;
        }
        let interior_count = self
            .full_max
            .saturating_sub(self.horizon)
            .saturating_sub(1);
        if interior_count == 1 {
            unique[self.full_physical_state_count as usize] = self.full_physical_state_count;
        }
        for mask_state in self.full_physical_state_count + 1..self.mask_state_count() {
            let tail_index = mask_state - self.full_physical_state_count - 1;
            let remaining = self.horizon - tail_index;
            let consumed = self.full_max - remaining;
            unique[mask_state as usize] = self.full_physical_state_count + consumed - 1;
        }
        unique
    }
}

impl VirtualZeroMinUnitRepeatRuntime {
    pub(super) fn new(
        body: U8Set,
        max: usize,
        terminal: TerminalID,
        num_terminals: u32,
        physical_state_count: u32,
        root_state: u32,
    ) -> Option<Self> {
        if body.is_empty()
            || max == 0
            || physical_state_count == 0
            || root_state >= physical_state_count
            || !virtual_unit_repeat_state_ids_fit(max, physical_state_count)
            || terminal >= num_terminals
        {
            return None;
        }
        let mut accepting = BitSet::new(num_terminals as usize);
        accepting.set(terminal as usize);
        let live = accepting.clone();
        Some(Self {
            body,
            max,
            physical_state_count,
            root_state,
            terminal,
            accepting,
            live,
            dead: BitSet::new(num_terminals as usize),
            accepting_list: vec![terminal].into_boxed_slice(),
        })
    }

    #[inline]
    pub(super) fn body(&self) -> U8Set {
        self.body
    }

    #[inline]
    pub(super) fn max(&self) -> usize {
        self.max
    }

    #[inline]
    pub(super) fn physical_state_count(&self) -> u32 {
        self.physical_state_count
    }

    #[inline]
    pub(super) fn root_state(&self) -> u32 {
        self.root_state
    }

    #[inline]
    pub(super) fn handles_state(&self, state: u32) -> bool {
        state == self.root_state || self.is_virtual_state(state)
    }

    #[inline]
    pub(super) fn terminal(&self) -> TerminalID {
        self.terminal
    }

    #[inline]
    fn consumed(&self, state: u32) -> Option<usize> {
        if state == self.root_state {
            return Some(0);
        }
        if state < self.physical_state_count {
            return None;
        }
        let index = state.checked_sub(self.physical_state_count)? as usize;
        (index < self.max).then_some(index + 1)
    }

    fn encode_positive(&self, consumed: usize) -> Option<u32> {
        if consumed == 0 || consumed > self.max {
            return None;
        }
        let index = consumed - 1;
        let id = self
            .physical_state_count
            .checked_add(u32::try_from(index).ok()?)?;
        (id < VIRTUAL_STATE_LIMIT).then_some(id)
    }

    pub(super) fn step(&self, state: u32, byte: u8) -> Option<u32> {
        let consumed = self.consumed(state)?;
        if consumed >= self.max || !self.body.contains(byte) {
            return None;
        }
        self.encode_positive(consumed.checked_add(1)?)
    }

    pub(super) fn is_virtual_state(&self, state: u32) -> bool {
        state >= self.physical_state_count && self.consumed(state).is_some()
    }

    pub(super) fn finalizers(&self, state: u32) -> Option<&BitSet> {
        (state >= self.physical_state_count).then(|| {
            if self.is_virtual_state(state) { &self.accepting } else { &self.dead }
        })
    }

    pub(super) fn finalizer_list(&self, state: u32) -> Option<&[TerminalID]> {
        (state >= self.physical_state_count).then(|| {
            if self.is_virtual_state(state) {
                self.accepting_list.as_ref()
            } else {
                &[]
            }
        })
    }

    pub(super) fn futures(&self, state: u32) -> Option<&BitSet> {
        if state != self.root_state && state < self.physical_state_count {
            return None;
        }
        let Some(consumed) = self.consumed(state) else {
            return Some(&self.dead);
        };
        Some(if consumed < self.max { &self.live } else { &self.dead })
    }

    pub(super) fn transition_target(&self, state: u32) -> Option<u32> {
        let consumed = self.consumed(state)?;
        (consumed < self.max)
            .then(|| self.encode_positive(consumed + 1))
            .flatten()
    }
}

#[cfg(test)]
mod tests {
    use super::{virtual_unit_repeat_state_ids_fit, VIRTUAL_STATE_LIMIT};

    #[test]
    fn virtual_unit_repeat_state_id_boundary_is_exact() {
        let physical = 17u32;
        let last_fitting = (VIRTUAL_STATE_LIMIT - physical) as usize;
        assert!(virtual_unit_repeat_state_ids_fit(last_fitting, physical));
        assert!(!virtual_unit_repeat_state_ids_fit(
            last_fitting + 1,
            physical,
        ));
    }
}
