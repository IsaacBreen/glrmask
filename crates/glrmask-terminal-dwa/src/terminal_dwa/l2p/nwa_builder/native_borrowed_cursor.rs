//! Induced coordinate restriction of the shared immutable byte action.
use super::*;
use std::cell::Cell;

pub(super) struct BorrowedScalarCache<'a> {
    source: &'a Tokenizer,
    map: BorrowedObservation<'a>,
    bytes: U8Set,
    flat: &'a ByteRows,
    pub(super) failed: Cell<bool>,
}

impl<'a> BorrowedScalarCache<'a> {
    pub(super) fn new(
        source: &'a Tokenizer, map: BorrowedObservation<'a>, bytes: U8Set,
        flat: Option<&'a ByteRows>,
    ) -> Option<Self> {
        let width = bytes.iter().count();
        if width == 0 || map.len().checked_mul(width)? > 1_000_000 { return None; }
        let flat = flat.filter(|rows| rows.num_states() == source.num_states() as usize)?;
        Some(Self { source, map, bytes, flat, failed: Cell::new(false) })
    }

    fn contains(&self, raw: u32) -> bool {
        let present = self.map.logical(raw).is_some();
        if !present { self.failed.set(true); }
        present
    }
}

impl CursorSource for BorrowedScalarCache<'_> {
    fn has_epsilon(&self, state: u32) -> bool {
        self.contains(state) && self.source.state_has_epsilon_transitions(state)
    }

    fn step(&self, state: u32, byte: u8) -> Option<u32> {
        if !self.bytes.contains(byte) {
            self.failed.set(true);
            return None;
        }
        if !self.contains(state) { return None; }
        let target = self.flat.trans(state as usize, byte as usize);
        self.map.logical(target).map(|_| target)
    }

    fn check_config(&self, states: &[u32]) -> bool {
        states.iter().all(|&state| self.contains(state))
    }

    fn visit_finalizers(&self, state: u32, visit: impl FnMut(TerminalID)) -> bool {
        if !self.contains(state) { return false; }
        self.source.matched_terminals_iter(state).for_each(visit);
        true
    }
}

impl NfaTrieScanCache<'_> {
    pub(super) fn execute_native_borrowed_into(
        &mut self, action: &mut LexicalStateAction<'_>, input: &[u8], start: u32,
        matches: &mut Vec<TokenizerMatch>, prepared: &BorrowedScalarCache<'_>, validate: bool,
    ) -> SmallVec<[u32; 1]> {
        let ends = self.execute_cursor::<true, _>(action, prepared, input, start, matches);
        if !prepared.failed.get() && (validate || self.strict_reference) {
            self.verify_scalar_reference(action, input, start, matches, &ends);
        }
        ends
    }
}
