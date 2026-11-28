//! Exact finite-byte observations over a borrowed immutable tokenizer.
//! Logical IDs are the existing induced-view IDs; NFA configuration IDs and
//! ordered matches remain in original source coordinates until publication.
use super::*;
use std::cell::Cell;
use crate::ds::bitset::BitSet;

pub(super) struct BorrowedScalarCache<'a> {
    map:BorrowedObservation<'a>,
    byte_column:[u16;256],
    stride:usize,
    transitions:Vec<u32>,
    epsilon:Vec<bool>,
    finalizers:Vec<&'a BitSet>,
    pub(super) failed:Cell<bool>,
}
impl<'a> BorrowedScalarCache<'a> {
    pub(super) fn new(source:&'a Tokenizer,map:BorrowedObservation<'a>,bytes:U8Set,flat:Option<&ByteRows>)->Option<Self>{
        let bytes=bytes.iter().collect::<Vec<_>>();
        let cells=map.len().checked_mul(bytes.len())?;
        if bytes.is_empty()||cells>1_000_000{return None;}
        let flat=flat.filter(|rows|rows.num_states()==source.num_states()as usize)?;
        let mut byte_column=[u16::MAX;256];for (column,&byte)in bytes.iter().enumerate(){byte_column[byte as usize]=column as u16;}
        let mut transitions=Vec::with_capacity(cells);
        let mut epsilon=Vec::with_capacity(map.len());let mut finalizers=Vec::with_capacity(map.len());
        for &raw in map.logical_to_original {
            epsilon.push(source.state_has_epsilon_transitions(raw));
            finalizers.push(source.matched_terminal_bitset(raw));
            for &byte in &bytes {
                let target=flat.trans(raw as usize, byte as usize);
                transitions.push(if map.logical(target).is_some(){target}else{u32::MAX});
            }
        }
        Some(Self{map,byte_column,stride:bytes.len(),transitions,epsilon,finalizers,failed:Cell::new(false)})
    }
    fn logical(&self,raw:u32)->Option<usize>{
        let logical=self.map.logical(raw).map(|q|q as usize);
        if logical.is_none(){self.failed.set(true);}logical
    }
    fn has_epsilon(&self,raw:u32)->bool {self.logical(raw).is_some_and(|q|self.epsilon[q])}
    fn step(&self,raw:u32,byte:u8)->Option<u32>{
        let column=self.byte_column[byte as usize];
        if column==u16::MAX{self.failed.set(true);return None;}
        let target=self.transitions[self.logical(raw)?*self.stride+column as usize];
        (target!=u32::MAX).then_some(target)
    }
    fn finalizers(&self,raw:u32)->Option<&BitSet>{self.logical(raw).map(|q|self.finalizers[q])}
    fn check_config(&self,states:&[u32])->bool {
        if states.iter().any(|&raw|self.map.logical(raw).is_none()){self.failed.set(true);false}else{true}
    }
}

impl CursorSource for BorrowedScalarCache<'_> {
    fn has_epsilon(&self, state: u32) -> bool { self.has_epsilon(state) }
    fn step(&self, state: u32, byte: u8) -> Option<u32> { self.step(state, byte) }
    fn check_config(&self, states: &[u32]) -> bool { self.check_config(states) }
    fn visit_finalizers(&self, state: u32, mut visit: impl FnMut(TerminalID)) -> bool {
        let Some(finalizers) = self.finalizers(state) else { return false; };
        for terminal in finalizers.iter() {
            visit(terminal as TerminalID);
        }
        true
    }
}

impl NfaTrieScanCache<'_> {
    pub(super) fn execute_native_borrowed_into(
        &mut self, action: &mut LexicalStateAction<'_>, input: &[u8], start: u32, matches: &mut Vec<TokenizerMatch>,
        prepared: &BorrowedScalarCache<'_>, validate: bool,
    ) -> SmallVec<[u32; 1]> {
        let ends = self.execute_cursor::<true, _>(action, prepared, input, start, matches);
        if !prepared.failed.get() && (validate || self.strict_reference) {
            self.verify_scalar_reference(action, input, start, matches, &ends);
        }
        ends
    }
}
