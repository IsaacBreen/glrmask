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
    pub(super) fn new(source:&'a Tokenizer,map:BorrowedObservation<'a>,bytes:U8Set,flat:Option<&[u32]>)->Option<Self>{
        let bytes=bytes.iter().collect::<Vec<_>>();
        let cells=map.len().checked_mul(bytes.len())?;
        if bytes.is_empty()||cells>1_000_000{return None;}
        let flat=flat.filter(|rows|rows.len()==source.num_states()as usize*256)?;
        let mut byte_column=[u16::MAX;256];for (column,&byte)in bytes.iter().enumerate(){byte_column[byte as usize]=column as u16;}
        let mut transitions=Vec::with_capacity(cells);
        let mut epsilon=Vec::with_capacity(map.len());let mut finalizers=Vec::with_capacity(map.len());
        for &raw in map.logical_to_original {
            epsilon.push(source.state_has_epsilon_transitions(raw));
            finalizers.push(source.matched_terminal_bitset(raw));
            for &byte in &bytes {
                let target=flat[raw as usize*256+byte as usize];
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

impl NfaTrieScanCache<'_> {
    pub(super) fn execute_native_borrowed_into(
        &mut self,input:&[u8],start:u32,matches:&mut Vec<TokenizerMatch>,
        prepared:&BorrowedScalarCache<'_>,validate:bool,
    )->SmallVec<[u32;1]>{
        self.match_stamp=self.match_stamp.wrapping_add(1);
        if self.match_stamp==0 {self.match_seen_stamp.fill(0);self.match_stamp=1;}
        let stamp=self.match_stamp;self.touched_terminals.clear();matches.clear();
        let mut scalar=(!prepared.has_epsilon(start)).then_some(start);
        let mut config=if scalar.is_none(){self.config_for_raw_start(start)}else{NFA_CONFIG_UNKNOWN};
        if scalar.is_none()&&!prepared.check_config(&self.configs[config as usize]){return SmallVec::new();}
        let mut alive=true;
        for(index,&byte)in input.iter().enumerate(){
            if let Some(state)=scalar {
                let Some(target)=prepared.step(state,byte)else{alive=false;break;};
                if prepared.has_epsilon(target){scalar=None;config=self.config_for_raw_start(target);}
                else{scalar=Some(target);}
            }else{
                let Some(next)=self.step_config(config,byte)else{alive=false;break;};config=next;
                let closure=&self.configs[config as usize];
                if !prepared.check_config(closure){alive=false;break;}
                if closure.len()==1&&!prepared.has_epsilon(closure[0]){scalar=Some(closure[0]);}
            }
            let width=index+1;
            if let Some(state)=scalar {
                let Some(finalizers)=prepared.finalizers(state)else{alive=false;break;};
                for t in finalizers.iter(){let terminal=t as TerminalID;
                    if self.active_terminals.as_ref().is_none_or(|active|active.get(t).copied().unwrap_or(false)){
                        self.record_native_match(terminal,state,width,stamp);
                    }
                }
            }else{
                if !prepared.check_config(&self.configs[config as usize]){alive=false;break;}
                self.ensure_observations(config);
                let n=self.observations[config as usize].as_ref().unwrap().len();
                for i in 0..n{let(terminal,state)=self.observations[config as usize].as_ref().unwrap()[i];self.record_native_match(terminal,state,width,stamp);}
            }
        }
        for &terminal in &self.touched_terminals{
            let t=terminal as usize;let width=self.match_width[t];
            matches.extend(self.match_end_states[t].iter().copied().map(|end_state|TokenizerMatch{id:terminal,width,end_state}));
        }
        let ends=if !alive{SmallVec::new()}else if let Some(raw)=scalar{SmallVec::from_buf([raw])}else{SmallVec::from_slice(&self.configs[config as usize])};
        if !prepared.failed.get()&&(validate||self.strict_reference){
            let mut reference=Vec::new();let expected=self.execute_into(input,start,&mut reference);
            assert_eq!(ends,expected,"borrowed compact scalar cursor changed end states");
            assert_eq!(*matches,reference,"borrowed compact scalar cursor changed longest matches");
        }
        ends
    }
}
