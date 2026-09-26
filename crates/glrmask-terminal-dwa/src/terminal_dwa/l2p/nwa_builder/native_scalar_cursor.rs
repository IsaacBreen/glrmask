//! Exact ordered lexical observations without constructing singleton powersets.
//! A raw state with no epsilon transitions has closure {state}. Its transition
//! may be read directly until a genuinely epsilon-bearing target is reached.
//! The general cache remains responsible for all nontrivial configurations.
use super::*;

impl NfaTrieScanCache<'_> {
    pub(super) fn record_native_match(&mut self, terminal:TerminalID, end_state:u32, width:usize, stamp:u32) {
        let t=terminal as usize;
        if self.match_seen_stamp[t]!=stamp {
            self.match_seen_stamp[t]=stamp;
            self.match_width[t]=width;
            self.match_end_states[t].clear();
            self.match_end_states[t].push(end_state);
            self.touched_terminals.push(terminal);
        }else if width>self.match_width[t] {
            self.match_width[t]=width;
            self.match_end_states[t].clear();
            self.match_end_states[t].push(end_state);
        }else if width==self.match_width[t]&&!self.match_end_states[t].contains(&end_state) {
            self.match_end_states[t].push(end_state);
        }
    }

    pub(super) fn execute_native_scalar_into(
        &mut self, input:&[u8], start:u32, matches:&mut Vec<TokenizerMatch>,
        flat:Option<&[u32]>, validate:bool,
    )->SmallVec<[u32;1]> {
        let flat=flat.filter(|table|table.len()==self.tokenizer.num_states()as usize*256);
        self.match_stamp=self.match_stamp.wrapping_add(1);
        if self.match_stamp==0 {self.match_seen_stamp.fill(0);self.match_stamp=1;}
        let stamp=self.match_stamp;
        self.touched_terminals.clear();matches.clear();
        let tokenizer=self.tokenizer;
        let mut scalar=(!tokenizer.state_has_epsilon_transitions(start)).then_some(start);
        let mut config=if scalar.is_none(){self.config_for_raw_start(start)}else{NFA_CONFIG_UNKNOWN};
        let mut alive=true;
        for(index,&byte)in input.iter().enumerate(){
            if let Some(state)=scalar {
                let target=match flat {
                    Some(table)=>{let t=table[state as usize*256+byte as usize];(t!=u32::MAX).then_some(t)},
                    None=>tokenizer.step(state,byte),
                };
                let Some(target)=target else{alive=false;break};
                if tokenizer.state_has_epsilon_transitions(target){
                    scalar=None;config=self.config_for_raw_start(target);
                }else{scalar=Some(target)}
            }else{
                let Some(next)=self.step_config(config,byte)else{alive=false;break};
                config=next;
                let closure=&self.configs[config as usize];
                if closure.len()==1&&!tokenizer.state_has_epsilon_transitions(closure[0]){
                    scalar=Some(closure[0]);
                }
            }
            let width=index+1;
            if let Some(state)=scalar {
                for terminal in tokenizer.matched_terminals_iter(state){
                    if self.active_terminals.as_ref().is_none_or(|active|active.get(terminal as usize).copied().unwrap_or(false)){
                        self.record_native_match(terminal,state,width,stamp);
                    }
                }
            }else{
                self.ensure_observations(config);
                let n=self.observations[config as usize].as_ref().unwrap().len();
                for i in 0..n{
                    let(terminal,state)=self.observations[config as usize].as_ref().unwrap()[i];
                    self.record_native_match(terminal,state,width,stamp);
                }
            }
        }
        for &terminal in &self.touched_terminals{
            let t=terminal as usize;let width=self.match_width[t];
            matches.extend(self.match_end_states[t].iter().copied().map(|end_state|TokenizerMatch{id:terminal,width,end_state}));
        }
        let ends=if !alive{SmallVec::new()}else if let Some(raw)=scalar{SmallVec::from_buf([raw])}
            else{SmallVec::from_slice(&self.configs[config as usize])};
        if validate||self.strict_reference {
            let mut reference=Vec::new();
            let expected=self.execute_into(input,start,&mut reference);
            assert_eq!(ends,expected,"hybrid scalar cursor changed ordered end states");
            assert_eq!(*matches,reference,"hybrid scalar cursor changed ordered longest matches");
        }
        ends
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    use crate::automata::lexer::ast::{bytes,choice,plus};
    use crate::automata::lexer::compile::{build_regex,build_regex_partitioned};
    #[test]
    fn scalar_cursor_matches_all_ordered_observations_and_stamp_wrap(){
        let exprs=vec![choice(vec![bytes(b"a"),bytes(b"aba"),bytes(b"abc")]),
            plus(choice(vec![bytes(b"b"),bytes(b"ca")])),bytes(b"!"),plus(bytes(b" ")),
            choice(vec![bytes(b"ab!"),bytes(b"bc"),bytes(b"c")]),bytes(&[0,255,128])];
        let mut rng=713927u64;let mut next=||{rng=rng.wrapping_mul(6364136223846793005).wrapping_add(1);(rng>>32)as usize};
        let mut comparisons=0;
        for partitioned in [false,true]{
            let tokenizer=if partitioned{build_regex_partitioned(&exprs,&[0,1,2,3,4,5])}else{build_regex(&exprs)}
                .into_tokenizer(6,Some(Arc::from(exprs.clone())));
            let flat=crate::terminal_dwa::l1::build_flat_transition_table(&tokenizer);
            for case in 0..32{
                let active=(0..6).map(|t|case%3!=1||t!=4).collect::<Vec<_>>();
                let mut ordinary=NfaTrieScanCache::new(&tokenizer,Some(active.clone()));
                let mut candidate=NfaTrieScanCache::new(&tokenizer,Some(active));
                if case==0{ordinary.match_stamp=u32::MAX-1;candidate.match_stamp=u32::MAX-1;}
                let mut words=vec![vec![],b"a".to_vec(),b"abc!".to_vec(),b"ab !a".to_vec(),b"bc".to_vec(),vec![0,255,128]];
                for _ in 0..16{let n=next()%16;words.push((0..n).map(|_|b"abc! \0\xff\x80"[next()%8]).collect());}
                for start in 0..tokenizer.num_states(){for word in &words{for supplied in [None,Some(flat.as_slice()),Some(&flat[..flat.len()-1])]{
                    let mut a=Vec::new();let mut b=Vec::new();
                    let x=ordinary.execute_into(word,start,&mut a);
                    let y=candidate.execute_native_scalar_into(word,start,&mut b,supplied,false);
                    assert_eq!(x,y,"end case={case} part={partitioned} start={start} word={word:?}");
                    assert_eq!(a,b,"matches case={case} part={partitioned} start={start} word={word:?}");comparisons+=1;
                }}}
            }
        }
        assert!(comparisons>20000);eprintln!("NATIVE_SCALAR_ORDERED comparisons={comparisons}");
    }
}
