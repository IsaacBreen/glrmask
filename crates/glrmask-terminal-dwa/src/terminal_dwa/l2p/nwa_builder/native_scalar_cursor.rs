//! Scalar entry to the shared longest-match recurrence.
use super::*;

impl NfaTrieScanCache<'_> {
    pub(super) fn execute_native_scalar_into(
        &mut self, input: &[u8], start: u32, matches: &mut Vec<TokenizerMatch>,
        flat: Option<&[u32]>, validate: bool,
    ) -> SmallVec<[u32; 1]> {
        let source = RawCursorSource {
            tokenizer: self.tokenizer,
            flat: flat.filter(|table| table.len() == self.tokenizer.num_states() as usize * 256),
        };
        let ends = self.execute_cursor::<true, _>(&source, input, start, matches);
        if validate || self.strict_reference {
            self.verify_scalar_reference(input, start, matches, &ends);
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
                    candidate.verify_reference(word,start,&b,&y);
                    assert_eq!(x,y,"end case={case} part={partitioned} start={start} word={word:?}");
                    assert_eq!(a,b,"matches case={case} part={partitioned} start={start} word={word:?}");comparisons+=1;
                }}}
            }
        }
        assert!(comparisons>20000);eprintln!("NATIVE_SCALAR_ORDERED comparisons={comparisons}");
    }
}
