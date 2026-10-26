//! Compare public canonical forcing with its retained reference at arbitrary
//! raw byte frontiers, including frontiers not reachable by a complete model token.
use glrmask::{BuildOptions,Constraint,Grammar,Optimization,Token,Vocab};
use std::time::Instant;
fn words(out:&mut Vec<Vec<u8>>, current:&mut Vec<u8>, depth:usize) {
    out.push(current.clone());if depth==0{return;}
    for b in b"abc" {current.push(*b);words(out,current,depth-1);current.pop();}
}
fn main() {
    let mut prefixes=Vec::new();words(&mut prefixes,&mut Vec::new(),5);
    let mut spellings=Vec::new();words(&mut spellings,&mut Vec::new(),4);
    let spellings:Vec<_>=spellings.into_iter().filter(|b|!b.is_empty()).collect();
    let mut checked=0usize;let begin=Instant::now();
    let grammars=[
        "start root nt root::=\"abc\" | \"ac\" | \"abcc\"",
        "start root t A::=/a|ab/ nt root::=A \"c\"",
        "start root t A::=/a|ab/ t B::=/a|ac/ nt root::=A B",
        "start root t A::=(/a|ab/ - (/a|ab/ b/./s+)) nt root::=A /[bc]{1,3}/",
        "start root t A::=/[ab]{1,3}/ - /ab|baa/ nt root::=A \"c\"",
        "start root t A::=/[a-c]{1,3}/ & /a.*/ nt root::=A /bc?/",
        "start root nt root::=\"a\"? \"b\"? \"c\"?",
        "start root t A::=/a+/ t B::=/ab/ nt root::=A \"c\" | B \"b\"",
        "start root t A::=/[a-c]+/ t I::=/c/ ignore I nt root::=\"a\" \"b\"",
        "start root nt root::=\"ab\" | \"a\" \"b\" | \"abc\"",
    ];
    for dense in [false,true] {
      let mut entries:Vec<(u32,Vec<u8>)>=spellings.iter().enumerate()
        .filter(|(i,b)|dense || b.len()>1 || i%2==0)
        .map(|(i,b)|((i*3) as u32,b.clone())).collect();
      for j in 0..512 {entries.push((1000+j, format!("z{j:04}").into_bytes()));}
      entries.push((9000,b"a".to_vec()));entries.push((9001,b"a".to_vec()));
      let vocab=Vocab::new(entries).with_special_tokens([10000,10001]);
      for (gno,source) in grammars.iter().enumerate() {
        for mode in [Optimization::FastBuild,Optimization::Balanced,Optimization::FastRuntime] {
          let c=Grammar::from_glrm(source).compile_with(&vocab,BuildOptions::default().optimization(mode)).unwrap();
          let loaded=Constraint::load(c.save(), &vocab).unwrap();
          for (is_loaded,c) in [(false,&c),(true,&loaded)] {
            for prefix in &prefixes {
              let mut state=c.start();if !state.commit_bytes(prefix){continue;}
              let actual=state.forced_tokens().unwrap_or_else(|e|panic!("g{gno} {mode:?} dense={dense} loaded={is_loaded} prefix={prefix:?}: {e}"));
              let expected=state.forced_canonical_reference();
              assert_eq!(actual,expected,"g{gno} {mode:?} dense={dense} loaded={is_loaded} prefix={prefix:?}");
              checked+=1;
            }
          }
        }
      }
      // Both ordinary byte paths and event alternatives remain live.
      let mut g=Grammar::from_glrm("extern token E start root nt root::=E \"b\" | \"a\" \"c\"");g.bind("E",TokenSet::from(9000)).unwrap();
      for mode in [Optimization::FastBuild,Optimization::Balanced,Optimization::FastRuntime] {
        let c=g.compile_with(&vocab,BuildOptions::default().optimization(mode)).unwrap();
        for prefix in &prefixes {
          let mut state=c.start();if !state.commit_bytes(prefix){continue;}
          assert_eq!(state.forced_tokens(),state.forced_canonical_reference(),"event branch {mode:?} prefix={prefix:?}");checked+=1;
        }
      }
    }
    println!("PASS {checked} arbitrary byte-frontier comparisons, including aliases/sparse-ID/unreachable-by-token states; seconds={:.3}",begin.elapsed().as_secs_f64());
}
