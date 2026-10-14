//! Independent oracle corpus for sound lexical-alphabet pruning in bounded queries.
//! No private metadata assumptions; no performance claims.
use glrmask::{BuildOptions,Constraint,Grammar,Optimization,Token,Vocab};
fn examine(c:&Constraint,prefix:&[u8],n:&mut usize) {
    let mut state=c.start();if state.commit_bytes(prefix).is_err(){return;}
    let before=state.marker_events().to_vec();let accepting=state.is_accepting();
    // Cold forcing first, full-mask oracle only afterwards.
    let got=state.forced_tokens().unwrap_or_else(|e|panic!("prefix {prefix:?}: {e}"));
    let expected=state.forced_token_ids_reference();assert_eq!(got,expected,"prefix={prefix:?}");
    assert_eq!(before,state.marker_events());assert_eq!(accepting,state.is_accepting());
    let full=state.mask();assert_eq!(state.forced_tokens().unwrap(),expected);assert_eq!(state.mask(),full);
    let mut cursor=state.clone();for id in got{assert!(!cursor.is_accepting());let m=cursor.mask();assert_eq!(m.iter().map(|v|v.count_ones()).sum::<u32>(),1);cursor.commit_token(id).unwrap();}
    *n+=1;
}
fn main(){
    let modes=[Optimization::Auto,Optimization::FastBuild,Optimization::Balanced,Optimization::FastRuntime];
    let sources=[
        "grammar C { start x nt x=\"xy\" } start r nt r=\"A\" C \"B\"",
        "grammar C { start x nt x=\"xy\"? } start r nt r=C \"!\"",
        "grammar C { start x t W=/[ \\t]+/ ignore W nt x=\"xy\" } start r nt r=\"A\" C \"B\"",
        "grammar C { start x nt x=\"xy\" } start r t W=/ +/ ignore W nt r=\"A\" C \"B\"",
        "grammar C { grammar D { start d nt d=\"yz\"? } start c nt c=\"x\" D } start r nt r=\"A\" C \"!\"",
        "grammar C { start x t T=shortest { /a|ab/ } nt x=T \"c\" } start r nt r=\"!\" C \"?\"",
        "grammar C { start x t T=/a|ab/ nt x=T \"c\"? } start r nt r=C \"!\"",
        "grammar C { start x t T=/[a-z]+/ - /bad/ nt x=T } start r nt r=\"[\" C \"]\"",
        "grammar C { start x nt x=\"\u{e9}\" } start r nt r=\"A\" C \"!\"",
        "grammar C { start x nt x=\"!\" | \"!?\" } start r nt r=C \"!\"",
        "grammar C { start x nt x=@mark(1) \"xy\" @mark(2) } start r nt r=@mark(0) \"A\" C \"!\" @mark(3)",
    ];
    let mut n=0usize;let mut compiled=0;
    for alias in [false,true]{
        let mut words=(0..256u32).map(|i|(i,vec![i as u8])).collect::<Vec<_>>();
        for i in 0..512{words.push((300+i,format!("word{i:04}").into_bytes()));}
        if alias{for (i,s) in ["A","xy","Axy","xyB","yz!","!?","\u{e9}","A\u{e9}","bad","badly"].iter().enumerate(){words.push((1000+i as u32,s.as_bytes().to_vec()));}}
        let v=Vocab::new_with_exact_token_ids(words,[9000]);
        for mode in modes{for source in sources{
            let c=Grammar::from_glrm(source).compile_with(&v,BuildOptions::default().optimization(mode)).unwrap();compiled+=1;
            let restored=Constraint::load(c.save()).unwrap();
            for c in [&c,&restored]{
                for text in ["","A","AxyB","xy!","A xy B","A\txyB","Axyz!","!ac?","ac!","abc!","[badly]","A\u{e9}!","!?!"]{
                    for end in 0..=text.len(){examine(c,&text.as_bytes()[..end],&mut n);}
                }
            }
        }
        let mut child=Grammar::from_glrm("extern token E start x nt x=\"x\" E \"y\"");child.bind("E",Token::new(9000)).unwrap();
        let mut parent=Grammar::from_glrm("extern grammar C start r nt r=\"A\" C \"B\"");parent.bind("C",&child).unwrap();
        let c=parent.compile_with(&v,BuildOptions::default().optimization(mode)).unwrap();
        for c in [&c,&Constraint::load(c.save()).unwrap()]{let mut s=c.start();for id in [65,120,9000,121,66]{let before=s.marker_events().to_vec();assert_eq!(s.forced_tokens().unwrap(),s.forced_token_ids_reference());assert_eq!(before,s.marker_events());s.commit_token(id).unwrap();n+=1;}assert!(s.is_accepting());}
        }
    }
    println!("PASS {n} nested/nullable/ignore/Unicode/shortest/alias/exact-event frontier comparisons across {compiled} compiled cases and all 4 modes, each source + loaded");
}
