use glrmask::{BuildOptions,Constraint,Grammar,Optimization,Token,Vocab};
fn main(){
 let mut queries=0;
 let modes=[Optimization::Auto,Optimization::FastBuild,Optimization::Balanced,Optimization::FastRuntime];
 for aliases in [false,true]{
  let mut entries=vec![(0,b"a".to_vec()),(1,b"b".to_vec()),(2,b"c".to_vec()),(3,b" ".to_vec())];
  if aliases{entries.extend([(7,b"a".to_vec()),(8,b"ab".to_vec()),(9,b"bc".to_vec())]);}
  for i in 0..512{entries.push((100+i,format!("z{i:04}").into_bytes()));}
  let v=Vocab::new(entries).with_special_tokens([9000]);
  for mode in modes{
   for source in [
    "start root nt root=@mark(0) \"a\" @mark(1) \"b\" @mark(2) \"c\" @mark(3)",
    "start root nt root=@mark(0) (\"a\" @mark(1) \"b\" | \"a\" @mark(2) \"c\") @mark(3)",
    "start root t A=/a|ab/ nt root=@mark(0) A @mark(1) \"c\" @mark(2)",
    "start root t A=(/a|ab/ - (/a|ab/ b/./s+)) nt root=@mark(0) A @mark(1) \"b\" @mark(2)",
    "start root t WS=/ +/ ignore WS nt root=@mark(0) \"a\" @mark(1) \"b\" @mark(2)",
   ]{
    let c=Grammar::from_glrm(source).compile_with(&v,BuildOptions::default().optimization(mode)).unwrap();
    let loaded=Constraint::load(c.save(), &vocab).unwrap();
    for c in [&c,&loaded]{
     for ids in [&[][..],&[0][..],&[0,1][..],&[0,2][..],&[0,3][..],&[0,3,1][..]]{
      let mut state=c.start();let mut alive=true;
      for &id in ids{if !state.commit_token(id){alive=false;break;}}
      if !alive{continue;}
      let before=state.marker_events().to_vec();let accepting=state.is_accepting();
      let actual=state.forced_tokens();assert_eq!(before,state.marker_events(),"query modified event batch {mode:?}");
      assert_eq!(accepting,state.is_accepting());assert_eq!(actual,state.forced_canonical_reference());
      let mut reference=state.clone();for id in actual{assert!(state.commit_token(id));assert!(reference.commit_token(id));assert_eq!(state.marker_events(),reference.marker_events());}
      assert_eq!(state.mask(),reference.mask());queries+=1;
     }
    }
   }
   let mut g=Grammar::from_glrm("extern token E start root nt root=@mark(0) \"a\" @mark(1) E @mark(2) \"b\" @mark(3)");g.bind("E",TokenSet::from(9000)).unwrap();
   let c=g.compile_with(&v,BuildOptions::default().optimization(mode)).unwrap();let mut state=c.start();assert!(state.commit_token(0));
   let before=state.marker_events().to_vec();let forced=state.forced_tokens();assert_eq!(forced,state.forced_canonical_reference());assert_eq!(before,state.marker_events());
   assert_eq!(forced,vec![9000,1]);for id in forced{assert!(state.commit_token(id));}assert!(state.is_accepting());queries+=1;
  }
 }
 println!("PASS {queries} marker+bounded-query snapshots (all4modes,loaded,aliases,ignore,shortest,event-only IDs); markers never published by queries");
}
