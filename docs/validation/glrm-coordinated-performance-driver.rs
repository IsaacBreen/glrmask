use std::{collections::BTreeSet, error::Error, hint::black_box, time::Instant};
use glrmask::{BuildOptions, Constraint, Grammar, Optimization, Vocab};
use serde_json::{json,Value};
#[cfg(feature="redesign")] use glrmask::PreparedGrammar as Prepared;
#[cfg(not(feature="redesign"))] use glrmask::UnlinkedConstraint as Prepared;
const JSON_SOURCE:&str=r#"{"type":"object","properties":{"name":{"type":"string","pattern":"^[a-z]+$"},"events":{"type":"array","items":{"type":"integer","minimum":0,"maximum":99},"maxItems":5},"active":{"type":"boolean"}},"required":["name","events","active"],"additionalProperties":false}"#;
const LARK_SOURCE:&str="start: expr\n?expr: expr \"+\" term | expr \"-\" term | term\n?term: term \"*\" atom | term \"/\" atom | atom\n?atom: INT | \"(\" expr \")\"\nINT: /[0-9]+/\n%ignore /[ \\t\\n]+/";
const SIMPLE:&str="glrm 1; start root; t ITEM ::= /[a-z]+/; nt root ::= \"[\" ITEM (\",\" ITEM)* \"]\";";
const CHILD:&str="glrm 1; start root; t WS ::= /[ \\t]+/; ignore WS; t ITEM ::= /[a-z]+/; nt root ::= \"[\" ITEM (\",\" ITEM)* \"]\";";
const TEXTS:[(&str,&str);5]=[
    ("json_schema",r#"{"name": "alice", "events": [1, 2, 3], "active": true}"#),
    ("lark_arithmetic","((17 + 28) * (3 - 4)) / 9"),
    ("glrm_list","[alice,bob,charlie,delta]"),
    ("scoped_child","BEGIN:[alpha,beta,gamma]:END"),
    ("exact_token_set","ab"),
];
fn time<T>(f:impl FnOnce()->T)->(T,f64){let t=Instant::now();let v=black_box(f());(v,t.elapsed().as_secs_f64()*1e9)}
fn summary(xs:&[f64])->Value {let mut v=xs.to_vec();v.sort_by(f64::total_cmp);let q=|a:usize,b:usize|v[(v.len()*a).div_ceil(b).saturating_sub(1)];json!({"n":v.len(),"p50_ns":q(1,2),"p95_ns":q(95,100),"p99_ns":q(99,100),"p999_ns":q(999,1000),"max_ns":v[v.len()-1]})}
fn input_vocab(n:usize)->Vec<(u32,Vec<u8>)>{
    let mut all=BTreeSet::<Vec<u8>>::new();
    for (_,text) in TEXTS {for i in 0..text.len(){for width in 2..=8 {if i+width<=text.len(){all.insert(text.as_bytes()[i..i+width].to_vec());}}}}
    let mut words=(0..256).map(|i|(i,vec![i as u8])).collect::<Vec<_>>();
    for word in all {words.push((words.len() as u32,word));}
    let mut i=0;
    while words.len()<n {words.push((words.len() as u32,format!("word{i:06}").into_bytes()));i+=1;}
    words
}
fn trace(text:&str,vocab:&[(u32,Vec<u8>)],exact:bool)->Vec<u32>{
    if exact{return text.bytes().map(u32::from).collect();}
    let mut ids=Vec::new();let mut pos=0;
    while pos<text.len(){let (id,b)=vocab.iter().filter(|(_,b)|!b.is_empty()&&b.len()<=3&&text.as_bytes()[pos..].starts_with(b)).max_by_key(|(id,b)|(b.len(),std::cmp::Reverse(*id))).unwrap();ids.push(*id);pos+=b.len();}
    ids
}
fn grammar(case:&str,vocab:&Vocab)->Result<Grammar<'static>,Box<dyn Error>>{
    Ok(match case {
        "json_schema"=>Grammar::from_json_schema(JSON_SOURCE),
        "lark_arithmetic"=>Grammar::from_lark(LARK_SOURCE),
        "glrm_list"=>Grammar::from_glrm(SIMPLE),
        "scoped_child"=>{
            let p=if cfg!(feature="redesign") {"glrm 1; start root; extern grammar CHILD; nt root ::= \"BEGIN:\" CHILD \":END\";"}
                else {"glrm 1; start root; extern CHILD; nt root ::= \"BEGIN:\" CHILD \":END\";"};
            Grammar::from_glrm(p).bind("CHILD",Grammar::from_glrm(CHILD))?
        },
        "exact_token_set"=>{
            let p=if cfg!(feature="redesign") {"glrm 1; start root; extern token MARK; nt root ::= MARK MARK;"}
                else {"glrm 1; start root; extern MARK; nt root ::= MARK MARK;"};
            Grammar::from_glrm(p).bind("MARK",glrmask::TokenSet::from_ranges([97..=122]))?
        },
        _=>unreachable!(),
    })
}
fn prepare(g:&Grammar<'_>,v:&Vocab)->glrmask::Result<Prepared>{
    #[cfg(feature="redesign")] {g.prepare(v)}
    #[cfg(not(feature="redesign"))] {g.compile_unlinked(v)}
}
fn mask_trace(c:&Constraint,ids:&[u32])->u64{
    let mut s=c.start();let mut h=0xcbf29ce484222325u64;
    for (step,&id) in ids.iter().enumerate() {let m=s.mask();assert!(m[id as usize/32]&(1<<(id%32))!=0,"trace step {step} ID {id} rejected");for word in m {h=(h^u64::from(word)).wrapping_mul(0x100000001b3);}s.commit_token(id).unwrap();}
    assert!(s.is_accepting());h
}
fn run(case:&str,text:&str,mode:Optimization,n:usize,reps:usize)->Result<Value,Box<dyn Error>>{
    let (entries,vocab_input_ns)=time(||input_vocab(n));let ids=trace(text,&entries,case=="exact_token_set");
    let (v,vocab_ns)=time(||Vocab::new(entries));
    let (g,source_ns)=time(||grammar(case,&v));let g=g?;
    let options=BuildOptions::default().optimization(mode);
    let (c,compile_ns)=time(||g.compile_with(&v,options.clone()));let c=c?;
    let fingerprint=mask_trace(&c,&ids);
    let mut builds=Vec::new();for _ in 0..5 {let (next,t)=time(||g.compile_with(&v,options.clone()));assert_eq!(mask_trace(&next?,&ids),fingerprint);builds.push(t);}
    let (prepared,prepare_ns)=time(||prepare(&g,&v));let prepared=prepared?;
    let (linked,link_ns)=time(||prepared.link_with(options.clone()));assert_eq!(mask_trace(&linked?,&ids),fingerprint);
    let mut links=Vec::new();for _ in 0..7 {let (next,t)=time(||prepared.link_with(options.clone()));assert_eq!(mask_trace(&next?,&ids),fingerprint);links.push(t);}
    let (saved,save_ns)=time(||c.save());let bytes=saved.len();
    let (loaded,load_ns)=time(||Constraint::load(saved));assert_eq!(mask_trace(&loaded?,&ids),fingerprint);
    let (psaved,prepared_save_ns)=time(||prepared.save());let prepared_bytes=psaved.len();
    let (ploaded,prepared_load_ns)=time(||Prepared::load_with_vocab(psaved,&v));assert_eq!(mask_trace(&ploaded?.link_with(options.clone())?,&ids),fingerprint);
    // Validation and allocator warmup are outside the sampled trace loop.
    for _ in 0..20 {black_box(mask_trace(&c,&ids));}
    let mut start_times=Vec::new();let mut masks=Vec::new();let mut commits=Vec::new();let mut total=Vec::new();let mut output=vec![0;c.mask_len()];
    for _ in 0..reps {
        let begin=Instant::now();let (mut s,t)=time(||c.start());start_times.push(t);
        for &id in &ids {
            let (_,t)=time(||s.fill_mask(black_box(&mut output)));masks.push(t);
            assert!(output[id as usize/32]&(1<<(id%32))!=0);
            let (r,t)=time(||s.commit_token(black_box(id)));r?;commits.push(t);
        }
        total.push(begin.elapsed().as_secs_f64()*1e9);assert!(s.is_accepting());
    }
    Ok(json!({"case":case,"mode":format!("{mode:?}"),"vocab_target":n,"trace":ids,"mask_fingerprint":format!("{fingerprint:016x}"),"vocab_input_ns":vocab_input_ns,"vocab_ns":vocab_ns,"source_create_and_bind_ns":source_ns,"first_compile_ns":compile_ns,"warm_compile":summary(&builds),"prepare_with_warm_vocab_ns":prepare_ns,"first_link_ns":link_ns,"warm_link":summary(&links),"save_ns":save_ns,"load_ns":load_ns,"artifact_bytes":bytes,"prepared_save_ns":prepared_save_ns,"prepared_load_ns":prepared_load_ns,"prepared_artifact_bytes":prepared_bytes,"start":summary(&start_times),"mask":summary(&masks),"commit":summary(&commits),"trace_total":summary(&total)}))
}
fn main()->Result<(),Box<dyn Error>>{
    let args=std::env::args().skip(1).collect::<Vec<_>>();
    let n=args.first().map(|s|s.parse()).transpose()?.unwrap_or(8192usize);
    let reps=args.get(1).map(|s|s.parse()).transpose()?.unwrap_or(2000usize);
    assert!(n>=2048&&reps>=20&&!cfg!(debug_assertions));
    let mut rows=Vec::new();
    for mode in [Optimization::FastBuild,Optimization::Balanced,Optimization::FastRuntime] {
        for (case,text) in TEXTS {eprintln!("RUN {case} {mode:?}");rows.push(run(case,text,mode,n,reps)?);}
    }
    println!("{}",serde_json::to_string_pretty(&json!({"kind":"glrm-redesign-matched-public-api-v1","redesign":cfg!(feature="redesign"),"profile":"cargo-release-defaults","vocabulary":"deterministic synthetic bytes + corpus ngrams + ASCII identifiers","repetitions":reps,"notes":["Same source driver and input token traces; only typed extern spelling and prepare/compile_unlinked name differ.","Parse/binding, vocabulary, compile, prepare, final link, save/load and runtime operations timed separately.","Warm compile/link samples share vocabulary caches; small sample counts describe these cases, not global tails.","Runtime measured in release with reusable mask buffer, new state per trace and explicit warmup.","Not a full schema-corpus or real model benchmark. Mask fingerprints and trace acceptance compared exactly."],"rows":rows}))?);Ok(())
}
