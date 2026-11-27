//! Developer-only static frontier capacity comparison.
//! Usage: cargo run --profile fasttest --features internal-api --example frontier_capacity_bench -- <vocab.json> <corpus.json>
//! Corpus: {"problems": [{"problem_id": "...", "schema": {}, "valid_inputs": ["..."]}]}.
#![recursion_limit = "512"]
use glrmask::{Grammar,Vocab,Constraint,ConstraintState,CompileOptions,InternalMaskStrategy};
use glrmask::__private::{ConstraintStateExt, VocabExt};
use serde_json::Value;
use std::{collections::HashMap, hint::black_box, path::{Path,PathBuf}, time::Instant};
use std::hash::{Hash,Hasher};

fn decode(s:&str)->Vec<u8>{s.as_bytes().as_chunks::<2>().0.iter().map(|p|u8::from_str_radix(std::str::from_utf8(p).unwrap(),16).unwrap()).collect()}
fn tokenize(bytes:&[u8], map:&HashMap<Vec<u8>,u32>,max:usize)->Vec<u32>{
    let mut result=vec![]; let mut i=0;
    while i<bytes.len(){
        let (n,id)=(1..=max.min(bytes.len()-i)).rev().find_map(|n|map.get(&bytes[i..i+n]).map(|id|(n,*id))).unwrap();
        result.push(id); i+=n;
    }
    result
}
fn compiled(name:&str,g:Grammar,v:&Vocab,cache:&Path)->Result<Constraint,String>{
    let path=cache.join(format!("{}.constraint",name.replace(['/','\\'],"_")));
    if path.exists(){return Constraint::load(&std::fs::read(path).map_err(|e|e.to_string())?,v).map_err(|e|e.to_string());}
    let start=Instant::now();
    let c=g.compile_with(v,CompileOptions::default().internal_mask_strategy(InternalMaskStrategy::ParserDwa)).map_err(|e|e.to_string())?;
    eprintln!("BUILD,{name},{}",start.elapsed().as_millis());
    std::fs::write(path,c.save()).map_err(|e|e.to_string())?;
    Ok(c)
}
fn replay(name:&str,c:&Constraint,ids:&[u32]){
    let mut hash=std::collections::hash_map::DefaultHasher::new();
    for run in 0..4{
        let mut state=c.start();
        for (i,&id) in ids.iter().enumerate(){
            // Run 0 warms scratch and records frontier geometry separately.
            if run==0 {
                let (width,distinct,ptr,equal,keys)=state.frontier_shape();
                println!("SHAPE,{name},{i},{width},{distinct},{ptr},{equal},{keys:?}");
            }
            let t=Instant::now();
            let mut next=black_box(state.clone());
            let ok=black_box(next.commit_token(black_box(id)));
            let commit=t.elapsed().as_nanos();
            // Deliberately uncached: mask on the source, clone has fresh cache.
            let t=Instant::now();let mask=black_box(state.mask());let mask_ns=t.elapsed().as_nanos();
            if run==0 { mask.hash(&mut hash); ok.hash(&mut hash); }
            if run>0 {println!("TIME,{name},{i},{run},{commit},{mask_ns},{ok}");}
            if !ok {eprintln!("REJECT,{name},{i},{id}");break;}
            state=next;
        }
        if run==0 {println!("HASH,{name},{:016x},{}",hash.finish(),state.is_accepting());}
    }
}
fn main(){
    let root=PathBuf::from(std::env::args().nth(1).expect("canonical vocab.json"));
    let corpus_path=std::env::args().nth(2).unwrap_or_else(|| "frontier-runtime-tail-corpus.json".to_owned());
    let corpus:Value=serde_json::from_str(&std::fs::read_to_string(corpus_path).unwrap()).unwrap();
    let vocab_data:Value=serde_json::from_str(&std::fs::read_to_string(root).unwrap()).unwrap();
    let entries:Vec<_>=vocab_data["tokens_hex"].as_array().unwrap().iter().enumerate().map(|(i,t)|(i as u32,decode(t.as_str().unwrap()))).collect();
    let mut lookup=HashMap::new();let mut longest=0;
    for (id,b) in &entries {if !b.is_empty(){longest=longest.max(b.len());lookup.entry(b.clone()).or_insert(*id);}}
    let vocab=Vocab::new(entries).unwrap();
    vocab.prepare_for_compile();
    let cache=Path::new("frontier-static-cache");std::fs::create_dir_all(cache).unwrap();
    println!("LAYOUT,{}",std::mem::size_of::<ConstraintState>());
    let js=Grammar::from_glrm(include_str!("../../src/programmatic_js/javascript.glrm")).unwrap();
    match compiled("javascript",js,&vocab,cache){
        Ok(c)=>for f in ["16_dangling_else_and_nested_prefixes.js","19_deep_nesting.js","22_comprehensive_integration.js"]{
            let text=std::fs::read(Path::new("tests/data/javascript").join(f)).unwrap();
            let ids=tokenize(&text,&lookup,longest);
            replay(f,&c,&ids);
        },
        Err(e)=>eprintln!("BUILD_FAIL,javascript,{e}"),
    }
    for p in corpus["problems"].as_array().unwrap(){
        let name=p["problem_id"].as_str().unwrap();
        let Some(inputs)=p["valid_inputs"].as_array() else {continue};
        if inputs.is_empty(){eprintln!("NO_VALID_INPUT,{name}");continue;}
        let g=match Grammar::from_json_schema(&p["schema"].to_string()){Ok(g)=>g,Err(e)=>{eprintln!("GRAMMAR_FAIL,{name},{e}");continue}};
        match compiled(name,g,&vocab,cache){
            Ok(c)=>for (i,input) in inputs.iter().enumerate(){let ids=tokenize(input.as_str().unwrap().as_bytes(),&lookup,longest);replay(&format!("{name}/{i}"),&c,&ids);},
            Err(e)=>eprintln!("BUILD_FAIL,{name},{e}"),
        }
    }
}
