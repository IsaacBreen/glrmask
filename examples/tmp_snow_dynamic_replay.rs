use std::{collections::BTreeMap, fs, time::Instant};
use glrmask::{DynamicConstraint, Grammar, Vocab};

fn hex_to_bytes(s: &str) -> Vec<u8> {
    let b=s.as_bytes(); let mut out=Vec::with_capacity(b.len()/2); let mut i=0;
    while i<b.len(){ let h=(b[i] as char).to_digit(16).unwrap(); let l=(b[i+1] as char).to_digit(16).unwrap(); out.push(((h<<4)|l) as u8); i+=2; }
    out
}
fn fingerprint(mask:&[u32])->u64 { mask.iter().fold(0xcbf29ce484222325u64, |h,&w| (h ^ u64::from(w)).wrapping_mul(0x100000001b3)) }
fn main(){
    let schema=fs::read_to_string("/tmp/snow_schema.json").unwrap();
    let raw: BTreeMap<u32,String>=serde_json::from_str(&fs::read_to_string("/tmp/llama3_vocab_hex.json").unwrap()).unwrap();
    let vocab=Vocab::new(raw.into_iter().map(|(id,h)|(id,hex_to_bytes(&h))).collect());
    let ids: Vec<u32>=serde_json::from_str(&fs::read_to_string("/tmp/snow_example0_ids.json").unwrap()).unwrap();
    let build=Instant::now(); let constraint=DynamicConstraint::compile(Grammar::json_schema(&schema), &vocab).unwrap();
    eprintln!("build_ms={:.3}", build.elapsed().as_secs_f64()*1000.0);
    let mut state=constraint.start();
    for (i,&tok) in ids.iter().enumerate(){
        let st=Instant::now(); let mask=state.mask(); let us=st.elapsed().as_secs_f64()*1e6;
        println!("{} {:.3} {:016x} {}", i, us, fingerprint(&mask), mask.iter().map(|x|x.count_ones() as usize).sum::<usize>());
        if let Err(e)=state.commit_token(tok){ eprintln!("commit failed i={i} tok={tok}: {e}"); break; }
    }
}
