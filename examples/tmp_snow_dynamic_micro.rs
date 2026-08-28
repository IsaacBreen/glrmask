use std::{collections::BTreeMap, fs, time::Instant};
use glrmask::{DynamicConstraint, Grammar, Vocab};
fn hex_to_bytes(s:&str)->Vec<u8>{let b=s.as_bytes();let mut o=Vec::with_capacity(b.len()/2);let mut i=0;while i<b.len(){o.push((((b[i]as char).to_digit(16).unwrap()<<4)|(b[i+1]as char).to_digit(16).unwrap())as u8);i+=2;}o}
fn fp(mask:&[u32])->u64{mask.iter().fold(0xcbf29ce484222325u64,|h,&w|(h^u64::from(w)).wrapping_mul(0x100000001b3))}
fn main(){
 unsafe { std::env::set_var("GLRMASK_DISABLE_DYNAMIC_MASK_CACHE","1"); }
 let target:usize=std::env::args().nth(1).and_then(|s|s.parse().ok()).unwrap_or(17);
 let reps:usize=std::env::args().nth(2).and_then(|s|s.parse().ok()).unwrap_or(80);
 let mode=std::env::args().nth(3).unwrap_or_else(||"alternate".to_owned());
 let schema=fs::read_to_string("/tmp/snow_schema.json").unwrap();
 let raw:BTreeMap<u32,String>=serde_json::from_str(&fs::read_to_string("/tmp/llama3_vocab_hex.json").unwrap()).unwrap();
 let vocab=Vocab::new(raw.into_iter().map(|(id,h)|(id,hex_to_bytes(&h))).collect());
 let ids:Vec<u32>=serde_json::from_str(&fs::read_to_string("/tmp/snow_example0_ids.json").unwrap()).unwrap();
 let c=DynamicConstraint::compile(Grammar::json_schema(&schema),&vocab).unwrap(); let mut st=c.start();
 for &tok in ids.iter().take(target){st.commit_token(tok).unwrap();}
 let mut ref_fp=0; let mut off=Vec::new(); let mut on=Vec::new();
 for r in 0..reps {
   let force=match mode.as_str(){"full_exact"=>true,"generic"=>false,_=>r%2==1};
   unsafe { if force { std::env::set_var("GLRMASK_EXPERIMENT_DYNAMIC_FORCE_FULL_EXACT_WALK","1"); } else { std::env::remove_var("GLRMASK_EXPERIMENT_DYNAMIC_FORCE_FULL_EXACT_WALK"); } }
   // ConstraintState::clone deliberately starts with an empty per-generation
   // mask cache. Clone before the timer so each sample exercises mask
   // generation rather than the public fill_mask recurrence cache.
   let probe=st.clone();
   let t=Instant::now(); let m=probe.mask(); let us=t.elapsed().as_secs_f64()*1e6; let f=fp(&m);
   if r==0 {ref_fp=f;} else {assert_eq!(f,ref_fp);}
   if force {on.push(us)} else {off.push(us)}
 }
 off.sort_by(|a,b|a.partial_cmp(b).unwrap()); on.sort_by(|a,b|a.partial_cmp(b).unwrap());
 fn q(v:&[f64],p:f64)->f64{v[((v.len()-1)as f64*p).round()as usize]}
 match mode.as_str(){
   "full_exact"=>eprintln!("target={target} reps={reps} mode=full_exact fp={ref_fp:016x} p50={:.3} p90={:.3}",q(&on,0.5),q(&on,0.9)),
   "generic"=>eprintln!("target={target} reps={reps} mode=generic fp={ref_fp:016x} p50={:.3} p90={:.3}",q(&off,0.5),q(&off,0.9)),
   _=>eprintln!("target={target} reps={reps} fp={ref_fp:016x} off_p50={:.3} off_p90={:.3} on_p50={:.3} on_p90={:.3} ratio={:.2}x",q(&off,0.5),q(&off,0.9),q(&on,0.5),q(&on,0.9),q(&off,0.5)/q(&on,0.5)),
 }
}
