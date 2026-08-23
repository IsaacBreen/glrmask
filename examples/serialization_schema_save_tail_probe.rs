use std::{collections::BTreeMap, path::Path, time::Instant};
use glrmask::{Constraint, Vocab};
use glrmask::__private::VocabExt;
fn hex_to_bytes(s:&str)->Vec<u8>{(0..s.len()).step_by(2).map(|i|u8::from_str_radix(&s[i..i+2],16).unwrap()).collect()}
fn load_vocab(p:&Path)->Vocab{let r=std::fs::read_to_string(p).unwrap();let e:BTreeMap<u32,String>=serde_json::from_str(&r).unwrap();Vocab::new(e.into_iter().map(|(i,s)|(i,hex_to_bytes(&s))).collect())}
fn pct(x:&[u128],q:f64)->f64{x[((x.len()-1) as f64*q).round() as usize] as f64/1e6}
fn main(){
 let a=std::env::args().collect::<Vec<_>>();
 let schema=std::fs::read_to_string(&a[1]).unwrap(); let vocab=load_vocab(Path::new(&a[2])); vocab.prepare_for_compile();
 let t=Instant::now(); let c=Constraint::from_json_schema(&schema,&vocab).unwrap(); println!("compile_ms={:.3}",t.elapsed().as_secs_f64()*1000.0);
 let n=a.get(3).and_then(|s|s.parse().ok()).unwrap_or(31usize); let mut xs=Vec::with_capacity(n); let mut bytes=0;
 for _ in 0..n { let t=Instant::now(); let out=c.save(); xs.push(t.elapsed().as_nanos()); bytes=out.len(); std::hint::black_box(out); }
 xs.sort_unstable(); println!("n={} bytes={} p50_ms={:.3} p90_ms={:.3} p99_ms={:.3} max_ms={:.3}",n,bytes,pct(&xs,0.5),pct(&xs,0.9),pct(&xs,0.99),pct(&xs,1.0));
}
