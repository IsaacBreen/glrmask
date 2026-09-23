//! Real selected10 static/dynamic composition equality and runtime control.
//! Build:  build COMPONENT_CACHE OUTPUT_ARTIFACT
//! Replay: replay VOCAB_DUMP STATIC_ARTIFACT DYNAMIC_ARTIFACT TRACES OUTPUT [LIMIT]
//! No construction, validation, hashing or output is included in mask/TBM time.
use glrmask::{Constraint, Vocab};
use glrmask::__private::{ConstraintExt, ConstraintStateExt};
use serde::Deserialize;
use std::{fs, io::{BufWriter, Write}, path::Path, time::Instant};

fn vocab(path: &Path) -> Result<Vocab, Box<dyn std::error::Error>> {
    let data=fs::read(path)?;let mut at=0usize;
    let read=|at:&mut usize| {let n=u32::from_le_bytes(data[*at..*at+4].try_into().unwrap());*at+=4;n};
    let count=read(&mut at);let mut entries=Vec::new();
    for _ in 0..count {let id=read(&mut at);let n=read(&mut at) as usize;entries.push((id,data[at..at+n].to_vec()));at+=n;}
    assert_eq!(at,data.len());Ok(Vocab::new(entries))
}
#[derive(Deserialize)]
struct Trace {label:String,token_ids:Vec<u32>}
fn main() -> Result<(),Box<dyn std::error::Error>> {
    let args:Vec<_>=std::env::args().collect();
    if args.get(1).map(String::as_str)==Some("build") {
        assert_eq!(args.len(),4,"build CACHE OUTPUT");
        let cache=Path::new(&args[2]);let vocabulary=vocab(&cache.join("vocab_dump.bin"))?;
        let core=Constraint::load_with_vocab(fs::read(cache.join("core.bin"))?,&vocabulary)?;
        let dispatch=Constraint::load_with_vocab(fs::read(cache.join("dispatch-literal.bin"))?,&vocabulary)?;
        let started=Instant::now();
        let bound=core.compose_compiled_subgrammars_dynamic(&[("PROGRAMMATIC_TOOL_SUFFIX",&dispatch)],&vocabulary)?;
        let elapsed=started.elapsed().as_nanos();
        let bytes=bound.save();fs::write(&args[3],&bytes)?;
        println!("BUILD dynamic_boundary_ns={elapsed} bytes={}",bytes.len());return Ok(());
    }
    assert!(args.get(1).map(String::as_str)==Some("replay") && args.len()>=7,
        "replay VOCAB STATIC DYNAMIC TRACES OUTPUT [LIMIT]");
    let vocabulary=vocab(Path::new(&args[2]))?;
    let static_constraint=Constraint::load_with_vocab(fs::read(&args[3])?,&vocabulary)?;
    let dynamic_constraint=Constraint::load_with_vocab(fs::read(&args[4])?,&vocabulary)?;
    let traces:Vec<Trace>=serde_json::from_slice(&fs::read(&args[5])?)?;
    let limit=args.get(7).map(|s|s.parse::<usize>()).transpose()?.unwrap_or(0);
    let dynamic_reference=std::env::var_os("DYNAMIC_REFERENCE").is_some();
    let profile_step=std::env::var("PROFILE_STEP").ok().map(|s|s.parse::<usize>()).transpose()?;
    let profile_trace=std::env::var("PROFILE_TRACE").ok().map(|s|s.parse::<usize>()).transpose()?.unwrap_or(0);
    let profile_previous_masks=std::env::var_os("PROFILE_PREVIOUS_MASKS").is_some();
    let profile_previous_traces=std::env::var_os("PROFILE_PREVIOUS_TRACES").is_some();
    let mut out=BufWriter::new(fs::File::create(&args[6])?);
    writeln!(out,"trace,step,token,static_mask_ns,dynamic_mask_ns,static_commit_ns,dynamic_commit_ns,hash")?;
    let words=static_constraint.mask_len().max(dynamic_constraint.mask_len());
    let mut a_mask=vec![0;words];let mut b_mask=vec![0;words];let mut samples=0usize;
    'traces: for (trace_id,trace) in traces.iter().enumerate() {
        if profile_step.is_some() && trace_id!=profile_trace {
            if profile_previous_traces && trace_id<profile_trace {
                let mut prior=dynamic_constraint.start();
                for &token in &trace.token_ids {prior.fill_mask(&mut b_mask);prior.commit_token(token)?;}
                prior.fill_mask(&mut b_mask);
            }
            continue;
        }
        let mut a=static_constraint.start();let mut b=dynamic_constraint.start();
        eprintln!("TRACE_BEGIN {trace_id} {} {}",trace.label,trace.token_ids.len());
        for step in 0..=trace.token_ids.len() {
            if limit>0 && samples>=limit {break 'traces;}
            if let Some(target)=profile_step {
                if step<target {
                    if profile_previous_masks {
                        a.fill_mask(&mut a_mask); b.fill_mask(&mut b_mask);
                        assert_eq!(a_mask,b_mask,"profile history must remain exact");
                    }
                    let token=trace.token_ids[step];a.commit_token(token)?;b.commit_token(token)?;continue;
                }
                a.fill_mask(&mut a_mask);
                eprintln!("PROFILE_ROOTS trace={trace_id} step={step} prior_masks={profile_previous_masks} roots={} paths={}",b.parser_root_count(),b.parser_path_count(4096));
                for repeat in 0..3 {
                    eprintln!("PROFILE_BEGIN trace={trace_id} step={step} repeat={repeat}");
                    let started=Instant::now();b.fill_mask(&mut b_mask);
                    let elapsed=started.elapsed().as_nanos();
                    assert_eq!(a_mask,b_mask,"profile mask equality");
                    eprintln!("PROFILE_END elapsed_ns={elapsed}");
                }
                return Ok(());
            }
            let started=Instant::now();
            if dynamic_reference { a_mask=a.fill_mask_dynamic_vec(); }
            else { a.fill_mask(&mut a_mask); }
            let a_ns=started.elapsed().as_nanos();
            let started=Instant::now();b.fill_mask(&mut b_mask);let b_ns=started.elapsed().as_nanos();
            if a_mask!=b_mask || a.is_accepting()!=b.is_accepting() {
                out.flush()?;
                let different:Vec<_>=a_mask.iter().zip(&b_mask).enumerate().flat_map(|(w,(&a,&b))|
                    (0..32).filter_map(move |bit| (((a^b)&(1<<bit))!=0).then_some((w*32+bit,a&(1<<bit)!=0,b&(1<<bit)!=0)))).take(16).collect();
                return Err(format!("mask mismatch trace={trace_id} step={step} first={different:?}").into());
            }
            let hash=b_mask.iter().fold(0xcbf29ce484222325u64,|h,&w|(h^w as u64).wrapping_mul(0x100000001b3));
            let token=trace.token_ids.get(step).copied();
            let (ca,cb)=if let Some(token)=token {
                assert!(a_mask[token as usize/32]&(1<<(token%32))!=0,"trace token not admitted");
                let t=Instant::now();let ra=a.commit_token(token);let ca=t.elapsed().as_nanos();
                let t=Instant::now();let rb=b.commit_token(token);let cb=t.elapsed().as_nanos();
                ra?;rb?;(ca.to_string(),cb.to_string())
            } else {(String::new(),String::new())};
            writeln!(out,"{trace_id},{step},{},{a_ns},{b_ns},{ca},{cb},{hash:016x}",token.map(|t|t.to_string()).unwrap_or_default())?;
            samples+=1;
            if step%128==0 {out.flush()?;eprintln!("PROGRESS trace={trace_id} step={step} mask_us={:.3}",b_ns as f64/1000.);}
        }
    }
    out.flush()?;println!("REPLAY masks_equal=true samples={samples} dynamic_reference={dynamic_reference} partial={}",limit>0 && samples==limit);Ok(())
}
