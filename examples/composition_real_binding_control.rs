//! Binding-only comparison for the saved real JS/ten-schema components.
//! Preparation is explicitly reported, not silently excluded from raw costs.
//! Usage: CACHE_DIR [REPEATS=7]
use glrmask::{Constraint,Vocab};
use glrmask::__private::ConstraintExt;
use std::{fs,path::Path,sync::Arc,time::Instant};
fn main()->Result<(),Box<dyn std::error::Error>> {
    let args:Vec<_>=std::env::args().collect();
    let cache=Path::new(args.get(1).ok_or("CACHE_DIR [REPEATS]")?);
    let repeats=args.get(2).map(|s|s.parse::<usize>()).transpose()?.unwrap_or(7);
    assert!(repeats>0);
    let raw=fs::read(cache.join("vocab_dump.bin"))?;let mut pos=0usize;
    let read=|pos:&mut usize|{let n=u32::from_le_bytes(raw[*pos..*pos+4].try_into().unwrap());*pos+=4;n};
    let count=read(&mut pos);let mut tokens=Vec::new();
    for _ in 0..count{let id=read(&mut pos);let n=read(&mut pos) as usize;tokens.push((id,raw[pos..pos+n].to_vec()));pos+=n;}
    assert_eq!(pos,raw.len());let vocab=Vocab::new(tokens);
    let core_bytes=fs::read(cache.join("core.bin"))?;
    let dispatch_bytes=fs::read(cache.join("dispatch-literal.bin"))?;
    println!("prepared,shared,repeat,load_ns,prepare_core_ns,prepare_child_ns,parent_clone_ns,bind_ns");
    for prepared in [false,true] {
        for shared in [false,true] {
            if std::env::var("BIND_PREPARED").is_ok_and(|s| (s=="1")!=prepared) {continue;}
            if std::env::var("BIND_SHARED").is_ok_and(|s| (s=="1")!=shared) {continue;}
            eprintln!("BIND_CASE prepared={prepared} shared={shared} load_start");
            let load=Instant::now();
            let mut core=Constraint::load_with_vocab(&core_bytes,&vocab)?;
            let mut dispatch=Constraint::load_with_vocab(&dispatch_bytes,&vocab)?;
            let load_ns=load.elapsed().as_nanos();
            eprintln!("BIND_LOAD_DONE ns={load_ns}");
            let t=Instant::now();
            if prepared {core.prepare_for_dynamic_composition(&vocab)?;}
            let core_ns=t.elapsed().as_nanos();
            eprintln!("BIND_CORE_PREP_DONE ns={core_ns}");
            let t=Instant::now();
            if prepared {dispatch.prepare_for_dynamic_composition(&vocab)?;}
            let child_ns=t.elapsed().as_nanos();
            eprintln!("BIND_CHILD_PREP_DONE ns={child_ns}");
            let dispatch=Arc::new(dispatch);
            for repeat in 0..repeats {
                let t=Instant::now();let parent=core.clone();let clone_ns=t.elapsed().as_nanos();
                eprintln!("BIND_START repeat={repeat} clone_ns={clone_ns}");
                let t=Instant::now();
                let result=if shared {
                    parent.compose_compiled_subgrammars_dynamic_shared(
                        &[("PROGRAMMATIC_TOOL_SUFFIX",Arc::clone(&dispatch))],&vocab)
                } else {
                    parent.compose_compiled_subgrammars_dynamic(
                        &[("PROGRAMMATIC_TOOL_SUFFIX",dispatch.as_ref())],&vocab)
                }?;
                let bind_ns=t.elapsed().as_nanos();
                std::hint::black_box(&result);
                println!("{prepared},{shared},{repeat},{load_ns},{core_ns},{child_ns},{clone_ns},{bind_ns}");
            }
        }
    }
    Ok(())
}
