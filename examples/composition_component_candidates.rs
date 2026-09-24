//! Export a reusable outgoing-token upper bound from ONE compiled component.
//! No link is constructed or even represented by this program's arguments.
use glrmask::{Constraint,Vocab};
use glrmask::__private::{ConstraintExt,debug_component_boundary_candidate_ids};
use std::{fs,path::Path,time::Instant};

fn read_vocab(path:&Path)->Vocab{
    let bytes=fs::read(path).expect("read model vocabulary");let mut offset=0usize;
    let u32_at=|offset:&mut usize|{
        let end=offset.checked_add(4).expect("vocab offset overflow");
        let value=u32::from_le_bytes(bytes.get(*offset..end).expect("truncated vocabulary").try_into().unwrap());
        *offset=end;value
    };
    let count=u32_at(&mut offset) as usize;let mut entries=Vec::with_capacity(count);
    for _ in 0..count{
        let id=u32_at(&mut offset);let n=u32_at(&mut offset) as usize;
        let end=offset.checked_add(n).expect("token length overflow");
        entries.push((id,bytes.get(offset..end).expect("truncated token").to_vec()));offset=end;
    }
    assert_eq!(offset,bytes.len(),"vocab trailing bytes");Vocab::new(entries)
}

fn main(){
    let args=std::env::args().collect::<Vec<_>>();
    assert!(matches!(args.len(),4|5),"component-candidates COMPONENT.bin VOCAB.bin OUTPUT.json [PREPARED_COMPONENT.bin]");
    let bytes=fs::read(&args[1]).expect("read one component");
    let vocab=read_vocab(Path::new(&args[2]));
    let started=Instant::now();let mut component=Constraint::load_with_vocab(&bytes,&vocab).expect("load one component");
    let load_ms=started.elapsed().as_secs_f64()*1000.0;
    let started=Instant::now();
    if args.len()==5 { component.prepare_for_composition(&vocab).expect("prepare one component"); }
    let ids=debug_component_boundary_candidate_ids(&component,&vocab).expect("component outgoing certificate");
    let prepare_ms=started.elapsed().as_secs_f64()*1000.0;
    let started=Instant::now();
    let saved=if args.len()==5 {
        let saved=component.save();fs::write(&args[4],&saved).expect("write prepared component");Some(saved)
    }else{None};
    let save_ms=started.elapsed().as_secs_f64()*1000.0;
    let record=serde_json::json!({"schema":"glrmask.component-outgoing-tokens.v1",
        "component_file":args[1],"component_blake3":blake3::hash(&bytes).to_hex().to_string(),
        "vocab_file":args[2],"vocab_blake3":blake3::hash(&fs::read(&args[2]).unwrap()).to_hex().to_string(),
        "raw_states":component.num_tokenizer_states(),"terminals":component.num_terminals(),
        "load_ms":load_ms,"prepare_ms":prepare_ms,"save_ms":save_ms,
        "prepared_bytes":saved.as_ref().map(Vec::len),
        "prepared_blake3":saved.as_ref().map(|b|blake3::hash(b).to_hex().to_string()),"ids":ids});
    fs::write(&args[3],serde_json::to_vec(&record).unwrap()).expect("write component-only descriptor");
    println!("COMPONENT_OUTGOING states={} terminals={} tokens={} load_ms={load_ms:.3} prepare_ms={prepare_ms:.3}",
        component.num_tokenizer_states(),component.num_terminals(),record["ids"].as_array().unwrap().len());
}
