//! Full-vocabulary recursive epsilon/virtual mask control over saved artifacts.
//! Usage: composition_virtual_control VOCAB_JSON CACHE_DIR [samples=11]
//! First run creates the fixtures. Later binaries load the same bytes.
use glrmask::{Constraint, Grammar, Vocab};
use std::{collections::BTreeMap, fs, hint::black_box, path::Path, time::Instant};

fn percentile(v: &mut [u64], q: f64) -> f64 {
    v.sort_unstable(); v[((v.len()-1) as f64*q).round() as usize] as f64/1000.
}
fn main() -> Result<(),Box<dyn std::error::Error>> {
    let args:Vec<_>=std::env::args().collect();
    assert!(args.len()>=3,"VOCAB_JSON CACHE_DIR [samples]");
    let repeats:usize=args.get(3).map(|s|s.parse()).transpose()?.unwrap_or(11); assert!(repeats>0);
    let input:BTreeMap<String,String>=serde_json::from_slice(&fs::read(&args[1])?)?;
    let entries=input.into_iter().map(|(id,hex)| {
        let bytes=(0..hex.len()).step_by(2).map(|i|u8::from_str_radix(&hex[i..i+2],16).unwrap()).collect();
        (id.parse().unwrap(),bytes)
    }).collect();
    let vocab=Vocab::new(entries); let cache=Path::new(&args[2]); fs::create_dir_all(cache)?;
    println!("case,prefix,repeats,p50_us,p90_us,p99_us,p100_us,allowed,mask_hash");
    for (name, source, json, prefixes) in [
        ("virtual_uri",r#"{"type":"string","format":"uri","minLength":1,"maxLength":5000}"#,true,
         vec!["","X","X\"","X\"x:","X\"x:hello","X\"x:hello\"","X\"x:hello\"!"]),
        ("epsilon",r#"start payload; lexer group a ::= A; lexer group b ::= B; t A ::= "a"+; t B ::= "ab"+; nt payload ::= A | B;"#,false,
         vec!["","X","Xa","Xab","Xaa","Xab!"]),
    ] {
        let file=cache.join(format!("{name}.bin"));
        if !file.exists() {
            eprintln!("BUILD_BEGIN {name}");
            let start=Instant::now();
            let leaf=Constraint::compile(if json {Grammar::json_schema(source)} else {Grammar::glrm(source)},&vocab)?;
            let parent=Constraint::compile(Grammar::glrm(
                r#"glrm 1; start root; extern grammar payload; nt root = "X" payload "!";"#),&vocab)?;
            let components_ms=start.elapsed().as_secs_f64()*1000.;
            let start=Instant::now();let bound=parent.bind_grammar_dynamic_boundary("payload",leaf)?;
            let link_ms=start.elapsed().as_secs_f64()*1000.;
            fs::write(&file,bound.save())?;
            eprintln!("BUILD_DONE {name} components_ms={components_ms} link_ms={link_ms}");
        }
        let bound=Constraint::load_with_vocab(fs::read(&file)?,&vocab)?;
        for prefix in prefixes {
            eprintln!("MASK_BEGIN {name} {prefix:?}");
            let mut template=bound.start();template.commit_bytes(prefix.as_bytes())?;
            let expected=template.mask();
            let allowed:u32=expected.iter().map(|w|w.count_ones()).sum();
            let hash=expected.iter().fold(0xcbf29ce484222325u64,|h,&w|(h^w as u64).wrapping_mul(0x100000001b3));
            let mut times=Vec::new();
            for _ in 0..repeats {
                let state=template.clone();let mut mask=vec![0;expected.len()];
                let start=Instant::now();state.fill_mask(black_box(&mut mask));
                times.push(start.elapsed().as_nanos() as u64);
                assert_eq!(mask,expected,"unstable mask {name} {prefix:?}");
            }
            let hex=prefix.as_bytes().iter().map(|b|format!("{b:02x}")).collect::<String>();
            println!("{name},{hex},{repeats},{:.3},{:.3},{:.3},{:.3},{allowed},{hash:016x}",
                percentile(&mut times,0.5),percentile(&mut times,0.9),percentile(&mut times,0.99),percentile(&mut times,1.0));
        }
    }
    Ok(())
}
