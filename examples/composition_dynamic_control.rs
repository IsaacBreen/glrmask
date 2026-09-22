//! Same-language ordinary-vs-recursive dynamic mask and build control.
//! Usage: composition_dynamic_control LLAMA_JSON OUTPUT_STEM [samples=101]
//! Run with GLRMASK_DISABLE_DYNAMIC_MASK_CACHE=1 for recomputation timings.
//! Build, prefix setup, token choice and validation are outside mask/commit
//! timers. Dynamic compilation and each link are reported separately.
use glrmask::{DynamicConstraint, Grammar, Vocab};
use std::{collections::BTreeMap, fs, hint::black_box, time::Instant};

fn stats(values: &[u64]) -> serde_json::Value {
    let mut v=values.to_vec(); v.sort_unstable();
    let p=|q:f64| v[((v.len()-1) as f64*q).round() as usize] as f64/1000.;
    serde_json::json!({"n":v.len(),"p50_us":p(0.5),"p90_us":p(0.9),
        "p99_us":p(0.99),"p100_us":p(1.0)})
}

fn main() -> Result<(),Box<dyn std::error::Error>> {
    let args:Vec<_>=std::env::args().collect();
    assert!(args.len()>=3,"LLAMA_JSON OUTPUT_STEM [samples]");
    let repeats:usize=args.get(3).map(|s|s.parse()).transpose()?.unwrap_or(101);
    assert!(repeats>0);
    let encoded:BTreeMap<String,String>=serde_json::from_slice(&fs::read(&args[1])?)?;
    let entries=encoded.into_iter().map(|(id,hex)| {
        let bytes=(0..hex.len()).step_by(2).map(|i|u8::from_str_radix(&hex[i..i+2],16).unwrap()).collect();
        (id.parse().unwrap(),bytes)
    }).collect();
    let vocab=Vocab::new(entries);
    let mut rows=Vec::<serde_json::Value>::new();
    let mut builds=Vec::<serde_json::Value>::new();
    let case_filter=std::env::var("CONTROL_CASE").ok();
    let middle=r#"glrm 1; start middle; extern grammar leaf; nt middle = "[" leaf "]";"#;
    let outer=r#"glrm 1; start outer; extern grammar middle; nt outer = "X" middle "!";"#;
    for (name,leaf,flat,prefixes) in [
        ("literal",r#"glrm 1; start leaf; nt leaf = "a";"#,
         r#"glrm 1; start document; nt document = "X" "[" "a" "]" "!";"#,
         vec!["","X","X[","X[a","X[a]","X[a]!"]),
        ("text",r#"glrm 1; start leaf; t TEXT = /[a-zA-Z0-9 ]{1,64}/; nt leaf = TEXT;"#,
         r#"glrm 1; start document; t TEXT = /[a-zA-Z0-9 ]{1,64}/; nt document = "X" "[" TEXT "]" "!";"#,
         vec!["","X","X[","X[hello","X[hello]","X[hello]!"]),
        ("nullable",r#"glrm 1; start leaf; nt leaf = ("a")?;"#,
         r#"glrm 1; start document; nt document = "X" "[" ("a")? "]" "!";"#,
         vec!["","X","X[","X[]","X[a","X[a]","X[a]!","X[]!"]),
    ] {
        if case_filter.as_deref().is_some_and(|filter|filter!=name) { continue; }
        let mut built=None;
        for repeat in 0..5 {
            eprintln!("BUILD_STAGE case={name} repeat={repeat} stage=components");
            let t=Instant::now(); let l=DynamicConstraint::compile(Grammar::glrm(leaf),&vocab)?; let leaf_ns=t.elapsed().as_nanos() as u64;
            let t=Instant::now(); let m=DynamicConstraint::compile(Grammar::glrm(middle),&vocab)?; let middle_ns=t.elapsed().as_nanos() as u64;
            let t=Instant::now(); let o=DynamicConstraint::compile(Grammar::glrm(outer),&vocab)?; let outer_ns=t.elapsed().as_nanos() as u64;
            eprintln!("BUILD_STAGE case={name} repeat={repeat} stage=inner_link");
            let t=Instant::now(); let m=m.bind_grammar_dynamic_boundary("leaf",l)?; let inner_link_ns=t.elapsed().as_nanos() as u64;
            eprintln!("BUILD_STAGE case={name} repeat={repeat} stage=outer_link inner_link_ns={inner_link_ns}");
            let t=Instant::now(); let c=o.bind_grammar_dynamic_boundary("middle",m)?; let outer_link_ns=t.elapsed().as_nanos() as u64;
            eprintln!("BUILD_STAGE case={name} repeat={repeat} stage=ordinary outer_link_ns={outer_link_ns}");
            let t=Instant::now(); let f=DynamicConstraint::compile(Grammar::glrm(flat),&vocab)?; let flat_ns=t.elapsed().as_nanos() as u64;
            builds.push(serde_json::json!({"case":name,"repeat":repeat,"leaf_ns":leaf_ns,"middle_ns":middle_ns,"outer_ns":outer_ns,
                "inner_link_ns":inner_link_ns,"outer_link_ns":outer_link_ns,"flat_ns":flat_ns}));
            fs::write(format!("{}.builds.json",args[2]),serde_json::to_vec_pretty(&builds)?)?;
            built=Some((c,f));
        }
        let (composed,ordinary)=built.unwrap();
        for prefix in prefixes {
            let mut a=composed.start(); a.commit_bytes(prefix.as_bytes())?;
            let mut b=ordinary.start(); b.commit_bytes(prefix.as_bytes())?;
            let expected=a.mask(); assert_eq!(expected,b.mask(),"{name} at {prefix:?}");
            assert_eq!(a.is_accepting(),b.is_accepting());
            let token=expected.iter().enumerate().find_map(|(i,w)|(*w!=0).then(||i as u32*32+w.trailing_zeros()));
            for (backend,c) in [("composed",&composed),("ordinary",&ordinary)] {
                let mut masks=Vec::new();let mut commits=Vec::new();let mut tbms=Vec::new();
                for rep in 0..repeats+4 {
                    let mut s=c.start(); s.commit_bytes(prefix.as_bytes())?;
                    let mut output=vec![0;expected.len()];
                    let t=Instant::now();s.fill_mask(black_box(&mut output));let mask_ns=t.elapsed().as_nanos() as u64;
                    assert_eq!(output,expected,"{name}/{backend}/{prefix:?}");
                    let commit_ns=if let Some(tid)=token {let t=Instant::now();s.commit_token(black_box(tid))?;t.elapsed().as_nanos() as u64} else {0};
                    if rep>=4 {masks.push(mask_ns);commits.push(commit_ns);tbms.push(mask_ns+commit_ns);}
                }
                rows.push(serde_json::json!({"case":name,"backend":backend,"prefix":prefix,"mask":stats(&masks),"commit":stats(&commits),"tbm":stats(&tbms),
                    "mask_ns":masks,"commit_ns":commits,"has_commit":token.is_some()}));
                eprintln!("CONTROL {name} {backend} {prefix:?} {}",rows.last().unwrap()["mask"]);
                fs::write(format!("{}.partial.json",args[2]),serde_json::to_vec_pretty(&serde_json::json!({
                    "valid":false,"partial":true,"builds":builds,"samples":rows}))?)?;
            }
        }
    }
    let result=serde_json::json!({"valid":true,"cache_disabled_env":std::env::var("GLRMASK_DISABLE_DYNAMIC_MASK_CACHE").ok(),
        "vocab":args[1],"builds":builds,"samples":rows,"p100_scope":"observed on these exact prefixes and repeats"});
    fs::write(format!("{}.json",args[2]),serde_json::to_vec_pretty(&result)?)?;
    Ok(())
}
