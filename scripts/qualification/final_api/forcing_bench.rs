//! Identical external harness for canonical forcing before/after the Module API.
//! Link each copy against the corresponding production-release glrmask rlib.
use glrmask::{BuildOptions, Grammar, Optimization, Token, Vocab};
use std::{hint::black_box, time::Instant};

fn source(case: usize) -> Grammar<'static> {
    match case {
        0 | 2 => Grammar::from_ebnf("start ::= \"abcdefghijklmnopqrstuvwx\""),
        1 => Grammar::from_ebnf("start ::= \"ab\""),
        3 => {
            let mut g=Grammar::from_glrm("extern token EVENT start root nt root=EVENT{24}");
            g.bind("EVENT",TokenSet::from(1000)).unwrap();g
        },
        4 => Grammar::from_ebnf("start ::= \"a\" | \"ab\""),
        5 => Grammar::from_glrm("start root nt root=@mark(7) \"ab\" @mark(9)"),
        _ => unreachable!(),
    }
}
fn vocabulary(case:usize) -> Vocab {
    let mut entries=(0..256).map(|n|(n,vec![n as u8])).collect::<Vec<_>>();
    entries.push((1000,Vec::new()));
    if case==2 {
        let text=b"abcdefghijklmnopqrstuvwx";
        for i in 0..text.len() {for n in 2..=6 {
            if i+n<=text.len() {entries.push((1100+entries.len() as u32,text[i..i+n].to_vec()));}
        }}
    }
    Vocab::new(entries)
}
fn quantile(xs:&[u128],a:usize,b:usize)->u128 {xs[(xs.len()*a).div_ceil(b).saturating_sub(1)]}
fn main() -> Result<(),Box<dyn std::error::Error>> {
    assert!(!cfg!(debug_assertions),"link this harness with production release library");
    let reps=std::env::args().nth(1).map(|s|s.parse::<usize>()).transpose()?.unwrap_or(2000);
    assert!(reps>=20);
    let names=["chain24_single_ids","two_chars","chain24_multi_spellings","exact_event24","accepting_prefix","passive_markers"];
    print!("{{\"semantics\":\"canonical-forced-output\",\"repetitions\":{reps},\"rows\":[");
    let mut first=true;
    for mode in [Optimization::Auto,Optimization::FastBuild,Optimization::Balanced,Optimization::FastRuntime] {
        for (case,name) in names.iter().enumerate() {
            let vocab=vocabulary(case);let grammar=source(case);
            let constraint=grammar.compile_with(&vocab,BuildOptions::default().optimization(mode))?;
            let reference=constraint.start().forced_tokens()?;
            for warm in [false,true] {
                for _ in 0..20 {black_box(constraint.start().forced_tokens()?);}
                let mut times=Vec::with_capacity(reps);
                for _ in 0..reps {
                    let state=constraint.start();
                    if warm {black_box(state.mask());}
                    let accepting=state.is_accepting();let rejected=state.is_rejected();
                    let events=state.marker_events().to_vec();
                    let t=Instant::now();let ids=black_box(state.forced_tokens()?);times.push(t.elapsed().as_nanos());
                    assert_eq!(ids,reference);assert_eq!(state.is_accepting(),accepting);
                    assert_eq!(state.is_rejected(),rejected);assert_eq!(state.marker_events(),events);
                }
                times.sort_unstable();let sum:u128=times.iter().sum();
                if !first {print!(",");}first=false;
                print!("{{\"case\":\"{name}\",\"mode\":\"{mode:?}\",\"warm_mask\":{warm},\"ids\":{reference:?},\"samples\":{reps},\"p50_ns\":{},\"p90_ns\":{},\"p95_ns\":{},\"p99_ns\":{},\"p999_ns\":{},\"p9999_ns\":{},\"max_ns\":{},\"mean_ns\":{}}}",
                    quantile(&times,1,2),quantile(&times,9,10),quantile(&times,95,100),quantile(&times,99,100),quantile(&times,999,1000),quantile(&times,9999,10000),times[times.len()-1],sum as f64/reps as f64);
            }
        }
    }
    println!("]}}");Ok(())
}
