//! Independent liveness probe for caller-owned Rayon + cold shared event cache.
//! Compile externally against a frozen/verified crate; not a speed benchmark.
use glrmask::{BuildOptions,Grammar,Optimization,PreparedGrammar,Vocab};
use rayon::prelude::*;
use std::time::Instant;

fn configured(template:&PreparedGrammar, value:glrmask::ExactTokens)->PreparedGrammar {
    #[cfg(candidate)] {
        let mut instance=template.clone();instance.bind("E",value).unwrap();instance
    }
    #[cfg(not(candidate))] {template.bind("E",value).unwrap()}
}
fn main() {
    let argv:Vec<_>=std::env::args().collect();
    let threads:usize=argv.get(1).map(|s|s.parse().unwrap()).unwrap_or(2);
    let repeats:usize=argv.get(2).map(|s|s.parse().unwrap()).unwrap_or(16);
    let mut words=(0..256).map(|id|(id,vec![id as u8])).collect::<Vec<_>>();
    while words.len()<4096 {let id=words.len() as u32;words.push((id,format!("token{id:06}").into_bytes()));}
    let v=Vocab::new(words);
    let g=Grammar::from_glrm("extern token E start root nt root=E");
    let selection=v.tokens((0..128).map(|i|2*i+1)).unwrap();
    let pool=rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
    for round in 0..repeats {
        // A fresh prepared definition gives a cold adapter cache each round.
        let template=g.prepare(&v).unwrap();
        let started=Instant::now();eprintln!("BEGIN threads={threads} round={round}");
        pool.install(|| (0..threads*8).into_par_iter().for_each(|_|{
            let instance=configured(&template,selection.clone());
            let c=instance.link_with(BuildOptions::default().optimization(Optimization::FastBuild)).unwrap();
            let mut state=c.start();let mask=state.mask();
            assert_eq!(mask.iter().map(|w|w.count_ones() as usize).sum::<usize>(),128);
            state.commit_token(1).unwrap();assert!(state.is_accepting());
        }));
        eprintln!("DONE threads={threads} round={round} ms={:.3}",started.elapsed().as_secs_f64()*1000.0);
    }
}
