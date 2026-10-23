//! Matched public-API setup/scaling probe. Not a model-token mask benchmark.
//!
//! The small Apply implementations let the *same source* measure both the old
//! return-new bind contract and the mutable contract. They compile to assignment
//! for the old API and to a no-op for the new API; no runtime version detection.
use glrmask::{BuildOptions, Grammar, MaskStrategy, PreparedGrammar, TokenSet, Vocab};
use rayon::prelude::*;
use serde_json::json;
use std::{error::Error, hint::black_box, time::Instant};

trait Apply<G> { fn apply(self, receiver: &mut G); }
impl Apply<Grammar> for Grammar {
    fn apply(self, receiver: &mut Grammar) { *receiver = self; }
}
impl Apply<Grammar> for () { fn apply(self, _: &mut Grammar) {} }
impl Apply<PreparedGrammar> for PreparedGrammar {
    fn apply(self, receiver: &mut PreparedGrammar) { *receiver = self; }
}
impl Apply<PreparedGrammar> for () { fn apply(self, _: &mut PreparedGrammar) {} }

fn source(slots: usize) -> Grammar {
    let mut text = (0..slots).map(|i| format!("extern token E{i}\n")).collect::<String>();
    text.push_str("start root nt root=E0");
    Grammar::from_glrm(text).unwrap()
}

fn bind_source(g: &mut Grammar, name: &str, id: u32) -> glrmask::Result<()> {
    g.bind(name, TokenSet::from(id))?.apply(g);
    Ok(())
}
fn bind_prepared(g: &mut PreparedGrammar, name: &str, id: u32) -> glrmask::Result<()> {
    g.bind(name, TokenSet::from(id))?.apply(g);
    Ok(())
}

fn measure<F>(label: &str, slots: usize, workers: usize, jobs_per_worker: usize, f: F)
    -> Result<serde_json::Value, Box<dyn Error>>
where F: Fn() -> glrmask::Result<()> + Send + Sync {
    let pool = rayon::ThreadPoolBuilder::new().num_threads(workers).build()?;
    pool.install(|| (0..workers).into_par_iter().try_for_each(|_| f()))?;
    let mut elapsed = Vec::new();
    for _ in 0..5 {
        let start = Instant::now();
        pool.install(|| (0..workers).into_par_iter().try_for_each(|_| {
            for _ in 0..jobs_per_worker { f()?; }
            Ok::<_, glrmask::Error>(())
        }))?;
        elapsed.push(start.elapsed().as_secs_f64());
    }
    elapsed.sort_by(f64::total_cmp);
    let jobs = workers * jobs_per_worker;
    let median = elapsed[2];
    eprintln!("{label}: slots={slots} workers={workers} jobs={jobs} median={median:.6}s");
    Ok(json!({"case":label,"slots":slots,"workers":workers,"jobs_per_trial":jobs,
        "jobs_per_worker":jobs_per_worker,"wall_seconds":elapsed,"median_seconds":median,
        "throughput_jobs_per_second":jobs as f64/median,
        "aggregate_ns_per_job":median*1e9/jobs as f64}))
}

fn main() -> Result<(), Box<dyn Error>> {
    assert!(!cfg!(debug_assertions), "use canonical release for performance measurements");
    let mut args = std::env::args().skip(1);
    let largest: usize = args.next().unwrap_or_else(|| "16".into()).parse()?;
    let jobs: usize = args.next().unwrap_or_else(|| "64".into()).parse()?;
    assert!(largest > 0 && jobs > 0);
    let v = Vocab::new(vec![(0,b"a".to_vec()),(1,b"b".to_vec()),(2,Vec::new())]);
    let options = BuildOptions::default().optimization(MaskStrategy::TokenTrie);
    let workers = [1,2,4,8,16,32,64,128].into_iter().filter(|n| *n<=largest).collect::<Vec<_>>();
    let mut rows = Vec::new();
    for slots in [1,32,128,512] {
        let g = source(slots);
        let p = g.prepare(&v)?;
        let names = (0..slots).map(|i|format!("E{i}")).collect::<Vec<_>>();
        for &threads in &workers {
            rows.push(measure("source_clone_one_bind", slots, threads, jobs, || {
                let mut request = g.clone(); bind_source(&mut request,"E0",2)?;black_box(request);Ok(())
            })?);
            rows.push(measure("prepared_clone_one_bind", slots, threads, jobs, || {
                let mut request = p.clone();bind_prepared(&mut request,"E0",2)?;black_box(request);Ok(())
            })?);
            rows.push(measure("requirements", slots, threads, jobs, || {
                black_box(p.requirements()?);Ok(())
            })?);
        }
        rows.push(measure("source_bind_all", slots, 1, (jobs/8).max(1), || {
            let mut request = g.clone();
            for name in &names {bind_source(&mut request,name,2)?;}
            assert!(request.requirements()?.is_empty());black_box(request);Ok(())
        })?);
    }
    let tiny=Grammar::from_glrm("start root nt root=\"ab\"")?;
    for &threads in &workers {
        rows.push(measure("independent_tiny_compile", 0, threads, (jobs/4).max(1), || {
            let c=tiny.compile_with_build_options(&v,options.clone())?;
            let mut s=c.start();assert!(s.commit_token(0));assert!(s.commit_token(1));
            assert!(s.is_accepting());black_box(c);Ok(())
        })?);
    }
    let mut p=Grammar::from_glrm("extern token A extern token B extern token C extern token D start root nt root=A B C D")?.prepare(&v)?;
    for name in ["A","B","C","D"] {bind_prepared(&mut p,name,2)?;}
    for &threads in &workers {
        rows.push(measure("relink_four_events",4,threads,(jobs/4).max(1),|| {
            let c=p.link_with(options.clone())?;let mut s=c.start();
            for _ in 0..4{assert!(s.commit_token(2));}assert!(s.is_accepting());black_box(c);Ok(())
        })?);
    }
    println!("{}",serde_json::to_string_pretty(&json!({"kind":"glrm-core-batch-native-scaling-v1",
        "profile":"canonical cargo release","available_logical_parallelism":std::thread::available_parallelism()?.get(),
        "notes":["Persistent caller-owned Rayon pool per measured case; pool creation excluded.",
            "Weak scaling: fixed jobs per worker; five trials, median wall time.",
            "Setup/requirements/compile/link, not token-mask latency. Hardware oversubscription is reported, not extra physical cores.",
            "Same benchmark source supports return-new and mutable bind APIs."],"rows":rows}))?);
    Ok(())
}
