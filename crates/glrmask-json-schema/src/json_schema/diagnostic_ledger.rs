//! Opt-in diagnostic spans retained in memory. The private profiling runner
//! drains them after stopping its outer import timer, so printing is untimed.
use std::cell::Cell;
use std::sync::{Mutex, OnceLock};
use std::thread::ThreadId;
use std::time::Instant;
use serde::Serialize;

struct RawRecord { phase: &'static str, depth: usize, thread: ThreadId, ns: u64 }
static RECORDS: Mutex<Vec<RawRecord>> = Mutex::new(Vec::new());
thread_local! { static DEPTH: Cell<usize> = const { Cell::new(0) }; }

fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("GLRMASK_DIAG_IMPORT_LEDGER").is_some())
}

pub(crate) struct Timer { phase: &'static str, started: Option<Instant>, depth: usize }
impl Timer {
    pub fn start(phase: &'static str) -> Self {
        if !enabled() { return Self { phase, started: None, depth: 0 }; }
        let depth = DEPTH.with(|d| { let old=d.get(); d.set(old+1); old });
        Self { phase, started: Some(Instant::now()), depth }
    }
}
impl Drop for Timer {
    fn drop(&mut self) {
        if let Some(started) = self.started {
            let ns=started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
            DEPTH.with(|d| { debug_assert_eq!(d.get(),self.depth+1); d.set(self.depth); });
            RECORDS.lock().unwrap().push(RawRecord {
                phase:self.phase, depth:self.depth, thread:std::thread::current().id(), ns,
            });
        }
    }
}

#[derive(Serialize)]
pub struct Record { pub phase: &'static str, pub depth: usize, pub thread: String, pub ns: u64 }

/// Profiling-only API: call after the import has returned, never inside a
/// measured phase. Thread/depth metadata prevents summing overlapping spans.
pub fn drain() -> Vec<Record> {
    let raw=std::mem::take(&mut *RECORDS.lock().unwrap());
    raw.into_iter().map(|r| Record { phase:r.phase, depth:r.depth,
        thread:format!("{:?}",r.thread), ns:r.ns }).collect()
}
