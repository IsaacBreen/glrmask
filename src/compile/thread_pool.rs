//! Compile-thread-pool selection.
//!
//! The compile graph has several naturally parallel phase groups.  This module
//! owns the optional private rayon pool used to keep those groups from
//! accidentally oversubscribing platforms where the default rayon pool is too
//! large for this workload.

use once_cell::sync::Lazy;

use crate::compile::options::compile_thread_count;

static COMPILE_THREAD_POOL: Lazy<Option<rayon::ThreadPool>> = Lazy::new(|| {
    let thread_count = compile_thread_count()?;
    rayon::ThreadPoolBuilder::new()
        .num_threads(thread_count)
        .build()
        .ok()
});

pub(crate) fn run_with_compile_thread_pool<F, R>(f: F) -> R
where
    F: FnOnce() -> R + Send,
    R: Send,
{
    if let Some(pool) = &*COMPILE_THREAD_POOL {
        pool.install(f)
    } else {
        f()
    }
}
