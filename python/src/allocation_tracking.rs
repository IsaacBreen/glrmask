use std::alloc::{GlobalAlloc, Layout};
use std::cell::Cell;
use std::time::Instant;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct AllocationStats {
    pub(crate) alloc_calls: u64,
    pub(crate) alloc_zeroed_calls: u64,
    pub(crate) realloc_calls: u64,
    pub(crate) dealloc_calls: u64,
    pub(crate) allocated_bytes: u64,
    pub(crate) reallocated_bytes: u64,
    pub(crate) deallocated_bytes: u64,
    pub(crate) alloc_ns: u64,
    pub(crate) max_alloc_ns: u64,
    pub(crate) realloc_ns: u64,
    pub(crate) max_realloc_ns: u64,
    pub(crate) dealloc_ns: u64,
    pub(crate) max_dealloc_ns: u64,
}

thread_local! {
    static ENABLED: Cell<bool> = const { Cell::new(false) };
    static STATS: Cell<AllocationStats> = const { Cell::new(AllocationStats {
        alloc_calls: 0,
        alloc_zeroed_calls: 0,
        realloc_calls: 0,
        dealloc_calls: 0,
        allocated_bytes: 0,
        reallocated_bytes: 0,
        deallocated_bytes: 0,
        alloc_ns: 0,
        max_alloc_ns: 0,
        realloc_ns: 0,
        max_realloc_ns: 0,
        dealloc_ns: 0,
        max_dealloc_ns: 0,
    }) };
}

#[inline]
fn tracking_enabled() -> bool {
    ENABLED.with(Cell::get)
}

#[inline]
fn elapsed_ns(start: Option<Instant>) -> u64 {
    start.map_or(0, |started| {
        started.elapsed().as_nanos().min(u64::MAX as u128) as u64
    })
}

#[inline]
fn update(f: impl FnOnce(&mut AllocationStats)) {
    ENABLED.with(|enabled| {
        if !enabled.get() {
            return;
        }
        STATS.with(|stats| {
            let mut value = stats.get();
            f(&mut value);
            stats.set(value);
        });
    });
}

pub(crate) struct TrackingAllocator(pub(crate) mimalloc::MiMalloc);

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let started = tracking_enabled().then(Instant::now);
        let ptr = unsafe { self.0.alloc(layout) };
        let duration_ns = elapsed_ns(started);
        update(|stats| {
            stats.alloc_calls += 1;
            stats.allocated_bytes += layout.size() as u64;
            stats.alloc_ns += duration_ns;
            stats.max_alloc_ns = stats.max_alloc_ns.max(duration_ns);
        });
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let started = tracking_enabled().then(Instant::now);
        let ptr = unsafe { self.0.alloc_zeroed(layout) };
        let duration_ns = elapsed_ns(started);
        update(|stats| {
            stats.alloc_zeroed_calls += 1;
            stats.allocated_bytes += layout.size() as u64;
            stats.alloc_ns += duration_ns;
            stats.max_alloc_ns = stats.max_alloc_ns.max(duration_ns);
        });
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let started = tracking_enabled().then(Instant::now);
        unsafe { self.0.dealloc(ptr, layout) };
        let duration_ns = elapsed_ns(started);
        update(|stats| {
            stats.dealloc_calls += 1;
            stats.deallocated_bytes += layout.size() as u64;
            stats.dealloc_ns += duration_ns;
            stats.max_dealloc_ns = stats.max_dealloc_ns.max(duration_ns);
        });
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let started = tracking_enabled().then(Instant::now);
        let result = unsafe { self.0.realloc(ptr, layout, new_size) };
        let duration_ns = elapsed_ns(started);
        update(|stats| {
            stats.realloc_calls += 1;
            stats.reallocated_bytes += new_size as u64;
            stats.realloc_ns += duration_ns;
            stats.max_realloc_ns = stats.max_realloc_ns.max(duration_ns);
        });
        result
    }
}

pub(crate) fn measure<T>(operation: impl FnOnce() -> T) -> (T, AllocationStats) {
    STATS.with(|stats| stats.set(AllocationStats::default()));
    ENABLED.with(|enabled| enabled.set(true));
    let result = operation();
    ENABLED.with(|enabled| enabled.set(false));
    let stats = STATS.with(Cell::get);
    (result, stats)
}
