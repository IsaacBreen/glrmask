pub(crate) const DEEP_STACK_RED_ZONE: usize = 4 * 1024 * 1024;
pub(crate) const DEEP_STACK_SIZE: usize = 16 * 1024 * 1024;

/// Runs `work` with at least `DEEP_STACK_RED_ZONE` bytes of stack, growing onto a
/// fresh `DEEP_STACK_SIZE` segment when the current thread is short.
///
/// Public entry points that compile, prepare, load or save call this. They may be
/// called from threads with small stacks (Python worker threads, default 2 MiB
/// rayon workers), and rayon work stealing can stack several such calls on one
/// thread, so each entry point checks again.
pub(crate) fn with_deep_stack<T>(work: impl FnOnce() -> T) -> T {
    stacker::maybe_grow(DEEP_STACK_RED_ZONE, DEEP_STACK_SIZE, work)
}
