//! Internal invariant policy.

#[macro_export]
macro_rules! glrmask_debug_invariant { ($cond:expr, $($arg:tt)+) => { debug_assert!($cond, $($arg)+) }; }
#[macro_export]
macro_rules! glrmask_invariant { ($cond:expr, $($arg:tt)+) => { if !$cond { panic!("glrmask invariant failed: {}", format!($($arg)+)); } }; }
