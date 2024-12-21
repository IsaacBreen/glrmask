#[macro_export]
macro_rules! debug {
    ($level:expr, $($arg:tt)*) => {{
        pub const DEBUG_LEVEL: usize = 1;
        if $level <= DEBUG_LEVEL {
            #[cfg(feature = "debug")]
            println!("[DEBUG {}] {}", $level, format!($($arg)*));
        }
    }};
}