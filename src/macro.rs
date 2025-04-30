/// Macro for creating a sequence of parsers
#[macro_export]
macro_rules! seq_fast {
    ($($x:expr),* $(,)?) => {
        $crate::interface::tokenizer_combinators::seq_fast(vec![$($x),*])
    };
}

/// Macro for creating a choice of parsers
#[macro_export]
macro_rules! choice_fast {
    ($($x:expr),* $(,)?) => {
        $crate::interface::tokenizer_combinators::choice_fast(vec![$($x),*])
    };
}

#[macro_export]
macro_rules! debug {
    ($level:expr, $fmt:literal $(, $($arg:tt)*)?) => {{
        const MACRO_DEBUG_LEVEL: usize = 3; // Replace with crate::DEBUG_LEVEL or similar

        if $level <= MACRO_DEBUG_LEVEL {
            // #[cfg(feature = "debug")] // Enable this line if using a feature flag
            println!(concat!("[DEBUG {}] ", $fmt), $level $(, $($arg)*)?);
        }
    }};

    ($level:expr, $msg:expr) => {{
        const MACRO_DEBUG_LEVEL: usize = 3; // Replace with crate::DEBUG_LEVEL or similar

        if $level <= MACRO_DEBUG_LEVEL {
            // #[cfg(feature = "debug")] // Enable this line if using a feature flag
            println!("[DEBUG {}] {:?}", $level, $msg);
        }
    }};
}
