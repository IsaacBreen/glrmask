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
    ($level:expr, $($arg:tt)*) => {{
        pub const DEBUG_LEVEL: usize = 1;
        if $level <= DEBUG_LEVEL {
            #[cfg(feature = "debug")]
            println!("[DEBUG {}] {}", $level, format!($($arg)*));
        }
    }};
}