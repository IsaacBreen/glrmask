pub mod core;
pub mod context;
pub mod coordinator;
pub mod passes;
pub mod optimizer;
pub mod metrics;

// Re-exports to make consumption ergonomic.
pub use core::*;
pub use context::*;
pub use coordinator::*;
pub use passes::*;
pub use optimizer::optimize_trie3_size;
pub use metrics::*;

pub use CoordinatorConfig as Trie3Config;
