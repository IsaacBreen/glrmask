//! Environment-variable helpers. New environment reads should go through this module.

pub(crate) fn flag(name: &str) -> bool { std::env::var_os(name).is_some() }
pub(crate) fn truthy(name: &str) -> Option<bool> { let value=std::env::var(name).ok()?; match value.trim().to_ascii_lowercase().as_str() { "1"|"true"|"yes"|"on"=>Some(true), "0"|"false"|"no"|"off"=>Some(false), _=>None } }
pub(crate) fn usize_var(name: &str) -> Option<usize> { std::env::var(name).ok()?.trim().parse().ok() }
pub(crate) fn u64_var(name: &str) -> Option<u64> { std::env::var(name).ok()?.trim().parse().ok() }
pub(crate) fn string_var(name: &str) -> Option<String> { std::env::var(name).ok() }
