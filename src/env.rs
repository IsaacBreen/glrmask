//! Configuration reads for the process.
//!
//! GLRMask reads a large number of tuning and diagnostic variables from the
//! environment, including on the mask and commit hot paths. Reading the
//! process environment allocates and locks, so every `GLRMASK_*` and
//! `O21137_*` name is served from a snapshot taken once per process. Other
//! names, such as `RAYON_NUM_THREADS`, are not GLRMask configuration and are
//! forwarded to [`std::env`] unchanged.

use std::collections::HashMap;
use std::ffi::{OsStr, OsString};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{LazyLock, Mutex, OnceLock};

/// Names observed by GLRMask itself, captured on first use.
static SNAPSHOT: OnceLock<HashMap<OsString, OsString>> = OnceLock::new();

/// Explicit changes layered over the snapshot, used by tests and tooling.
static OVERRIDES: LazyLock<Mutex<HashMap<OsString, Option<OsString>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Avoids taking the override lock while no override exists.
static HAS_OVERRIDES: AtomicBool = AtomicBool::new(false);

fn is_cached(key: &OsStr) -> bool {
    let key = key.as_encoded_bytes();
    key.starts_with(b"GLRMASK_") || key.starts_with(b"O21137_")
}

fn overrides() -> std::sync::MutexGuard<'static, HashMap<OsString, Option<OsString>>> {
    OVERRIDES.lock().unwrap_or_else(|error| error.into_inner())
}

fn snapshot() -> &'static HashMap<OsString, OsString> {
    SNAPSHOT.get_or_init(|| {
        std::env::vars_os()
            .filter(|(key, _)| is_cached(key))
            .collect()
    })
}

fn resolve(
    key: &OsStr,
    snapshot: &HashMap<OsString, OsString>,
    overrides: Option<&HashMap<OsString, Option<OsString>>>,
) -> Option<OsString> {
    if let Some(overrides) = overrides {
        if let Some(value) = overrides.get(key) {
            return value.clone();
        }
    }
    snapshot.get(key).cloned()
}

/// Reads a configuration variable, exactly like [`std::env::var_os`].
///
/// GLRMask's own names are answered from the process snapshot, so a value can
/// only change through [`std::env::set_var`] before the first read or through
/// the `__private` helper that records an override.
pub(crate) fn var_os<K: AsRef<OsStr>>(key: K) -> Option<OsString> {
    let key = key.as_ref();
    if !is_cached(key) {
        return std::env::var_os(key);
    }
    if HAS_OVERRIDES.load(Ordering::Relaxed) {
        return resolve(key, snapshot(), Some(&overrides()));
    }
    resolve(key, snapshot(), None)
}

/// Reads a configuration variable, exactly like [`std::env::var`].
pub(crate) fn var<K: AsRef<OsStr>>(key: K) -> Result<String, std::env::VarError> {
    match var_os(key) {
        Some(value) => value.into_string().map_err(std::env::VarError::NotUnicode),
        None => Err(std::env::VarError::NotPresent),
    }
}

/// Records a value that [`var_os`] must observe even after the snapshot.
pub(crate) fn record_override(key: &OsStr, value: Option<OsString>) {
    overrides().insert(key.to_os_string(), value);
    HAS_OVERRIDES.store(true, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn override_is_seen() {
        let key = "GLRMASK_ENV_TEST_OVERRIDE";
        unsafe { crate::__private::set_env_var(key, Some("override")) };
        assert_eq!(var_os(key), Some(OsString::from("override")));
        assert_eq!(var(key).as_deref(), Ok("override"));
        unsafe { crate::__private::set_env_var(key, None) };
        assert_eq!(var_os(key), None);
    }

    #[test]
    fn removal_override_hides_snapshot_value() {
        let key = OsString::from("GLRMASK_ENV_TEST_SNAPSHOT_VALUE");
        let snapshot = HashMap::from([(key.clone(), OsString::from("snapshot"))]);
        let mut overrides = HashMap::new();
        assert_eq!(
            resolve(&key, &snapshot, Some(&overrides)),
            Some(OsString::from("snapshot"))
        );
        overrides.insert(key.clone(), None);
        assert_eq!(resolve(&key, &snapshot, Some(&overrides)), None);
        assert_eq!(resolve(&key, &snapshot, None), Some(OsString::from("snapshot")));
    }

    #[test]
    fn unprefixed_name_reads_live_environment() {
        let key = "ENV_TEST_UNPREFIXED";
        unsafe { std::env::set_var(key, "live") };
        assert_eq!(var_os(key), Some(OsString::from("live")));
        assert_eq!(var(key).as_deref(), Ok("live"));
        unsafe { std::env::remove_var(key) };
        assert_eq!(var_os(key), None);
    }
}
