//! Diagnostic-only capture; callers explicitly enable the ledger.
static ROWS: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
pub(crate) fn record(row: String) {
    ROWS.lock().expect("core ledger lock poisoned").push(row);
}
pub(crate) fn take() -> Vec<String> {
    std::mem::take(&mut *ROWS.lock().expect("core ledger lock poisoned"))
}
