#![deny(warnings)]

#[derive(Debug)]
struct InternalInvariantViolation {
    message: String,
}

fn fail_internal_invariant(message: impl Into<String>) -> ! {
    std::panic::resume_unwind(Box::new(InternalInvariantViolation {
        message: message.into(),
    }))
}

fn catch_internal_invariant_message<T>(f: impl FnOnce() -> T) -> Result<T, String> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(value) => Ok(value),
        Err(payload) => match payload.downcast::<InternalInvariantViolation>() {
            Ok(violation) => Err(violation.message),
            Err(payload) => std::panic::resume_unwind(payload),
        },
    }
}

// Deterministic work budget for template builds. Each top-level template
// attempt owns one handle, and that handle is installed on every thread that
// performs work for that attempt and on no other compile. Charges outside any
// installed handle are no-ops, so TokenTrie and unbudgeted paths are
// unaffected and concurrent compiles cannot corrupt each other.
#[derive(Debug)]
pub struct TemplateBuildBudget {
    units: std::sync::atomic::AtomicU64,
    limit: u64,
}

impl TemplateBuildBudget {
    pub fn new(limit: u64) -> std::sync::Arc<Self> {
        std::sync::Arc::new(Self {
            units: std::sync::atomic::AtomicU64::new(0),
            limit,
        })
    }

    pub fn units(&self) -> u64 {
        self.units.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn limit(&self) -> u64 {
        self.limit
    }

    fn charge(&self, stage: &'static str, units: u64) {
        let total = self
            .units
            .fetch_add(units, std::sync::atomic::Ordering::Relaxed)
            .saturating_add(units);
        if total > self.limit {
            fail_compilation_resource_limit(format!("{stage} build budget exhausted"));
        }
    }
}

thread_local! {
    // The budget of the top-level template attempt currently executing on this
    // thread, or `None` when this thread is not doing budgeted work.
    static TEMPLATE_BUILD_BUDGET_HANDLE:
        std::cell::RefCell<Option<std::sync::Arc<TemplateBuildBudget>>> =
        const { std::cell::RefCell::new(None) };
}

// 10x the largest measured real-grammar peak (671,896 units for o9838 with
// ParserDwa) after the charges were calibrated to inner-loop work. The extra
// headroom above the 10x floor leaves room for grammar-preparation charges on
// the same real grammars.
pub const DEFAULT_TEMPLATE_BUILD_BUDGET: u64 = 10_000_000;

pub fn set_template_build_budget_handle(handle: std::sync::Arc<TemplateBuildBudget>) {
    TEMPLATE_BUILD_BUDGET_HANDLE.with(|slot| *slot.borrow_mut() = Some(handle));
}

pub fn clear_template_build_budget_handle() {
    TEMPLATE_BUILD_BUDGET_HANDLE.with(|slot| *slot.borrow_mut() = None);
}

pub fn template_build_budget_present() -> bool {
    TEMPLATE_BUILD_BUDGET_HANDLE.with(|slot| slot.borrow().is_some())
}

pub fn template_build_budget_units() -> u64 {
    TEMPLATE_BUILD_BUDGET_HANDLE
        .with(|slot| slot.borrow().as_ref().map_or(0, |handle| handle.units()))
}

pub fn charge_template_build_budget(stage: &'static str, units: u64) {
    TEMPLATE_BUILD_BUDGET_HANDLE.with(|slot| {
        if let Some(handle) = slot.borrow().as_ref() {
            handle.charge(stage, units);
        }
    });
}

// Expected resource exhaustion is distinct from an internal invariant failure.
// This payload crosses the existing compiler unwind boundary without catching
// unrelated application panics or returning a partial artifact.
#[derive(Debug)]
struct CompilationResourceLimit { message: String }

fn fail_compilation_resource_limit(message: impl Into<String>) -> ! {
    std::panic::resume_unwind(Box::new(CompilationResourceLimit { message: message.into() }))
}

fn catch_compilation_resource_limit<T>(f: impl FnOnce() -> T) -> Result<T, String> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(value) => Ok(value),
        Err(payload) => match payload.downcast::<CompilationResourceLimit>() {
            Ok(limit) => Err(limit.message),
            Err(payload) => std::panic::resume_unwind(payload),
        },
    }
}

/// Implementation details shared by the GLRMask workspace.
#[doc(hidden)]
pub mod __private {
    pub fn fail_compilation_resource_limit(message: impl Into<String>) -> ! {
        super::fail_compilation_resource_limit(message)
    }

    pub fn catch_compilation_resource_limit<T>(f: impl FnOnce() -> T) -> Result<T, String> {
        super::catch_compilation_resource_limit(f)
    }

    pub fn fail_internal_invariant(message: impl Into<String>) -> ! {
        super::fail_internal_invariant(message)
    }

    pub fn catch_internal_invariant_message<T>(f: impl FnOnce() -> T) -> Result<T, String> {
        super::catch_internal_invariant_message(f)
    }

    pub fn template_build_budget_units() -> u64 {
        super::template_build_budget_units()
    }

    pub fn charge_template_build_budget(stage: &'static str, units: u64) {
        super::charge_template_build_budget(stage, units)
    }

    pub fn set_template_build_budget_handle(handle: std::sync::Arc<super::TemplateBuildBudget>) {
        super::set_template_build_budget_handle(handle)
    }

    pub fn clear_template_build_budget_handle() {
        super::clear_template_build_budget_handle()
    }

    pub fn template_build_budget_present() -> bool {
        super::template_build_budget_present()
    }
}

#[cfg(all(test, feature = "internal-api"))]
mod resource_limit_tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn resource_payload_is_caught_and_success_is_unchanged() {
        assert_eq!(catch_compilation_resource_limit(|| 42), Ok(42));
        let error = catch_compilation_resource_limit(|| -> () {
            fail_compilation_resource_limit("projection budget exhausted")
        }).unwrap_err();
        assert_eq!(error, "projection budget exhausted");
    }

    #[test]
    fn resource_unwind_runs_destructors() {
        struct DropMark<'a>(&'a Cell<bool>);
        impl Drop for DropMark<'_> {
            fn drop(&mut self) { self.0.set(true); }
        }
        let dropped = Cell::new(false);
        let _ = catch_compilation_resource_limit(|| -> () {
            let _mark = DropMark(&dropped);
            fail_compilation_resource_limit("limited")
        });
        assert!(dropped.get());
    }

    #[test]
    fn unrelated_panics_keep_the_original_payload() {
        let outer = std::panic::catch_unwind(|| {
            let _ = catch_compilation_resource_limit(|| -> () {
                std::panic::resume_unwind(Box::new(37_u32));
            });
        }).unwrap_err();
        assert_eq!(*outer.downcast::<u32>().unwrap(), 37);
    }

    #[test]
    fn resource_and_invariant_payloads_remain_distinct() {
        let limit = catch_compilation_resource_limit(|| {
            catch_internal_invariant_message(|| -> () {
                fail_compilation_resource_limit("limited")
            })
        });
        assert_eq!(limit, Err("limited".to_owned()));
        let invariant = catch_compilation_resource_limit(|| {
            catch_internal_invariant_message(|| -> () {
                fail_internal_invariant("broken invariant")
            })
        });
        assert_eq!(invariant, Ok(Err("broken invariant".to_owned())));
    }
}
