#![deny(warnings)]

#[cfg(feature = "internal-api")]
#[derive(Debug)]
struct InternalInvariantViolation {
    message: String,
}

#[cfg(feature = "internal-api")]
fn fail_internal_invariant(message: impl Into<String>) -> ! {
    std::panic::resume_unwind(Box::new(InternalInvariantViolation {
        message: message.into(),
    }))
}

#[cfg(feature = "internal-api")]
fn catch_internal_invariant_message<T>(f: impl FnOnce() -> T) -> Result<T, String> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(value) => Ok(value),
        Err(payload) => match payload.downcast::<InternalInvariantViolation>() {
            Ok(violation) => Err(violation.message),
            Err(payload) => std::panic::resume_unwind(payload),
        },
    }
}

// Expected resource exhaustion is distinct from an internal invariant failure.
// This payload crosses the existing compiler unwind boundary without catching
// unrelated application panics or returning a partial artifact.
#[cfg(feature = "internal-api")]
#[derive(Debug)]
struct CompilationResourceLimit { message: String }

#[cfg(feature = "internal-api")]
fn fail_compilation_resource_limit(message: impl Into<String>) -> ! {
    std::panic::resume_unwind(Box::new(CompilationResourceLimit { message: message.into() }))
}

#[cfg(feature = "internal-api")]
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
#[cfg(feature = "internal-api")]
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
