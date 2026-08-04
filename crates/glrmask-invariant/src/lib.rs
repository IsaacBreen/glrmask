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

/// Implementation details shared by the GLRMask workspace.
#[cfg(feature = "internal-api")]
#[doc(hidden)]
pub mod __private {
    pub fn fail_internal_invariant(message: impl Into<String>) -> ! {
        super::fail_internal_invariant(message)
    }

    pub fn catch_internal_invariant_message<T>(f: impl FnOnce() -> T) -> Result<T, String> {
        super::catch_internal_invariant_message(f)
    }
}
