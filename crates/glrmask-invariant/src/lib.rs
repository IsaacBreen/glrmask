#![deny(warnings)]

#[derive(Debug)]
struct InternalInvariantViolation {
    message: String,
}

pub fn fail_internal_invariant(message: impl Into<String>) -> ! {
    std::panic::resume_unwind(Box::new(InternalInvariantViolation {
        message: message.into(),
    }))
}

pub fn catch_internal_invariant_message<T>(f: impl FnOnce() -> T) -> Result<T, String> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(value) => Ok(value),
        Err(payload) => match payload.downcast::<InternalInvariantViolation>() {
            Ok(violation) => Err(violation.message),
            Err(payload) => std::panic::resume_unwind(payload),
        },
    }
}
