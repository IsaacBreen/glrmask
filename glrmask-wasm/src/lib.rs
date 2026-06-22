#![deny(warnings)]

use std::alloc::{alloc, dealloc, Layout};
use std::cell::RefCell;

use glrmask_runtime::{RuntimeConstraint, Session};

struct WasmSession {
    inner: Session,
    mask: Vec<u32>,
}

thread_local! {
    static CONSTRAINTS: RefCell<Vec<Option<RuntimeConstraint>>> = const { RefCell::new(Vec::new()) };
    static SESSIONS: RefCell<Vec<Option<WasmSession>>> = const { RefCell::new(Vec::new()) };
    static LAST_ERROR: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

fn set_error(error: impl std::fmt::Display) {
    LAST_ERROR.with(|slot| {
        *slot.borrow_mut() = error.to_string().into_bytes();
    });
}

fn clear_error() {
    LAST_ERROR.with(|slot| slot.borrow_mut().clear());
}

fn handle_index(handle: u32) -> Option<usize> {
    handle.checked_sub(1).map(|value| value as usize)
}

fn with_session_mut<T>(handle: u32, f: impl FnOnce(&mut WasmSession) -> T) -> Option<T> {
    let index = handle_index(handle)?;
    SESSIONS.with(|sessions| {
        let mut sessions = sessions.borrow_mut();
        sessions.get_mut(index)?.as_mut().map(f)
    })
}

fn with_constraint<T>(handle: u32, f: impl FnOnce(&RuntimeConstraint) -> T) -> Option<T> {
    let index = handle_index(handle)?;
    CONSTRAINTS.with(|constraints| {
        let constraints = constraints.borrow();
        constraints.get(index)?.as_ref().map(f)
    })
}

fn insert_constraint(constraint: RuntimeConstraint) -> u32 {
    CONSTRAINTS.with(|constraints| {
        let mut constraints = constraints.borrow_mut();
        if let Some((index, slot)) = constraints.iter_mut().enumerate().find(|(_, slot)| slot.is_none()) {
            *slot = Some(constraint);
            (index + 1) as u32
        } else {
            constraints.push(Some(constraint));
            constraints.len() as u32
        }
    })
}

fn insert_session(inner: Session) -> u32 {
    let session = WasmSession {
        mask: vec![0; inner.mask_len()],
        inner,
    };
    SESSIONS.with(|sessions| {
        let mut sessions = sessions.borrow_mut();
        if let Some((index, slot)) = sessions.iter_mut().enumerate().find(|(_, slot)| slot.is_none()) {
            *slot = Some(session);
            (index + 1) as u32
        } else {
            sessions.push(Some(session));
            sessions.len() as u32
        }
    })
}

/// Allocate caller-owned linear-memory storage. JavaScript writes an artifact into
/// this buffer, calls `glrmask_session_new`, and then releases it with `glrmask_dealloc`.
#[unsafe(no_mangle)]
pub extern "C" fn glrmask_alloc(length: u32) -> u32 {
    if length == 0 {
        return 0;
    }
    let layout = Layout::from_size_align(length as usize, 1).expect("byte alignment is valid");
    unsafe { alloc(layout) as usize as u32 }
}

/// Free storage returned by `glrmask_alloc`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn glrmask_dealloc(pointer: u32, length: u32) {
    if pointer == 0 || length == 0 {
        return;
    }
    let layout = Layout::from_size_align(length as usize, 1).expect("byte alignment is valid");
    // SAFETY: JS must pass exactly a pointer and byte length returned by
    // `glrmask_alloc`; the layout is therefore identical to the allocation.
    unsafe { dealloc(pointer as usize as *mut u8, layout) };
}

/// Load a versioned, fully compiled constraint artifact for reuse across sessions.
/// Returns zero on error; retrieve UTF-8 diagnostics via `glrmask_last_error_*`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn glrmask_constraint_load(artifact_pointer: u32, artifact_length: u32) -> u32 {
    clear_error();
    if artifact_pointer == 0 || artifact_length == 0 {
        set_error("empty compiled glrmask artifact");
        return 0;
    }
    // SAFETY: JavaScript has copied `artifact_length` bytes into a live Wasm allocation.
    let bytes = unsafe {
        std::slice::from_raw_parts(artifact_pointer as usize as *const u8, artifact_length as usize)
    };
    let constraint = match RuntimeConstraint::from_bytes(bytes.to_vec()) {
        Ok(constraint) => constraint,
        Err(error) => {
            set_error(error);
            return 0;
        }
    };

    insert_constraint(constraint)
}

/// Release a loaded constraint and its shared runtime caches.
#[unsafe(no_mangle)]
pub extern "C" fn glrmask_constraint_free(handle: u32) {
    if let Some(index) = handle_index(handle) {
        CONSTRAINTS.with(|constraints| {
            if let Some(slot) = constraints.borrow_mut().get_mut(index) {
                *slot = None;
            }
        });
    }
}

/// Start one independent decoder session sharing an already-loaded constraint.
#[unsafe(no_mangle)]
pub extern "C" fn glrmask_session_new_from_constraint(constraint_handle: u32) -> u32 {
    clear_error();
    let Some(inner) = with_constraint(constraint_handle, RuntimeConstraint::start) else {
        set_error("invalid glrmask constraint handle");
        return 0;
    };
    insert_session(inner)
}

/// Construct a one-off session from a versioned, fully compiled constraint artifact.
/// For more than one session per artifact, prefer `glrmask_constraint_load` plus
/// `glrmask_session_new_from_constraint`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn glrmask_session_new(artifact_pointer: u32, artifact_length: u32) -> u32 {
    let constraint_handle = unsafe { glrmask_constraint_load(artifact_pointer, artifact_length) };
    if constraint_handle == 0 {
        return 0;
    }
    let session_handle = glrmask_session_new_from_constraint(constraint_handle);
    glrmask_constraint_free(constraint_handle);
    session_handle
}

/// Drop a browser session and all its parser/lexer state.
#[unsafe(no_mangle)]
pub extern "C" fn glrmask_session_free(handle: u32) {
    if let Some(index) = handle_index(handle) {
        SESSIONS.with(|sessions| {
            if let Some(slot) = sessions.borrow_mut().get_mut(index) {
                *slot = None;
            }
        });
    }
}

/// Recompute the exact original-vocabulary mask and return its linear-memory pointer.
/// The session owns and reuses this allocation across token steps. Call
/// `glrmask_mask_len` immediately after this function, then copy the u32 words.
#[unsafe(no_mangle)]
pub extern "C" fn glrmask_mask(handle: u32) -> u32 {
    clear_error();
    let Some(pointer) = with_session_mut(handle, |session| {
        session.inner.fill_mask(&mut session.mask);
        session.mask.as_ptr() as usize as u32
    }) else {
        set_error("invalid glrmask session handle");
        return 0;
    };
    pointer
}

/// Length in 32-bit words of the most recently materialized vocabulary mask.
#[unsafe(no_mangle)]
pub extern "C" fn glrmask_mask_len(handle: u32) -> u32 {
    with_session_mut(handle, |session| session.mask.len() as u32).unwrap_or(0)
}

/// Commit one sampled BPE token ID. Returns 1 on success and 0 on rejection/error.
#[unsafe(no_mangle)]
pub extern "C" fn glrmask_commit(handle: u32, token_id: u32) -> u32 {
    clear_error();
    let Some(result) = with_session_mut(handle, |session| session.inner.commit_token(token_id)) else {
        set_error("invalid glrmask session handle");
        return 0;
    };
    match result {
        Ok(()) => 1,
        Err(error) => {
            set_error(error);
            0
        }
    }
}

/// Whether end-of-sequence is currently grammatically admissible.
#[unsafe(no_mangle)]
pub extern "C" fn glrmask_eos_allowed(handle: u32) -> u32 {
    with_session_mut(handle, |session| u32::from(session.inner.eos_allowed())).unwrap_or(0)
}

/// Restore a session to its artifact's initial parser/lexer state.
#[unsafe(no_mangle)]
pub extern "C" fn glrmask_reset(handle: u32) -> u32 {
    clear_error();
    let Some(()) = with_session_mut(handle, |session| session.inner.reset()) else {
        set_error("invalid glrmask session handle");
        return 0;
    };
    1
}

#[unsafe(no_mangle)]
pub extern "C" fn glrmask_last_error_ptr() -> u32 {
    LAST_ERROR.with(|error| error.borrow().as_ptr() as usize as u32)
}

#[unsafe(no_mangle)]
pub extern "C" fn glrmask_last_error_len() -> u32 {
    LAST_ERROR.with(|error| error.borrow().len() as u32)
}
