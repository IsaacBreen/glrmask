#![recursion_limit = "256"]

//! Private ownership and validation support shared by the C and Node bindings.
//! No language binding reimplements the grammar compiler or decoding engine.

use std::collections::BTreeSet;
use std::fmt;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use glrmask::__private::ConstraintStateExt as _;
pub use glrmask;

// C status codes are fixed by glrmask.h; a new kind needs a new status value there.
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorKind { InvalidArgument = 1, Engine = 2, Panic = 3, BufferTooSmall = 4, Closed = 5 }

#[derive(Debug)]
pub struct Error { pub kind: ErrorKind, pub message: String }
pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self { Self { kind, message: message.into() } }
    pub fn invalid(message: impl Into<String>) -> Self { Self::new(ErrorKind::InvalidArgument, message) }
    pub fn closed() -> Self { Self::new(ErrorKind::Closed, "object has been disposed") }
    pub fn engine(error: impl fmt::Display) -> Self { Self::new(ErrorKind::Engine, error.to_string()) }
}
impl fmt::Display for Error { fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.write_str(&self.message) } }
impl std::error::Error for Error {}
impl From<glrmask::Error> for Error { fn from(error: glrmask::Error) -> Self { Self::engine(error) } }

/// Contain unwinding Rust panics at a foreign-language boundary. This cannot
/// recover process aborts (including allocation failure), or invalid C pointers.
pub fn guarded<T>(f: impl FnOnce() -> Result<T>) -> Result<T> {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(result) => result,
        Err(payload) => {
            let message = payload.downcast_ref::<String>().map(String::as_str)
                .or_else(|| payload.downcast_ref::<&str>().copied()).unwrap_or("unknown Rust panic");
            Err(Error::new(ErrorKind::Panic, format!("GLRMask internal panic: {message}")))
        }
    }
}

pub fn vocab(entries: Vec<(u32, Vec<u8>)>, exact_only: Vec<u32>) -> Result<glrmask::Vocab> {
    let mut ids = BTreeSet::new();
    for (id, _) in &entries {
        if !ids.insert(*id) { return Err(Error::invalid(format!("duplicate vocabulary ID {id}"))); }
    }
    for id in &exact_only {
        if !ids.insert(*id) { return Err(Error::invalid(format!("duplicate or byte-backed exact-only ID {id}"))); }
    }
    guarded(|| Ok(glrmask::Vocab::new(entries).with_special_tokens(exact_only)))
}

pub fn mask_strategy(val: u32) -> Result<glrmask::MaskStrategy> {
    match val {
        0 => Ok(glrmask::MaskStrategy::Auto),
        1 => Ok(glrmask::MaskStrategy::ParserDwa),
        2 => Ok(glrmask::MaskStrategy::TokenTrie(glrmask::TokenTrie::new())),
        _ => Err(Error::invalid("unknown mask strategy; expected 0, 1 or 2")),
    }
}

pub fn mask_strategy_value(strategy: glrmask::MaskStrategy) -> u32 {
    match strategy {
        glrmask::MaskStrategy::Auto => 0,
        glrmask::MaskStrategy::ParserDwa => 1,
        glrmask::MaskStrategy::TokenTrie(_) => 2,
        _ => 0,
    }
}

pub fn boundary_mask_strategy(val: u32) -> Result<glrmask::BoundaryMaskStrategy> {
    match val {
        0 => Ok(glrmask::BoundaryMaskStrategy::Auto),
        1 => Ok(glrmask::BoundaryMaskStrategy::ParserDwa),
        2 => Ok(glrmask::BoundaryMaskStrategy::TokenTrie),
        _ => Err(Error::invalid("unknown boundary mask strategy; expected 0, 1 or 2")),
    }
}

pub fn boundary_mask_strategy_value(strategy: glrmask::BoundaryMaskStrategy) -> u32 {
    match strategy {
        glrmask::BoundaryMaskStrategy::Auto => 0,
        glrmask::BoundaryMaskStrategy::ParserDwa => 1,
        glrmask::BoundaryMaskStrategy::TokenTrie => 2,
        _ => 0,
    }
}

pub fn parser_backend(value: u32) -> Result<glrmask::ParserBackend> {
    match value {
        0 => Ok(glrmask::ParserBackend::LrTable),
        1 => Ok(glrmask::ParserBackend::TemplateDfa),
        _ => Err(Error::invalid("unknown parser backend; expected 0 or 1")),
    }
}

pub fn compile_options(
    mask_strategy: glrmask::MaskStrategy,
    boundary_mask_strategy: glrmask::BoundaryMaskStrategy,
    token_trie: glrmask::TokenTrie,
) -> glrmask::CompileOptions {
    let mask_strategy = match mask_strategy {
        glrmask::MaskStrategy::TokenTrie(_) => glrmask::MaskStrategy::TokenTrie(token_trie),
        other => other,
    };
    glrmask::CompileOptions::default()
        .mask_strategy(mask_strategy)
        .boundary_mask_strategy(boundary_mask_strategy)
}

type BorrowedState<'a> = glrmask::ConstraintState<'a>;
self_cell::self_cell!(
    struct StateCell {
        owner: Arc<glrmask::Constraint>,
        #[not_covariant]
        dependent: BorrowedState,
    }
);

/// Holds the compiled constraint alive independently of every foreign handle.
/// A panicking operation poisons this state; it must then be discarded.
pub struct State { cell: StateCell, poisoned: AtomicBool }

impl State {
    pub fn start(owner: Arc<glrmask::Constraint>) -> Result<Self> {
        guarded(|| Ok(Self { cell: StateCell::new(owner, |c| c.start()), poisoned: AtomicBool::new(false) }))
    }
    fn check(&self) -> Result<()> {
        if self.poisoned.load(Ordering::Relaxed) {
            Err(Error::new(ErrorKind::Panic, "state is unusable after an internal panic; discard it"))
        } else { Ok(()) }
    }
    fn read<T>(&self, f: impl FnOnce(&BorrowedState<'_>) -> Result<T>) -> Result<T> {
        self.check()?;
        let result = guarded(|| self.cell.with_dependent(|_, state| f(state)));
        if result.as_ref().err().is_some_and(|e| e.kind == ErrorKind::Panic) { self.poisoned.store(true, Ordering::Relaxed); }
        result
    }
    fn write<T>(&mut self, f: impl FnOnce(&mut BorrowedState<'_>) -> Result<T>) -> Result<T> {
        self.check()?;
        let result = guarded(|| self.cell.with_dependent_mut(|_, state| f(state)));
        if result.as_ref().err().is_some_and(|e| e.kind == ErrorKind::Panic) { self.poisoned.store(true, Ordering::Relaxed); }
        result
    }
    pub fn try_clone(&self) -> Result<Self> {
        self.check()?;
        guarded(|| {
            let mut other = Self::start(self.cell.borrow_owner().clone())?;
            self.cell.with_dependent(|_, source| {
                other.cell.with_dependent_mut(|_, destination| destination.copy_snapshot_from(source));
            });
            Ok(other)
        })
    }
    pub fn mask_len(&self) -> usize { self.cell.borrow_owner().mask_len() }
    pub fn mask(&self) -> Result<Vec<u32>> { self.read(|s| Ok(s.mask())) }
    pub fn fill_mask(&self, output: &mut [u32]) -> Result<()> {
        let required = self.mask_len();
        if output.len() < required {
            return Err(Error::new(ErrorKind::BufferTooSmall, format!("mask needs at least {required} u32 words")));
        }
        self.read(|s| { s.fill_mask(output); Ok(()) })
    }
    pub fn commit_token(&mut self, id: u32) -> Result<bool> { self.write(|s| Ok(s.commit_token(id))) }
    pub fn commit_tokens(&mut self, ids: &[u32]) -> Result<bool> { self.write(|s| Ok(s.commit_tokens(ids))) }
    pub fn commit_bytes(&mut self, bytes: &[u8]) -> Result<bool> { self.write(|s| Ok(s.commit_bytes(bytes))) }
    pub fn is_accepting(&self) -> Result<bool> { self.read(|s| Ok(s.is_accepting())) }
    pub fn is_terminated(&self) -> Result<bool> { self.read(|s| Ok(s.is_terminated())) }
    pub fn forced_tokens(&self) -> Result<Vec<u32>> { self.read(|s| Ok(s.forced_tokens())) }
    pub fn marker_events(&self) -> Result<Vec<glrmask::MarkerEvent>> { self.read(|s| Ok(s.marker_events().to_vec())) }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn state_retains_owner_and_clones_independently() {
        let vocab = vocab(vec![(0,b"a".to_vec()), (1,b"b".to_vec()), (31,b"ab".to_vec()), (32,b"a".to_vec()), (65,b"z".to_vec())], vec![]).unwrap();
        for mode in 0..3 {
            let grammar = glrmask::Grammar::from_ebnf("start ::= \"a\" \"b\"").unwrap();
            let module = glrmask::Module::new(&grammar);
            let options = compile_options(
                mask_strategy(mode).unwrap(),
                glrmask::BoundaryMaskStrategy::Auto,
                glrmask::TokenTrie::new(),
            );
            let c = Arc::new(module.compile_with(&vocab, options).unwrap());
            let expected = c.start().mask();
            let mut a = State::start(c).unwrap();
            assert_eq!(a.mask().unwrap(), expected);
            let b = a.try_clone().unwrap();
            assert!(a.commit_token(31).unwrap());
            assert!(a.is_accepting().unwrap());
            assert!(!b.is_accepting().unwrap());
            let mut words = [u32::MAX;3];
            b.fill_mask(&mut words).unwrap();
            assert_eq!(&words[..expected.len()], expected.as_slice());
            assert!(words[expected.len()..].iter().all(|&w| w == 0));
            assert_eq!(b.fill_mask(&mut []).unwrap_err().kind, ErrorKind::BufferTooSmall);
        }
    }
    #[test]
    fn duplicate_ids_and_panics_are_errors() {
        assert!(vocab(vec![(1,vec![]),(1,vec![1])],vec![]).is_err());
        assert_eq!(guarded::<()>(|| panic!("test containment")).unwrap_err().kind, ErrorKind::Panic);
    }
}
