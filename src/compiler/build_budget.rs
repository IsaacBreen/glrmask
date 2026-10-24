//! Deterministic work budget for template builds.
//!
//! Template construction can enumerate a combinatorial number of states for
//! grammars whose dynamic TokenTrie build is small. The budget charges coarse
//! structural events (states, items, rules) in the stages that dominate those
//! builds, so a declining template attempt stops deterministically instead of
//! exhausting CPU or memory. TokenTrie itself is never charged.
//!
//! Each top-level template attempt owns one [`glrmask_invariant::TemplateBuildBudget`]
//! and installs it on a dedicated Rayon pool, so every worker that performs
//! work for that attempt observes the same handle and no concurrent compile
//! can reset or share it. Nested scopes (for example a composed child compile)
//! reuse the enclosing attempt's handle instead of starting a new one.

use std::cell::Cell;
use std::sync::Arc;

use glrmask_invariant::__private::{
    charge_template_build_budget, clear_template_build_budget_handle,
    set_template_build_budget_handle, template_build_budget_present,
    template_build_budget_units,
};
use glrmask_invariant::TemplateBuildBudget;

pub(crate) const DEFAULT_LIMIT: u64 = glrmask_invariant::DEFAULT_TEMPLATE_BUILD_BUDGET;

thread_local! {
    // Test-only pending limit for the next top-level scope entered on this
    // thread. It is per-thread so a forced tiny budget cannot perturb
    // unrelated concurrent compiles in another test.
    static PENDING_TEST_LIMIT: Cell<Option<u64>> = const { Cell::new(None) };
}

fn resolved_limit() -> u64 {
    PENDING_TEST_LIMIT
        .with(Cell::get)
        .or_else(|| {
            crate::env::var("GLRMASK_TEMPLATE_BUILD_BUDGET")
                .ok()
                .and_then(|value| value.trim().parse::<u64>().ok())
                .filter(|&value| value > 0)
        })
        .unwrap_or(DEFAULT_LIMIT)
}

/// Charge work to the template-build budget installed on the current thread.
///
/// Outside a [`BuildBudgetScope`] this is a no-op, so TokenTrie and other
/// unbudgeted compilation paths are unaffected.
pub(crate) fn charge(stage: &'static str, units: u64) {
    charge_template_build_budget(stage, units);
}

/// Owns one top-level template attempt's budget and the pool that runs it.
///
/// [`BuildBudgetScope::install`] must wrap the attempt itself. Entering a scope
/// from inside an already-budgeted thread (a nested compose) shares the outer
/// handle and runs inline.
pub(crate) struct BuildBudgetScope {
    handle: Option<Arc<TemplateBuildBudget>>,
    pool: Option<rayon::ThreadPool>,
    owns_caller_handle: bool,
    /// Test-only experiment override captured on the caller thread and
    /// propagated to pool workers, whose thread-locals are separate.
    #[cfg(test)]
    epsfree_prepass_override: Option<bool>,
}

impl BuildBudgetScope {
    pub(crate) fn enter() -> Self {
        Self::enter_with_limit(resolved_limit())
    }

    fn enter_with_limit(limit: u64) -> Self {
        if template_build_budget_present() {
            // Nested scope: share the enclosing compile's budget.
            return Self {
                handle: None,
                pool: None,
                owns_caller_handle: false,
                #[cfg(test)]
                epsfree_prepass_override:
                    crate::compiler::grammar::transforms::epsfree_prepass_override_value(),
            };
        }
        let handle = TemplateBuildBudget::new(limit);
        let worker_handle = Arc::clone(&handle);
        let mut builder = rayon::ThreadPoolBuilder::new()
            .stack_size(super::pipeline::COMPILE_WORKER_STACK_SIZE)
            .start_handler(move |_| {
                set_template_build_budget_handle(Arc::clone(&worker_handle));
            });
        if let Some(threads) = super::pipeline::compile_thread_count() {
            builder = builder.num_threads(threads);
        }
        let pool = builder.build().ok();
        // Install on the caller as well: the pool's workers are the only place
        // template work normally runs, but an inline fallback or a nested
        // scope must still see the handle.
        set_template_build_budget_handle(Arc::clone(&handle));
        if crate::env::var_os("GLRMASK_PROFILE_TEMPLATE_BUILD_BUDGET").is_some() {
            eprintln!(
                "[glrmask/profile][template_build_budget] limit={} units=0",
                handle.limit()
            );
        }
        Self {
            handle: Some(handle),
            pool,
            owns_caller_handle: true,
            #[cfg(test)]
            epsfree_prepass_override:
                crate::compiler::grammar::transforms::epsfree_prepass_override_value(),
        }
    }

    /// Run the template attempt on this scope's pool, with the budget visible
    /// to every worker that participates.
    pub(crate) fn install<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        match &self.pool {
            Some(pool) => {
                #[cfg(test)]
                {
                    let override_value = self.epsfree_prepass_override;
                    pool.install(move || {
                        crate::compiler::grammar::transforms::with_epsfree_prepass_override(
                            override_value,
                            f,
                        )
                    })
                }
                #[cfg(not(test))]
                {
                    pool.install(f)
                }
            }
            None => f(),
        }
    }

    #[cfg(test)]
    pub(crate) fn enter_inline_for_test(limit: u64) -> Self {
        let handle = TemplateBuildBudget::new(limit);
        set_template_build_budget_handle(Arc::clone(&handle));
        Self {
            handle: Some(handle),
            pool: None,
            owns_caller_handle: true,
            #[cfg(test)]
            epsfree_prepass_override:
                crate::compiler::grammar::transforms::epsfree_prepass_override_value(),
        }
    }
}

impl Drop for BuildBudgetScope {
    fn drop(&mut self) {
        if self.owns_caller_handle {
            if crate::env::var_os("GLRMASK_PROFILE_TEMPLATE_BUILD_BUDGET").is_some() {
                eprintln!(
                    "[glrmask/profile][template_build_budget] complete units={}",
                    template_build_budget_units()
                );
            }
            clear_template_build_budget_handle();
        }
        let _ = self.handle.take();
    }
}

#[cfg(test)]
pub(crate) struct TestBudgetLimit(());

#[cfg(test)]
impl TestBudgetLimit {
    pub(crate) fn set(limit: u64) -> Self {
        PENDING_TEST_LIMIT.with(|slot| slot.set(Some(limit)));
        Self(())
    }
}

#[cfg(test)]
impl Drop for TestBudgetLimit {
    fn drop(&mut self) {
        PENDING_TEST_LIMIT.with(|slot| slot.set(None));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::catch_internal_invariant;

    #[test]
    fn charge_under_limit_is_fine() {
        let _scope = BuildBudgetScope::enter_inline_for_test(64);
        charge("unit_test", 10);
        charge("unit_test", 20);
        assert_eq!(template_build_budget_units(), 30);
    }

    #[test]
    fn charge_over_limit_returns_a_resource_limit_error() {
        let error = catch_internal_invariant(|| {
            let _scope = BuildBudgetScope::enter_inline_for_test(10);
            charge("unit_test", 11);
        })
        .expect_err("the budget must decline");
        match error.kind() {
            crate::ErrorKind::ResourceLimit(message) => {
                assert_eq!(message, "unit_test build budget exhausted");
            }
            other => panic!("expected a resource-limit error, got {other:?}"),
        }
    }

    #[test]
    fn charge_outside_a_scope_is_a_no_op() {
        charge("unit_test", 100);
        assert_eq!(template_build_budget_units(), 0);
    }
}
