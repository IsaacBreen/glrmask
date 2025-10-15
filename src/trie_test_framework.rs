// src/trie_test_framework.rs

//! A testing framework for trie transformations.
//!
//! This framework simplifies the process of verifying that a complex, in-place transformation
//! on a `Trie` graph is equivalent to a simpler, reference transformation defined on raw
//! stacks (`Vec<EK>`).
//!
//! # Usage
//!
//! 1.  **Set up your test harness**: Define your `EK`, `EV`, `T` types. Create a `God`
//!     instance and build your initial test trie using helpers like `harness::new_node`
//!     and `harness::add_edge`.
//!
//! 2.  **Define transformations**:
//!     -   `stack_transform`: A function `Fn(Vec<EK>) -> Vec<EK>` that represents the
//!         "gold standard" or reference implementation of your logic, operating on a
//!         single, simple path.
//!     -   `trie_transform`: A function `Fn(&God, &[Trie2Index])` that performs the
//!         equivalent, but more complex, in-place transformation on the trie graph.
//!
//! 3.  **Define optional helpers**:
//!     -   `canonicalizer`: A function `Fn(Vec<EK>) -> Vec<EK>` to normalize stacks
//!         before comparison. This is useful if multiple stack representations are
//!         semantically equivalent (e.g., merging no-op edges).
//!     -   `assertion`: A function `Fn(&BTreeSet<Vec<EK>>)` to run additional checks
//!         on the final set of stacks produced by both transformations.
//!
//! 4.  **Run the test**:
//!     ```ignore
//!     use crate::trie_test_framework::TrieTestFramework;
//!
//!     // ... (setup god, roots, and transform functions)
//!
//!     TrieTestFramework::new(&god, &roots)
//!         .with_stack_transform(my_stack_transform)
//!         .with_trie_transform(my_trie_transform)
//!         .with_stack_canonicalizer(my_canonicalizer)
//!         .run();
//!     ```
//!
//! The `run` method will automatically:
//! - Flatten the initial trie into a set of stacks.
//! - Apply the `stack_transform` to get the expected result.
//! - Clone the trie, apply the `trie_transform` to the clone.
//! - Flatten the transformed trie.
//! - Canonicalize both sets of stacks (if a canonicalizer is provided).
//! - Assert that the two sets of stacks are equal.
//! - Run any additional assertions.

use crate::datastructures::trie::{GodWrapper, Trie, Trie2Index};
use std::collections::BTreeSet;
use std::fmt::Debug;

/// Maps a function over a BTreeSet of stacks.
fn map_to_stacks<EK: Ord + Clone>(
    f: impl Fn(Vec<EK>) -> Vec<EK>,
    stacks: &BTreeSet<Vec<EK>>,
) -> BTreeSet<Vec<EK>> {
    stacks.iter().map(|s| f(s.clone())).collect()
}

/// The main entry point for the testing framework.
pub struct TrieTestFramework<'a, EK, EV, T>
where
    EK: Ord + Clone + Debug,
    EV: Clone + PartialEq,
    T: Clone + PartialEq,
{
    god: &'a GodWrapper<EK, EV, T>,
    roots: &'a [Trie2Index],
    stack_canonicalizer: Option<Box<dyn Fn(Vec<EK>) -> Vec<EK>>>,
    assertions: Vec<Box<dyn Fn(&BTreeSet<Vec<EK>>)>>,
}

impl<'a, EK, EV, T> TrieTestFramework<'a, EK, EV, T>
where
    EK: Ord + Clone + Debug,
    EV: Clone + PartialEq,
    T: Clone + PartialEq,
{
    /// Creates a new test framework instance.
    ///
    /// # Arguments
    /// * `god`: The arena containing the initial trie.
    /// * `roots`: The roots of the trie to be tested.
    pub fn new(god: &'a GodWrapper<EK, EV, T>, roots: &'a [Trie2Index]) -> Self {
        Self {
            god,
            roots,
            stack_canonicalizer: None,
            assertions: Vec::new(),
        }
    }

    /// Sets the canonicalizer function used to normalize stacks before comparison.
    pub fn with_stack_canonicalizer(mut self, f: impl Fn(Vec<EK>) -> Vec<EK> + 'static) -> Self {
        self.stack_canonicalizer = Some(Box::new(f));
        self
    }

    /// Adds an assertion to be run on the final, canonicalized stacks from both the
    /// reference and the trie-under-test. The assertion will be run on both sets
    /// independently.
    pub fn with_assertion(mut self, f: impl Fn(&BTreeSet<Vec<EK>>) + 'static) -> Self {
        self.assertions.push(Box::new(f));
        self
    }

    /// Runs a test comparing a stack transformation with a trie transformation.
    ///
    /// # Arguments
    /// * `stack_transform`: The reference implementation that transforms a single stack.
    /// * `trie_transform`: The implementation under test that transforms the trie in-place.
    pub fn test_transform(
        &self,
        stack_transform: impl Fn(Vec<EK>) -> Vec<EK>,
        trie_transform: impl Fn(&GodWrapper<EK, EV, T>, &[Trie2Index]),
    ) {
        // 1. Get the initial set of stacks from the original trie.
        let initial_stacks = Trie::get_all_paths(self.god, self.roots);

        // 2. Compute the expected stacks using the reference stack_transform.
        let mut expected_stacks = map_to_stacks(stack_transform, &initial_stacks);

        // 3. Create a deep copy of the trie to run the transform on.
        let (god_clone, roots_clone, _) = Trie::deep_copy_subtrees(self.god, self.roots);

        // 4. Run the trie transform on the cloned trie.
        trie_transform(&god_clone, &roots_clone);

        // 5. Get the actual stacks by flattening the transformed trie.
        let mut actual_stacks = Trie::get_all_paths(&god_clone, &roots_clone);

        // 6. Canonicalize both sets of stacks if a canonicalizer is provided.
        if let Some(canonicalizer) = &self.stack_canonicalizer {
            expected_stacks = map_to_stacks(canonicalizer, &expected_stacks);
            actual_stacks = map_to_stacks(canonicalizer, &actual_stacks);
        }

        // 7. Run assertions on both sets of stacks.
        for assertion in &self.assertions {
            assertion(&expected_stacks);
            assertion(&actual_stacks);
        }

        // 8. Assert that the final sets of stacks are identical.
        assert_eq!(
            expected_stacks, actual_stacks,
            "The stacks from the trie transformation do not match the expected stacks from the reference stack transformation."
        );
    }
}

/// A collection of helper functions to make building test tries easier.
pub mod harness {
    use super::*;
    use crate::datastructures::trie::Trie;

    /// Creates a new empty node in the trie.
    pub fn new_node<EK, EV, T>(god: &GodWrapper<EK, EV, T>, value: T) -> Trie2Index
    where
        EK: Ord + Clone,
    {
        Trie2Index::new(god.insert(Trie::new(value)))
    }

    /// Adds a directed edge between two nodes in the trie.
    pub fn add_edge<EK, EV, T>(
        god: &GodWrapper<EK, EV, T>,
        from: Trie2Index,
        to: Trie2Index,
        key: EK,
        edge_value: EV,
    ) where
        EK: Ord + Clone,
    {
        let mut w = from
            .write(god)
            .expect("Arena write poisoned while adding edge");
        let mut ev_opt = Some(edge_value);
        w.try_insert(key, &mut ev_opt, to);
    }
}
