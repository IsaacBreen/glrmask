use super::*;
// Still needed for macro use perhaps?
use crate::datastructures::hybrid_bitset::HybridBitset;
use std::iter::FromIterator;


// Helper to create byte vecs from strings for tests.
fn b(s: &str) -> Vec<u8> {
    s.as_bytes().to_vec()
}

#[test]
fn test_empty_tree() {
    let tokens: Vec<(usize, Vec<u8>)> = vec![];
    let tree = VocabPrefixTree::build(&tokens);
    assert_eq!(tree.root.token_id, 0); // Root defaults to 0
    assert_eq!(tree.root.prefix_length, 0); // Root length is 0
    assert!(!tree.has_empty_string_token()); // No empty token provided
    assert_eq!(tree.max_token_id(), 0); // Max ID is 0 for empty input
    assert!(tree.root.children.is_empty());
    assert_eq!(tree.find_token(b"a"), None);
    assert_eq!(tree.find_token(b""), None); // Empty query returns None (flag is false)
    // Root's reachable IDs should be empty (bit 0 removed as it's conventional)
    assert!(tree.root.reachable_token_ids.is_empty());
}

#[test]
fn test_single_token() {
    let tokens = vec![(1, b("hello"))];
    let tree = VocabPrefixTree::build(&tokens);

    assert_eq!(tree.root.token_id, 0); // Root ID remains 0
    assert_eq!(tree.root.prefix_length, 0);
    assert!(!tree.has_empty_string_token()); // No empty token provided
    assert_eq!(tree.max_token_id(), 1);
    assert_eq!(tree.root.children.len(), 1);
    assert!(tree.root.children.contains_key(&b("hello")));

    let node = tree.root.children.get(&b("hello")).unwrap();
    assert_eq!(node.token_id, 1);
    assert_eq!(node.prefix_length, 5); // "hello" has length 5
    assert!(node.children.is_empty());
    // Node "hello" should only have its own ID reachable
    let expected_node_bits = RangeSetBlaze::from_iter(vec![1]);
    assert_eq!(node.reachable_token_ids, expected_node_bits);

    assert_eq!(tree.find_token(&b("hello")), Some(1));
    assert_eq!(tree.find_token(&b("hell")), None);
    assert_eq!(tree.find_token(&b("hello world")), None);
    assert_eq!(tree.find_token(b""), None); // Flag is false

    // Root's reachable IDs should contain only ID 1 (ID 0 is conventional)
    let expected_root_bits = RangeSetBlaze::from_iter(vec![1]);
    assert_eq!(tree.root.reachable_token_ids, expected_root_bits);
}

 #[test]
fn test_empty_string_token() {
    // Assign ID 99 to the empty string
    let tokens = vec![(99, b("")), (1, b("a"))];
    let tree = VocabPrefixTree::build(&tokens);

    assert_eq!(tree.root.token_id, 99); // Root gets the ID for ""
    assert_eq!(tree.root.prefix_length, 0); // Empty string has length 0
    assert!(tree.has_empty_string_token()); // Empty token WAS provided
    assert_eq!(tree.max_token_id(), 99);
    assert_eq!(tree.root.children.len(), 1);
    assert!(tree.root.children.contains_key(&b("a")));

    let node_a = tree.root.children.get(&b("a")).unwrap();
    assert_eq!(node_a.token_id, 1);
    // Node "a" should only have its own ID reachable
    let expected_node_a_bits = RangeSetBlaze::from_iter(vec![1]);
    assert_eq!(node_a.reachable_token_ids, expected_node_a_bits);

    assert_eq!(tree.find_token(&b("")), Some(99)); // Query for "" returns its ID (flag is true)
    assert_eq!(tree.find_token(&b("a")), Some(1));

    // Root's reachable IDs should contain 1 (from child) and 99 (itself)
    let expected_root_bits = RangeSetBlaze::from_iter(vec![1, 99]);
    assert_eq!(tree.root.reachable_token_ids, expected_root_bits);
}

#[test]
fn test_empty_string_token_with_id_zero() {
    // Assign ID 0 to the empty string explicitly
    let tokens = vec![(0, b("")), (1, b("a"))];
    let tree = VocabPrefixTree::build(&tokens);

    assert_eq!(tree.root.token_id, 0); // Root gets the ID 0 for ""
    assert_eq!(tree.root.prefix_length, 0);
    assert!(tree.has_empty_string_token()); // Empty token WAS provided
    assert_eq!(tree.max_token_id(), 1);
    assert_eq!(tree.root.children.len(), 1);
    assert!(tree.root.children.contains_key(&b("a")));

    let node_a = tree.root.children.get(&b("a")).unwrap();
    assert_eq!(node_a.token_id, 1);
    // Node "a" should only have its own ID reachable
    let expected_node_a_bits = RangeSetBlaze::from_iter(vec![1]);
    assert_eq!(node_a.reachable_token_ids, expected_node_a_bits);

    assert_eq!(tree.find_token(&b("")), Some(0)); // Query for "" returns its ID 0 (flag is true)
    assert_eq!(tree.find_token(&b("a")), Some(1));

    // Root's reachable IDs should contain 1 (from child) and 0 (itself, as it's explicit)
    let expected_root_bits = RangeSetBlaze::from_iter(vec![0, 1]);
    assert_eq!(tree.root.reachable_token_ids, expected_root_bits);
}


#[test]
fn test_simple_prefix() {
    // "a" is prefix of "ab"
    let tokens = vec![(1, b("a")), (2, b("ab"))];
    let tree = VocabPrefixTree::build(&tokens);

    assert_eq!(tree.root.token_id, 0);
    assert_eq!(tree.root.prefix_length, 0);
    assert!(!tree.has_empty_string_token());
    assert_eq!(tree.max_token_id(), 2);
    assert_eq!(tree.root.children.len(), 1);
    assert!(tree.root.children.contains_key(&b("a")));

    let node_a = tree.root.children.get(&b("a")).unwrap();
    assert_eq!(node_a.token_id, 1);
    assert_eq!(node_a.prefix_length, 1); // "a" length 1
    assert_eq!(node_a.children.len(), 1);
    assert!(node_a.children.contains_key(&b("b")));

    let node_ab = node_a.children.get(&b("b")).unwrap();
    assert_eq!(node_ab.prefix_length, 2); // "ab" length 2
    assert_eq!(node_ab.token_id, 2);
    assert!(node_ab.children.is_empty());
    // Node "ab" reachable IDs: {2}
    let expected_ab_bits = RangeSetBlaze::from_iter(vec![2]);
    assert_eq!(node_ab.reachable_token_ids, expected_ab_bits);

    // Node "a" reachable IDs: {1, 2}
    let expected_a_bits = RangeSetBlaze::from_iter(vec![1, 2]);
    assert_eq!(node_a.reachable_token_ids, expected_a_bits);

    assert_eq!(tree.find_token(&b("a")), Some(1));
    assert_eq!(tree.find_token(&b("ab")), Some(2));
    assert_eq!(tree.find_token(&b("b")), None);
    assert_eq!(tree.find_token(&b("abc")), None);
    assert_eq!(tree.find_token(b""), None); // Flag is false

    // Root reachable IDs: {1, 2} (0 is conventional)
    assert_eq!(tree.root.reachable_token_ids, expected_a_bits);
}

#[test]
fn test_multiple_prefixes() {
    // "a", "ab", "abc"
    let tokens = vec![(1, b("a")), (2, b("ab")), (3, b("abc"))];
    let tree = VocabPrefixTree::build(&tokens);

    assert_eq!(tree.root.token_id, 0);
    assert_eq!(tree.root.prefix_length, 0);
    assert!(!tree.has_empty_string_token());
    assert_eq!(tree.max_token_id(), 3);
    assert_eq!(tree.root.children.len(), 1);
    let node_a = tree.root.children.get(&b("a")).unwrap();
    assert_eq!(node_a.token_id, 1);
    assert_eq!(node_a.prefix_length, 1);
    assert_eq!(node_a.children.len(), 1);

    let node_ab = node_a.children.get(&b("b")).unwrap();
    assert_eq!(node_ab.token_id, 2);
    assert_eq!(node_ab.prefix_length, 2);
    assert_eq!(node_ab.children.len(), 1);

    let node_abc = node_ab.children.get(&b("c")).unwrap();
    assert_eq!(node_abc.prefix_length, 3);
    assert_eq!(node_abc.token_id, 3);
    assert!(node_abc.children.is_empty());
    // Node "abc" reachable: {3}
    let expected_abc_bits = RangeSetBlaze::from_iter(vec![3]);
    assert_eq!(node_abc.reachable_token_ids, expected_abc_bits);

    // Node "ab" reachable: {2, 3}
    let expected_ab_bits = RangeSetBlaze::from_iter(vec![2, 3]);
    assert_eq!(node_ab.reachable_token_ids, expected_ab_bits);

    // Node "a" reachable: {1, 2, 3}
    let expected_a_bits = RangeSetBlaze::from_iter(vec![1, 2, 3]);
    assert_eq!(node_a.reachable_token_ids, expected_a_bits);

    assert_eq!(tree.find_token(&b("a")), Some(1));
    assert_eq!(tree.find_token(&b("ab")), Some(2));
    assert_eq!(tree.find_token(&b("abc")), Some(3));
    assert_eq!(tree.find_token(&b("b")), None);
    assert_eq!(tree.find_token(&b("abcd")), None);

    // Root reachable: {1, 2, 3}
    assert_eq!(tree.root.reachable_token_ids, expected_a_bits);
    assert_eq!(tree.find_token(b""), None); // Flag is false
}

#[test]
fn test_shared_prefix_branching_where_prefix_is_token() {
    let tokens_with_prefix = vec![(5, b("app")), (10, b("apple")), (20, b("apply"))];
    let tree_with_prefix = VocabPrefixTree::build(&tokens_with_prefix);

    assert_eq!(tree_with_prefix.root.token_id, 0);
    assert_eq!(tree_with_prefix.root.prefix_length, 0);
    assert!(!tree_with_prefix.has_empty_string_token());
    assert_eq!(tree_with_prefix.max_token_id(), 20);
    assert_eq!(tree_with_prefix.root.children.len(), 1);
    assert!(tree_with_prefix.root.children.contains_key(&b("app")));

    let node_app = tree_with_prefix.root.children.get(&b("app")).unwrap();
    assert_eq!(node_app.token_id, 5);
    assert_eq!(node_app.prefix_length, 3); // "app" length 3
    assert_eq!(node_app.children.len(), 2);

    assert!(node_app.children.contains_key(&b("le")));
    let node_apple = node_app.children.get(&b("le")).unwrap();
    assert_eq!(node_apple.token_id, 10);
    assert_eq!(node_apple.prefix_length, 5); // "apple" length 5
    assert!(node_apple.children.is_empty());
    // Node "apple" reachable: {10}
    let expected_apple_bits = RangeSetBlaze::from_iter(vec![10]);
    assert_eq!(node_apple.reachable_token_ids, expected_apple_bits);

    assert!(node_app.children.contains_key(&b("ly")));
    let node_apply = node_app.children.get(&b("ly")).unwrap();
    assert_eq!(node_apply.prefix_length, 5); // "apply" length 5
    assert_eq!(node_apply.token_id, 20);
    assert!(node_apply.children.is_empty());
    // Node "apply" reachable: {20}
    let expected_apply_bits = RangeSetBlaze::from_iter(vec![20]);
    assert_eq!(node_apply.reachable_token_ids, expected_apply_bits);

    // Node "app" reachable: {5, 10, 20}
    let expected_app_bits = RangeSetBlaze::from_iter(vec![5, 10, 20]);
    assert_eq!(node_app.reachable_token_ids, expected_app_bits);

    assert_eq!(tree_with_prefix.find_token(&b("app")), Some(5));
    assert_eq!(tree_with_prefix.find_token(&b("apple")), Some(10));
    assert_eq!(tree_with_prefix.find_token(&b("apply")), Some(20));
    assert_eq!(tree_with_prefix.find_token(&b("appl")), None);
    assert_eq!(tree_with_prefix.find_token(&b("ap")), None);
    // Root reachable: {5, 10, 20}
    assert_eq!(tree_with_prefix.root.reachable_token_ids, expected_app_bits);
    assert_eq!(tree_with_prefix.find_token(b""), None); // Flag is false
}

 #[test]
fn test_shared_prefix_branching_where_prefix_is_not_token() {
    let tokens = vec![(10, b("apple")), (20, b("apply"))];
    let tree = VocabPrefixTree::build(&tokens);

    assert_eq!(tree.root.token_id, 0);
    assert_eq!(tree.root.prefix_length, 0);
    assert!(!tree.has_empty_string_token());
    assert_eq!(tree.max_token_id(), 20);
    assert_eq!(tree.root.children.len(), 2); // "apple" and "apply" are direct children of root
    assert!(tree.root.children.contains_key(&b("apple")));
    assert!(tree.root.children.contains_key(&b("apply")));

    let node_apple = tree.root.children.get(&b("apple")).unwrap();
    assert_eq!(node_apple.prefix_length, 5);
    assert_eq!(node_apple.token_id, 10);
    assert!(node_apple.children.is_empty());
    // Node "apple" reachable: {10}
    let expected_apple_bits = RangeSetBlaze::from_iter(vec![10]);
    assert_eq!(node_apple.reachable_token_ids, expected_apple_bits);

    let node_apply = tree.root.children.get(&b("apply")).unwrap();
    assert_eq!(node_apply.prefix_length, 5);
    assert_eq!(node_apply.token_id, 20);
    assert!(node_apply.children.is_empty());
    // Node "apply" reachable: {20}
    let expected_apply_bits = RangeSetBlaze::from_iter(vec![20]);
    assert_eq!(node_apply.reachable_token_ids, expected_apply_bits);

    assert_eq!(tree.find_token(&b("apple")), Some(10));
    assert_eq!(tree.find_token(&b("apply")), Some(20));
    assert_eq!(tree.find_token(&b("app")), None);
    assert_eq!(tree.find_token(&b("appl")), None);

    // Root reachable: {10, 20}
    let expected_root_bits = RangeSetBlaze::from_iter(vec![10, 20]);
    assert_eq!(tree.root.reachable_token_ids, expected_root_bits);
    assert_eq!(tree.find_token(b""), None); // Flag is false
}


#[test]
fn test_complex_case() {
    let tokens = vec![
        (1, b("a")),
        (2, b("b")),
        (10, b("ape")),
        (11, b("apple")),
        (12, b("apply")),
        (20, b("banana")),
        (99, b("")), // Add empty token
    ];
    let tree = VocabPrefixTree::build(&tokens);

    assert_eq!(tree.root.token_id, 99); // Root ID set by "" token
    assert_eq!(tree.root.prefix_length, 0);
    assert!(tree.has_empty_string_token()); // Flag is true
    assert_eq!(tree.max_token_id(), 99);
    assert_eq!(tree.root.children.len(), 2); // "a" and "b" are direct children
    assert!(tree.root.children.contains_key(&b("a")));
    assert!(tree.root.children.contains_key(&b("b")));

    // Check branch 'a'
    let node_a = tree.root.children.get(&b("a")).unwrap();
    assert_eq!(node_a.token_id, 1);
    assert_eq!(node_a.prefix_length, 1);
    assert_eq!(node_a.children.len(), 1); // "a" is prefix of "ape", "apple", "apply". "ape" is shortest.
                                         // So, "a" -> "pe" (for "ape")
                                         // "ape" -> "le" (for "apple")
                                         // "ape" -> "ly" (for "apply")
                                         // This structure is due to merge_nodes logic.
    assert!(node_a.children.contains_key(&b("pe"))); // Edge from "a" to "ape" node is "pe"

    let node_ape = node_a.children.get(&b("pe")).unwrap();
    assert_eq!(node_ape.token_id, 10);
    assert_eq!(node_ape.prefix_length, 3); // "ape"
    assert_eq!(node_ape.children.len(), 2); // "apple" and "apply" are children of "ape"
    assert!(node_ape.children.contains_key(&b("le"))); // Edge from "ape" to "apple" is "le"
    assert!(node_ape.children.contains_key(&b("ly"))); // Edge from "ape" to "apply" is "ly"

    let node_apple = node_ape.children.get(&b("le")).unwrap();
    assert_eq!(node_apple.token_id, 11);
    assert_eq!(node_apple.prefix_length, 5); // "apple"

    let node_apply = node_ape.children.get(&b("ly")).unwrap();
    assert_eq!(node_apply.token_id, 12);
    assert_eq!(node_apply.prefix_length, 5); // "apply"


    // Node "a" reachable: {1, 10, 11, 12}
    let expected_a_bits = RangeSetBlaze::from_iter(vec![1, 10, 11, 12]);
    assert_eq!(node_a.reachable_token_ids, expected_a_bits);


    // Check branch 'b'
    let node_b = tree.root.children.get(&b("b")).unwrap();
    assert_eq!(node_b.token_id, 2);
    assert_eq!(node_b.prefix_length, 1);
    assert_eq!(node_b.children.len(), 1);
    assert!(node_b.children.contains_key(&b("anana")));
    assert_eq!(node_b.children.get(&b("anana")).unwrap().token_id, 20);
    let node_banana = node_b.children.get(&b("anana")).unwrap();
    assert_eq!(node_banana.prefix_length, 6); // "banana"

    // Node "b" reachable: {2, 20}
    let expected_b_bits = RangeSetBlaze::from_iter(vec![2, 20]);
    assert_eq!(node_b.reachable_token_ids, expected_b_bits);

    // Test lookups
    assert_eq!(tree.find_token(&b("a")), Some(1));
    assert_eq!(tree.find_token(&b("ape")), Some(10));
    assert_eq!(tree.find_token(&b("apple")), Some(11));
    assert_eq!(tree.find_token(&b("apply")), Some(12));
    assert_eq!(tree.find_token(&b("b")), Some(2));
    assert_eq!(tree.find_token(&b("banana")), Some(20));
    assert_eq!(tree.find_token(&b("app")), None);
    assert_eq!(tree.find_token(&b("ban")), None);
    assert_eq!(tree.find_token(&b("c")), None);

    // Root reachable: {1, 2, 10, 11, 12, 20, 99}
    let expected_root_bits = RangeSetBlaze::from_iter(vec![1, 2, 10, 11, 12, 20, 99]);
    assert_eq!(tree.root.reachable_token_ids, expected_root_bits);
    assert_eq!(tree.find_token(b""), Some(99)); // Query for "" returns its ID (flag is true)
}

 #[test]
fn test_duplicate_token_bytes() {
    let tokens = vec![(1, b("a")), (2, b("ab")), (3, b("a"))];
    let tree = VocabPrefixTree::build(&tokens);

    assert_eq!(tree.root.token_id, 0);
    assert_eq!(tree.root.prefix_length, 0);
    assert!(!tree.has_empty_string_token());
    assert_eq!(tree.max_token_id(), 3);
    assert_eq!(tree.root.children.len(), 1);
    assert!(tree.root.children.contains_key(&b("a")));

    let node_a = tree.root.children.get(&b("a")).unwrap();
    assert_eq!(node_a.token_id, 3); // Last ID wins
    assert_eq!(node_a.prefix_length, 1); // Length of "a"
    assert_eq!(node_a.children.len(), 1);

    assert!(node_a.children.contains_key(&b("b")));
    let node_ab = node_a.children.get(&b("b")).unwrap();
    assert_eq!(node_ab.prefix_length, 2); // Length of "ab"
    assert_eq!(node_ab.token_id, 2);
    assert!(node_ab.children.is_empty());
    // Node "ab" reachable: {2}
    let expected_ab_bits = RangeSetBlaze::from_iter(vec![2]);
    assert_eq!(node_ab.reachable_token_ids, expected_ab_bits);

    // Node "a" reachable: {2, 3} (ID 1 was overwritten)
    let expected_a_bits = RangeSetBlaze::from_iter(vec![2, 3]);
    assert_eq!(node_a.reachable_token_ids, expected_a_bits);

    assert_eq!(tree.find_token(&b("a")), Some(3));
    assert_eq!(tree.find_token(&b("ab")), Some(2));
    // Root reachable: {2, 3}
    assert_eq!(tree.root.reachable_token_ids, expected_a_bits);
    assert_eq!(tree.find_token(b""), None); // Flag is false
}

#[test]
fn test_children_iteration() {
    let tokens = vec![
        (1, b("a")),
        (2, b("b")),
        (10, b("ape")), // "a" is prefix of "ape"
        (20, b("banana")), // "b" is prefix of "banana"
    ];
    let tree = VocabPrefixTree::build(&tokens);

    // Iterate root children: "a" and "b"
    let mut root_children_iter = tree.root_children();

    let (edge_a, node_a_ref) = root_children_iter.next().unwrap();
    assert_eq!(edge_a, &b("a"));
    assert_eq!(node_a_ref.token_id, 1);
    assert_eq!(node_a_ref.prefix_length, 1);

    let (edge_b, node_b_ref) = root_children_iter.next().unwrap();
    assert_eq!(edge_b, &b("b"));
    assert_eq!(node_b_ref.token_id, 2);
    assert_eq!(node_b_ref.prefix_length, 1);

    assert!(root_children_iter.next().is_none());

    // Iterate children of node 'a': "pe" (for "ape")
    let mut node_a_children_iter = node_a_ref.iter_children();
    let (edge_pe, node_ape_ref) = node_a_children_iter.next().unwrap();
    assert_eq!(edge_pe, &b("pe"));
    assert_eq!(node_ape_ref.token_id, 10);
    assert_eq!(node_ape_ref.prefix_length, 3); // "ape"
    assert!(node_a_children_iter.next().is_none());

    // Iterate children of node 'b': "anana" (for "banana")
    let mut node_b_children_iter = node_b_ref.iter_children();
    let (edge_anana, node_banana_ref) = node_b_children_iter.next().unwrap();
    assert_eq!(edge_anana, &b("anana"));
    assert_eq!(node_banana_ref.token_id, 20);
    assert_eq!(node_banana_ref.prefix_length, 6); // "banana"
    assert!(node_b_children_iter.next().is_none());
}

#[test]
fn test_find_longest_prefix_token() {
    let tokens = vec![
        (1, b("a")),
        (10, b("ape")),
        (11, b("apple")),
        (12, b("apply")),
        (20, b("banana")),
        (99, b("")), // Empty string token
    ];
    let tree = VocabPrefixTree::build(&tokens);

    // Test case 1: Exact match for "apple"
    assert_eq!(tree.find_longest_prefix_token(b"apple"), Some((11, &b("apple")[..])));

    // Test case 2: Input is longer than any token, "apple" is longest prefix
    assert_eq!(tree.find_longest_prefix_token(b"applepie"), Some((11, &b("apple")[..])));

    // Test case 3: Input is "apply", exact match
    assert_eq!(tree.find_longest_prefix_token(b"apply"), Some((12, &b("apply")[..])));

    // Test case 4: Input is "ape", exact match
    assert_eq!(tree.find_longest_prefix_token(b"ape"), Some((10, &b("ape")[..])));

    // Test case 5: Input is "ap", "a" is the longest prefix token
    assert_eq!(tree.find_longest_prefix_token(b"ap"), Some((1, &b("a")[..])));

    // Test case 6: Input is "application", "apply" is not a prefix, "a" is.
    assert_eq!(tree.find_longest_prefix_token(b"application"), Some((1, &b("a")[..])));


    // Test case 7: Input is "banana", exact match
    assert_eq!(tree.find_longest_prefix_token(&b("banana")), Some((20, &b("banana")[..])));

    // Test case 8: Input is "bananatart", "banana" is longest prefix
    assert_eq!(tree.find_longest_prefix_token(&b("bananatart")), Some((20, &b("banana")[..])));

    // Test case 9: Input is "b", no token starts with "b" other than "banana"
    // Since "" is a token, it should be returned.
    assert_eq!(tree.find_longest_prefix_token(b"b"), Some((99, &b("")[..])));


    // Test case 10: Input is "c", no token starts with "c". "" is a token.
    assert_eq!(tree.find_longest_prefix_token(b"c"), Some((99, &b("")[..])));

    // Test case 11: Input is "", "" is a token.
    assert_eq!(tree.find_longest_prefix_token(b""), Some((99, &b("")[..])));

    // Test case 12: Tree without empty string token
    let tokens_no_empty = vec![
        (1, b("a")),
        (11, b("apple")),
    ];
    let tree_no_empty = VocabPrefixTree::build(&tokens_no_empty);

    assert_eq!(tree_no_empty.find_longest_prefix_token(b"applepie"), Some((11, &b("apple")[..])));
    assert_eq!(tree_no_empty.find_longest_prefix_token(b"ax"), Some((1, &b("a")[..])));
    // Input "b", no token is a prefix, and "" is not a token.
    assert_eq!(tree_no_empty.find_longest_prefix_token(b"b"), None);
    // Input "", "" is not a token.
    assert_eq!(tree_no_empty.find_longest_prefix_token(b""), None);

    // Test case 13: Only empty string token
    let tokens_only_empty = vec![(55, b(""))];
    let tree_only_empty = VocabPrefixTree::build(&tokens_only_empty);
    assert!(tree_only_empty.has_empty_string_token());
    assert_eq!(tree_only_empty.find_longest_prefix_token(b"abc"), Some((55, &b("")[..])));
    assert_eq!(tree_only_empty.find_longest_prefix_token(b""), Some((55, &b("")[..])));

    // Test case 14: Empty tree
    let empty_tokens: Vec<(usize, Vec<u8>)> = vec![];
    let tree_empty = VocabPrefixTree::build(&empty_tokens);
    assert!(!tree_empty.has_empty_string_token());
    assert_eq!(tree_empty.find_longest_prefix_token(b"abc"), None);
    assert_eq!(tree_empty.find_longest_prefix_token(b""), None);

    // Test case 15: Tokens with spaces
    let space_tokens = vec![
        (31, b(" ")),
        (32, b("  ")),
        (34, b("    ")),
    ];
    let tree_spaces = VocabPrefixTree::build(&space_tokens);
    assert!(!tree_spaces.has_empty_string_token());

    // Exact match for one space
    assert_eq!(tree_spaces.find_longest_prefix_token(b" "), Some((31, &b(" ")[..])));
    // Exact match for two spaces
    assert_eq!(tree_spaces.find_longest_prefix_token(b"  "), Some((32, &b("  ")[..])));
    // Input is three spaces, longest prefix is two spaces
    assert_eq!(tree_spaces.find_longest_prefix_token(b"   "), Some((32, &b("  ")[..])));
    // Exact match for four spaces
    assert_eq!(tree_spaces.find_longest_prefix_token(b"    "), Some((34, &b("    ")[..])));
    // Input is five spaces, longest prefix is four spaces
    assert_eq!(tree_spaces.find_longest_prefix_token(b"     "), Some((34, &b("    ")[..])));
    // No match
    assert_eq!(tree_spaces.find_longest_prefix_token(b"a"), None);
    // Empty input, no empty token
    assert_eq!(tree_spaces.find_longest_prefix_token(b""), None);
}