use std::collections::{HashMap, HashSet, BinaryHeap};
use std::collections::BTreeMap;
use std::fmt::Debug;
use std::sync::{Arc, Mutex};
use std::cmp::Ordering;

#[derive(Debug, Clone)]
pub struct TrieNode<E, T> {
    pub value: T,
    children: BTreeMap<E, Arc<Mutex<TrieNode<E, T>>>>,
    max_depth: usize,
}

#[derive(Debug)]
pub enum InsertError {
    CycleDetected,
}

impl<T, E: Ord> TrieNode<E, T> {
    pub fn new(value: T) -> TrieNode<E, T> {
        TrieNode {
            value,
            children: BTreeMap::new(),
            max_depth: 1,
        }
    }

    fn would_create_cycle(&self, new_child: &Arc<Mutex<TrieNode<E, T>>>) -> bool {
        let mut visited = HashSet::new();
        let mut stack = Vec::new();
        
        // Start from the new child
        stack.push(new_child.clone());
        visited.insert(Arc::as_ptr(new_child));
        
        // The node we're trying to add to
        let self_ptr = self as *const TrieNode<E, T>;
        
        while let Some(current) = stack.pop() {
            let current_node = current.try_lock().unwrap();
            
            for child in current_node.children.values() {
                let child_ptr = Arc::as_ptr(child);
                
                // If we find the current node in the child's descendants, we have a cycle
                if child_ptr as *const _ == self_ptr {
                    return true;
                }
                
                if visited.insert(child_ptr) {
                    stack.push(child.clone());
                }
            }
        }
        
        false
    }

    fn update_max_depths(&mut self) {
        let mut visited = HashSet::new();
        let mut stack = vec![(self as *const TrieNode<E, T>, 1)];
        let mut new_depths = HashMap::new();

        while let Some((node_ptr, depth)) = stack.pop() {
            if !visited.insert(node_ptr) {
                continue;
            }

            // Safety: We know the pointer is valid as it comes from our trie
            let node = unsafe { &*(node_ptr) };
            let current_max = new_depths.entry(node_ptr).or_insert(depth);
            *current_max = (*current_max).max(depth);

            for child in node.children.values() {
                let child_ptr = &*child.try_lock().unwrap() as *const TrieNode<E, T>;
                stack.push((child_ptr, depth + 1));
            }
        }

        // Update all the max_depths
        for (node_ptr, new_depth) in new_depths {
            // Safety: We know the pointer is valid as it comes from our trie
            unsafe {
                let node = &mut *(node_ptr as *mut TrieNode<E, T>);
                node.max_depth = new_depth;
            }
        }
    }

    pub fn insert(&mut self, edge: E, child: Arc<Mutex<TrieNode<E, T>>>) -> Result<(), InsertError> {
        // Check for cycles before making any modifications
        if self.would_create_cycle(&child) {
            return Err(InsertError::CycleDetected);
        }

        // Insert the new child
        self.children.insert(edge, child);
        
        // Update max_depths for all affected nodes
        self.update_max_depths();
        
        Ok(())
    }

    pub fn get(&self, edge: &E) -> Option<Arc<Mutex<TrieNode<E, T>>>> {
        self.children.get(edge).cloned()
    }

    pub fn children(&self) -> &BTreeMap<E, Arc<Mutex<TrieNode<E, T>>>> {
        &self.children
    }

    pub fn is_empty(&self) -> bool {
        self.children.is_empty()
    }

    pub fn max_depth(&self) -> usize {
        self.max_depth
    }

    pub fn all_nodes(root: Arc<Mutex<TrieNode<E, T>>>) -> Vec<Arc<Mutex<TrieNode<E, T>>>> {
        let mut node_ptrs_in_order: Vec<*const TrieNode<E, T>> = Vec::new();
        let mut nodes: BTreeMap<*const TrieNode<E, T>, Arc<Mutex<TrieNode<E, T>>>> = BTreeMap::new();
        let mut queue: Vec<Arc<Mutex<TrieNode<E, T>>>> = Vec::new();
        queue.push(root);
        while let Some(node) = queue.pop() {
            if node_ptrs_in_order.contains(&(&*node.try_lock().unwrap() as *const TrieNode<E, T>)) {
                continue;
            }
            node_ptrs_in_order.push(&*node.try_lock().unwrap() as *const TrieNode<E, T>);
            nodes.insert(&*node.try_lock().unwrap() as *const TrieNode<E, T>, node.clone());
            let node = node.try_lock().unwrap();
            for (_, child) in &node.children {
                queue.push(child.clone());
            }
        }
        node_ptrs_in_order.into_iter().map(|ptr| nodes.get(&ptr).unwrap().clone()).collect()
    }
}

#[derive(Debug)]
struct QueueItem<E, T, V> {
    max_depth: usize,
    node: Arc<Mutex<TrieNode<E, T>>>,
    value: V,
}

impl<E, T, V> PartialEq for QueueItem<E, T, V> {
    fn eq(&self, other: &Self) -> bool {
        self.max_depth == other.max_depth
    }
}

impl<E, T, V> Eq for QueueItem<E, T, V> {}

impl<E, T, V> PartialOrd for QueueItem<E, T, V> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<E, T, V> Ord for QueueItem<E, T, V> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse order for min-heap
        other.max_depth.cmp(&self.max_depth)
    }
}


impl<T: Clone, E: Ord + Clone> TrieNode<E, T> {
    pub fn special_map<V>(
        initial_node: Arc<Mutex<TrieNode<E, T>>>,
        initial_value: V,
        mut step: impl FnMut(&V, &E, &TrieNode<E, T>) -> V,
        mut merge: impl FnMut(Vec<V>) -> V,
        mut process: impl FnMut(&T, &V),
    ) where
        V: Clone,
        E: Ord,
    {
        // Priority queue ordered by max_depth (min heap)
        let mut queue = BinaryHeap::new();
        let mut processed = HashSet::new();
        
        // Initialize with root node
        queue.push(QueueItem {
            max_depth: initial_node.try_lock().unwrap().max_depth,
            node: initial_node.clone(),
            value: initial_value
        });

        while let Some(QueueItem { max_depth: _, node: node_arc, value }) = queue.pop() {
            let node = node_arc.try_lock().unwrap();
            let node_ptr = &*node as *const TrieNode<E, T>;

            if !processed.insert(node_ptr) {
                continue;
            }

            // Process current node
            process(&node.value, &value);

            // Process children
            for (edge, child_arc) in &node.children {
                let child = child_arc.try_lock().unwrap();
                let new_value = step(&value, edge, &child);
                
                queue.push(QueueItem {
                    max_depth: child.max_depth,
                    node: child_arc.clone(),
                    value: new_value
                });
            }
        }
    }

    pub fn merge<T2>(
        node: Arc<Mutex<TrieNode<E, T>>>,
        other: Arc<Mutex<TrieNode<E, T2>>>,
        t_merge: impl Fn(T, T2) -> T,
        t_init: impl Fn() -> T,
    )
    where
        T2: Clone,
    {
        // A map to track the mapping of nodes from `other` to `self`
        let mut node_map: HashMap<*const TrieNode<E, T2>, Arc<Mutex<TrieNode<E, T>>>> = HashMap::new();
        let mut already_merged_values: HashSet<*const TrieNode<E, T>> = HashSet::new();

        // Special case: merge T for the root node
        let existing_value = node.try_lock().unwrap().value.clone();
        let new_value = t_merge(existing_value, other.try_lock().unwrap().value.clone());
        node.try_lock().unwrap().value = new_value;

        TrieNode::special_map(
            other.clone(),
            (),
            // Step function
            |current_nodes: &(), edge: &E, dest_other_node: &TrieNode<E, T2>| {
                let mut new_nodes = Vec::new();
                let current_nodes = vec![node.clone()];

                for current_self_node in current_nodes {
                    let mut current_self_node_guard = current_self_node.try_lock().unwrap();

                    // Check if the current node has an equivalent edge
                    if let Some(child) = current_self_node_guard.get(edge) {
                        if !already_merged_values.contains(&(&*child.try_lock().unwrap() as *const TrieNode<E, T>)) {
                            // Merge the values
                            let child_value = child.try_lock().unwrap().value.clone();
                            let merged_value = t_merge(child_value, dest_other_node.value.clone());
                            child.try_lock().unwrap().value = merged_value;
                        }
                        new_nodes.push(child);
                    } else {
                        // Check if the `other` node is already mapped
                        let other_node_ptr = dest_other_node as *const TrieNode<E, T2>;
                        if let Some(mapped_node) = node_map.get(&other_node_ptr) {
                            // Add the mapped node as a child
                            current_self_node_guard.insert(edge.clone(), mapped_node.clone());
                            new_nodes.push(mapped_node.clone());
                        } else {
                            // Create a new node and map it
                            let new_node = Arc::new(Mutex::new(TrieNode::new(t_merge(t_init(), dest_other_node.value.clone()))));
                            current_self_node_guard.insert(edge.clone(), new_node.clone());
                            node_map.insert(other_node_ptr, new_node.clone());
                            new_nodes.push(new_node);
                        }
                    }
                }
                ()
            },
            |_: Vec<()>| {
                ()
            },
            // Process function
            |_, _| {}
        );
    }
}

pub(crate) fn dump_structure<E, T>(root: Arc<Mutex<TrieNode<E, T>>>) where E: Debug, T: Debug {
    let mut queue = Vec::new();
    let mut seen = HashSet::new();

    queue.push(root);

    while let Some(node) = queue.pop() {
        let node = node.try_lock().unwrap();
        let node_ptr = &*node as *const TrieNode<E, T>;
        println!("{:?}: max_depth: {}", node_ptr, node.max_depth);
        for (edge, child) in &node.children {
            let child_ptr = &*child.try_lock().unwrap() as *const TrieNode<E, T>;
            println!("  - {:?} -> {:?}", edge, child_ptr);
            if !seen.contains(&child_ptr) {
                seen.insert(child_ptr);
                queue.push(child.clone());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cycle_detection() {
        let mut a = TrieNode::new("a");
        let b = Arc::new(Mutex::new(TrieNode::new("b")));
        let c = Arc::new(Mutex::new(TrieNode::new("c")));

        // Create a->b->c
        b.try_lock().unwrap().insert("b->c", c.clone()).unwrap();
        a.insert("a->b", b.clone()).unwrap();

        // Try to create a cycle by making c->a
        let a_arc = Arc::new(Mutex::new(a));
        assert!(matches!(
            c.try_lock().unwrap().insert("c->a", a_arc.clone()),
            Err(InsertError::CycleDetected)
        ));

        // Verify max_depths weren't changed by the failed insertion
        assert_eq!(a_arc.try_lock().unwrap().max_depth(), 3);
        assert_eq!(b.try_lock().unwrap().max_depth(), 2);
        assert_eq!(c.try_lock().unwrap().max_depth(), 1);
    }

    #[test]
    fn test_max_depth_updates() {
        let mut root = TrieNode::new("root");
        let child1 = Arc::new(Mutex::new(TrieNode::new("child1")));
        let child2 = Arc::new(Mutex::new(TrieNode::new("child2")));
        let grandchild = Arc::new(Mutex::new(TrieNode::new("grandchild")));

        // Add child1 and verify depths
        root.insert("root->child1", child1.clone()).unwrap();
        assert_eq!(root.max_depth(), 2);
        assert_eq!(child1.try_lock().unwrap().max_depth(), 1);

        // Add child2 and verify depths
        root.insert("root->child2", child2.clone()).unwrap();
        assert_eq!(root.max_depth(), 2);
        assert_eq!(child2.try_lock().unwrap().max_depth(), 1);

        // Add grandchild to child1 and verify depths update
        child1.try_lock().unwrap().insert("child1->grandchild", grandchild.clone()).unwrap();
        assert_eq!(root.max_depth(), 3);
        assert_eq!(child1.try_lock().unwrap().max_depth(), 2);
        assert_eq!(grandchild.try_lock().unwrap().max_depth(), 1);
    }

    #[test]
    fn test_special_map_depth_order() {
        let root = Arc::new(Mutex::new(TrieNode::new(0)));
        let child1 = Arc::new(Mutex::new(TrieNode::new(1)));
        let child2 = Arc::new(Mutex::new(TrieNode::new(2)));
        let grandchild = Arc::new(Mutex::new(TrieNode::new(3)));

        root.try_lock().unwrap().insert("0->1", child1.clone()).unwrap();
        root.try_lock().unwrap().insert("0->2", child2.clone()).unwrap();
        child1.try_lock().unwrap().insert("1->3", grandchild.clone()).unwrap();

        let mut processed_order = Vec::new();
        TrieNode::special_map(
            root.clone(),
            (),
            |_, _, _| (),
            |_| (),
            |value, _| processed_order.push(*value)
        );

        // Verify nodes are processed in order of increasing depth
        assert_eq!(processed_order, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_all_nodes() {
        let root = Arc::new(Mutex::new(TrieNode::new("root")));
        let child1 = Arc::new(Mutex::new(TrieNode::new("child1")));
        let child2 = Arc::new(Mutex::new(TrieNode::new("child2")));
        let grandchild = Arc::new(Mutex::new(TrieNode::new("grandchild")));

        root.try_lock().unwrap().insert("r->c1", child1.clone()).unwrap();
        root.try_lock().unwrap().insert("r->c2", child2.clone()).unwrap();
        child1.try_lock().unwrap().insert("c1->gc", grandchild.clone()).unwrap();

        let all_nodes = TrieNode::all_nodes(root.clone());
        
        // Check that all nodes are present
        assert_eq!(all_nodes.len(), 4);

        // Check that the root is present
        assert!(all_nodes.iter().any(|node| Arc::ptr_eq(node, &root)));
        assert!(all_nodes.iter().any(|node| Arc::ptr_eq(node, &child1)));
        assert!(all_nodes.iter().any(|node| Arc::ptr_eq(node, &child2)));
        assert!(all_nodes.iter().any(|node| Arc::ptr_eq(node, &grandchild)));
    }
}