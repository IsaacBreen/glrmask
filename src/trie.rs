use std::collections::{HashMap, HashSet, VecDeque};
use std::collections::BTreeMap;
use std::fmt::Debug;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone)]
pub struct TrieNode<E, T> {
    pub value: T,
    children: BTreeMap<E, Arc<Mutex<TrieNode<E, T>>>>,
    num_parents: usize,
}

impl<T, E: Ord> TrieNode<E, T> {
    pub fn new(value: T) -> TrieNode<E, T> {
        TrieNode {
            value,
            children: BTreeMap::new(),
            num_parents: 0,
        }
    }

    pub fn insert(&mut self, edge: E, child: Arc<Mutex<TrieNode<E, T>>>) -> Option<Arc<Mutex<TrieNode<E, T>>>> {
        child.try_lock().unwrap().num_parents += 1;
        if let Some(existing_child) = self.children.insert(edge, child) {
            existing_child.try_lock().unwrap().num_parents -= 1;
            Some(existing_child)
        } else {
            None
        }
    }

    pub fn insert_with(&mut self, edge: E, child: Arc<Mutex<TrieNode<E, T>>>, combine: impl FnOnce(&mut T, T)) {
        todo!()
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

    pub fn deep_clone(&self) -> Arc<Mutex<TrieNode<E, T>>> where T: Clone, E: Clone {
        let mut new_children = BTreeMap::new();
        for (edge, child) in &self.children {
            let new_child = child.try_lock().unwrap().deep_clone();
            new_children.insert(edge.clone(), new_child);
        }
        Arc::new(Mutex::new(TrieNode {
            value: self.value.clone(),
            children: new_children,
            num_parents: 0,
        }))
    }

    pub fn shallow_clone(&self) -> Arc<Mutex<TrieNode<E, T>>> where T: Clone, E: Clone {
        let mut new_children = BTreeMap::new();
        for (edge, child) in self.children.clone() {
            new_children.insert(edge, child);
        }
        Arc::new(Mutex::new(TrieNode {
            value: self.value.clone(),
            children: new_children,
            num_parents: 0,
        }))
    }

    pub fn replace_child_with_clone(&mut self, edge: &E) -> Arc<Mutex<TrieNode<E, T>>> where T: Clone, E: Clone {
        let child = self.children.get(edge).unwrap();
        let new_child = child.try_lock().unwrap().shallow_clone();
        self.insert(edge.clone(), new_child.clone());
        new_child
    }

    pub fn all_nodes(root: Arc<Mutex<TrieNode<E, T>>>) -> Vec<Arc<Mutex<TrieNode<E, T>>>> {
        let mut node_ptrs_in_order: Vec<*const TrieNode<E, T>> = Vec::new();
        let mut nodes: BTreeMap<*const TrieNode<E, T>, Arc<Mutex<TrieNode<E, T>>>> = BTreeMap::new();
        let mut queue: VecDeque<Arc<Mutex<TrieNode<E, T>>>> = VecDeque::new();
        queue.push_back(root);
        while let Some(node) = queue.pop_front() {
            if node_ptrs_in_order.contains(&(&*node.try_lock().unwrap() as *const TrieNode<E, T>)) {
                continue;
            }
            node_ptrs_in_order.push(&*node.try_lock().unwrap() as *const TrieNode<E, T>);
            nodes.insert(&*node.try_lock().unwrap() as *const TrieNode<E, T>, node.clone());
            let node = node.try_lock().unwrap();
            for (_, child) in &node.children {
                queue.push_back(child.clone());
            }
        }
        node_ptrs_in_order.into_iter().map(|ptr| nodes.get(&ptr).unwrap().clone()).collect()
    }

    pub fn map_t<F, U>(self, f: F) -> Arc<Mutex<TrieNode<E, U>>>
    where
        T: Clone,
        E: Clone,
        // todo: is it 'proper' to use `Copy` here?
        F: Copy + Fn(T) -> U,
    {
        let mut active_states: Vec<(Arc<Mutex<TrieNode<E, T>>>, Arc<Mutex<TrieNode<E, U>>>)> = Vec::new();
        let mut dormant_states: HashMap<*const TrieNode<E, T>, (usize, Arc<Mutex<TrieNode<E, U>>>)> = HashMap::new();
        let root = Arc::new(Mutex::new(TrieNode::new(f(self.value.clone()))));
        active_states.push((Arc::new(Mutex::new(self)), root.clone()));

        while let Some((node, new_node)) = active_states.pop() {
            let node = node.try_lock().unwrap();
            for (edge, child_arc) in &node.children {
                let child = child_arc.try_lock().unwrap();
                if let Some((num_parents_seen, new_child)) = dormant_states.get_mut(&(&*child as *const TrieNode<E, T>)) {
                    new_node.try_lock().unwrap().insert(edge.clone(), new_child.clone());
                    *num_parents_seen += 1;
                    if *num_parents_seen == child.num_parents {
                        active_states.push((child_arc.clone(), new_child.clone()));
                    }
                } else {
                    let new_child = Arc::new(Mutex::new(TrieNode::new(f(child.value.clone()))));
                    new_node.try_lock().unwrap().insert(edge.clone(), new_child.clone());
                    if child.num_parents == 1 {
                        active_states.push((child_arc.clone(), new_child.clone()));
                    } else {
                        dormant_states.insert(&*child as *const TrieNode<E, T>, (1, new_child.clone()));
                    }
                }
            }
        }

        root
    }

    pub fn flatten<F>(&self, is_terminal: F) -> BTreeMap<Vec<E>, T>
    where
        E: Clone,
        T: Clone,
        F: Copy + Fn(&T) -> bool,
    {
        let mut result = BTreeMap::new();
        self.flatten_recursive(&mut result, Vec::new(), is_terminal);
        result
    }

    fn flatten_recursive<F>(
        &self,
        result: &mut BTreeMap<Vec<E>, T>,
        path: Vec<E>,
        is_terminal: F,
    )
    where
        E: Clone,
        T: Clone,
        F: Copy + Fn(&T) -> bool,
    {
        if is_terminal(&self.value) {
            result.insert(path.clone(), self.value.clone());
        }
        for (edge, child) in &self.children {
            let mut new_path = path.clone();
            new_path.push(edge.clone());
            child.try_lock().unwrap().flatten_recursive(result, new_path, is_terminal);
        }
    }

    fn can_reach(&self, target: *const TrieNode<E, T>) -> bool {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        queue.push_back(self as *const TrieNode<E, T>);
        visited.insert(self as *const TrieNode<E, T>);

        while let Some(current_ptr) = queue.pop_front() {
            if current_ptr == target {
                return true;
            }

            // Safety: We need to ensure that the pointer is valid.
            // In this context, it's assumed that the TrieNode instances
            // are managed such that pointers to them remain valid as long
            // as they are reachable from the root.
            unsafe {
                if let Some(current) = current_ptr.as_ref() {
                    for child in current.children.values() {
                        let child_ptr = &*child.try_lock().unwrap() as *const TrieNode<E, T>;
                        if visited.insert(child_ptr) {
                            queue.push_back(child_ptr);
                        }
                    }
                }
            }
        }

        false
    }
}

impl<T: Clone, E: Ord + Clone> TrieNode<E, T> {
    // Simplified special_map function using a stack instead of a queue
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
        let mut stack: Vec<(Arc<Mutex<TrieNode<E, T>>>, V)> = Vec::new();
        let mut dormant_states: HashMap<*const TrieNode<E, T>, Vec<V>> = HashMap::new();

        stack.push((initial_node.clone(), initial_value));

        while let Some((node_arc, value)) = stack.pop() {
            let node = node_arc.try_lock().unwrap();
            process(&node.value, &value);

            for (edge, child_arc) in &node.children {
                let child = child_arc.try_lock().unwrap();
                let new_value = step(&value, edge, &child);
                let child_ptr = &*child as *const TrieNode<E, T>;

                let entry = dormant_states.entry(child_ptr).or_default();
                entry.push(new_value.clone());

                if entry.len() == child.num_parents {
                    let merged_value = merge(entry.clone());
                    dormant_states.remove(&child_ptr);
                    stack.push((child_arc.clone(), merged_value));
                }
            }
        }

        if !dormant_states.is_empty() {
            for (node_ptr, values) in &dormant_states {
                println!("dormant state: {:?}", node_ptr)
            }
            panic!("Leftover dormant states");
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
        let mut node_map: HashMap<*const TrieNode<E, T2>, Arc<Mutex<TrieNode<E, T>>>> = HashMap::new();
        let mut already_merged_values: HashSet<*const TrieNode<E, T>> = HashSet::new();

        let existing_value = node.try_lock().unwrap().value.clone();
        let new_value = t_merge(existing_value, other.try_lock().unwrap().value.clone());
        node.try_lock().unwrap().value = new_value;

        // Use a stack for iterative traversal
        let mut stack: Vec<(Vec<Arc<Mutex<TrieNode<E, T>>>>, Arc<Mutex<TrieNode<E, T2>>>)> = Vec::new();
        stack.push((vec![node.clone()], other.clone()));

        while let Some((current_nodes, other_node_arc)) = stack.pop() {
            let other_node = other_node_arc.try_lock().unwrap();

            for (edge, dest_other_node_arc) in &other_node.children {
                let dest_other_node = dest_other_node_arc.try_lock().unwrap();
                let mut new_nodes = Vec::new();

                for current_self_node in current_nodes.iter() {
                    let mut current_self_node_guard = current_self_node.try_lock().unwrap();

                    if let Some(child) = current_self_node_guard.get(edge) {
                        if !already_merged_values.contains(&(&*child.try_lock().unwrap() as *const TrieNode<E, T>)) {
                            let child_value = child.try_lock().unwrap().value.clone();
                            let merged_value = t_merge(child_value, dest_other_node.value.clone());
                            child.try_lock().unwrap().value = merged_value;
                        }
                        new_nodes.push(child.clone());
                    } else {
                        let other_node_ptr = &*dest_other_node as *const TrieNode<E, T2>;
                        if let Some(mapped_node) = node_map.get(&other_node_ptr) {
                            current_self_node_guard.insert(edge.clone(), mapped_node.clone());
                            new_nodes.push(mapped_node.clone());
                        } else {
                            let new_node = Arc::new(Mutex::new(TrieNode::new(t_merge(t_init(), dest_other_node.value.clone()))));
                            current_self_node_guard.insert(edge.clone(), new_node.clone());
                            node_map.insert(other_node_ptr, new_node.clone());
                            new_nodes.push(new_node);
                        }
                    }
                }
                stack.push((new_nodes, dest_other_node_arc.clone()));
            }
        }
    }
}

pub(crate) fn dump_structure<E, T>(root: Arc<Mutex<TrieNode<E, T>>>) where E: Debug, T: Debug {
    // TODO: modify this to use letter names "a" "b" "c"... for nodes rather than raw pointers.
    // TODO: make it possible to print edge value and node's internal value if they implement Debug
    let mut queue: VecDeque<Arc<Mutex<TrieNode<E, T>>>> = VecDeque::new();
    let mut seen: HashSet<*const TrieNode<E, T>> = HashSet::new();

    queue.push_back(root);

    while let Some(node) = queue.pop_front() {
        let node = node.try_lock().unwrap();
        let node_ptr = &*node as *const TrieNode<E, T>;
        println!("{:?}: num_parents: {}", node_ptr, node.num_parents);
        for (edge, child) in &node.children {
            let child_ptr = &*child.try_lock().unwrap() as *const TrieNode<E, T>;
            println!("  - {:?} -> {:?}", edge, child_ptr);
            if !seen.contains(&child_ptr) {
                seen.insert(child_ptr);
                queue.push_back(child.clone());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};
    use crate::trie::{dump_structure, TrieNode};

    #[test]
    fn test_trie() {
        let mut a = TrieNode::new("a");
        let mut b = TrieNode::new("b");
        let c = TrieNode::new("c");

        b.insert("b->c", Arc::new(Mutex::new(c)));
        a.insert("a->b", Arc::new(Mutex::new(b)));

        let mut a2 = TrieNode::new("a");
        let mut b2 = TrieNode::new("b");
        let d = TrieNode::new("d");

        b2.insert("b->d", Arc::new(Mutex::new(d)));
        a2.insert("a->b", Arc::new(Mutex::new(b2)));

        let a = Arc::new(Mutex::new(a));
        let a2 = Arc::new(Mutex::new(a2));

        println!("a structure (before)");
        dump_structure(a.clone());
        println!("a2 structure (before)");
        dump_structure(a2.clone());

        let merged = TrieNode::merge(
            a.clone(),
            a2.clone(),
            |x, y| { if x.is_empty() { y } else { assert_eq!(x, y); x } },
            || { "" }
        );

        println!("a structure (after)");
        dump_structure(a.clone());
        println!("a2 structure (after)e");
        dump_structure(a2.clone());
    }
}