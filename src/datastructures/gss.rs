use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GSSNode<T> {
    value: T,
    predecessors: Vec<Arc<GSSNode<T>>>,
}

impl<T> GSSNode<T> {
    pub fn new(value: T) -> Self {
        Self {
            value,
            predecessors: Vec::new(),
        }
    }

    pub fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let mut iter = iter.into_iter();
        let mut root = Self::new(iter.next().unwrap());
        for value in iter {
            root = root.push(value);
        }
        root
    }

    pub fn push(self, value: T) -> Self {
        let mut new_node = Self::new(value);
        new_node.predecessors.push(Arc::new(self));
        new_node
    }

    pub fn pop(&self) -> Vec<Arc<Self>> {
        self.predecessors.clone()
    }

    pub fn popn(&self, n: usize) -> Vec<Arc<Self>>
    where
        T: Clone,
    {
        if n == 0 {
            return vec![Arc::new(self.clone())];
        }
        let mut nodes = Vec::new();
        for popped in self.pop() {
            nodes.extend(popped.popn(n - 1));
        }
        nodes
    }

    pub fn peek(&self) -> &T {
        &self.value
    }

    pub fn flatten(&self) -> Vec<Vec<T>>
    where
        T: Clone,
    {
        let mut result = Vec::new();
        let mut stack = Vec::new();
        stack.push((self, Vec::new()));
        while let Some((node, mut path)) = stack.pop() {
            path.push(node.value.clone());
            if node.predecessors.is_empty() {
                result.push(path);
            } else {
                for predecessor in &node.predecessors {
                    stack.push((predecessor, path.clone()));
                }
            }
        }
        result
    }

    pub fn flatten_bulk(nodes: &[Self]) -> Vec<Vec<T>>
    where
        T: Clone,
    {
        nodes.iter().flat_map(|node| node.flatten()).collect()
    }

    pub fn merge(&mut self, mut other: Self)
    where
        T: PartialEq,
    {
        assert!(self.value == other.value);
        self.predecessors.append(&mut other.predecessors);
    }

    pub fn map<F, U>(&self, f: F) -> GSSNode<U>
    where
        F: Copy + Fn(&T) -> U,
    {
        GSSNode {
            value: f(&self.value),
            predecessors: self.predecessors.clone().into_iter()
                .map(|node| Arc::new(node.map(f)))
                .collect(),
        }
    }
}

impl<T> Drop for GSSNode<T> {
    fn drop(&mut self) {
        let mut cur_nodes = std::mem::take(&mut self.predecessors);
        while let Some(node) = cur_nodes.pop() {
            if let Ok(mut inner_node) = Arc::try_unwrap(node) {
                cur_nodes.append(&mut inner_node.predecessors);
            }
        }
    }
}

pub trait GSSTrait<T: Clone> {
    type Peek<'a> where T: 'a, Self: 'a;
    fn peek(&self) -> Self::Peek<'_>;
    fn push(&self, value: T) -> GSSNode<T>;
    fn pop(&self) -> Vec<Arc<GSSNode<T>>>;
    fn popn(&self, n: usize) -> Vec<Arc<GSSNode<T>>>;
}

impl<T: Clone> GSSTrait<T> for GSSNode<T> {
    type Peek<'a> = &'a T where T: 'a;

    fn peek(&self) -> Self::Peek<'_> {
        &self.value
    }

    fn push(&self, value: T) -> GSSNode<T> {
        let mut new_node = GSSNode::new(value);
        new_node.predecessors.push(Arc::new(self.clone()));
        new_node
    }

    fn pop(&self) -> Vec<Arc<GSSNode<T>>> {
        self.predecessors.clone()
    }

    fn popn(&self, n: usize) -> Vec<Arc<GSSNode<T>>> {
        if n == 0 {
            return vec![Arc::new(self.clone())];
        }
        let mut nodes = Vec::new();
        for popped in self.pop() {
            nodes.extend(popped.popn(n - 1));
        }
        nodes
    }
}

impl<T: Clone> GSSTrait<T> for Arc<GSSNode<T>> {
    type Peek<'a> = &'a T where T: 'a;

    fn peek(&self) -> Self::Peek<'_> {
        &self.value
    }

    fn push(&self, value: T) -> GSSNode<T> {
        let mut new_node = GSSNode::new(value);
        new_node.predecessors.push(self.clone());
        new_node
    }

    fn pop(&self) -> Vec<Arc<GSSNode<T>>> {
        self.predecessors.clone()
    }

    fn popn(&self, n: usize) -> Vec<Arc<GSSNode<T>>> {
        if n == 0 {
            return vec![self.clone()];
        }
        let mut nodes = Vec::new();
        for popped in self.pop() {
            nodes.extend(popped.popn(n - 1));
        }
        nodes
    }
}

impl<T: Clone> GSSTrait<T> for Option<Arc<GSSNode<T>>> {
    type Peek<'a> = Option<&'a T> where T: 'a;

    fn peek(&self) -> Self::Peek<'_> {
        self.as_ref().map(|node| node.peek())
    }

    fn push(&self, value: T) -> GSSNode<T> {
        self.clone().map(|node| node.push(value.clone())).unwrap_or_else(|| GSSNode::new(value))
    }

    fn pop(&self) -> Vec<Arc<GSSNode<T>>> {
        self.as_ref().map(|node| node.pop()).unwrap_or_default()
    }

    fn popn(&self, n: usize) -> Vec<Arc<GSSNode<T>>> {
        self.as_ref().map(|node| node.popn(n)).unwrap_or_default()
    }
}

impl<T: Clone> GSSTrait<T> for Option<GSSNode<T>> {
    type Peek<'a> = Option<&'a T> where T: 'a;

    fn peek(&self) -> Self::Peek<'_> {
        self.as_ref().map(|node| node.peek())
    }

    fn push(&self, value: T) -> GSSNode<T> {
        self.clone().map(|node| node.push(value.clone())).unwrap_or_else(|| GSSNode::new(value))
    }

    fn pop(&self) -> Vec<Arc<GSSNode<T>>> {
        self.as_ref().map(|node| node.pop()).unwrap_or_default()
    }

    fn popn(&self, n: usize) -> Vec<Arc<GSSNode<T>>> {
        self.as_ref().map(|node| node.popn(n)).unwrap_or_default()
    }
}

pub trait BulkMerge<T> {
    fn bulk_merge(&mut self);
}

impl<T: Clone + Ord> BulkMerge<T> for Vec<Arc<GSSNode<T>>> {
    fn bulk_merge(&mut self) {
        // todo: should be possible to avoid cloning T in some cases by using &T in this map,
        //  but we need to be careful about lifetimes. If we use `node.as_ref().value`, then node
        //  will go out of bounds while the reference to its value is still inside `groups`.
        let mut groups: BTreeMap<T, HashMap<_, Arc<GSSNode<T>>>> = BTreeMap::new();
        for node in self.drain(..) {
            groups.entry(node.value.clone()).or_default().entry(Arc::as_ptr(&node)).or_insert(node);
        }
        for mut group in groups.into_values() {
            let mut group = group.into_values().collect::<Vec<_>>();
            let mut first = group.pop().unwrap();
            if group.is_empty() {
                self.push(first);
            } else {
                let mut predecessors_set: BTreeMap<_, _> = BTreeMap::new();
                for sibling in group {
                    for predecessor in &sibling.predecessors {
                        predecessors_set.insert(Arc::as_ptr(predecessor), predecessor.clone());
                    }
                }
                let first_mut_ref = Arc::make_mut(&mut first);
                first_mut_ref.predecessors = predecessors_set.into_values().collect();
                self.push(first);
            }
        }
    }
}