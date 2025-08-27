//! A simple arena allocator for storing nodes of a graph-like structure.

use crate::json_serialization::{JSONConvertible, JSONNode};
use serde::{de::DeserializeOwned, Deserialize, Serialize};

/// A unique identifier for a node within an `Arena`.
pub type NodeId = usize;

/// An arena allocator that stores nodes of type `T` and provides stable `NodeId`s.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Arena<T> {
    nodes: Vec<T>,
}

impl<T> Arena<T> {
    /// Creates a new, empty `Arena`.
    pub fn new() -> Self {
        Arena { nodes: Vec::new() }
    }

    /// Allocates a new node in the arena and returns its stable `NodeId`.
    pub fn alloc(&mut self, node: T) -> NodeId {
        let id = self.nodes.len();
        self.nodes.push(node);
        id
    }

    /// Returns a reference to the node with the given `NodeId`.
    ///
    /// # Panics
    ///
    /// Panics if `id` is out of bounds.
    pub fn get(&self, id: NodeId) -> &T {
        &self.nodes[id]
    }

    /// Returns a mutable reference to the node with the given `NodeId`.
    ///
    /// # Panics
    ///
    /// Panics if `id` is out of bounds.
    pub fn get_mut(&mut self, id: NodeId) -> &mut T {
        &mut self.nodes[id]
    }

    /// Returns the number of nodes currently in the arena.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns `true` if the arena contains no nodes.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Returns an iterator over the nodes and their `NodeId`s.
    pub fn iter(&self) -> impl Iterator<Item = (NodeId, &T)> {
        self.nodes.iter().enumerate()
    }
}

impl<T> Default for Arena<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: JSONConvertible> JSONConvertible for Arena<T> {
    fn to_json(&self) -> JSONNode {
        let nodes_json: Vec<JSONNode> = self.nodes.iter().map(|n| n.to_json()).collect();
        JSONNode::Array(nodes_json)
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Array(arr) => {
                let nodes: Result<Vec<T>, String> =
                    arr.into_iter().map(T::from_json).collect();
                nodes.map(|nodes| Arena { nodes })
            }
            _ => Err("Expected JSONNode::Array for Arena".to_string()),
        }
    }
}