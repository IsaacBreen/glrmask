//! Graph-Structured Stack (GSS).
//!
//! Used during GLR parsing to handle nondeterminism efficiently.
//! Multiple parse stacks share common prefixes via a DAG structure.

#![allow(dead_code)]

/// A node in the Graph-Structured Stack.
#[derive(Debug, Clone)]
pub struct GssNode {
    /// The state at this stack level.
    pub state: u32,
    /// Links to predecessor nodes (shared prefixes).
    pub predecessors: Vec<u32>,
}

/// A Graph-Structured Stack.
#[derive(Debug, Clone)]
pub struct Gss {
    /// All nodes in the GSS.
    nodes: Vec<GssNode>,
    /// Active frontier node IDs.
    frontier: Vec<u32>,
}

impl Gss {
    /// Create a new GSS with an initial node at the given state.
    pub fn new(initial_state: u32) -> Self {
        Self {
            nodes: vec![GssNode {
                state: initial_state,
                predecessors: Vec::new(),
            }],
            frontier: vec![0],
        }
    }

    /// Get the active frontier.
    pub fn frontier(&self) -> &[u32] {
        &self.frontier
    }

    /// Get a node by ID.
    pub fn get_node(&self, id: u32) -> &GssNode {
        &self.nodes[id as usize]
    }

    /// Add a new node and return its ID.
    pub fn add_node(&mut self, state: u32, predecessors: Vec<u32>) -> u32 {
        let id = self.nodes.len() as u32;
        self.nodes.push(GssNode {
            state,
            predecessors,
        });
        id
    }

    /// Set the frontier.
    pub fn set_frontier(&mut self, frontier: Vec<u32>) {
        self.frontier = frontier;
    }
}
