use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub(crate) enum PrecomputeGSSNode<GrammarToken, Leaf> {
    Internal(BTreeMap<GrammarToken, PrecomputeGSSNode<GrammarToken, Leaf>>),
    Leaf(Leaf),
}

impl<GrammarToken, Leaf> PrecomputeGSSNode<GrammarToken, Leaf>
where
    GrammarToken: Clone + Ord,
    Leaf: Clone,
{
    /// Flattens the tree structure into a map of token sequences to leaf values
    pub(crate) fn flatten(&self) -> BTreeMap<Vec<GrammarToken>, Leaf> {
        let mut result = BTreeMap::new();
        self.flatten_recursive(&mut result, Vec::new());
        result
    }

    /// Recursive helper method for flattening the tree
    fn flatten_recursive(
        &self,
        result: &mut BTreeMap<Vec<GrammarToken>, Leaf>,
        path: Vec<GrammarToken>,
    ) {
        match self {
            PrecomputeGSSNode::Internal(children) => {
                for (token, child) in children {
                    let mut new_path = path.clone();
                    new_path.push(token.clone());
                    child.flatten_recursive(result, new_path);
                }
            }
            PrecomputeGSSNode::Leaf(leaf) => {
                result.insert(path, leaf.clone());
            }
        }
    }
    
    /// Maps leaf values using a provided transformation function
    pub(crate) fn map<F, U>(&self, f: F) -> PrecomputeGSSNode<GrammarToken, U>
    where
        F: Copy + Fn(&Leaf) -> U,
    {
        match self {
            PrecomputeGSSNode::Internal(children) => {
                let mapped_children = children
                    .iter()
                    .map(|(token, child)| (token.clone(), child.map(f)))
                    .collect();
                PrecomputeGSSNode::Internal(mapped_children)
            }
            PrecomputeGSSNode::Leaf(leaf) => {
                PrecomputeGSSNode::Leaf(f(leaf))
            }
        }
    }
}