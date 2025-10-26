**Internal edges**
`HashSet<(Some(NonTerminalID), StateID, TerminalID, (usize, NonTerminalID))>`
*We've reduced by this nonterminal, now this is a possible revealed state. We keep parsing with this terminal ID and eventually we reduce below the bottom by this amount with this nonterminal.*

**Escape edges**
`HashSet<(NonTerminalID, StateID, TerminalID, StateID, StateID)>`
*We've reduced by this nonterminal, now this is a possible revealed state and goto state. We keep parsing with this terminal ID and eventually we find a shift to this other state.*

**Start reduce edges**
`HashSet<(None(NonTerminalID), StateID, TerminalID, (usize, NonTerminalID))>`
*We start in this state with this terminal. We keep parsing until eventually we reduce below the bottom by this amount with this nonterminal.*

**Start shift edges**
`HashSet<(StateID, TerminalID, StateID)>`
*We start in this state with this terminal. We immediately shift to this state.*


Can unify internal edges and start reduce edges.
`HashSet<(Option<NonTerminalID>, StateID, TerminalID, (usize, NonTerminalID))>`
Can also unify escape and start shift edges as
`HashSet<(Option<NonTerminalID>, StateID, TerminalID, Vec<StateID>)>`
Can unify both as
`HashSet<(Option<NonTerminalID>, StateID, TerminalID, (usize, NonTerminalID) | Vec<StateID>)>`


There is one start/end node (combiend). For each nonterminal, there is internal node.
Node IDs can then just be `Option<NonTerminalID>`, with the start/end node ID being `None`.


**Super edges**
`HashSet<(Option<NonTerminalID>, TerminalID, (usize, NonTerminalID), LLMTokenBV, PrecomputeNodeIndex1, PrecomputeNodeIndex1)>`
If `None`, it's the start node.

Initial nodes and values.
Values are
`HashSet<(Vec<StateID>, LLMTokenBV, PrecomputeNodeIndex1)>`

Initial value for tokenizer ID/precompute1 node: empty state vec, all LLM tokens, precompute1 node.

Map from tokenizer state ID to precompute1 node index.


There's two parts to computing this.
**First stage**
The first stage is this. It doesn't use the precompute1 tree.
- Loop through (nonterminal, terminal, state)
    - Initialize a stack with the state
    - Perform a goto for the nonterminal.
    - Continue parsing as normal.
    - Once we pop below bottom of stack, note the nonterminal R2 we are reducing by and the pop number n *below the stack*.
    - Add an edge to node R2 with (terminal, state, n)

In implementing the parser, don't be fancy. Use a rudimentary approach. Just a queue of stacks, and when we hit a split we literally copy the stacks and put them into the queue.
Process one queue item at a time.
Once we pop below zero, we add an edge to the set.

What about shift?
Once we hit a shift, what should we do?
Shift to the stack. (Fun fact: stack size will either be 2 or 3. But that doesn't matter.)
`HashSet<(Option<NonTerminalID>, StateID, TerminalID, (usize, NonTerminalID) | Vec<StateID>)>`
So, the first nonterminal is the current node (src), state ID is the revealed state ID, and the vector is the top two states - a goto state and the shifted state.


**Second stage**
Here's the second stage. It's basically for 'super' edges.
Now suppose we do special map over precompute1 tree. Edge processing like this.
The special map values are `HashSet<(Vec<StateID>, LLMTokenBV, PrecomputeNode1Index)>`
- We have a terminal and a LLM bv for this precompute1 edge, and we know pci1'
- Loop through our state vec/LLM bv/pci1
    - start at the start node.
    - process start reduce edges
        - initialize a queue of `(Option<NonTerminalID>, Vec<StateID>, LLMTokenBV)`
        - Initialize with None, this state vec, this bv
        - Process the queue. Pop an item.
            - Look for all reduce edges whose nonterminal/terminal matches and state ID matches top of state vec. Pop the number from the stack.
                - If there's still anything left on the stack, add to the queue the remainder and the start reduce edge's dest nonterminal ID.
                - If the stack is empty, add a super edge from the src nt ID to the dst, with this terminal, the remainder of the pop n (could be zero), the same LLM bv, pci1 as the first index, and the dest is pci1'
            - Look for all escape edges whose nonterminal/terminal/state match.
                - Add to the output set. Push the edge's final state ID vec to the current state vec (ie extend). Intersect the LLM token bv. Keep pci1 the same.

The precompute merge fn simply merges the hashsets.

...

Now let's look at get_mask4. It'll use this tree. Its output should be identical to get_mask3.
Obviously it won't use trie's special map. It'll manage its own queue.
In the initial nodes and values, the values will include the current precompute1 index. It'll also include optional nonterminals - ie node indices in this new tree. Specifically, it'll be `HashSet<(PrecomputeNode1Index, Option<NonTerminalID>, LLMTokenBV)>`.
We traverse all compatible edges, normal and super.
If super, it's only compatible if the first pci1 index matches the current index. If it does, the current pci1 index gets replaced by the edge's second one.

---

# UPDATES, CORRECTIONS, CLARIFICATIONS

Here are some changes and clarifications. These superseded the above.

Get rid of LLM only edges. Shouldn't exist.
Add a graphviz visualization

Basically, in get_mask4, there is a 'state' at each node `Option<NonTerminalID>`, and that state contains `(PrecomputeNodeIndex1, LLMTokenBV)`.
Why the precompute1 node index, given we're not actually using the precompute1 node? Because it tells us which *order* to process queue items in.
Specifically, we process in the topological order of the precompute1 tree.
For now, you can assume precompute1 indices are already in this order.


One node per nonterminal with index `Some(...)`, and one **start node** overall with index `None`.

**Three** edge types:

**Super edges:**
`HashSet<(Option<NonTerminalID>, TerminalID, PrecomputeNodeIndex1, usize, NonTerminalID, LLMTokenBV, PrecomputeNodeIndex1)>`
*Edge from this (optional) ntid, conditional on this tid and on the current precompute1 node being this, pops n, destination noe (for queue item) is last nonterminal ID, we filter by LLM bv mask, and set the precompute1 node index for the queue item to this new one.*

**Reduce edges:**
`HashSet<(Option<NonTerminalID>, TerminalID, StateID, usize, NonTerminalID)>`
*Edge from this (optional) ntid, conditional on this tid and the current state being this, pops n, destination is last nonterminal ID.*

**Shift edges:**
`HashSet<(Option<NonTerminalID>, TerminalID, StateID, Vec<StateID>)>`
*Edge from this (optional) ntid, conditional on this tid and current state being this, pushes this vector to the stack.*


I don't like these two section names "Reduce-cross groups" and "Shift groups". What we're doing there is computing the reduce edges and the shift edges.
Building the initial stacks shouldn't be like that. In fact, what does `gotos_for` do?? Doesn't make sense. There's only one goto for each state/nonterm combination.

Ok let's redo the first stage

**First stage**
- Loop through (nonterminal, terminal, state)
    - Initialize a stack with the state
    - Perform a goto for the nonterminal.
    - Continue parsing as normal.
    - There are two ways this can end. Either we encounter a shift, or we eventually reduce below the bottom of the parse stack. (Or we don't find an action for this state.)
    - Once a shift is encountered:
        - Shift to the stack as normal.
        - Fun fact: At this point, the stack size can either be 2 or 3, because there are no zero-reductions in the table.
            - One item for the initial state, one for the goto, and one for the shift from the goto. OR
            - One item for the initial state, one for an immediate shift from the initial.
        - Add a shift edge `HashSet<(Option<NonTerminalID>, TerminalID, StateID, Vec<StateID>)>` with this nonterminal/terminal and initial state (top of the loop), and the 1 or 2 other items on the stack (excluding the bottom initial item - the vector is items that will be pushed to the stack, whereas the lone state is what'll be checked, a condition for using this edge).
    - Once we pop below bottom of stack, note the nonterminal R2 we are reducing by and the pop number n *below the stack*.
        - Add an edge `HashSet<(Option<NonTerminalID>, TerminalID, StateID, usize, NonTerminalID)>` with this nonterminal/terminal and initial state (top of the loop), the number of pops remaining to do below the bottom of the stack, and with the final nonterminal being the nonterminal R2 we're reducing with.

**Second stage**
Use `Trie::special_map_grouped` on the precompute1 tree where the initial values are `HashSet<(Vec<StateID>, LLMTokenBV, PrecomputeNode1Index)>`.
Here's the edge function.
- We have a terminal and a LLM bv for this precompute1 edge, and we know pci1' - the destination precompute1 node. (In reality we get `&V, &EK, &OrderedHashMap<Trie2Index, EV>`.)
- We have our current values at this node, `HashSet<(Vec<StateID>, LLMTokenBV, PrecomputeNode1Index)>`.
- Loop through and intersect inplace each LLM token bv in the current values with this edge's LLM mask.
- Loop through the current values' state vec/LLM bv/pci1
    - initialize a queue of `(Option<NonTerminalID>, Vec<StateID>, LLMTokenBV)`
    - start at the **start node**, the node with index `None`, ie initialize the queue with None, this state vec, this bv
    - process a queue item `(Option<NonTerminalID>, Vec<StateID>, LLMTokenBV)`
        - Look for all reduce edges whose nonterminal/terminal matches and state ID matches top of state vec (its last item). Pop the number on the reduce edge from the stack. SPECIAL CASE FOR END NODE: If the destination precompute1 node is the end node, clear the stack (triggering the second case below).
            - If there's still anything left on the stack, add to the queue its remainder, with the start reduce edge's dest nonterminal ID, with the same LLM bv.
            - If the stack is empty, add a super edge `HashSet<(Option<NonTerminalID>, TerminalID, PrecomputeNodeIndex1, usize, NonTerminalID, LLMTokenBV, PrecomputeNodeIndex1)>` from the src nt ID, with this terminal, with initial precompute1 node index pci1 (not pci1'), the remainder of the pop n (could be zero), the reduce's nonterminal, the same LLM bv, and finally pci1' as the new precompute1 node index.
        - Look for all escape edges whose nonterminal/terminal/state match.
            - Add to the output set `HashSet<(Vec<StateID>, LLMTokenBV, PrecomputeNode1Index)>`. Push the escape edge's state vec to the current state vec (ie extend). Intersect the LLM token bv. Set the precompute node index to pci1 (not pci1').

The special map merge fn simply merges the hashsets.

The process fn -- what does that do?
Actually, nothing. Not needed.

Now, `Trie::special_map_grouped` needs initial nodes and values. The nodes will be the precompute1 nodes...
The states should be all possible states in singleton vecs.
The LLM bvs should just be `LLMTokenBV::max_ones()`.


But then how does get_mask4 know when to end (and add its LLM bv to the final mask)?
It simply looks at the precompute1 node's `end`.

In get_mask4, the 'state' is the current precompute1 node index AND the current terminal. Initialize with ALL possible current terminals AND the GSS.
The initial precompute1 node index is under the entry for the tokenizer state ID in the precompute1 roots map.
The initial terminal will be every possible terminal.
The start precompute special nodes will all be the start node with index None.
So, we'll have a lot of 'states' all sitting at the start precompute special node.


Note that the terminal in the super edge isn't a terminal filter, unlike for the other edges. It's the 'new' terminal for the get_mask4 state, the one that *will* be used to check the (reduce) edge conditions.
In get_mask4, we only traverse reduce edges and super edges, not shift edges.


FYI:
```rust
impl Trie {
    pub fn compute_traversal_data(
        arena: &Arena<Trie<EK, EV, T>>,
        initial_nodes: &[Trie2Index],
    ) -> Option<TrieTraversalData> { ... }

    pub fn special_map_grouped<V, S, I>(  
        arena: &Arena<Trie<EK, EV, T>>,  
        traversal_data: &TrieTraversalData,  
        initial_nodes_and_values: Vec<(Trie2Index, V)>,  
        mut step: S,  
        mut merge: impl FnMut(&mut V, V),  
        mut process: impl FnMut(&Trie<EK, EV, T>, &mut V) -> bool,  
    )  
    where  
        V: Clone,  
        S: FnMut(  
            &V, &EK, &OrderedHashMap<Trie2Index, EV>  
        ) -> I,  
        I: IntoIterator<Item = (Trie2Index, V)>,  
    { ... }
}
```