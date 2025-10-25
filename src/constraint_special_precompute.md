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
Shift to the stack. Stack size can either be 2 or 3.
Shifts can be characterized by nonterminal, terminal, state, shift state, and optional middle state.
By 'state' here I mean 'state used to initialize stack'.
`HashSet<(Option<NonTerminalID>, StateID, TerminalID, Vec<StateID>)>`


**Second stage**
Here's the second stage. It's basically for 'super' edges.
Now suppose we do special map over precompute1 tree. Edge processing like this.
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

...

Now let's look at get_mask4. It'll use this tree. Its output should be identical to get_mask3.
Obviously it won't use trie's special map. It'll manage its own queue.
In the initial nodes and values, the values will include the current precompute1 index. It'll also include optional nonterminals - ie node indices in this new tree. Specifically, it'll be `HashSet<(PrecomputeNode1Index, Option<NonTerminalID>, LLMTokenBV)>`.
We traverse all compatible edges, normal and super.
If super, it's only compatible if the first pci1 index matches the current index. If it does, the current pci1 index gets replaced by the edge's second one.