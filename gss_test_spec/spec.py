import inspect
from typing import Protocol, TypeVar, Generic, Optional, Iterable

# Generic type variables for stack values and accumulators.
T = TypeVar("T")
Acc = TypeVar("Acc")

# A canonical representation of a single stack's state.
StackState = tuple[tuple[T, ...], Acc]


class GSS(Protocol, Generic[T, Acc]):
    """
    A protocol defining the interface for a Graph-Structured Stack (GSS).

    A GSS is a collection of stacks that can share common tails, forming a directed
    acyclic graph. Each active stack is identified by a unique integer ID and has
    an associated accumulator value.
    """

    def push(self, from_stack_id: Optional[int], value: T, acc: Acc) -> int:
        """
        Pushes a new value onto a stack, creating a new stack.

        Args:
            from_stack_id: The ID of the parent stack. If None, creates a new
                           stack from the empty root.
            value: The value to push onto the stack.
            acc: The accumulator value for the new stack.

        Returns:
            The integer ID of the newly created stack.
        """
        ...

    def pop(self, stack_id: int, count: int) -> list[tuple[int, Acc]]:
        """
        Pops one or more values from a stack, creating new stacks for each path.

        If the pop operation traverses a node that was the result of a merge (i.e.,
        has multiple parents), this will result in multiple new stacks.

        Args:
            stack_id: The ID of the stack to pop from.
            count: The number of elements to pop.

        Returns:
            A list of (new_stack_id, original_accumulator) tuples for each
            resulting stack.
        """
        ...

    def mutate(self, stack_id: int, acc: Acc) -> None:
        """
        Updates the accumulator value for a given stack.

        Args:
            stack_id: The ID of the stack to modify.
            acc: The new accumulator value.
        """
        ...

    def merge(self, from_stack_id: int, to_stack_id: int) -> None:
        """
        Merges the history of `from_stack_id` into `to_stack_id`.

        After the merge, `to_stack_id` is updated to represent a state that
        has the parents of both its original head and the head of `from_stack_id`.
        The `from_stack_id` remains unchanged.

        Args:
            from_stack_id: The source stack for the merge.
            to_stack_id: The destination stack, which will be modified.
        """
        ...

    def peek(self, stack_id: int) -> tuple[T, Acc]:
        """
        Retrieves the top value and accumulator of a stack without modifying it.

        Args:
            stack_id: The ID of the stack to peek.

        Returns:
            A tuple containing the top value and the accumulator.
        """
        ...

    def get_all_stacks(self) -> list[StackState[T, Acc]]:
        """
        Returns a canonical representation of the entire GSS state.

        The state is represented as a sorted list of all unrolled stacks. Each
        stack is a tuple containing its path of values and its accumulator.

        Returns:
            A sorted list of StackState tuples.
        """
        ...


class StackGSS(GSS[T, Acc]):
    """
    A reference implementation of the GSS protocol using a node-based graph.

    Internals:
    - Nodes are tuples of (value, frozenset_of_parent_node_ids).
    - Stacks are pointers (by ID) to a head node and an accumulator.
    - Node structures are canonicalized to ensure that identical nodes (same
      value and same parents) are represented by a single ID.
    """
    _ROOT_NODE_ID = 0

    def __init__(self):
        # Node structure: (value, frozenset[parent_ids])
        self._nodes: dict[int, tuple[Optional[T], frozenset[int]]] = {
            self._ROOT_NODE_ID: (None, frozenset())
        }
        # Reverse map for node deduplication
        self._node_to_id: dict[tuple[Optional[T], frozenset[int]], int] = {
            (None, frozenset()): self._ROOT_NODE_ID
        }
        self._next_node_id = 1

        # Active stacks: stack_id -> head_node_id
        self._stacks: dict[int, int] = {}
        self._accs: dict[int, Acc] = {}
        self._next_stack_id = 0

    def _get_or_create_node(self, value: Optional[T], parents: frozenset[int]) -> int:
        node_struct = (value, parents)
        if node_struct in self._node_to_id:
            return self._node_to_id[node_struct]

        node_id = self._next_node_id
        self._nodes[node_id] = node_struct
        self._node_to_id[node_struct] = node_id
        self._next_node_id += 1
        return node_id

    def push(self, from_stack_id: Optional[int], value: T, acc: Acc) -> int:
        parent_node_id = self._stacks[from_stack_id] if from_stack_id is not None else self._ROOT_NODE_ID
        new_node_id = self._get_or_create_node(value, frozenset([parent_node_id]))

        new_stack_id = self._next_stack_id
        self._stacks[new_stack_id] = new_node_id
        self._accs[new_stack_id] = acc
        self._next_stack_id += 1
        return new_stack_id

    def pop(self, stack_id: int, count: int) -> list[tuple[int, Acc]]:
        if stack_id not in self._stacks:
            raise KeyError(f"Stack ID {stack_id} not found.")

        q = {self._stacks[stack_id]}
        for _ in range(count):
            if not q: break
            next_q = set()
            for node_id in q:
                if node_id == self._ROOT_NODE_ID: continue
                _, parents = self._nodes[node_id]
                next_q.update(parents)
            q = next_q

        if not q:
            q = {self._ROOT_NODE_ID}

        original_acc = self._accs[stack_id]
        results = []
        for head_node_id in q:
            new_stack_id = self._next_stack_id
            self._stacks[new_stack_id] = head_node_id
            self._accs[new_stack_id] = original_acc
            self._next_stack_id += 1
            results.append((new_stack_id, original_acc))
        return results

    def mutate(self, stack_id: int, acc: Acc) -> None:
        if stack_id not in self._stacks:
            raise KeyError(f"Stack ID {stack_id} not found.")
        self._accs[stack_id] = acc

    def merge(self, from_stack_id: int, to_stack_id: int) -> None:
        if from_stack_id not in self._stacks or to_stack_id not in self._stacks:
            raise KeyError("One or both stack IDs not found for merge.")

        from_head_id = self._stacks[from_stack_id]
        to_head_id = self._stacks[to_stack_id]

        _, from_parents = self._nodes[from_head_id]
        to_value, to_parents = self._nodes[to_head_id]

        merged_parents = from_parents.union(to_parents)
        new_head_id = self._get_or_create_node(to_value, merged_parents)
        self._stacks[to_stack_id] = new_head_id

    def peek(self, stack_id: int) -> tuple[T, Acc]:
        if stack_id not in self._stacks:
            raise KeyError(f"Stack ID {stack_id} not found.")
        head_node_id = self._stacks[stack_id]
        if head_node_id == self._ROOT_NODE_ID:
            raise IndexError("Cannot peek an empty stack.")
        value, _ = self._nodes[head_node_id]
        return value, self._accs[stack_id]

    def get_all_stacks(self) -> list[StackState[T, Acc]]:
        memo = {}

        def unroll(node_id: int) -> set[tuple[T, ...]]:
            if node_id in memo:
                return memo[node_id]
            if node_id == self._ROOT_NODE_ID:
                return {tuple()}

            value, parents = self._nodes[node_id]
            paths = set()
            for parent_id in parents:
                parent_paths = unroll(parent_id)
                for path in parent_paths:
                    paths.add(path + (value,))
            memo[node_id] = paths
            return paths

        all_stack_states = set()
        for stack_id, head_node_id in self._stacks.items():
            acc = self._accs[stack_id]
            paths = unroll(head_node_id)
            for path in paths:
                all_stack_states.add((path, acc))

        return sorted(list(all_stack_states))


def generate_gss_states() -> Iterable[tuple[list[StackState], int]]:
    """
    A generator that yields canonical GSS states after a series of operations.

    This function defines a sequence of GSS operations, serving as a test
    specification. After each modification, it yields the complete GSS state
    and the line number of the yield statement.
    """
    gss: GSS[str, int] = StackGSS()
    yield gss.get_all_stacks(), inspect.currentframe().f_lineno

    # Basic push from empty
    s1 = gss.push(None, "A", 10)
    yield gss.get_all_stacks(), inspect.currentframe().f_lineno

    # Push onto existing stack
    s2 = gss.push(s1, "B", 20)
    yield gss.get_all_stacks(), inspect.currentframe().f_lineno

    # Another push from s1
    s3 = gss.push(s1, "C", 30)
    yield gss.get_all_stacks(), inspect.currentframe().f_lineno

    # Mutate accumulator
    gss.mutate(s2, 25)
    yield gss.get_all_stacks(), inspect.currentframe().f_lineno

    # Pop one level
    p1_results = gss.pop(s2, 1)
    assert len(p1_results) == 1
    yield gss.get_all_stacks(), inspect.currentframe().f_lineno

    # Merge s3's history into s2. s2's head node is now ('B', {parents_of_A, parents_of_C})
    # which is effectively ('B', {node_A}) because C was also pushed on A.
    # Let's create a more interesting merge.
    s4 = gss.push(None, "X", 40)
    s5 = gss.push(s4, "B", 50) # s5 is (X, B), s2 is (A, B)
    yield gss.get_all_stacks(), inspect.currentframe().f_lineno

    gss.merge(s2, s5) # s5's head 'B' now has parents from s2's 'B' (node A) and its own (node X)
    yield gss.get_all_stacks(), inspect.currentframe().f_lineno

    # Pop from the merged stack s5 - this should cause a fork
    p2_results = gss.pop(s5, 1)
    assert len(p2_results) == 2 # Results in stacks pointing to A and X
    yield gss.get_all_stacks(), inspect.currentframe().f_lineno

    # Pop past the root
    p3_results = gss.pop(s1, 5)
    assert len(p3_results) == 1 # Results in one empty stack
    yield gss.get_all_stacks(), inspect.currentframe().f_lineno

    # Pop 0
    p4_results = gss.pop(s3, 0)
    assert len(p4_results) == 1 # Results in a stack identical to s3
    yield gss.get_all_stacks(), inspect.currentframe().f_lineno


if __name__ == "__main__":
    print("--- GSS Test Specification Runner ---")
    print("Running operations and printing canonical state at each step.\n")
    for i, (state, line) in enumerate(generate_gss_states()):
        print(f"--- Step {i+1} (from line {line}) ---")
        if not state:
            print("GSS is empty.")
        else:
            for path, acc in state:
                print(f"  Path: {path}, Acc: {acc}")
        print()
