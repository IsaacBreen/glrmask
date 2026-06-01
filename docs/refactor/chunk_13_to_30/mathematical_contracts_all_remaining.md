# Mathematical contracts for Chunks 13-30

## 1. Template DFA contract

Let `G` be the GLR stack-transition relation and let `T` be a compiled template DFA for a finite family of stack templates.  For every represented stack language `L` and terminal `a`, template advance is valid exactly when:

```text
T(L, a) = G(L, a)
```

If the template executor cannot prove that the live stack frontier lies in a represented template class, it must return `None` and use direct GLR execution.

## 2. Weight algebra contract

A `Weight` denotes a relation between an outer id space and an inner token-set id space.  Union, intersection, subtraction, and remapping are relation operations.  Interning is not semantic; it is representation sharing.  Therefore `Arc::ptr_eq` may be a fast equality witness but never the definition of equality.

## 3. GSS contract

A graph-structured stack denotes a finite set of stacks plus an accumulated value per path.  Merge is a semilattice operation over accumulators.  Pop/push/isolate operations must preserve the represented stack language.

## 4. ID-space contract

A many-to-one map `q: O -> I` defines a quotient of original ids into internal ids.  Any artifact expressed in internal coordinates must carry the quotient used to interpret it.  A compacted artifact is correct only if it is the conjugate of the old artifact under the quotient.

## 5. Mask contract

For an original token `o`, final Mask output is true iff at least one internal representative of `o` is allowed by the current runtime state.  Internal dense/sparse strategies are implementation details.

## 6. Commit contract

Commit is composition of byte scanning and parser-stack advancement:

```text
Commit(state, bytes) = Advance(ParserFrontier(state), Scan(TokenizerState(state), bytes))
```

Fast paths must equal this reference composition.

## 7. Serialization contract

Only semantic artifact fields are serialized.  Derived caches are reconstructed deterministically from semantic fields after load.
