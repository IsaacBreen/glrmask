# Runtime token space

This directory owns the final projection from the internal token quotient used
by the runtime back to the caller's original token ids.  The invariant is:

```text
original token o is allowed  iff  internal representative q(o) is allowed
```

with expansion over non-singleton quotient classes when several original tokens
share one internal representative.
