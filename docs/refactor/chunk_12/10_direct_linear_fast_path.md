# Direct linear fast path

The direct linear fast path handles byte fragments whose tokenization and parser effects can be consumed in a single left-to-right path without materializing the full queue. It carries a virtual stack where possible and materializes a GSS only when the path exits the shortcut.

The mathematical content is still the same sequence:

```text
scan prefix -> choose one terminal boundary -> apply stack effect -> continue
```

The path is valid only when the scanner and parser together determine a unique next semantic step. Ambiguity, ignored-terminal corner cases, or incompatible parser actions must force a fallback to the general transition.
