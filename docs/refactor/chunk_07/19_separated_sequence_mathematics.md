# Separated sequence mathematics

`SeparatedSequence` is used heavily by JSON Schema ordered-object lowering. It deserves its own module because its semantics are easy to get subtly wrong.

## Basic denotation

A separated sequence has:

```text
items = [(e1, required1), ..., (en, requiredn)]
separator = s
allow_empty = bool
```

It denotes ordered selections of items where all required items are present and optional items may be absent. Present items are joined by the separator.

## The nullable trap

Suppose `e2` is required but nullable.

```text
e1 , e2 , e3
```

If `e2` derives epsilon, the concrete text may look like:

```text
e1 , , e3
```

or depending on terminal details, it may consume no bytes between separators. This is not the same as the optional item being absent:

```text
e1 , e3
```

A lowering that treats nullable required items as absent changes the language.

## Recursive split

The implementation recursively splits the item list and returns:

```text
(symbol, can_be_empty)
```

The symbol recognizes the subtree when structurally present. The boolean says whether the subtree can be omitted in parent composition. The parent uses that flag to decide whether to add alternatives without left or right subtrees.

## No local epsilon rule

Do not emit `subtree -> ε` simply because a subtree can be empty. That creates dangling separator problems because parent contexts need to know whether the subtree is absent or present-but-empty.

## Repetition inside separated sequence

If an item is `x+`, inside a separated sequence with separator `,`, then it must lower as:

```text
x (',' x)*
```

not:

```text
x+
```

because the separator belongs between repeated item occurrences.
