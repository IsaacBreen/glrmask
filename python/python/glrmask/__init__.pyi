"""GLRMask is a grammar-constrained decoding library for high-throughput LLM serving. It is optimized for fast mask computation with extremely low tail latency, even for complex grammars.

GLRMask supports general context-free grammars, including ambiguous ones. It accepts JSON Schema, Lark, and GLRMask's native format: GLRM.

## Installation

```bash
python -m pip install glrmask
```

## Usage

GLRMask has four core types:

- `Grammar`: the source-language definition, without compilation settings.
- `Module`: a reusable grammar with bindings and any retained compilation work. It may still have unresolved externs.
- `Constraint`: a fully runnable grammar compiled against a specific vocabulary.
- `ConstraintState`: the mutable state for a single generated sequence.

## Quickstart

```python
import glrmask

vocab = glrmask.Vocab.from_id_to_bytes({0: b"hello", 1: b" ", 2: b"world"})
grammar = glrmask.Grammar.from_ebnf('start ::= "hello" " " "world"')
constraint = grammar.compile(vocab)
state = constraint.start()
for token_id, expected in [(0, [True, False, False]),
                           (1, [False, True, False]),
                           (2, [False, False, True])]:
    assert state.mask().tolist() == expected
    state.commit_token(token_id)
assert state.is_accepting()
```

Python masks are NumPy Boolean arrays indexed by original model token ID.

## Learn more

[Full guide](https://github.com/IsaacBreen/glrmask/blob/main/GUIDE.md) · [API reference](https://glrmask.com/api/python/)
"""
from __future__ import annotations
from collections.abc import Iterator, Mapping
from types import ModuleType
from typing import Any, ClassVar, Iterable, Literal, Optional, Sequence, Union, final, overload
import numpy as np
from numpy.typing import NDArray

_internal: ModuleType

@final
class MaskStrategy:
    """Mask computation strategy, trading build time against per-token latency."""
    AUTO: ClassVar[MaskStrategy]
    """Automatic selection: `PARSER_DWA`, or the retained `TOKEN_CLASS_TRIE` when one is available."""
    PARSER_DWA: ClassVar[MaskStrategy]
    """Static masks: slowest build, fastest per-token masks."""
    TOKEN_TRIE: ClassVar[MaskStrategy]
    """Trie walk per mask: fastest build."""
    TOKEN_CLASS_TRIE: ClassVar[MaskStrategy]
    """Trie walk over token equivalence classes."""
    FAST_BUILD: ClassVar[MaskStrategy]
    """Alias for `TOKEN_TRIE`."""
    FAST_RUNTIME: ClassVar[MaskStrategy]
    """Alias for `PARSER_DWA`."""

@final
class BoundaryMaskStrategy:
    """Boundary parser selection for crossings between composed grammars."""
    AUTO: ClassVar[BoundaryMaskStrategy]
    """Automatic selection: `PARSER_DWA`, falling back to `TOKEN_TRIE` when the static boundary parser declines on budget."""
    PARSER_DWA: ClassVar[BoundaryMaskStrategy]
    """Static boundary parser; faster crossings. An over-budget build raises ValueError."""
    TOKEN_TRIE: ClassVar[BoundaryMaskStrategy]
    """Segmented dynamic boundary token walk; faster build."""
    FAST_BUILD: ClassVar[BoundaryMaskStrategy]
    """Alias for TOKEN_TRIE."""
    FAST_RUNTIME: ClassVar[BoundaryMaskStrategy]
    """Alias for PARSER_DWA."""

@final
class TokenSet:
    """A set of model token IDs. IDs are validated against the compile vocabulary; an empty set matches nothing."""
    def __init__(self, ids: Iterable[int]) -> None:
        """Creates a set from token IDs; duplicates are ignored."""
        ...
    @staticmethod
    def from_ranges(ranges: Iterable[range]) -> TokenSet:
        """Creates a set from Python ranges with step 1 and exclusive stops.

        Overlapping or adjacent ranges merge; an empty or reversed range contributes nothing.
        """
        ...
    def __repr__(self) -> str:
        """The normalized ranges, for example `TokenSet[0..=9, 20..=29]`."""
        ...
    def __eq__(self, other: object) -> bool:
        """Whether both sets hold the same token IDs."""
        ...

@final
class Vocab:
    """The model vocabulary a constraint is compiled against.

    Entries map model token IDs to exact token bytes. Token IDs may be sparse; masks use the original model token IDs.
    """
    @staticmethod
    def from_dict(token_to_id: dict[Union[str, bytes], int]) -> Vocab:
        """Creates a vocabulary from `{token: id}`.

        String tokens are UTF-8 encoded and byte tokens are used verbatim; IDs must be unsigned 32-bit integers.
        """
        ...
    @staticmethod
    def from_id_to_bytes(id_to_bytes: dict[int, bytes]) -> Vocab:
        """Creates a vocabulary from `{token_id: bytes}`.

        Use the tokenizer's exact decoded bytes, including partial UTF-8 sequences, not display spellings.
        """
        ...
    @staticmethod
    def from_llama_cpp(llm: Any) -> Vocab:
        """Creates a vocabulary from a llama-cpp-python `Llama`.

        EOG, control, unused, and empty-piece tokens are omitted from byte matching; use `llama_cpp_end_token_ids` for the root stopping policy.
        """
        ...
    @property
    def llama_cpp_end_token_ids(self) -> list[int]:
        """EOG IDs discovered by `from_llama_cpp`; pass them as `end_tokens` when compiling."""
        ...
    def __len__(self) -> int:
        """Number of byte-backed entries; special IDs are not counted."""
        ...

@final
class Extern:
    """A declared extern slot on a `Grammar` or `Module`."""
    @property
    def name(self) -> str:
        """The declared name."""
        ...
    @property
    def kind(self) -> Literal["token", "grammar"]:
        """The declared kind: token or grammar."""
        ...
    def __repr__(self) -> str:
        """The debug representation."""
        ...
    def __eq__(self, other: object) -> bool:
        """Compares name and kind."""
        ...

@final
class Externs(Mapping[str, Extern]):
    """A view over a `Grammar`'s or `Module`'s declared externs, in declaration order.

    Implements `collections.abc.Mapping`.
    """
    @property
    def unbound(self) -> Externs:
        """The same view restricted to unresolved externs."""
        ...
    def __getitem__(self, key: str) -> Extern:
        """The extern for `key`; raises `KeyError` if absent."""
        ...
    def __iter__(self) -> Iterator[str]:
        """The declared extern names in declaration order."""
        ...
    def __len__(self) -> int:
        """Number of externs in this view."""
        ...
    def __repr__(self) -> str:
        """The debug representation."""
        ...
    def __eq__(self, other: object) -> bool:
        """Compares the externs in declaration order."""
        ...

@final
class Expr:
    """An immutable unordered union or concatenation of independently scoped grammars."""
    @staticmethod
    def choice(*parts: Union[Grammar, Module, Expr, TokenSet]) -> Expr:
        """Creates a union without ordered-choice priority.

        An empty choice matches nothing; open parts raise `ValueError` immediately.
        """
        ...
    @staticmethod
    def sequence(*parts: Union[Grammar, Module, Expr, TokenSet]) -> Expr:
        """Creates a concatenation; a model token may cross any number of operand boundaries.

        An empty sequence is epsilon; open parts raise `ValueError` immediately.
        """
        ...
    @staticmethod
    def literal(value: Union[str, bytes], /) -> Expr:
        """Matches exact bytes, with no whitespace or ignore inserted.

        `str` is UTF-8 encoded.
        """
        ...

@final
class Module:
    """A reusable grammar with bindings and any retained compilation work.

    May still have unresolved externs.
    """
    def __init__(self, source: Grammar, /) -> None:
        """Creates a module from a source grammar."""
        ...
    @overload
    def bind(self, name: str, value: Union[Grammar, Module, Expr, TokenSet], /) -> None:
        """Binds the extern `name` to `value`; the keyword form binds several atomically, and none is applied if any fails.

        Fails if an extern is undeclared or already bound; names containing `::` are rejected.
        """
        ...
    @overload
    def bind(self, **bindings: Union[Grammar, Module, Expr, TokenSet]) -> None:
        """Binds the extern `name` to `value`; the keyword form binds several atomically, and none is applied if any fails.

        Fails if an extern is undeclared or already bound; names containing `::` are rejected.
        """
        ...
    def precompile(
        self,
        vocab: Vocab,
        *,
        mask_strategy: MaskStrategy = MaskStrategy.AUTO,
        boundary_mask_strategy: BoundaryMaskStrategy = BoundaryMaskStrategy.AUTO,
    ) -> None:
        """Retains vocabulary-specific compilation artifacts for this module.

        The strategies decide which static parts are built. A module resolved to `TOKEN_TRIE` retains no work and compiles from source.
        """
        ...
    def compile(
        self,
        vocab: Vocab,
        *,
        end_tokens: Optional[Sequence[int]] = None,
        mask_strategy: MaskStrategy = MaskStrategy.AUTO,
        boundary_mask_strategy: BoundaryMaskStrategy = BoundaryMaskStrategy.AUTO,
    ) -> Constraint:
        """Compiles the module assembly against `vocab`.

        Requires every extern to be bound; `end_tokens` are admitted once the grammar is complete and must be committed before stopping. The strategies apply to the whole compile, including every composed child, and never downgrade a part that already has a static preparation.
        """
        ...
    @property
    def externs(self) -> Externs:
        """A view over the module's declared externs, in declaration order."""
        ...
    def copy(self) -> Module:
        """An independent snapshot sharing immutable source and retained work."""
        ...
    def __copy__(self) -> Module:
        """An independent snapshot."""
        ...
    def __deepcopy__(self, memo: dict[int, Any]) -> Module:
        """An independent snapshot."""
        ...
    def save(self) -> bytes:
        """Serializes the module assembly and any retained precompiled artifacts."""
        ...
    @staticmethod
    def load(data: bytes) -> Module:
        """Deserializes a module assembly. Retained work is checked against the vocabulary supplied to `compile` or `precompile`."""
        ...

@final
class Grammar:
    """The source-language definition, without compilation settings.

    Source text is parsed lazily, so parse errors surface at compile time.
    """
    @staticmethod
    def from_ebnf(source: str) -> Grammar:
        """Creates a grammar from EBNF source text."""
        ...
    @staticmethod
    def from_glrm(source: str) -> Grammar:
        """Creates a grammar from GLRM source.

        GLRM supports typed extern slots; bind them before compiling a runnable root.
        """
        ...
    @staticmethod
    def from_json_schema(schema: Any) -> Grammar:
        """Creates a grammar from a JSON Schema document.

        Accepts JSON text or a JSON-serializable object; `format` is an annotation only.
        """
        ...
    @staticmethod
    def from_lark(source: str) -> Grammar:
        """Creates a grammar from Lark source.

        Named token atoms declare token externs; bind each through `Module.bind`.
        """
        ...
    @property
    def externs(self) -> Externs:
        """A view over the grammar's declared externs, in declaration order."""
        ...
    def precompile(
        self,
        vocab: Vocab,
        *,
        mask_strategy: MaskStrategy = MaskStrategy.AUTO,
        boundary_mask_strategy: BoundaryMaskStrategy = BoundaryMaskStrategy.AUTO,
    ) -> Module:
        """Retains vocabulary-specific compilation artifacts and returns the module.

        Equivalent to `Module(self).precompile(vocab, ...)`.
        """
        ...
    def compile(
        self,
        vocab: Vocab,
        *,
        end_tokens: Optional[Sequence[int]] = None,
        mask_strategy: MaskStrategy = MaskStrategy.AUTO,
        boundary_mask_strategy: BoundaryMaskStrategy = BoundaryMaskStrategy.AUTO,
    ) -> Constraint:
        """Compiles the grammar against `vocab`.

        Equivalent to `Module(self).compile(vocab, ...)`.
        """
        ...
    def __copy__(self) -> Grammar:
        """An independent handle."""
        ...
    def __deepcopy__(self, memo: dict[int, Any]) -> Grammar:
        """An independent handle."""
        ...

@final
class Constraint:
    """A fully runnable grammar compiled against a specific vocabulary.

    Immutable and safe to share; each `start()` creates independent mutable state.
    """
    def marker_ids(self) -> list[int]:
        """Sorted, distinct application marker IDs."""
        ...
    def start(self) -> ConstraintState:
        """Creates independent state for one generated sequence."""
        ...
    def mask_len(self) -> int:
        """Number of packed `u32` words in a mask, for `fill_mask`; not the Boolean-array length returned by `mask`."""
        ...
    def save(self) -> bytes:
        """Serializes the constraint without its vocabulary.

        Load with `Constraint.load(data, vocab)` using the exact original token-ID/byte mapping, not merely the same size. Artifact compatibility across GLRMask releases is not guaranteed.
        """
        ...
    @staticmethod
    def load(data: bytes, vocab: Vocab) -> Constraint:
        """Deserializes a constraint.

        `vocab` must match the original mapping. Rejects malformed or incompatible artifacts.
        """
        ...

@final
class ConstraintState:
    """The mutable state for a single generated sequence."""
    def marker_events(self) -> list[tuple[int, int]]:
        """Initial or most recent commit's observations as `(marker_id, absolute_byte_offset)`.

        Sorted by offset then ID; a failed commit clears the batch and mask or forcing queries leave it unchanged.
        """
        ...
    def clone(self) -> ConstraintState:
        """An independent snapshot at the current prefix."""
        ...
    def mask(self, size: Optional[int] = None) -> NDArray[np.bool_]:
        """The current allowed-token mask as a new Boolean array indexed by original token ID.

        `size` must cover the constraint's whole token range; extra positions are false.
        """
        ...
    def fill_mask(self, bitmask: NDArray[np.int32]) -> None:
        """Writes the current mask into a contiguous, writable 1-D NumPy `int32` array.

        The array must hold at least `Constraint.mask_len()` words; extra words are cleared.
        """
        ...
    def commit_token(self, token_id: int) -> None:
        """Commits one model token event.

        Returns an error if the token is not allowed; the state is then unusable, so clone it first to try a token speculatively.
        """
        ...
    def commit_tokens(self, token_ids: Sequence[int]) -> None:
        """Commits model token IDs in order.

        Stops with an error at the first disallowed token; the state is then unusable.
        """
        ...
    def commit_bytes(self, data: bytes) -> None:
        """Commits raw bytes rather than a model token event.

        Does not substitute for committing a token-set or end-token event.
        """
        ...
    def is_accepting(self) -> bool:
        """Whether the input so far is a complete match.

        With end tokens configured, this becomes true only once an end token is committed.
        """
        ...
    def forced_tokens(self) -> list[int]:
        """A read-only prefix of library-canonical forced output, at most 10,000 IDs.

        Uses the longest admitted nonempty token spelling, then the lowest original token ID; token-set events retain their IDs. The query stops at root acceptance or an ambiguous event boundary.
        """
        ...
