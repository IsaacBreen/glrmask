"""GLRMask is a grammar-constrained decoding library for high-throughput LLM serving. It is optimized for fast mask computation with extremely low tail latency, even for complex grammars.

GLRMask supports general context-free grammars, including ambiguous ones. It accepts JSON Schema, Lark, and GLRMask's native format: GLRM.

## Installation

```bash
python -m pip install glrmask
```

## Usage

GLRMask has four core types:

- `Grammar`: the source-language definition, without compilation settings.
- `Module`: a reusable, independently configured grammar with bindings and any retained compilation work. It may still have unresolved externs.
- `Constraint`: a fully runnable grammar compiled against a specific vocabulary.
- `ConstraintState`: the mutable state for a single generated sequence.

Use direct `Grammar.compile` for a closed source, or a `Module` when local
settings, late-bound children, or retained work are needed. `Expr.choice` and
`Expr.sequence` group grammar leaves without adding an extra lexer scope at each
operator. See [modules and expressions](https://github.com/IsaacBreen/glrmask/blob/main/docs/module-api.md)
for construction, open precompilation, and persistence examples.

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

The next compatibility release targets ordinary, GIL-enabled CPython 3.9
through 3.15. Its `cp39-abi3` wheels use Python's stable extension ABI, so one
wheel per operating-system/architecture combination can serve multiple Python
versions. This does not imply support for free-threaded Python (`cp314t`,
`cp315t` or `abi3t`): the extension explicitly retains the GIL requirement.

The expanded build matrix covers Windows x64/ARM64, macOS Intel/Apple Silicon,
and Linux x64/ARM64 with either glibc or musl. Linux glibc Python wheels retain
the manylinux2014 baseline; native C++/Node binaries use glibc 2.28 or newer.
See [the compatibility guide](https://github.com/IsaacBreen/glrmask/blob/main/docs/compatibility.md)
for runtime requirements and the distinction between source support and
published package availability.

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
    "Mask generation strategy: trade off compilation time against per-token masking latency."
    AUTO: ClassVar[MaskStrategy]
    "Library chooses the strategy."
    PARSER_DWA: ClassVar[MaskStrategy]
    "Static parser-DWA masks (slowest build, fastest masks)."
    TOKEN_TRIE: ClassVar[MaskStrategy]
    "Dynamic token-trie walk (fastest build)."
    TOKEN_CLASS_TRIE: ClassVar[MaskStrategy]
    "Token-trie walk over token equivalence classes."
    FAST_BUILD: ClassVar[MaskStrategy]
    "Alias for TOKEN_TRIE."
    FAST_RUNTIME: ClassVar[MaskStrategy]
    "Alias for PARSER_DWA."

@final
class BoundaryMaskStrategy:
    "Boundary mask detail level: minimum metadata to build for composition boundaries."
    FAST_BUILD: ClassVar[BoundaryMaskStrategy]
    "Conservative token-set boundary trigger (faster build)."
    FAST_RUNTIME: ClassVar[BoundaryMaskStrategy]
    "Full exact proper-prefix boundary trigger (lower latency)."

@final
class Token:
    "An original token ID with no vocabulary identity. Only ID membership is checked against the target vocabulary; the same ID can refer to different bytes. Use `vocab.tokens([id])` for a vocabulary-bound singleton."
    def __init__(self, id: int) -> None:
        ...
    @property
    def id(self) -> int:
        ...

@final
class ExactTokens:
    "One token event chosen from a vocabulary-bound set. Vocabularies must have the same ID-to-byte mapping and exact-only IDs, even for a singleton. An empty set matches nothing."
    @property
    def ids(self) -> list[int]:
        "Sorted IDs, materialized on access. Prefer `ranges` for large sets."
        ...
    @property
    def ranges(self) -> list[range]:
        "Normalized, disjoint Python ranges with exclusive stops; does not expand the IDs."
        ...
    def __len__(self) -> int:
        "Number of IDs, rather than ranges."
        ...

@final
class Vocab:
    "Original model IDs mapped to exact token bytes. IDs may be sparse; masks keep their original positions. Reuse one vocabulary across components to share compilation caches."
    @staticmethod
    def from_dict(token_to_id: dict[Union[str, bytes], int]) -> Vocab:
        "String keys are UTF-8 encoded; byte keys are used verbatim. IDs must fit an unsigned 32-bit integer."
        ...
    @staticmethod
    def from_id_to_bytes(id_to_bytes: dict[int, bytes]) -> Vocab:
        "Use the tokenizer’s exact decoded bytes, including partial UTF-8 sequences, rather than display spellings."
        ...
    @staticmethod
    def from_llama_cpp(llm: Any) -> Vocab:
        "Extract the byte vocabulary from a llama-cpp-python `Llama`. EOG, control, unused and empty-piece tokens are omitted from byte matching; use `llama_cpp_end_token_ids` for the root stopping policy."
        ...
    @property
    def llama_cpp_end_token_ids(self) -> list[int]:
        "EOG IDs discovered by `from_llama_cpp`; pass them as `end_tokens` when compiling."
        ...
    def tokens(self, ids: Iterable[int]) -> ExactTokens:
        "Match one of the supplied IDs. Unknown or duplicate IDs are errors; empty input matches nothing. The result retains vocabulary identity, including for a singleton. Contiguous ranges remain compact; other iterables are consumed once."
        ...
    def token_ranges(self, ranges: Iterable[range]) -> ExactTokens:
        "Union of nonempty Python ranges with step 1 or -1 and exclusive stops. Overlapping or adjacent ranges merge; unknown IDs are errors. Empty input matches nothing. Use `tokens` for arbitrary strides."
        ...
    def __len__(self) -> int:
        "Counts byte-backed entries; exact-only control IDs are excluded."
        ...

@final
class Extern:
    "One declared extern contract on a Module."
    @property
    def name(self) -> str:
        "Declared extern name."
        ...
    @property
    def kind(self) -> Literal["token", "grammar"]:
        "A token extern accepts exact events; a grammar extern accepts grammar/module/expr components."
        ...
    @property
    def span(self) -> Optional[tuple[int, int, int, int]]:
        "`(start_byte, end_byte, line, column)` in the declaring source, or `None`. The byte interval is half-open; line and Unicode-scalar column are one-based."
        ...
    def __repr__(self) -> str:
        ...
    def __eq__(self, other: object) -> bool:
        ...

@final
class Externs(Mapping[str, Extern]):
    "Read-only snapshot view of a Module's declared externs. Implements collections.abc.Mapping."
    @property
    def unbound(self) -> Externs:
        "Filtered view containing only externs not yet bound."
        ...
    def __getitem__(self, key: str) -> Extern:
        ...
    def __iter__(self) -> Iterator[str]:
        ...
    def __len__(self) -> int:
        ...
    def __repr__(self) -> str:
        ...
    def __eq__(self, other: object) -> bool:
        ...

@final
class Expr:
    "Immutable scope-preserving union, concatenation, or byte-literal leaves."
    @staticmethod
    def choice(*parts: Union[Grammar, Module, Expr, Token, ExactTokens]) -> Expr:
        "Unordered context-free union. Empty choice matches nothing. Open parts raise ValueError immediately."
        ...
    @staticmethod
    def sequence(*parts: Union[Grammar, Module, Expr, Token, ExactTokens]) -> Expr:
        "Concatenation of language pieces. Empty sequence matches epsilon. Open parts raise ValueError immediately."
        ...
    @staticmethod
    def literal(value: Union[str, bytes], /) -> Expr:
        "Match exact bytes without whitespace or ignore inserted (str is UTF-8 encoded)."
        ...

@final
class Module:
    "An owned, independently configured reusable grammar with snapshot-safe mutable bindings and retained compilation work."
    def __init__(
        self,
        source: Grammar,
        /,
        *,
        mask_strategy: MaskStrategy = MaskStrategy.AUTO,
        boundary_mask_strategy: BoundaryMaskStrategy = BoundaryMaskStrategy.FAST_BUILD,
    ) -> None:
        "Create a cold module from a Grammar with settings."
        ...
    @property
    def mask_strategy(self) -> MaskStrategy:
        "Mask generation strategy for this module."
        ...
    @mask_strategy.setter
    def mask_strategy(self, value: MaskStrategy) -> None:
        ...
    @property
    def boundary_mask_strategy(self) -> BoundaryMaskStrategy:
        "Boundary mask detail level for this module."
        ...
    @boundary_mask_strategy.setter
    def boundary_mask_strategy(self, value: BoundaryMaskStrategy) -> None:
        ...
    @overload
    def bind(self, name: str, value: Union[Grammar, Module, Expr, Token, ExactTokens], /) -> None:
        "Fill one extern. Each extern can be bound once; names containing `::` are rejected."
        ...
    @overload
    def bind(self, **bindings: Union[Grammar, Module, Expr, Token, ExactTokens]) -> None:
        "Fill several externs atomically, in order: if any binding fails, none is applied. Mixing positional and keyword forms raises TypeError."
        ...
    def precompile(self, vocab: Vocab) -> None:
        "Perform and retain definition-local work for this vocabulary even while externs are open. Existing clones are unchanged."
        ...
    def compile(
        self,
        vocab: Vocab,
        *,
        end_tokens: Optional[Sequence[int]] = None,
    ) -> Constraint:
        "Create a runnable constraint. Requires all externs resolved."
        ...
    @property
    def externs(self) -> Externs:
        "Read-only snapshot mapping of this module's own declared externs in declaration order."
        ...
    def copy(self) -> Module:
        "Return an independent snapshot sharing immutable source and retained work."
        ...
    def __copy__(self) -> Module:
        "Return a cheap independent snapshot."
        ...
    def __deepcopy__(self, memo: dict[int, Any]) -> Module:
        "Return a logically independent snapshot."
        ...
    def save(self) -> bytes:
        "Serialize source, settings, bindings and retained work without compiling."
        ...
    @staticmethod
    def load(data: bytes, vocab: Optional[Vocab] = None) -> Module:
        "Load a saved module artifact, restoring settings and retained work."
        ...

@final
class Grammar:
    "Immutable authored source grammar without compilation settings or bindings."
    @staticmethod
    def from_ebnf(source: str) -> Grammar:
        "Parsing and compilation errors can be deferred until `compile`."
        ...
    @staticmethod
    def from_glrm(source: str) -> Grammar:
        "GLRM supports typed external declarations; bind them via `Module(g).bind(...)` before compiling."
        ...
    @staticmethod
    def from_json_schema(schema: Any) -> Grammar:
        "Accepts JSON text or a JSON-serializable object."
        ...
    @staticmethod
    def from_lark(
        source: str,
        *,
        profile: Literal["recognition", "llm"] = "recognition",
        token_names: Optional[dict[str, Union[int, Sequence[int]]]] = None,
    ) -> Grammar:
        "The default `recognition` profile and the `llm` extension profile differ in accepted syntax."
        ...
    def compile(
        self,
        vocab: Vocab,
        *,
        end_tokens: Optional[Sequence[int]] = None,
        mask_strategy: MaskStrategy = MaskStrategy.AUTO,
        boundary_mask_strategy: BoundaryMaskStrategy = BoundaryMaskStrategy.FAST_BUILD,
    ) -> Constraint:
        "Compile this grammar directly. Sugar for `Module(self, ...).compile(vocab, end_tokens=...)`."
        ...
    def __copy__(self) -> Grammar:
        ...
    def __deepcopy__(self, memo: dict[int, Any]) -> Grammar:
        ...

@final
class Constraint:
    "Immutable runnable grammar and root stopping policy. Reuse across sequences; each `start` creates independent mutable state."
    def marker_ids(self) -> list[int]:
        "Sorted, distinct application marker IDs, independent of model token IDs."
        ...
    def start(self) -> ConstraintState:
        ...
    def mask_len(self) -> int:
        "Packed 32-bit word count for `fill_mask`, not the Boolean-array length returned by `mask`."
        ...
    def save(self, *, external_vocab: bool = False) -> bytes:
        "Embeds the vocabulary by default. With `external_vocab=True`, loading requires the exact original token-ID/byte mapping, not merely the same vocabulary size."
        ...
    @staticmethod
    def load(data: bytes, vocab: Optional[Vocab] = None) -> Constraint:
        "Rejects malformed or incompatible artifacts. Supply `vocab` for external-vocabulary artifacts; it must match the original mapping."
        ...

@final
class ConstraintState:
    "Mutable state for one generated sequence."
    def marker_events(self) -> list[tuple[int, int]]:
        "Initial or most recent commit’s `(marker_id, absolute_byte_offset)` batch, sorted by offset then ID."
        ...
    def clone(self) -> ConstraintState:
        "Independent snapshot at the current prefix."
        ...
    def mask(self, size: Optional[int] = None) -> NDArray[np.bool_]:
        "New Boolean array indexed by original token ID."
        ...
    def fill_mask(self, bitmask: NDArray[np.int32]) -> None:
        "Fill a contiguous, writable 1-D NumPy `int32` array with at least `Constraint.mask_len()` words."
        ...
    def commit_token(self, token_id: int) -> None:
        "Advance by a token event. Unknown or forbidden IDs reject the state."
        ...
    def commit_bytes(self, data: bytes) -> None:
        "Advance by raw bytes rather than model token events."
        ...
    def is_accepting(self) -> bool:
        "The current prefix is a complete grammar match."
        ...
    def is_rejected(self) -> bool:
        "The prefix cannot be extended to a valid match."
        ...
    def forced_tokens(self) -> list[int]:
        "Read-only canonical prefix of forced tokens."
        ...
