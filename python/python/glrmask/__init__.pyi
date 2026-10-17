"""GLRMask is a grammar-constrained decoding library for high-throughput LLM serving. It is optimized for fast mask computation with extremely low tail latency, even for complex grammars.

GLRMask supports general context-free grammars, including ambiguous ones. It accepts JSON schemas and Lark grammars. GLRM is its internal grammar format.

## Installation

```bash
python -m pip install glrmask
```

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

## Core concepts

GLRMask has four core types:

- `Grammar`: source grammar semantics and bindings.
- `PreparedGrammar`: reusable vocabulary-specific compiled machinery, which can still have unresolved bindings.
- `Constraint`: a closed, runnable grammar compiled against a vocabulary.
- `ConstraintState`: the mutable state for one generated sequence.

Compile a `Grammar` directly, or use `prepare()` and `link()` for reusable components. Create a state with `start()`, then compute the next-token mask, sample an allowed token, and commit it.

`Grammar` and `PreparedGrammar` keep reusable definitions with mutable bindings. `bind()` fills one declared slot on the handle. Clones and previously attached children retain independent binding snapshots; a failed bind leaves the handle unchanged.

## Learn more

[Full guide](https://github.com/IsaacBreen/glrmask/blob/main/GUIDE.md) · [API reference](https://glrmask.com/api/python/)
"""
from __future__ import annotations
from types import ModuleType
from typing import Any, ClassVar, Iterable, Literal, Optional, Sequence, Union, final
import numpy as np
from numpy.typing import NDArray

_internal: ModuleType

@final
class Optimization:
    "Build-time versus per-token cost. The modes preserve the supported grammar language; `AUTO` chooses a strategy."
    AUTO: ClassVar[Optimization]
    "Choose a strategy automatically."
    FAST_BUILD: ClassVar[Optimization]
    "Favor shorter compilation."
    BALANCED: ClassVar[Optimization]
    "Balance compilation and masking cost."
    FAST_RUNTIME: ClassVar[Optimization]
    "Favor lower masking latency, accepting more compilation work."

@final
class BoundaryTriggerDetail:
    "Optional metadata for composition boundaries, independent of the masking optimization. A request at final link overrides the prepared request."
    NONE: ClassVar[BoundaryTriggerDetail]
    "Build no boundary metadata (the default)."
    TOKENS: ClassVar[BoundaryTriggerDetail]
    "Build a conservative, parser-state-independent token candidate set."
    EXACT: ClassVar[BoundaryTriggerDetail]
    "Build parser-state-sensitive boundary metadata where supported."

@final
class Token:
    "One original token ID, independent of its byte spelling. This value stores no vocabulary identity: the target checks ID membership, so the same ID can be accepted with different bytes. Use `vocab.tokens([id])` when a singleton must retain vocabulary identity."
    def __init__(self, id: int) -> None:
        ""
        ...
    @property
    def id(self) -> int:
        ""
        ...

@final
class ExactTokens:
    "One token event chosen from a vocabulary-bound set, not a sequence. Even a singleton retains the full token-ID/byte mapping; a different mapping is rejected. An empty set matches nothing, not the empty string."
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
        "EOG IDs discovered by `from_llama_cpp`; pass them as `end_tokens` when compiling or linking."
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
class Requirement:
    "An unresolved typed slot. Paths distinguish the same local name in different attached children."
    @property
    def name(self) -> str:
        "Qualified path joined with `::`."
        ...
    @property
    def path(self) -> list[str]:
        "Path segments from the assembly root to the unresolved slot."
        ...
    @property
    def kind(self) -> Literal["token", "grammar"]:
        "A token slot accepts exact events; a grammar slot also accepts a grammar component."
        ...
    @property
    def source_span(self) -> Optional[tuple[int, int, int, int]]:
        "`(start_byte, end_byte, line, column)` in the declaring source, or `None`. The byte interval is half-open; line and Unicode-scalar column are one-based."
        ...
    def __repr__(self) -> str:
        ""
        ...

@final
class Grammar:
    "Reusable source grammar with mutable bindings. `bind` changes this handle; clones and already attached children keep independent snapshots. Parsing and vocabulary checks can be deferred until preparation or compilation."
    @staticmethod
    def from_ebnf(source: str) -> Grammar:
        "Parsing and compilation errors can be deferred until `prepare` or `compile`."
        ...
    @staticmethod
    def from_glrm(source: str) -> Grammar:
        "GLRM supports typed external slots; fill them with `bind` before compiling a runnable root."
        ...
    @staticmethod
    def from_json_schema(schema: Any) -> Grammar:
        "Accepts JSON text or a JSON-serializable object. The importer implements a subset of JSON Schema: unsupported assertions may be silently weakened. Validate generated values separately when full schema conformance matters."
        ...
    @staticmethod
    def from_lark(
        source: str, *, profile: Literal["recognition", "llm"] = "recognition",
        token_names: Optional[dict[str, Union[int, Sequence[int]]]] = None,
    ) -> Grammar:
        "The default `recognition` profile and the `llm` extension profile differ in accepted syntax. Named token atoms require explicit `token_names` mappings; they are never inferred from tokenizer spellings."
        ...
    @staticmethod
    def exact(tokens: ExactTokens) -> Grammar:
        "Adapt one vocabulary-bound token event to a grammar fragment. Adds no byte spelling, ignore rule or root end-token policy."
        ...
    @property
    def requirements(self) -> list[Requirement]:
        "Includes unresolved slots in attached open children. Diagnostic expansion is bounded and may raise an error for very large shared graphs."
        ...
    def bind(self, name: str, value: Union[Grammar, Token, ExactTokens]) -> None:
        "Fill a declared local or `::`-qualified slot once. Accepts `Grammar`, `Token` or `ExactTokens`; a general grammar cannot fill a token slot. Captures the child’s current snapshot. Unknown, already filled, incompatible or cyclic bindings fail without changing this handle."
        ...
    def clone(self) -> Grammar:
        "Independent binding snapshot; later binds on either handle do not affect the other."
        ...
    def __copy__(self) -> Grammar:
        "Equivalent to `clone`."
        ...
    def __deepcopy__(self, memo: dict[int, Any]) -> Grammar:
        "Equivalent to `clone`; immutable compiled storage may remain shared."
        ...
    def prepare(
        self, vocab: Vocab, *, boundary_trigger: Optional[BoundaryTriggerDetail] = None,
    ) -> PreparedGrammar:
        "Compile vocabulary-specific machinery while retaining unresolved slots. Reuse it with different bindings, then `link`. Only the boundary metadata request is retained for final link."
        ...
    def compile(
        self, vocab: Vocab, *, end_tokens: Optional[Sequence[int]] = None,
        optimization: Optional[Optimization] = None,
        boundary_trigger: Optional[BoundaryTriggerDetail] = None,
    ) -> Constraint:
        "Requires every slot to be bound. `end_tokens` are root generation controls, allowed only at acceptance; commit the selected end token before stopping. Their policy is not inherited by an embedded child. `optimization` defaults to `AUTO`."
        ...

@final
class PreparedGrammar:
    "Reusable compiled machinery tied to one vocabulary, possibly with open slots. Bind prepared children or exact events, then link a runnable root. Bindings are mutable on this handle; clones and attached children retain independent snapshots."
    @property
    def requirements(self) -> list[Requirement]:
        "Unresolved paths, including slots inside attached children. Diagnostic expansion is bounded."
        ...
    def bind(
        self, name: str, value: Union[PreparedGrammar, Token, ExactTokens],
    ) -> None:
        "Fill a declared local or qualified slot once with `PreparedGrammar`, `Token` or `ExactTokens`. Prepared children must use the same vocabulary mapping. Captures a child snapshot without recompiling it. Invalid bindings leave this handle unchanged."
        ...
    def clone(self) -> PreparedGrammar:
        "Independent binding snapshot sharing immutable compiled machinery."
        ...
    def __copy__(self) -> PreparedGrammar:
        "Equivalent to `clone`."
        ...
    def __deepcopy__(self, memo: dict[int, Any]) -> PreparedGrammar:
        "Equivalent to `clone`; immutable compiled storage may remain shared."
        ...
    def link(
        self, *, end_tokens: Optional[Sequence[int]] = None,
        optimization: Optional[Optimization] = None,
        boundary_trigger: Optional[BoundaryTriggerDetail] = None,
    ) -> Constraint:
        "Requires all slots to be bound. Selects final optimization and root end tokens; an explicit boundary request overrides the prepared request. End tokens become allowed only at acceptance and must be committed before stopping."
        ...
    def save(self) -> bytes:
        "Preserves compiled machinery, vocabulary and unresolved slots. Shared definitions are serialized once."
        ...
    @staticmethod
    def load(data: bytes, vocab: Optional[Vocab] = None) -> PreparedGrammar:
        "Rejects malformed or incompatible artifacts. An optional vocabulary must match the saved token-ID/byte mapping and lets the loaded graph share its caches."
        ...

@final
class Constraint:
    "Immutable runnable grammar and root stopping policy. Reuse across sequences; each `start` creates independent mutable state."
    def marker_ids(self) -> list[int]:
        "Sorted, distinct application marker IDs, independent of model token IDs."
        ...
    def start(self) -> ConstraintState:
        ""
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
    "Mutable state for one generated sequence. Query a mask, sample an allowed original ID, then commit it. Clone to branch the sequence."
    def marker_events(self) -> list[tuple[int, int]]:
        "Initial or most recent commit’s `(marker_id, absolute_byte_offset)` batch, sorted by offset then ID, with duplicate pairs removed. These are possible observations, not captures or promises about the final parse. A failed commit clears the batch; mask and forcing queries leave it unchanged."
        ...
    def clone(self) -> ConstraintState:
        "Independent snapshot at the current prefix."
        ...
    def mask(self, size: Optional[int] = None) -> NDArray[np.bool_]:
        "Boolean array indexed by original token ID. The default covers the model ID domain; an explicit size cannot truncate it. Includes the root end-token policy."
        ...
    def fill_mask(self, bitmask: NDArray[np.int32]) -> None:
        "Write packed bits into a contiguous, writable one-dimensional NumPy `int32` array with at least `Constraint.mask_len()` words. Bit `id % 32` in word `id // 32` represents that original ID; signed words preserve all 32 bits."
        ...
    def commit_token(self, token_id: int) -> None:
        "An unknown ID raises an error. A known but grammatically forbidden ID rejects the state and yields an all-zero mask. Use `mask` before sampling."
        ...
    def commit_bytes(self, data: bytes) -> None:
        "Advance by raw bytes rather than model token events. This does not substitute for committing exact-token or end-token events."
        ...
    def is_accepting(self) -> bool:
        "The current prefix is a complete grammar match; further tokens may still be allowed. This alone does not commit an end token."
        ...
    def is_rejected(self) -> bool:
        "The prefix cannot be extended to a valid match."
        ...
    def forced_tokens(self) -> list[int]:
        "Read-only lookahead of up to 10,000 inevitable original IDs. Stops when the root can accept or multiple IDs are possible, including byte aliases. Does not commit or construct a full mask; proof/resource failures raise errors."
        ...
