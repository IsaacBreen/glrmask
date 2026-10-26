"""GLRMask is a grammar-constrained decoding library for high-throughput LLM serving. It is optimized for fast mask computation with extremely low tail latency, even for complex grammars.

GLRMask supports general context-free grammars, including ambiguous ones. It accepts JSON Schema, Lark, and GLRMask's native format: GLRM.

## Installation

```bash
python -m pip install glrmask
```

## Usage

GLRMask has four core types:

- `Grammar`: a grammar, written as source text or built from an expression.
- `Module`: a grammar plus its extern bindings and any precompiled work. Can be saved and loaded.
- `Constraint`: a grammar compiled for a vocabulary, ready to use.
- `ConstraintState`: the state of one generated sequence.

## Quickstart

```python
import glrmask

vocab = glrmask.Vocab.from_id_to_bytes({0: b"hello", 1: b" ", 2: b"world"})
grammar = glrmask.Grammar.from_lark('start: "hello" " " "world"')
constraint = grammar.compile(vocab)
state = constraint.start()
for token_id, expected in [(0, [True, False, False]),
                           (1, [False, True, False]),
                           (2, [False, False, True])]:
    assert state.mask().tolist() == expected
    assert state.commit_token(token_id)
assert state.is_accepting()
```

## Learn more

[Full guide](https://github.com/IsaacBreen/glrmask/blob/main/GUIDE.md) · [API reference](https://glrmask.com/api/python/)
"""
from __future__ import annotations
from collections.abc import Iterator, Mapping
from types import ModuleType
from typing import Any, ClassVar, Iterable, Literal, Optional, Sequence, TypeVar, Union, final, overload
import numpy as np
from numpy.typing import NDArray

_T = TypeVar("_T")

_internal: ModuleType

__version__: str
"""The GLRMask version, for example "0.5.0"."""
__all__: list[str] = [
    "BoundaryMaskStrategy",
    "Constraint",
    "ConstraintState",
    "Expr",
    "Extern",
    "Externs",
    "Grammar",
    "MaskStrategy",
    "Module",
    "ParserBackend",
    "TokenSet",
    "TokenTrie",
    "Vocab",
    "__version__",
]

@final
class MaskStrategy:
    """How masks are computed."""
    AUTO: ClassVar[MaskStrategy]
    """Reuses any precompiled work; otherwise `PARSER_DWA`, or `TOKEN_TRIE` when that would be too large. The default."""
    PARSER_DWA: ClassVar[MaskStrategy]
    """Builds a parser DWA for the grammar. Slowest to compile, with the fastest masks."""
    TOKEN_TRIE: ClassVar[MaskStrategy]
    """Computes each mask by walking the vocabulary. Fastest to compile, with slower masks. Pass a `TokenTrie` instead to change its settings."""
    FAST_COMPILE: ClassVar[MaskStrategy]
    """Same as `TOKEN_TRIE`."""
    FAST_MASKS: ClassVar[MaskStrategy]
    """Same as `PARSER_DWA`."""

@final
class BoundaryMaskStrategy:
    """How masks are computed at grammar boundaries."""
    AUTO: ClassVar[BoundaryMaskStrategy]
    """`PARSER_DWA`, or `TOKEN_TRIE` when that would be too large. The default."""
    PARSER_DWA: ClassVar[BoundaryMaskStrategy]
    """Builds a parser DWA across each boundary. Slower to compile, with faster masks. Raises `ValueError` if it would be too large."""
    TOKEN_TRIE: ClassVar[BoundaryMaskStrategy]
    """Walks the vocabulary at each boundary as needed. Faster to compile, with slower masks."""
    FAST_COMPILE: ClassVar[BoundaryMaskStrategy]
    """Same as `TOKEN_TRIE`."""
    FAST_MASKS: ClassVar[BoundaryMaskStrategy]
    """Same as `PARSER_DWA`."""

@final
class ParserBackend:
    """How the parser is represented and run when masks are computed with `MaskStrategy.TOKEN_TRIE`."""
    LR_TABLE: ClassVar[ParserBackend]
    """An LR table. Faster to compile, with slower masks. The default."""
    TEMPLATE_DFA: ClassVar[ParserBackend]
    """Template DFAs. Slower to compile, with faster masks."""

@final
class TokenTrie:
    """Settings for `MaskStrategy.TOKEN_TRIE`. Pass one as `mask_strategy`.

    `TokenTrie()` is the same as `MaskStrategy.TOKEN_TRIE`."""
    def __new__(cls, *, parser_backend: Optional[ParserBackend] = None, merge_equivalent_tokens: bool = False) -> TokenTrie: ...
    @property
    def parser_backend(self) -> ParserBackend:
        """How the parser is represented. Defaults to `ParserBackend.LR_TABLE`."""
        ...
    @property
    def merge_equivalent_tokens(self) -> bool:
        """Whether tokens that the grammar treats the same are merged, so masks walk fewer tokens. Defaults to `False`.

        Tokens are not merged in a composition that includes precompiled parts."""
        ...

@final
class TokenSet:
    """A set of token IDs."""
    def __new__(cls, ids: Iterable[int], /) -> TokenSet:
        """Creates a set from `ids`."""
        ...
    @staticmethod
    def from_ranges(ranges: Iterable[range], /) -> TokenSet:
        """Creates a set from `range` objects, which must have step 1."""
        ...
    def __repr__(self) -> str:
        """An expression that rebuilds the set, or a summary for large sets."""
        ...
    def __eq__(self, other: object, /) -> bool:
        """Whether both sets contain the same IDs."""
        ...

@final
class Vocab:
    """A model's tokens and their bytes. A token ID it doesn't contain never matches text, but a token set can still allow it."""
    @staticmethod
    def from_dict(token_to_id: dict[Union[str, bytes], int], /) -> Vocab:
        """Creates a vocabulary from a `{token: id}` dict.

        `str` tokens are UTF-8 encoded; `bytes` tokens are used as is.
        """
        ...
    @staticmethod
    def from_id_to_bytes(id_to_bytes: dict[int, bytes], /) -> Vocab:
        """Creates a vocabulary from an `{id: bytes}` dict."""
        ...
    @staticmethod
    def from_llama_cpp(llm: Any, /) -> Vocab:
        """Creates a vocabulary from a llama-cpp-python `Llama` model.

        Tokens with no text, such as control and end-of-generation tokens, become special tokens.
        """
        ...
    @property
    def llama_cpp_end_token_ids(self) -> list[int]:
        """The model's end-of-generation token IDs, or an empty list if the vocabulary was not created by `from_llama_cpp`. Put them in a `TokenSet` after the grammar to require one."""
        ...

@final
class Extern:
    """A named placeholder in a grammar."""
    @property
    def name(self) -> str:
        """The extern's name."""
        ...
    @property
    def kind(self) -> Literal["token", "grammar"]:
        """What the extern can be bound to: `"token"` or `"grammar"`."""
        ...
    def __repr__(self) -> str:
        """A representation for debugging."""
        ...
    def __eq__(self, other: object, /) -> bool:
        """Whether both have the same name and kind."""
        ...

@final
class Externs(Mapping[str, Extern]):
    """A grammar's or module's externs, in declaration order.

    A read-only mapping from name to `Extern`.
    """
    @property
    def unbound(self) -> Externs:
        """The externs not yet bound."""
        ...
    def __getitem__(self, key: str, /) -> Extern:
        """The extern named `key`. Raises `KeyError` if there is none."""
        ...
    def __iter__(self) -> Iterator[str]:
        """The extern names, in declaration order."""
        ...
    def __len__(self) -> int:
        """The number of externs."""
        ...
    @overload
    def get(self, key: str, /) -> Optional[Extern]:
        """The extern named `key`, or `default` if there is none."""
        ...
    @overload
    def get(self, key: str, default: Optional[Union[Extern, _T]], /) -> Union[Extern, _T]: ...
    def keys(self) -> list[str]:  # type: ignore[override]
        """The extern names, in declaration order."""
        ...
    def values(self) -> list[Extern]:  # type: ignore[override]
        """The externs, in declaration order."""
        ...
    def items(self) -> list[tuple[str, Extern]]:  # type: ignore[override]
        """`(name, extern)` pairs, in declaration order."""
        ...
    def __repr__(self) -> str:
        """A representation for debugging."""
        ...
    def __eq__(self, other: object, /) -> bool:
        """Whether both list the same externs in the same order."""
        ...

@final
class Expr:
    """A grammar expression."""
    @staticmethod
    def choice(*parts: Union[Grammar, Module, Expr, TokenSet]) -> Expr:
        """Matches any one of `parts`.

        Raises `ValueError` if a part has unbound externs.
        """
        ...
    @staticmethod
    def sequence(*parts: Union[Grammar, Module, Expr, TokenSet]) -> Expr:
        """Matches `parts` in order.

        Raises `ValueError` if a part has unbound externs.
        """
        ...
    @staticmethod
    def literal(value: Union[str, bytes], /) -> Expr:
        """Matches exactly `value`. A `str` is UTF-8 encoded."""
        ...

@final
class Module:
    """A grammar you can compose with others and partly compile before its externs are bound."""
    def __new__(cls, source: Grammar, /) -> Module:
        """Creates a module from a grammar."""
        ...
    @overload
    def bind(self, name: str, value: Union[Grammar, Module, Expr, TokenSet, int], /) -> None:
        """Binds the extern `name` to `value`.

        `bind(a=x, b=y)` binds several at once; if one fails, none is bound. Raises `ValueError` if a name is undeclared or already bound, or a value is the wrong kind.
        """
        ...
    @overload
    def bind(self, **bindings: Union[Grammar, Module, Expr, TokenSet, int]) -> None:
        """Binds the extern `name` to `value`.

        `bind(a=x, b=y)` binds several at once; if one fails, none is bound. Raises `ValueError` if a name is undeclared or already bound, or a value is the wrong kind.
        """
        ...
    def precompile(
        self,
        vocab: Vocab,
        *,
        mask_strategy: Optional[Union[MaskStrategy, TokenTrie]] = None,
        boundary_mask_strategy: Optional[BoundaryMaskStrategy] = None,
    ) -> None:
        """Compiles as much as possible for `vocab` now, even with externs unbound, so later compiles are faster.

        The keyword arguments choose how masks are computed; see `MaskStrategy`, `TokenTrie` and `BoundaryMaskStrategy`.
        """
        ...
    def compile(
        self,
        vocab: Vocab,
        *,
        mask_strategy: Optional[Union[MaskStrategy, TokenTrie]] = None,
        boundary_mask_strategy: Optional[BoundaryMaskStrategy] = None,
    ) -> Constraint:
        """Compiles the module for `vocab`, reusing any precompiled work. Raises `ValueError` if any extern is unbound or the module was precompiled for another vocabulary.

        The keyword arguments choose how masks are computed; see `MaskStrategy`, `TokenTrie` and `BoundaryMaskStrategy`.
        """
        ...
    @property
    def externs(self) -> Externs:
        """The module's externs."""
        ...
    def copy(self) -> Module:
        """A copy of the module. Binding the copy does not change the original."""
        ...
    def __copy__(self) -> Module:
        """Same as `copy()`."""
        ...
    def __deepcopy__(self, memo: dict[int, Any]) -> Module:
        """Same as `copy()`."""
        ...
    def save(self) -> bytes:
        """Saves the module as bytes, with its precompiled work but without the vocabulary. Bytes saved by one GLRMask version may not load in another."""
        ...
    @staticmethod
    def load(data: bytes, /) -> Module:
        """Loads a module saved with `save()`."""
        ...

@final
class Grammar:
    """A grammar, parsed from source or built from an Expr."""
    @staticmethod
    def from_glrm(source: str, /) -> Grammar:
        """Creates a grammar from GLRM source."""
        ...
    @staticmethod
    def from_json_schema(schema: Any, /) -> Grammar:
        """Creates a grammar from a JSON Schema, given as JSON text or a JSON-compatible object such as a dict."""
        ...
    @staticmethod
    def from_lark(source: str, /) -> Grammar:
        """Creates a grammar from Lark source."""
        ...
    @property
    def externs(self) -> Externs:
        """The grammar's externs."""
        ...
    def precompile(
        self,
        vocab: Vocab,
        *,
        mask_strategy: Optional[Union[MaskStrategy, TokenTrie]] = None,
        boundary_mask_strategy: Optional[BoundaryMaskStrategy] = None,
    ) -> Module:
        """Compiles as much as possible for `vocab` now, even with externs unbound, and returns a `Module` that keeps the work.

        The keyword arguments choose how masks are computed; see `MaskStrategy`, `TokenTrie` and `BoundaryMaskStrategy`.
        """
        ...
    def compile(
        self,
        vocab: Vocab,
        *,
        mask_strategy: Optional[Union[MaskStrategy, TokenTrie]] = None,
        boundary_mask_strategy: Optional[BoundaryMaskStrategy] = None,
    ) -> Constraint:
        """Compiles the grammar for `vocab`. Raises `ValueError` if any extern is unbound.

        The keyword arguments choose how masks are computed; see `MaskStrategy`, `TokenTrie` and `BoundaryMaskStrategy`.
        """
        ...
    @staticmethod
    def from_expr(expr: Union[Expr, Grammar, TokenSet], /) -> Grammar:
        """Creates a grammar from an expression."""
        ...
    @overload
    def bind(self, name: str, value: Union[Grammar, Expr, TokenSet, int], /) -> None:
        """Binds the extern `name` to `value`.

        `bind(a=x, b=y)` binds several at once; if one fails, none is bound. Raises `ValueError` if a name is undeclared or already bound, or a value is the wrong kind.
        """
        ...
    @overload
    def bind(self, **bindings: Union[Grammar, Expr, TokenSet, int]) -> None:
        """Binds the extern `name` to `value`.

        `bind(a=x, b=y)` binds several at once; if one fails, none is bound. Raises `ValueError` if a name is undeclared or already bound, or a value is the wrong kind.
        """
        ...
    def copy(self) -> Grammar:
        """A copy of the grammar. Binding the copy does not change the original."""
        ...
    def __copy__(self) -> Grammar:
        """Same as `copy()`."""
        ...
    def __deepcopy__(self, memo: dict[int, Any]) -> Grammar:
        """Same as `copy()`."""
        ...

@final
class Constraint:
    """A grammar compiled for a vocabulary. Call `start()` for each sequence you generate."""
    def start(self) -> ConstraintState:
        """A new state with nothing committed."""
        ...
    def mask_len(self) -> int:
        """The number of `int32` words that `fill_mask` needs."""
        ...
    def save(self) -> bytes:
        """Saves the constraint as bytes, without the vocabulary.

        Bytes saved by one GLRMask version may not load in another.
        """
        ...
    @staticmethod
    def load(data: bytes, vocab: Vocab) -> Constraint:
        """Loads a constraint saved with `save()`. Raises `ValueError` if `vocab` isn't the one it was compiled for."""
        ...

@final
class ConstraintState:
    """A constraint's state during generation."""
    def copy(self) -> ConstraintState:
        """A copy of the state."""
        ...
    def __copy__(self) -> ConstraintState:
        """Same as `copy()`."""
        ...
    def mask(self, *, size: Optional[int] = None) -> NDArray[np.bool_]:
        """The tokens allowed next, as a NumPy `bool` array indexed by token ID.

        `size` sets the array length, for example to match your logits; it must cover every token ID.
        """
        ...
    def fill_mask(self, bitmask: NDArray[np.int32], /) -> None:
        """Writes the mask into `bitmask`, zeroing any extra words. `bitmask` must be a contiguous, writable 1-D `int32` array of at least `Constraint.mask_len()` words. Raises `TypeError` if it isn't `int32`, or `ValueError` if it's too short or not contiguous.

        Token `id` is allowed when bit `id % 32` of word `id // 32` is set.
        """
        ...
    def commit_token(self, token_id: int, /) -> bool:
        """Commits a token. If it isn't allowed, returns `False` and terminates the state."""
        ...
    def commit_tokens(self, token_ids: Sequence[int], /) -> bool:
        """Commits each token in order. If one isn't allowed, returns `False` and terminates the state."""
        ...
    def commit_bytes(self, data: bytes, /) -> bool:
        """Commits raw bytes. If they aren't allowed, returns `False` and terminates the state."""
        ...
    def is_accepting(self) -> bool:
        """Whether everything committed so far is a complete match."""
        ...
    def is_terminated(self) -> bool:
        """Whether nothing more can be committed."""
        ...
    def commit_forced_tokens(self) -> list[int]:
        """Commits the tokens for what must come next, taking the longest token at each step, and returns them. Stops before anything that could vary, or after 1,000 tokens."""
        ...
