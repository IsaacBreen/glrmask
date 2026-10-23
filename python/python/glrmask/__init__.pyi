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
grammar = glrmask.Grammar.from_ebnf('start ::= "hello" " " "world"')
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
from typing import Any, ClassVar, Iterable, Literal, Optional, Sequence, Union, final, overload
import numpy as np
from numpy.typing import NDArray

_internal: ModuleType

@final
class MaskStrategy:
    """How masks are computed, trading compile time against mask speed."""
    AUTO: ClassVar[MaskStrategy]
    """`PARSER_DWA`, unless the module was precompiled with `TOKEN_CLASS_TRIE`. The default."""
    PARSER_DWA: ClassVar[MaskStrategy]
    """Precomputes masks: slowest compile, fastest masks."""
    TOKEN_TRIE: ClassVar[MaskStrategy]
    """Computes each mask by walking the vocabulary: fastest compile, slower masks."""
    TOKEN_CLASS_TRIE: ClassVar[MaskStrategy]
    """Like `TOKEN_TRIE`, but treats tokens the grammar cannot tell apart as one."""
    FAST_BUILD: ClassVar[MaskStrategy]
    """Same as `TOKEN_TRIE`."""
    FAST_RUNTIME: ClassVar[MaskStrategy]
    """Same as `PARSER_DWA`."""

@final
class BoundaryMaskStrategy:
    """How masks are computed where one grammar meets another."""
    AUTO: ClassVar[BoundaryMaskStrategy]
    """`PARSER_DWA`, falling back to `TOKEN_TRIE` when that would be too large. The default."""
    PARSER_DWA: ClassVar[BoundaryMaskStrategy]
    """Precomputes boundary masks: slower compile, faster masks.

Raises `ValueError` if they would be too large."""
    TOKEN_TRIE: ClassVar[BoundaryMaskStrategy]
    """Computes boundary masks as needed: faster compile, slower masks."""
    FAST_BUILD: ClassVar[BoundaryMaskStrategy]
    """Same as `TOKEN_TRIE`."""
    FAST_RUNTIME: ClassVar[BoundaryMaskStrategy]
    """Same as `PARSER_DWA`."""

@final
class TokenSet:
    """A set of token IDs.

    In a grammar, it matches any one of its tokens. Every ID must be in the vocabulary.
    """
    def __init__(self, ids: Iterable[int]) -> None:
        """Creates a set from `ids`."""
        ...
    @staticmethod
    def from_ranges(ranges: Iterable[range]) -> TokenSet:
        """Creates a set from `range` objects, which must have step 1."""
        ...
    def __repr__(self) -> str:
        """The set as ranges, for example `TokenSet[0..=9, 20..=29]`."""
        ...
    def __eq__(self, other: object) -> bool:
        """Whether both sets contain the same IDs."""
        ...

@final
class Vocab:
    """A model's vocabulary: each token ID and its bytes.

    Use each token's exact bytes, not its display form: `b" hello"`, not `"Ġhello"`. IDs need not be contiguous.
    """
    @staticmethod
    def from_dict(token_to_id: dict[Union[str, bytes], int], *, special_tokens: Optional[Sequence[int]] = None) -> Vocab:
        """Creates a vocabulary from a `{token: id}` dict.

        `str` tokens are UTF-8 encoded; `bytes` tokens are used as is. `special_tokens` adds tokens that have no bytes.
        """
        ...
    @staticmethod
    def from_id_to_bytes(id_to_bytes: dict[int, bytes], *, special_tokens: Optional[Sequence[int]] = None) -> Vocab:
        """Creates a vocabulary from an `{id: bytes}` dict.

        `special_tokens` adds tokens that have no bytes.
        """
        ...
    @staticmethod
    def from_llama_cpp(llm: Any) -> Vocab:
        """Creates a vocabulary from a llama-cpp-python `Llama` model.

        Tokens with no text, such as control and end-of-generation tokens, become special tokens.
        """
        ...
    @property
    def llama_cpp_end_token_ids(self) -> list[int]:
        """The model's end-of-generation token IDs. Put them in a `TokenSet` after the grammar to require one."""
        ...
    def __len__(self) -> int:
        """The number of tokens with bytes; special tokens are not counted."""
        ...

@final
class Extern:
    """An extern declared by a grammar: a named slot to bind later."""
    @property
    def name(self) -> str:
        """The extern's name."""
        ...
    @property
    def kind(self) -> Literal["token", "grammar"]:
        """What the extern can be bound to: `"token"` or `"grammar"`."""
        ...
    def __repr__(self) -> str:
        """The externs, for debugging."""
        ...
    def __eq__(self, other: object) -> bool:
        """Whether both have the same name and kind."""
        ...

@final
class Externs(Mapping[str, Extern]):
    """A grammar's or module's externs, in declaration order.

    A read-only mapping from name to `Extern`.
    """
    @property
    def unbound(self) -> Externs:
        """Only the externs that are not yet bound."""
        ...
    def __getitem__(self, key: str) -> Extern:
        """The extern named `key`. Raises `KeyError` if there is none."""
        ...
    def __iter__(self) -> Iterator[str]:
        """The extern names, in declaration order."""
        ...
    def __len__(self) -> int:
        """The number of externs."""
        ...
    def get(self, key: str, default: Optional[Extern] = None) -> Optional[Extern]:
        """The extern named `key`, or `default` if there is none."""
        ...
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
        """The debug representation."""
        ...
    def __eq__(self, other: object) -> bool:
        """Whether both list the same externs in the same order."""
        ...

@final
class Expr:
    """A choice or sequence of grammars, modules, token sets and literals.

    Each grammar keeps its own rules, including which whitespace it ignores.
    """
    @staticmethod
    def choice(*parts: Union[Grammar, Module, Expr, TokenSet]) -> Expr:
        """Matches any one of `parts`.

        Raises `ValueError` if a part has unbound externs.
        """
        ...
    @staticmethod
    def sequence(*parts: Union[Grammar, Module, Expr, TokenSet]) -> Expr:
        """Matches `parts` one after another.

        One token may span several parts. Raises `ValueError` if a part has unbound externs.
        """
        ...
    @staticmethod
    def literal(value: Union[str, bytes], /) -> Expr:
        """Matches exactly `value`, with no whitespace skipped. A `str` is UTF-8 encoded."""
        ...

@final
class Module:
    """A grammar plus its extern bindings and any precompiled work.

    Externs can stay unbound until `compile`.
    """
    def __init__(self, source: Grammar, /) -> None:
        """Creates a module from a grammar."""
        ...
    @overload
    def bind(self, name: str, value: Union[Grammar, Module, Expr, TokenSet], /) -> None:
        """Binds the extern `name` to `value`: a grammar, module, expression or token set.

        `bind(a=x, b=y)` binds several at once; if one fails, none is bound. Raises `ValueError` if a name is undeclared or already bound, or a value is the wrong kind for it. A token extern takes only a token set.
        """
        ...
    @overload
    def bind(self, **bindings: Union[Grammar, Module, Expr, TokenSet]) -> None:
        """Binds the extern `name` to `value`: a grammar, module, expression or token set.

        `bind(a=x, b=y)` binds several at once; if one fails, none is bound. Raises `ValueError` if a name is undeclared or already bound, or a value is the wrong kind for it. A token extern takes only a token set.
        """
        ...
    def precompile(
        self,
        vocab: Vocab,
        *,
        mask_strategy: MaskStrategy = MaskStrategy.AUTO,
        boundary_mask_strategy: BoundaryMaskStrategy = BoundaryMaskStrategy.AUTO,
    ) -> None:
        """Compiles as much as possible for `vocab` now, so later compiles are fast.

        Externs may still be unbound. The keyword arguments choose how masks are computed.
        """
        ...
    def compile(
        self,
        vocab: Vocab,
        *,
        mask_strategy: MaskStrategy = MaskStrategy.AUTO,
        boundary_mask_strategy: BoundaryMaskStrategy = BoundaryMaskStrategy.AUTO,
    ) -> Constraint:
        """Compiles the module for `vocab`, reusing any precompiled work.

        Every extern must be bound. The keyword arguments choose how masks are computed.
        """
        ...
    @property
    def externs(self) -> Externs:
        """The module's externs, in declaration order.

        Empty if the source has a syntax error; `bind` and `compile` report it.
        """
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
        """Saves the module, including precompiled work, as bytes.

        The vocabulary is not included. Bytes saved by one GLRMask version may not load in another.
        """
        ...
    @staticmethod
    def load(data: bytes) -> Module:
        """Loads a module saved with `save`.

        Precompiled work applies only to the vocabulary it was built for; compiling with a different one is an error.
        """
        ...

@final
class Grammar:
    """A grammar, written as source text or built from an `Expr`.

    Source text is parsed on first use, so syntax errors appear when you bind or compile.
    """
    @staticmethod
    def from_ebnf(source: str) -> Grammar:
        """Creates a grammar from EBNF source."""
        ...
    @staticmethod
    def from_glrm(source: str) -> Grammar:
        """Creates a grammar from GLRM, GLRMask's native format."""
        ...
    @staticmethod
    def from_json_schema(schema: Any) -> Grammar:
        """Creates a grammar from a JSON Schema, given as JSON text or a JSON-compatible object such as a dict.

        Standard keywords that GLRMask does not support are an error. Unknown keywords and `format` are ignored.
        """
        ...
    @staticmethod
    def from_lark(source: str) -> Grammar:
        """Creates a grammar from Lark source.

        A token name such as `<|end|>` declares a token extern, which must be bound before compiling.
        """
        ...
    @property
    def externs(self) -> Externs:
        """The grammar's externs, in declaration order.

        Empty if the source has a syntax error.
        """
        ...
    def precompile(
        self,
        vocab: Vocab,
        *,
        mask_strategy: MaskStrategy = MaskStrategy.AUTO,
        boundary_mask_strategy: BoundaryMaskStrategy = BoundaryMaskStrategy.AUTO,
    ) -> Module:
        """Compiles as much as possible for `vocab` now, and returns a `Module` that keeps the work.

        Externs may still be unbound. The keyword arguments choose how masks are computed.
        """
        ...
    def compile(
        self,
        vocab: Vocab,
        *,
        mask_strategy: MaskStrategy = MaskStrategy.AUTO,
        boundary_mask_strategy: BoundaryMaskStrategy = BoundaryMaskStrategy.AUTO,
    ) -> Constraint:
        """Compiles the grammar for `vocab`.

        The keyword arguments choose how masks are computed; see `MaskStrategy` and `BoundaryMaskStrategy`.
        """
        ...
    @staticmethod
    def from_expr(expr: Union[Expr, Grammar, TokenSet]) -> Grammar:
        """Creates a grammar from an expression.

        Use it to require an end token after a grammar: `Grammar.from_expr(Expr.sequence(grammar, TokenSet([eos_id])))`.

        The end token must be in the vocabulary.
        """
        ...
    def bind(self, name: str, value: Union[Grammar, Expr, TokenSet]) -> None:
        """Binds the extern `name` to `value`: a grammar, expression or token set.

        Raises `ValueError` if `name` is undeclared or already bound, or if `value` is the wrong kind for it.
        """
        ...
    def __copy__(self) -> Grammar:
        """A copy of the grammar."""
        ...
    def __deepcopy__(self, memo: dict[int, Any]) -> Grammar:
        """A copy of the grammar."""
        ...

@final
class Constraint:
    """A grammar compiled for a vocabulary, ready to constrain generation.

    Share one between threads, and call `start` for each sequence.
    """
    def marker_ids(self) -> list[int]:
        """The IDs of the grammar's markers, in ascending order."""
        ...
    def start(self) -> ConstraintState:
        """Starts a new sequence."""
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
        """Loads a constraint saved with `save`.

        `vocab` must be the vocabulary it was compiled for; a different one raises `ValueError`.
        """
        ...

@final
class ConstraintState:
    """The state of one generated sequence.

    Copy it to try tokens without changing the original.
    """
    def marker_events(self) -> list[tuple[int, int]]:
        """The markers passed by the most recent commit, or by `start()`, as `(marker_id, byte_offset)` pairs in byte order.

        A marker means some parse reached that point. That parse may still fail later.
        """
        ...
    def copy(self) -> ConstraintState:
        """A copy of the state, for trying tokens without changing this one."""
        ...
    def __copy__(self) -> ConstraintState:
        """A copy of the state, for trying tokens without changing this one."""
        ...
    def mask(self, size: Optional[int] = None) -> NDArray[np.bool_]:
        """The tokens allowed next, as a Boolean array indexed by token ID.

        `size` sets the array length, for example to match your logits; it must cover every token ID. All entries are `False` when nothing can follow: the sequence is complete, or a commit failed.
        """
        ...
    def fill_mask(self, bitmask: NDArray[np.int32]) -> None:
        """Writes the mask into `bitmask`, a contiguous, writable 1-D `int32` array of at least `Constraint.mask_len()` words.

        Token `id` is allowed when bit `id % 32` of word `id // 32` is set. Extra words are set to zero.
        """
        ...
    def commit_token(self, token_id: int) -> bool:
        """Commits one token.

        Returns `False` if it is not allowed, leaving the state unusable. To try a token safely, commit it on a clone.
        """
        ...
    def commit_tokens(self, token_ids: Sequence[int]) -> bool:
        """Commits the tokens in order.

        Returns `False` at the first token that is not allowed, leaving the state unusable.
        """
        ...
    def commit_bytes(self, data: bytes) -> bool:
        """Commits raw bytes instead of a token.

        Returns `False` if they are not allowed, leaving the state unusable. Bytes never match a token set; commit those tokens with `commit_token`.
        """
        ...
    def is_accepting(self) -> bool:
        """Whether the input committed so far is a complete match for the grammar.

        More input may still be allowed; see `is_terminated`.
        """
        ...
    def is_terminated(self) -> bool:
        """Whether nothing more can be committed.

        This is true when the sequence is finished, for example after its end token, and after a failed commit. `is_accepting` tells the two apart.

        A complete match is not terminated if more input could follow, such as whitespace the grammar ignores.
        """
        ...
    def forced_tokens(self) -> list[int]:
        """The tokens that must come next, without committing them.

        Stops where more than one token could follow, or after 10,000 tokens or bytes. When forced text could be tokenized in several ways, the longest token is chosen at each step.
        """
        ...
