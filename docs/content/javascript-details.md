## Vocabularies and token sets

`new Vocab(entries, options?)` takes `[id, bytes]` pairs, with each token's exact bytes as a `Uint8Array` or `Buffer`.
Use the bytes the tokenizer produces, not its display form: a token may be part of a UTF-8 character. IDs need not be
contiguous, and two tokens may have the same bytes. `specialTokenIds` adds tokens that have no bytes, such as
end-of-sequence or tool-call tokens. An ID can't appear twice.

```js
const vocab = new Vocab([
  [0, Uint8Array.of(0xc3)],
  [1, Uint8Array.of(0xa9)],
  [2, Buffer.from('é')],
], { specialTokenIds: [3] });
```

`vocab.size` counts the tokens with bytes. `new TokenSet(ids)` is a set of token IDs, and
`TokenSet.fromRanges([{start, end}])` builds one from ranges that exclude `end`. See [token
sets](https://github.com/IsaacBreen/glrmask/blob/main/docs/token-sets.md).

## Grammars

| Factory | Input |
| --- | --- |
| `Grammar.fromEbnf(source)` | EBNF |
| `Grammar.fromGlrm(source)` | [GLRM](https://github.com/IsaacBreen/glrmask/blob/main/docs/glrm-format.md) |
| `Grammar.fromJsonSchema(schema)` | JSON Schema, as JSON text or a value |
| `Grammar.fromLark(source)` | [Lark](https://github.com/IsaacBreen/glrmask/blob/main/docs/lark-dialect.md) |

## Composing grammars

```js
import { Module, Expr, Constraint } from '@glrmask/node';

const vocab = new Vocab(Array.from({ length: 256 }, (_, id) => [id, Uint8Array.of(id)]));
const weather = new Module(Grammar.fromJsonSchema({
  type: 'object',
  properties: { city: { type: 'string' } },
  required: ['city'],
  additionalProperties: false,
}));
await weather.precompile(vocab);

const call = new Module(Grammar.fromGlrm(`
  extern grammar ARGS
  start call
  nt call = "<weather>" ARGS "</weather>"
`));
call.bind('ARGS', weather);
const constraint = await call.compile(vocab);
```

`bind(name, value)` takes a `Grammar`, `Module`, `Expr`, `TokenSet` or token ID, and `bind({A: a, B: b})` binds several
at once; if one fails, none is bound. `bind` changes the grammar or module in place, so `clone()` it first to bind the
same one in different ways. `externs` is a read-only `Map` of the declared externs, and `externs.unbound` holds the ones
not yet bound.

`Expr.choice(...parts)`, `Expr.sequence(...parts)` and `Expr.literal(text)` combine grammars without writing a new one:

```js
const reply = Expr.choice(Grammar.fromEbnf('start ::= "yes"'), Grammar.fromEbnf('start ::= "no"'));
const grammar = Grammar.fromExpr(Expr.sequence(Expr.literal('Answer: '), reply));
```

## Mask strategies

`compile` and `precompile` take `{maskStrategy, boundaryMaskStrategy}`.

| `MaskStrategy` | Value | |
| --- | --- | --- |
| `Auto` | `'auto'` | The default. Reuses any precompiled work; otherwise `ParserDwa`, or `TokenTrie` when that would be too large. |
| `ParserDwa` | `'parser-dwa'` | Slowest to compile, with the fastest masks. |
| `TokenTrie` | `'token-trie'` | Fastest to compile, with slower masks. |

`FastMasks` and `FastCompile` are aliases for `ParserDwa` and `TokenTrie`. To change `TokenTrie`'s settings, pass
`{kind: 'token-trie', parserBackend: ParserBackend.TemplateDfa, mergeEquivalentTokens: true}` as `maskStrategy`.
`BoundaryMaskStrategy` has the same three values, for the boundaries where a grammar meets a bound extern. See [mask
strategies](https://github.com/IsaacBreen/glrmask/blob/main/GUIDE.md#mask-strategies).

## Generating

```js
const state = constraint.start();
const mask = new Uint32Array(constraint.maskLength);
state.fillMask(mask);
const allowed = (id) => ((mask[id >>> 5] >>> (id & 31)) & 1) === 1;
```

A mask has one bit per token ID: token `id` is allowed when bit `id % 32` of element `Math.floor(id / 32)` is set.
`fillMask` writes it into an array you reuse; `mask()` returns a new one.

`commitToken`, `commitTokens` and `commitBytes` return whether the input was allowed. A failed commit ends the state, so
`clone()` it first to try a token without committing to it. Commit special tokens and token sets with `commitToken`;
bytes never match them.

`isAccepting()` says whether the input so far is a complete match, and `isTerminated()` whether nothing more can be
committed. `forcedTokens()` returns the tokens that must come next. `markerEvents()` lists the markers the last commit
passed, each `{id, byteOffset}` with a `bigint` offset.

[examples/generate.mjs](https://github.com/IsaacBreen/glrmask/blob/main/bindings/node/examples/generate.mjs) is a
complete generation loop.

## Async and sync

`compile`, `precompile`, `save` and `load` return promises and run off the main thread. `compileSync`, `precompileSync`,
`saveSync` and `loadSync` do the same work and block. Everything else is synchronous.

An async call works on a snapshot of its inputs, so changing or disposing them afterwards doesn't affect it. If a module
is bound while its `precompile` is running, the promise rejects; precompile it again.

## Saving

```js
const bytes = await constraint.save();
const restored = await Constraint.load(bytes, vocab);

const moduleBytes = await call.save();
const module = await Module.load(moduleBytes);
```

Saved bytes are the same in every binding, so a constraint saved from Python loads in Node. A saved constraint doesn't
include its vocabulary: pass the same vocabulary to `load`, which checks that it matches exactly. A saved module records
the vocabulary its precompiled parts were built for, and `compile` checks it. Saved bytes are only guaranteed to load in
the same GLRMask version on the same architecture, so keep the grammar source to compile again after an upgrade. Don't
load saved bytes from untrusted sources.

## Errors

Arguments of the wrong type throw `TypeError`, and token IDs out of range throw `RangeError`. Everything else throws a
`GlrMaskError`, whose `code` is one of:

| Code | |
| --- | --- |
| `GLRMASK_ENGINE_ERROR` | A grammar, binding, compile or load failed. The message says why. |
| `GLRMASK_INVALID_ARGUMENT` | An argument is invalid. The message says which. |
| `GLRMASK_BUFFER_TOO_SMALL` | The array passed to `fillMask` is shorter than `maskLength`. |
| `GLRMASK_CLOSED` | The object was disposed. |
| `GLRMASK_PANIC` | An internal error. Discard the object; please report it. |

Async methods reject with the same errors.

## Freeing memory

Objects hold native memory, freed when they are garbage collected. `dispose()`, or a `using` declaration, frees it
straight away. Disposing twice is harmless. A state keeps its constraint alive, so disposing the constraint doesn't affect
states already started.

## API reference

[index.d.cts](https://github.com/IsaacBreen/glrmask/blob/main/bindings/node/index.d.cts) documents every class and method.
CommonJS and ESM export the same API.

## Building from source

You need Node.js 22 or newer, a stable Rust toolchain, and a C/C++ compiler. On Windows, use a Visual Studio Developer
PowerShell with the MSVC build tools.

```sh
npm ci --ignore-scripts --prefix bindings/node
npm run build --prefix bindings/node
```

`npm run build` puts the compiled addon next to the package's loader, where it takes priority over an installed platform
package. `npm pack --prefix bindings/node` makes a package you can install elsewhere. A build runs only on the platform it
was built for.
