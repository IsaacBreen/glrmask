# JavaScript and TypeScript

`@glrmask/node` is a native Node.js binding to GLRMask's Rust compiler and decoding engine. It supports CommonJS and ECMAScript modules, includes TypeScript declarations, and uses Node-API rather than embedding Python or launching a parser subprocess.

This is **Node.js support, not browser WebAssembly**. A browser, Bun or Deno target is not implied by the JavaScript interface. Bring your inference backend's exact token-ID-to-byte mapping; the package does not run a model, download a tokenizer, apply GPU logits, or choose a sampling policy.

## Owned assemblies and expressions

The public lifecycle is source `Grammar` -> `Module` (with per-module settings, occurrence bindings and precompilation) -> runnable `Constraint` -> per-sequence `ConstraintState`. `Expr` composes whole Grammar, Module, Expr, Token and ExactTokens pieces without merging their lexical scopes.

`Grammar` is immutable authored source. `Module` owns mutable assembly snapshots, per-module settings (`maskStrategy` and `boundaryMaskStrategy`), and any precompiled work.

```js
const {Vocab, Grammar, Module, Expr, MaskStrategy, BoundaryMaskStrategy} = require('@glrmask/node');
const vocab = new Vocab([[0,Buffer.from('xay')],[1,Buffer.from('xby')]]);
const child = new Module(Grammar.fromEbnf('start ::= "a"'), {
  maskStrategy: MaskStrategy.TokenClassTrie,
  boundaryMaskStrategy: BoundaryMaskStrategy.FastBuild,
});
await child.precompile(vocab);

const parent = new Module(Grammar.fromGlrm(
  'glrm 1; start start; extern grammar child; nt start = "x" child "y";'));
parent.bind('child', child); // Also accepts cold Grammar, Expr, Token and ExactTokens.
const restored = await Module.load(await parent.save(), vocab);
const constraint = await restored.compile(vocab);
const expression = Expr.choice(child, Grammar.fromEbnf('start ::= "b"'));
```

`bind` mutates atomically and returns `undefined`; `precompile` returns `Promise<void>`. Explicit `precompileSync`, `compileSync`, `saveSync` and `loadSync` methods block the caller. Async compile/save capture snapshots at scheduling, retaining their native owners after JS disposal. Async precompile checks the receiver revision at completion: concurrent mutation rejects the promise and preserves newer bindings; retry against the current snapshot.

Getters and setters for `maskStrategy` and `boundaryMaskStrategy` allow inspecting and modifying module settings. Retained local work is never discarded when settings change. Children retain their own settings; final compile accepts only `{ endTokens }`.

`externs` returns a read-only Map-like snapshot of `{ name, kind, span }`, with an `unbound` property of the same type. Copies share immutable work and retain independent mutable bindings.

Loading never compiles; the optional vocabulary validates exact saved target identity. Empty sequence accepts epsilon; empty choice fails when materialized.

## Installation

Use Node.js 22 or newer. This source revision prepares a thin root: `@glrmask/node` itself contains only JavaScript, TypeScript declarations and a loader, while the native binary for your OS/architecture/libc arrives automatically through an `optionalDependencies` entry (`@glrmask/node-<platform>`). No Rust compiler, C++ compiler, Python runtime or post-install download is needed on a supported platform. Prebuilt release binaries cover Windows x64 and arm64 (MSVC), macOS x64 and arm64 (macOS 11+), and Linux x64 and arm64 in both glibc (manylinux_2_28, minimum glibc 2.28) and musl (Alpine) variants. On an unsupported platform, or when the optional package was skipped with `--no-optional`/`--omit=optional`, requiring the package raises an explicit error naming the expected platform package; there is no silent fallback.

The existing **0.3.1 registry release** still uses the earlier universal package with five native binaries. The thin layout and expanded eight-target matrix described here will reach registry installations in a subsequent release; the public package name stays `@glrmask/node`.

Package publication is a separate release action. The source and CI commands below produce installable tarballs; they do not publish anything to npm. Install the thin-root tarball together with your platform tarball:

```sh
npm install /absolute/path/glrmask-node-0.3.1.tgz /absolute/path/glrmask-node-win32-x64-msvc-0.3.1.tgz
```

On Linux with musl libc, install the `linux-x64-musl` or `linux-arm64-musl` platform package. The loader inspects `process.report.getReport()` to detect musl rather than checking for glibc symbols.

### Build from source

You need the **complete GLRMask repository**, a current stable Rust toolchain, and the platform's native compiler/linker. On Windows, use Visual Studio Developer PowerShell with the MSVC build tools. Run:

```sh
npm ci --ignore-scripts --prefix bindings/node
npm run build --prefix bindings/node
npm test --prefix bindings/node
```

`npm run build` runs [scripts/build.cjs](scripts/build.cjs), compiling `crates/glrmask-node` and placing `glrmask.<tag>.node` next to the package's loader. Local source binaries always take priority over installed platform packages so the checkout is immediately runnable and testable.

For an owned incremental development cache, set `CARGO_TARGET_DIR` and use `npm run build:dev`. A specific Rust target can be passed with `npm run build -- --target aarch64-unknown-linux-gnu`; the corresponding Rust target, native compiler and linker must be installed. Cross-compilation is not inferred from an artifact filename.

## A decoding loop

```js
const { Vocab, Grammar } = require('@glrmask/node');

async function main() {
  const vocab = new Vocab([
    [0, Buffer.from('hello')],
    [1, Buffer.from(' ')],
    [2, Buffer.from('world')],
  ]);
  const grammar = Grammar.fromEbnf('start ::= "hello" " " "world"');
  const constraint = await grammar.compile(vocab);
  const state = constraint.start();
  const mask = new Uint32Array(constraint.maskLength);

  for (const id of [0, 1, 2]) {
    state.fillMask(mask);
    if ((mask[Math.floor(id / 32)] & (1 << (id % 32))) === 0)
      throw new Error(`Token ${id} is not permitted`);
    state.commitToken(id);
  }

  console.log('Accepted:', state.isAccepting());
}

main().catch(console.error);
```

The complete runnable example is in [examples/decode.mjs](examples/decode.mjs). In a serving loop, allocate the `Uint32Array` mask buffer once and reuse `state.fillMask(buffer)`. `state.mask()` allocates a new array each call.

## Vocabulary and packed masks

`Vocab` entries are `[id, bytes]` pairs where `id` is an integer in `0..4294967295` and `bytes` is a `Uint8Array` or Node `Buffer`. Partial UTF-8 bytes and NUL bytes are permitted. Do not normalize or decode/re-encode bytes.

```js
const vocab = new Vocab([
  [0, Buffer.from([0xc3])],
  [1, Buffer.from([0xa9])],
  [31, Buffer.from('é')],
], {
  exactOnlyIds: [99], // Control ID without byte spelling.
});
```

Duplicate IDs, duplicate exact-only IDs, and overlap between the two sets are rejected. `vocab.size` counts byte-backed entries, not exact-only controls.

Masks are packed **original-ID** bitsets. Bit `id % 32` of `Uint32Array` element `Math.floor(id / 32)` is set if token `id` is allowed. The length is `constraint.maskLength`. Extra elements in caller buffers passed to `fillMask` are zeroed. Undersized buffers throw `GLRMASK_BUFFER_TOO_SMALL`.

## Grammar sources and build choices

| Factory | Input |
| --- | --- |
| `Grammar.fromEbnf(source)` | EBNF text |
| `Grammar.fromGlrm(source)` | Native GLRM text |
| `Grammar.fromJsonSchema(schema)` | JSON Schema object or JSON text |
| `Grammar.fromLark(source, options)` | Lark text with optional profile/tokens |

These constructors copy and own their source text. Frontend capabilities are identical to the Rust core. See the [GLRM reference](../../docs/glrm-format.md) and [Lark dialect](../../docs/lark-dialect.md).

```js
const grammar = Grammar.fromLark('start: <|begin|>', {
  profile: LarkProfile.Llm,
  tokenNames: { begin: 99 },
});
```

## Asynchronous compilation and options

`compile`, `precompile`, `save` and `load` return promises and run on native worker threads. Their `Sync` counterparts block the calling thread. The mask/commit/state-query hot path is synchronous and native.

```js
const { Module, MaskStrategy, BoundaryMaskStrategy } = require('@glrmask/node');
const module = new Module(grammar, {
  maskStrategy: MaskStrategy.ParserDwa,
  boundaryMaskStrategy: BoundaryMaskStrategy.FastBuild,
});
const constraint = await module.compile(vocab, { endTokens: [99] });
```

| MaskStrategy | String value | Intent |
| --- | --- | --- |
| `MaskStrategy.Auto` | `auto` | Core chooses default trade-off. |
| `MaskStrategy.ParserDwa` | `parser-dwa` | Static parser-DWA masks. Alias: `FastRuntime`. |
| `MaskStrategy.TokenTrie` | `token-trie` | Dynamic token-trie walk. Alias: `FastBuild`. |
| `MaskStrategy.TokenClassTrie` | `token-class-trie` | Token-trie walk over equivalence classes. |

String literals matching these names are also accepted. `boundaryMaskStrategy` accepts `BoundaryMaskStrategy.FastBuild` (`fast-build`) and `BoundaryMaskStrategy.FastRuntime` (`fast-runtime`).

All Rust inputs are snapshotted **before** an async call returns its promise. Mutating or disposing the source objects afterward does not affect scheduled work.

## Mutable composition and precompilation

`Module.bind(name, value)` and `Module.bind({name: value, ...})` mutate atomically. If any binding fails, the receiving module is left unchanged. `copy()` produces an independent snapshot. Attaching a child also snapshots that occurrence.

```js
const composedVocab = new Vocab([
  [0, Buffer.from('[')], [1, Buffer.from('hello')],
  [2, Buffer.from(']')], [3, Buffer.from('[hello]')],
]);
const parent = new Module(Grammar.fromGlrm(
  'glrm 1; start root; extern grammar payload; nt root = "[" payload "]";'
));
const child = Grammar.fromEbnf('start ::= "hello"');
parent.bind('payload', child);
const runnable = await parent.compile(composedVocab);
```

For reusable compiled machinery, precompile the parent and child separately:

```js
const parentModule = new Module(Grammar.fromGlrm(
  'glrm 1; start root; extern grammar payload; nt root = "[" payload "]";'
));
await parentModule.precompile(composedVocab);
const childModule = new Module(child);
await childModule.precompile(composedVocab);
parentModule.bind('payload', childModule);
const runnable = await parentModule.compile(composedVocab);
```

Bindings accept `Module`, `Grammar`, `Expr`, `Token` or `ExactTokens`. Open child modules with unbound externs are rejected at bind time. Qualified descendant paths using `::` are rejected.

The `externs` getter returns an `ExternsMap`:

```ts
interface Extern {
  readonly name: string;
  readonly kind: 'token' | 'grammar';
  readonly span: {
    readonly startByte: number;
    readonly endByte: number;
    readonly line: number;
    readonly column: number;
  } | null;
}
```

`externs.unbound` returns an `ExternsMap` containing only unresolved externs.

## Exact-token sets and ranges

`new Token(id)` describes a raw original-ID event. `vocab.tokens(ids)` creates `ExactTokens` bound to a particular vocabulary.

```js
const exact = vocab.tokens([31]);
const control = new Module(Grammar.fromGlrm(
  'glrm 1; start root; extern token CONTROL; nt root = CONTROL;'
));
control.bind('CONTROL', exact);
```

`vocab.tokenRanges([{start, end}, ...])` uses **half-open ranges**. `end` may be 4,294,967,296 to include ID 4,294,967,295. Overlapping or adjacent ranges normalize without expanding IDs.

## State, termination and markers

An immutable `Constraint` can start many independent states. `state.clone()` copies the exact parser snapshot without sharing mutable sequence state.

`isAccepting()` means the current prefix is a complete grammar match. Configuring root `endTokens` allows stopping when an end token is sampled and committed.

`isRejected()` reports an irrecoverably rejected prefix. `forcedTokens()` performs read-only lookahead. `constraint.markerIds()` returns marker IDs; `state.markerEvents()` returns observations as `{ id: number, byteOffset: bigint }`.

## Save and load

```js
const bytes = await constraint.save();
const restored = await Constraint.load(bytes);

const external = await constraint.save({ externalVocab: true });
const rebound = await Constraint.load(external, vocab);

const moduleBytes = await parentModule.save();
const module = await Module.load(moduleBytes, vocab);
```

`saveSync`/`loadSync` provide synchronous execution. Artifacts can be exchanged between compatible builds of Rust, Python, C++ and Node.

## Errors and resource management

Wrong JavaScript types raise `TypeError`; invalid numeric IDs raise `RangeError`. Native failures are `GlrMaskError` with a `code`:

| Code | Meaning |
| --- | --- |
| `GLRMASK_INVALID_ARGUMENT` | Invalid native argument or unsupported buffer backing. |
| `GLRMASK_ENGINE_ERROR` | Core grammar/state/identity/artifact error. |
| `GLRMASK_PANIC` | An unwinding internal panic was contained. Discard affected state. |
| `GLRMASK_BUFFER_TOO_SMALL` | Caller mask capacity is insufficient. |
| `GLRMASK_CLOSED` | A disposed object was used. |

All native-owning classes support `dispose()` and `[Symbol.dispose]()`. TypeScript's `using` syntax can release them deterministically.

## API index

| Type | Public surface |
| --- | --- |
| `Vocab` | Constructor, `size`, `tokens`, `tokenRanges`, `copy`, disposal. |
| `Token` | Validated constructor and immutable `id`. |
| `ExactTokens` | `size`, `ids`, `ranges`, `copy`, disposal. |
| `Grammar` | Factories (`fromEbnf`, `fromGlrm`, `fromJsonSchema`, `fromLark`), `copy`, async/sync `compile`, disposal. |
| `Module` | Constructor, `maskStrategy`, `boundaryMaskStrategy`, `externs`, `copy`, `bind`, async/sync `precompile`, async/sync `compile`, async/sync `save`, static async/sync `load`, disposal. |
| `Expr` | `choice`, `sequence`, `literal`, `copy`, disposal. |
| `Constraint` | `maskLength`, `copy`, `start`, `markerIds`, async/sync `save`, static `load`, disposal. |
| `ConstraintState` | `maskLength`, `copy`, `mask`, `fillMask`, `commitToken`, `commitBytes`, `isAccepting`, `isRejected`, `forcedTokens`, `markerEvents`, disposal. |
| Options/errors | `MaskStrategy`, `BoundaryMaskStrategy`, `LarkProfile`, `GlrMaskError`, `version`. |

Exact signatures are in [index.d.cts](index.d.cts). Runnable examples are in [examples](examples/).
