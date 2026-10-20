# JavaScript and TypeScript

`@glrmask/node` is a native Node.js binding to GLRMask's Rust compiler and decoding engine. It supports CommonJS and ECMAScript modules, includes TypeScript declarations, and uses Node-API rather than embedding Python or launching a parser subprocess.

This is **Node.js support, not browser WebAssembly**. A browser, Bun or Deno target is not implied by the JavaScript interface. Bring your inference backend's exact token-ID-to-byte mapping; the package does not run a model, download a tokenizer, apply GPU logits, or choose a sampling policy.

## Owned assemblies and expressions

The Module/Expr interfaces in this section are unreleased and require a matching
source build. Published 0.3.1 packages retain the PreparedGrammar lifecycle.

`Grammar` retains source. `Module` owns mutable assembly snapshots, local options
and any reusable work. `Expr.choice(iterable)` and `Expr.sequence(iterable)` take
mixed Grammar, Module and Expr snapshots. Each leaf keeps its lexical scope;
paths use `leaf0`, `leaf1`, and so on, without operator path segments.

```js
const {Vocab,Grammar,Module,ModuleOptions,Expr,Optimization} = require('@glrmask/node');
const vocab = new Vocab([[0,Buffer.from('xay')],[1,Buffer.from('xby')]]);
const child = new Module(Grammar.fromEbnf('start ::= "a"'),
  new ModuleOptions({optimization:Optimization.Balanced}));
await child.precompile(vocab);
const parent = Grammar.fromGlrm(
  'glrm 1; start start; extern grammar child; nt start = "x" child "y";').module();
parent.bind('child',child); // Also accepts cold Grammar, Expr, Token and ExactTokens.
const restored = await Module.load(await parent.save(),vocab);
const constraint = await restored.compile(vocab);
const expression = Expr.choice([child,Grammar.fromEbnf('start ::= "b"')]);
```

`bind` mutates atomically and returns undefined; `precompile` returns
`Promise<void>`. Explicit `precompileSync`, `compileSync`, `saveSync` and
`loadSync` methods block the caller. Async compile/save capture snapshots at
scheduling, retaining their native owners after JS disposal. Async precompile
checks the receiver revision at completion: concurrent mutation rejects the
promise and preserves newer bindings; retry against the current snapshot.

`options` is a frozen ModuleOptions snapshot. `withOptions` returns an independent
configured copy; `setOptions` changes local settings atomically. Retained local
work prevents incompatible option changes. Children retain their own settings;
final compile accepts only `{endTokens}`. `unbound` and `externs` return frozen
entries with name, path, kind and sourceSpan. Copies share immutable work and
retain independent mutable bindings.

Loading never compiles; the optional vocabulary validates exact saved target
identity. Empty sequence accepts epsilon; empty choice fails when materialized.
`Grammar.fromExpr` and source Grammar.bind reject configured or prepared leaves.
Standalone FastBuild modules retain LR work; explicit composition edges require
native components and reject unsupported LR combinations without conversion.
The older PreparedGrammar API remains available for migration; new assembly code
uses Module.

## Installation

Use Node.js 22 or newer. This source revision prepares a thin root: `@glrmask/node`
itself contains only JavaScript, TypeScript declarations and a loader, while
the native binary for your OS/architecture/libc arrives automatically through
an `optionalDependencies` entry (`@glrmask/node-<platform>`). No Rust
compiler, C++ compiler, Python runtime or post-install download is needed on
a supported platform. Prebuilt release binaries cover Windows x64 and arm64
(MSVC), macOS x64 and arm64 (macOS 11+), and Linux x64 and arm64 in both
glibc (manylinux_2_28, minimum glibc 2.28) and musl (Alpine) variants. On an
unsupported platform, or when the optional package was skipped with
`--no-optional`/`--omit=optional`, requiring the package raises an explicit
error naming the expected platform package; there is no silent fallback.

The existing **0.3.1 registry release** still uses the earlier universal package
with five native binaries. The thin layout and expanded eight-target matrix
described here will reach registry installations in a subsequent release; the
public package name stays `@glrmask/node`.

Package publication is a separate release action. The source and CI commands below produce installable tarballs; they do not publish anything to npm. Install the thin-root tarball together with your platform tarball:

```sh
npm install /absolute/path/glrmask-node-0.3.1.tgz /absolute/path/glrmask-node-win32-x64-msvc-0.3.1.tgz
```

A registry install fetches the matching platform package automatically. Each
platform package carries `os`/`cpu`/`libc` routing, an exact version pin and
a checksummed manifest; a missing or incompatible native binary causes an
explicit error, never a silent Python, subprocess or WebAssembly fallback.

### Build from a source checkout

You need the **complete GLRMask repository**, a current stable Rust toolchain, and the platform's native compiler/linker. On Windows, use Visual Studio Developer PowerShell with the MSVC build tools. Run:

```sh
cd glrmask/bindings/node
npm ci --omit=optional
npm run build
npm test
npm run typecheck
node scripts/test-platform-artifact.cjs
```

The final command stages and packs the thin root and the actual host platform,
then tests both together in isolated temporary installations. For local
development, import this checkout after `npm run build`; the loader uses that
local native binary. For distribution, use the root and matching platform
tarballs from the CI artifact set. A thin-root `.tgz` alone intentionally does
not contain the binary. `npm ci` installs development tools; the package has no
JavaScript runtime dependencies other than its native optional package. Running
`npm install` against an unbuilt checkout is not a replacement for the build.

For an owned incremental development cache, set `CARGO_TARGET_DIR` and use `npm run build:dev`. A specific Rust target can be passed with `npm run build -- --target aarch64-unknown-linux-gnu`; the corresponding Rust target, native compiler and linker must be installed. Cross-compilation is not inferred from an artifact filename.

Packing refuses missing, mismatched or non-release artifacts. The thin root
ships no binaries: `prepack` checks that no `.node` file is packed, that
`optionalDependencies` pins all eight platform packages at the root version,
and that a staged `platform-manifest.json`, when present, records every
checksummed target. `GLRMASK_ALLOW_DEV_ARTIFACTS=1` is available for explicit
local development installation tests, not release qualification. Production
builds use portable target defaults rather than `target-cpu=native`. All
eight release targets also build from a source checkout with
`npm run build -- --target <rust-target>`.

## Quick start

Save this as an `.mjs` file:

```js
import { Vocab, Grammar } from '@glrmask/node';

const vocab = new Vocab([
  [0, Buffer.from('hello')],
  [1, Buffer.from(' ')],
  [2, Buffer.from('world')],
]);
const grammar = Grammar.fromEbnf('start ::= "hello" " " "world"');
const constraint = await grammar.compile(vocab);
const state = constraint.start();
const mask = new Uint32Array(constraint.maskLength);

// Replace this fixed sequence with your model's allowed-token sampler.
for (const id of [0, 1, 2]) {
  state.fillMask(mask);
  const permitted = (mask[Math.floor(id / 32)] & (1 << (id % 32))) !== 0;
  if (!permitted) throw new Error(`Token ${id} is not allowed`);
  state.commitToken(id);
}
console.log(state.isAccepting()); // true
state.dispose();
constraint.dispose();
grammar.dispose();
vocab.dispose();
```

CommonJS uses the same classes:

```js
const { Vocab, Grammar } = require('@glrmask/node');
```

The ESM and CommonJS exports share constructor identity, including their TypeScript types. Use named imports or namespace imports. The package does not define an ESM default export.

## Vocabularies and exact bytes

The constructor is `new Vocab(entries, { exactOnlyIds? })`. `entries` is an iterable of `[originalId, Uint8Array]` pairs; a `Map` is also suitable. `Buffer` is a `Uint8Array` and is accepted. Inputs are copied into owned Rust data.

```js
const vocab = new Vocab([
  [0, Uint8Array.of(0xc3)],
  [1, Uint8Array.of(0xa9)],
  [31, Buffer.from('é')],
], { exactOnlyIds: [99] });
```

Individual token pieces can be partial UTF-8 or contain NUL. Tokenizer bytes must not be normalized or reconstructed from a display string. Two different token IDs can have the same byte spelling. Duplicate IDs, duplicate exact-only IDs and overlap between byte-backed and exact-only domains are rejected.

`vocab.size` counts byte-backed entries, not exact-only controls. IDs must be JavaScript `number` values representing integers from 0 through 4,294,967,295. Negative values, fractions, infinities, `NaN`, strings and `bigint` IDs are rejected, not truncated or coerced.

SharedArrayBuffer-backed and detached input arrays are rejected before Rust borrows their memory. Async operations copy byte inputs before scheduling: changing or transferring the original input after the call does not change an in-flight load.

## Packed masks

Masks are `Uint32Array` values in the model's **original token-ID space**. Word `Math.floor(id / 32)`, bit `id % 32`, tells you whether a token is allowed. `constraint.maskLength` and `state.maskLength` count words, not tokens or bytes.

JavaScript bitwise operators use signed 32-bit intermediates. Test a bit with `!== 0`, not `> 0`, because bit 31 can produce a negative intermediate value. Reading the underlying `Uint32Array` still gives unsigned words.

`state.mask()` allocates a new packed array. `state.fillMask(output)` writes directly into a caller-owned `Uint32Array`, including a correctly aligned subarray, and zeroes surplus words. An undersized buffer fails without changing it. `Int32Array`, `DataView`, detached storage and SharedArrayBuffer-backed output are not accepted. Do not transfer or reuse an array concurrently with a native call.

Allocate the reusable mask once per decoding loop. The fill operation does not create a temporary bridge mask; it does not promise that the engine or JavaScript runtime never allocates. Output remains dense in the original-ID space: sparse or extremely large IDs can require a large mask even when an internal token set is compact.

## Grammar constructors

| Method | Meaning |
| --- | --- |
| `Grammar.fromEbnf(source)` | EBNF text. |
| `Grammar.fromGlrm(source)` | Native GLRM text. |
| `Grammar.fromJsonSchema(schema)` | JSON text, or a JSON-serializable value. |
| `Grammar.fromLark(source, options?)` | Lark text, with explicit optional LLM extensions. |
| `Grammar.exact(tokens)` | One event selected from a vocabulary-qualified exact-token set. |

These are descriptions; parsing/compilation errors may arise when inspecting requirements, binding, preparing or compiling. Frontend semantics and limitations are the same as the Rust core. See the [GLRM reference](../../docs/glrm-format.md) and [Lark dialect](../../docs/lark-dialect.md). This package does not turn the JSON Schema importer into a complete standards validator or expose parse trees/captures.

Lark defaults to the recognition profile. Named special tokens use only an explicit caller map:

```js
import { Grammar, LarkProfile } from '@glrmask/node';
const grammar = Grammar.fromLark('start: <|begin|>', {
  profile: LarkProfile.Llm,
  tokenNames: { begin: 99 },
});
```

A mapping value can be one ID or an iterable of IDs. The binding does not guess a token by model name or decoded spelling.

## Asynchronous compilation and options

`compile`, `prepare`, `link`, `save` and `load` return promises and run their heavy work on native workers. Their `Sync` counterparts intentionally block the calling JavaScript thread. The mask/commit/state-query hot path is synchronous and native.

```js
import { Optimization, BoundaryTriggerDetail } from '@glrmask/node';
const constraint = await grammar.compile(vocab, {
  optimization: Optimization.FastRuntime,
  endTokens: [99],
  boundaryTrigger: BoundaryTriggerDetail.Tokens,
});
```

| Optimization | String value | Intent |
| --- | --- | --- |
| `Optimization.Auto` | `auto` | Core default trade-off. |
| `Optimization.FastBuild` | `fast-build` | Ordinary Dynamic compiler/runtime. |
| `Optimization.Balanced` | `balanced` | Balanced/O2 path. |
| `Optimization.FastRuntime` | `fast-runtime` | Prefer lower per-token runtime cost. |

String literals with those exact values are also accepted. Unknown option names and values are rejected. Boundary detail is `none`, `tokens` or `exact`; omission preserves a prepared request, whereas explicit `none` clears it at final link. Unsupported engine combinations return errors, not a hidden fallback.

Final optimization and end-token policy belong to `compile`/`link`. `prepare(vocab, options)` accepts only `boundaryTrigger`. It does not silently retain final options on each source node. The interface follows the current core API rather than inventing a separate cross-language compilation policy.

All Rust inputs are snapshotted **before** an async call returns its promise. You can subsequently mutate or dispose the original source/prepared/vocabulary objects without changing that job. Async work does not retain a borrowed JavaScript buffer or call JavaScript from the worker. Choose appropriate build concurrency for your server; native compilation still consumes CPU, memory and worker capacity. Prepare/reuse common components rather than repeatedly compiling on a per-token path.

## Mutable composition and preparation

`Grammar.bind(name, value)` mutates the receiver and returns `undefined`. It is transactional: an invalid binding leaves the receiver unchanged. `clone()` produces an independent snapshot. Attaching a child also snapshots that occurrence, so later edits to the original child do not rewrite its parent.

```js
const composedVocab = new Vocab([
  [0, Buffer.from('[')], [1, Buffer.from('hello')],
  [2, Buffer.from(']')], [3, Buffer.from('[hello]')],
]);
const parent = Grammar.fromGlrm(
  'glrm 1; start root; extern grammar payload; nt root = "[" payload "]";'
);
const child = Grammar.fromEbnf('start ::= "hello"');
parent.bind('payload', child);
const runnable = await parent.compile(composedVocab);
```

For reusable compiled machinery, prepare an open parent and a child separately:

```js
const parentModule = await Grammar.fromGlrm(
  'glrm 1; start root; extern grammar payload; nt root = "[" payload "]";'
).prepare(composedVocab);
const childModule = await child.prepare(composedVocab);
parentModule.bind('payload', childModule);
const runnable = await parentModule.link();
```

A `Grammar` binding accepts `Grammar`, `Token` or `ExactTokens`. A `PreparedGrammar` binding accepts `PreparedGrammar`, `Token` or `ExactTokens`, not a source grammar or runnable `Constraint`. Preparing can retain open slots; compilation/final linking must produce a closed runnable root.

The `requirements` getter returns current, owned records:

```ts
interface Requirement {
  readonly name: string;
  readonly path: readonly string[];
  readonly kind: 'token' | 'grammar';
  readonly sourceSpan: {
    readonly startByte: number;
    readonly endByte: number;
    readonly line: number;
    readonly column: number;
  } | null;
}
```

Use actual requirement names such as `payload::TOOL` for descendant binding. Spans use a half-open byte range and one-based line/Unicode-scalar column. A missing span is `null`, not a fabricated location. Requirement records/path/span snapshots are frozen; mutating an output array does not mutate the native graph.

## Exact-token sets and ranges

`new Token(id)` describes a raw original-ID event. `vocab.tokens(ids)` creates `ExactTokens` bound to a particular vocabulary. Identical token bytes do not make two IDs interchangeable for an exact event.

```js
const exact = vocab.tokens([31]);
const grammar = Grammar.exact(exact);
const control = Grammar.fromGlrm(
  'glrm 1; start root; extern token CONTROL; nt root = CONTROL;'
);
control.bind('CONTROL', new Token(99));
```

`vocab.tokenRanges([{start, end}, ...])` uses **half-open ranges**, unlike the inclusive C/C++ endpoints. `end` may be 4,294,967,296 to include ID 4,294,967,295. Empty/reversed individual intervals are rejected. Overlapping or adjacent ranges normalize without expanding IDs. Each selected ID must belong to the vocabulary's byte-backed or exact-only domain.

`exact.ranges` stays compact; `exact.ids` intentionally materializes a `Uint32Array`. `exact.size` is cardinality. An empty set denotes an empty event language, not epsilon. Matching a set consumes one chosen token event, not every ID in a sequence. Empty tokens and special events must be committed as token IDs, not simulated with `commitBytes`.

## State, termination and markers

An immutable `Constraint` can start many independent states. `state.clone()` copies the exact parser snapshot without sharing mutable sequence state. Its native owner remains alive after `constraint.dispose()` or garbage collection. Materialized mask/ID arrays own their storage and also remain usable after the source handle is disposed.

`isAccepting()` means the current prefix is a complete grammar match. It is not synonymous with “finished”: continuation can still be legal. Applications may stop at first acceptance, or configure root `endTokens` and stop when one is sampled and committed. An allowed end token leaves the state accepting and non-rejected, empties the next mask and disallows further commits. Root end-token policy is not inherited by embedded children.

`isRejected()` reports an irrecoverably rejected prefix. Grammar commit failures follow the core's state semantics; they are not a general rollback API. Clone before speculative commits. Invalid JavaScript argument/buffer validation does not commit anything.

`forcedTokens()` performs read-only, bounded lookahead and returns a `Uint32Array`. It does not advance the state; the core can report a forcing error instead of looping indefinitely.

`constraint.markerIds()` returns marker IDs. `state.markerEvents()` returns the initial or latest commit's observations as `{ id: number, byteOffset: bigint }`. Byte offsets use `bigint` so long-running streams do not silently lose integer precision. Markers are possible recognition observations, including ambiguous branches, not final parse captures or a parse-tree API. Cloning preserves the current batch.

## Save and load

```js
const bytes = await constraint.save();
const restored = await Constraint.load(bytes);

const external = await constraint.save({ externalVocab: true });
const rebound = await Constraint.load(external, vocab);

const moduleBytes = await parentModule.save();
const module = await PreparedGrammar.load(moduleBytes, vocab);
```

`saveSync`/`loadSync` provide the same formats with blocking execution. The returned binary value is a `Uint8Array`-compatible Node buffer. Load copies input bytes before scheduling. An omitted external vocabulary is an error when the artifact requires it, and a supplied vocabulary must match exactly; equal vocabulary size is insufficient.

These are the Rust core's artifact formats. Artifacts can be exchanged between compatible builds of Rust, Python, C++ and Node. They are not promised to remain compatible across releases, architectures or unrelated builds. Keep source/configuration for long-term regeneration, and apply the core loader's limitations to untrusted artifacts. The binding does not claim that an internal artifact format is a security-hardened portable interchange standard.

## Errors and resource management

Wrong JavaScript types and option spellings raise `TypeError`; invalid numeric IDs raise `RangeError`. Native failures are `GlrMaskError` with a `code`:

| Code | Meaning |
| --- | --- |
| `GLRMASK_INVALID_ARGUMENT` | Invalid native argument or unsupported buffer backing. |
| `GLRMASK_ENGINE_ERROR` | Core grammar/state/identity/artifact error. |
| `GLRMASK_PANIC` | An unwinding internal panic was contained. Discard affected state. |
| `GLRMASK_BUFFER_TOO_SMALL` | Caller mask capacity is insufficient. |
| `GLRMASK_CLOSED` | A disposed object was used. |

Async methods reject their promises with these errors; even validation before scheduling becomes a rejection through the public async method. No Rust exception unwinds into JavaScript. Process aborts, allocation failures and stack overflows cannot be converted into ordinary JS errors by the panic boundary.

All native-owning classes support `dispose()` and `[Symbol.dispose]()`, so TypeScript's `using` syntax can release them deterministically. Disposal is idempotent. Normal garbage collection is also supported; explicit disposal is useful for large compiled objects but never invalidates retained states or queued native snapshots.

TypeScript 5.2 or newer is required for the disposable declarations. Both module systems re-export one canonical declaration surface so values are not assigned incompatible private brands merely because one file uses `import` and another uses `require`.

## API index

| Type | Public surface |
| --- | --- |
| `Vocab` | Constructor, `size`, `tokens`, `tokenRanges`, `clone`, disposal. |
| `Token` | Validated constructor and immutable `id`. |
| `ExactTokens` | `size`, `ids`, `ranges`, `clone`, disposal. |
| `Grammar` | Five factories, `requirements`, `bind`, `clone`, async/sync `prepare` and `compile`, disposal. |
| `PreparedGrammar` | `requirements`, `bind`, `clone`, async/sync `link`, `save`, static `load`, disposal. |
| `Constraint` | `maskLength`, `clone`, `start`, `markerIds`, async/sync `save`, static `load`, disposal. |
| `ConstraintState` | `maskLength`, `clone`, `mask`, `fillMask`, `commitToken`, `commitBytes`, `isAccepting`, `isRejected`, `forcedTokens`, `markerEvents`, disposal. |
| Options/errors | `Optimization`, `BoundaryTriggerDetail`, `LarkProfile`, `GlrMaskError`, `version`. |

Exact signatures are in [index.d.cts](index.d.cts); ESM declarations re-export that same surface. Runnable examples are in [examples](examples/). The private `Native*` addon classes and `.node` file are implementation details, not an additional supported public API.

## Validation and packaging

The native test suite covers all four modes, packed/high-bit/sparse masks, byte pieces, exact events/ranges, end tokens, source/prepared snapshots, artifacts, markers, forcing, disposal/GC, async input ownership, event-loop progress and invalid buffers. It also calls the private native layer directly to check that numeric truncation, shared backing and detached buffers are rejected there, not just by a JavaScript wrapper.

The repository additionally tests C++/Node mask equivalence and two-way artifact exchange, TypeScript consumers in both module modes, and installed npm tarballs (thin root plus platform package) outside the source tree, including offline installs and the missing/wrong-platform error paths. The native CI workflow builds the eight-target platform matrix, verifies checksums, stages per-platform packages plus the thin root, and tests installation without triggering a Rust build.
