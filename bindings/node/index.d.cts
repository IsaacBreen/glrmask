/// <reference lib="esnext.disposable" />
/** Native GLRMask bindings for Node.js. Masks use packed original-token-ID u32 words. */
export declare const version: string;

/** Mask computation strategy, trading build time against per-token latency. */
export declare const MaskStrategy: Readonly<{
  /** Automatic selection: `ParserDwa`, or the retained `TokenClassTrie` when one is available. */
  Auto: 'auto';
  /** Static masks: slowest build, fastest per-token masks. */
  ParserDwa: 'parser-dwa';
  /** Trie walk per mask: fastest build. */
  TokenTrie: 'token-trie';
  /** Trie walk over token equivalence classes. */
  TokenClassTrie: 'token-class-trie';
  /** Alias for `TokenTrie`. */
  FastBuild: 'token-trie';
  /** Alias for `ParserDwa`. */
  FastRuntime: 'parser-dwa';
}>;
/** A mask computation strategy value. */
export type MaskStrategy = typeof MaskStrategy[keyof typeof MaskStrategy];

/** Boundary parser selection for crossings between composed grammars. */
export declare const BoundaryMaskStrategy: Readonly<{
  /** Automatic selection: `ParserDwa`, falling back to `TokenTrie` when the static boundary parser declines on budget. */
  Auto: 'auto';
  /** Static boundary parser; faster crossings. An over-budget build raises `GlrMaskError`. */
  ParserDwa: 'parser-dwa';
  /** Segmented dynamic boundary token walk; faster build. */
  TokenTrie: 'token-trie';
  /** Alias for `TokenTrie`. */
  FastBuild: 'token-trie';
  /** Alias for `ParserDwa`. */
  FastRuntime: 'parser-dwa';
}>;
/** A boundary parser strategy; the legacy `fast-runtime`/`fast-build` strings are also accepted as input. */
export type BoundaryMaskStrategy =
  | typeof BoundaryMaskStrategy[keyof typeof BoundaryMaskStrategy]
  | 'fast-runtime'
  | 'fast-build';

/** Error codes carried by {@link GlrMaskError}. */
export type ErrorCode = 'GLRMASK_INVALID_ARGUMENT' | 'GLRMASK_ENGINE_ERROR' | 'GLRMASK_PANIC' | 'GLRMASK_BUFFER_TOO_SMALL' | 'GLRMASK_CLOSED';
/** A native GLRMask failure. */
export declare class GlrMaskError extends Error {
  /** Stable error code. */
  readonly code: ErrorCode;
  /** Creates an error; the bindings raise these rather than callers constructing them. */
  constructor(code: ErrorCode, message: string, options?: ErrorOptions);
}

/** A half-open token-ID interval. end can be 4294967296 to include token ID 4294967295. */
export interface TokenRange {
  /** First included token ID. */
  readonly start: number;
  /** Exclusive end token ID. */
  readonly end: number;
}
/** A declared extern slot on a Grammar or Module. */
export interface Extern {
  /** The declared name. */
  readonly name: string;
  /** The declared kind: token or grammar. */
  readonly kind: 'token' | 'grammar';
}
/** A read-only map of declared externs, in declaration order. */
export interface Externs extends ReadonlyMap<string, Extern> {
  /** The same view restricted to unresolved externs. */
  readonly unbound: Externs;
}

/** A possible boundary observation, not a selected parse node or capture. */
export interface MarkerEvent {
  /** The marker that was observed. */
  readonly id: number;
  /** Absolute byte offset in the stream supplied to the state. */
  readonly byteOffset: bigint;
}
/** Vocabulary construction options. */
export interface VocabOptions {
  /** Token IDs valid for exact-token bindings that have no byte spelling. */
  exactOnlyIds?: Iterable<number>;
}
/** Constraint serialization options. */
export interface SaveOptions {
  /** Omit the vocabulary; load with the exact original token-ID/byte mapping. */
  externalVocab?: boolean;
}
/** An original model token ID without vocabulary identity. Only ID membership is checked against the target vocabulary, so the same ID can refer to different bytes; use `vocab.tokens([id])` for a vocabulary-bound singleton. */
export declare class Token {
  /** The original model token ID. */
  readonly id: number;
  /** Creates a token event for `id`; membership is checked at compile. */
  constructor(id: number);
}

/** The model vocabulary a constraint is compiled against. Entries map model token IDs to exact token bytes; token IDs may be sparse. */
export declare class Vocab implements Disposable {
  private readonly __vocabBrand: void;
  /** Creates a vocabulary from `[token_id, bytes]` entries. */
  constructor(entries: Iterable<readonly [number, Uint8Array]>, options?: VocabOptions);
  /** Number of byte-backed entries; exact-only IDs are not counted. */
  readonly size: number;
  /** Creates a vocabulary-bound token event matching one of `ids`; unknown or duplicate IDs raise `GlrMaskError` and empty input matches nothing. */
  tokens(ids: Iterable<number>): ExactTokens;
  /** Creates a vocabulary-bound token event matching the union of half-open `ranges`; overlapping or adjacent ranges merge, unknown IDs raise `GlrMaskError`, and empty input matches nothing. */
  tokenRanges(ranges: Iterable<TokenRange>): ExactTokens;
  /** An independent handle. */
  copy(): Vocab;
  /** An independent handle. */
  clone(): Vocab;
  /** Releases the native handle. */
  dispose(): void;
  /** Releases the native handle when leaving a `using` block. */
  [Symbol.dispose](): void;
}

/** One token event chosen from a vocabulary-bound set. An empty set matches nothing. */
export declare class ExactTokens implements Disposable {
  private constructor();
  private readonly __exactTokensBrand: void;
  /** Number of allowed token IDs, not the number of ranges. */
  readonly size: number;
  /** The exact model token IDs, sorted in ascending order. */
  readonly ids: Uint32Array;
  /** The normalized, disjoint half-open token-ID ranges. */
  readonly ranges: TokenRange[];
  /** An independent handle. */
  copy(): ExactTokens;
  /** An independent handle. */
  clone(): ExactTokens;
  /** Releases the native handle. */
  dispose(): void;
  /** Releases the native handle when leaving a `using` block. */
  [Symbol.dispose](): void;
}

/** Strategy settings for a precompile or compile. */
export interface StrategyOptions {
  /** The mask computation strategy for the whole compile. */
  maskStrategy?: MaskStrategy;
  /** The boundary parser strategy for the whole compile. */
  boundaryMaskStrategy?: BoundaryMaskStrategy;
}

/** Root compile options. */
export interface GrammarCompileOptions extends StrategyOptions {
  /** Accepted end-of-sequence token IDs, applied only to this root compile and allowed only at acceptance. */
  endTokens?: Iterable<number>;
}

/** The source-language definition, without compilation settings. Source text is parsed lazily, so parse errors surface at compile. */
export declare class Grammar implements Disposable {
  private constructor();
  private readonly __grammarBrand: void;
  /** Creates a grammar from EBNF source text. */
  static fromEbnf(source: string): Grammar;
  /** Creates a grammar from GLRM source; typed extern slots must be bound before compiling a runnable root. */
  static fromGlrm(source: string): Grammar;
  /** Creates a grammar from JSON text or a JSON-serializable document; `format` is an annotation only. */
  static fromJsonSchema(schema: unknown): Grammar;
  /** Creates a grammar from Lark source; named token atoms declare token externs. */
  static fromLark(source: string): Grammar;
  /** A read-only map of the grammar's declared externs, in declaration order. */
  readonly externs: Externs;
  /** An independent handle. */
  copy(): Grammar;
  /** Retains vocabulary-specific compilation artifacts; equivalent to `new Module(this).precompile(vocab, ...)`. */
  precompile(vocab: Vocab, options?: StrategyOptions | CompileOptions): Promise<Module>;
  /** Retains vocabulary-specific compilation artifacts; equivalent to `new Module(this).precompileSync(vocab, ...)`. */
  precompileSync(vocab: Vocab, options?: StrategyOptions | CompileOptions): Module;
  /** Compiles the grammar against `vocab`; equivalent to `new Module(this).compile(vocab, ...)`. */
  compile(vocab: Vocab, options?: GrammarCompileOptions): Promise<Constraint>;
  /** Compiles the grammar against `vocab`; equivalent to `new Module(this).compileSync(vocab, ...)`. */
  compileSync(vocab: Vocab, options?: GrammarCompileOptions): Constraint;
  /** Releases the native handle. */
  dispose(): void;
  /** Releases the native handle when leaving a `using` block. */
  [Symbol.dispose](): void;
}

/** Root generation options. */
export interface RootSettings extends StrategyOptions {
  /** Accepted end-of-sequence token IDs, allowed only at acceptance and committed before stopping. */
  endTokens?: Iterable<number>;
}
/** Compilation settings for a module assembly. */
export declare class CompileOptions {
  /** Creates compile options; `endTokens` are allowed only at acceptance and must be committed before stopping. */
  constructor(options?: RootSettings);
  /** The accepted end-of-sequence token IDs. */
  readonly endTokens: readonly number[];
  /** The mask computation strategy for the whole compile. */
  readonly maskStrategy: MaskStrategy;
  /** The boundary parser strategy for the whole compile. */
  readonly boundaryMaskStrategy: BoundaryMaskStrategy;
}

/** A value accepted as an `Expr` operand. */
export type ExprPart = Grammar | Module | Expr | Token | ExactTokens;

/** An immutable unordered union or concatenation of independently scoped grammars. */
export declare class Expr implements Disposable {
  private constructor();
  private readonly __exprBrand: void;
  /** Creates a union without ordered-choice priority; an empty choice matches nothing. */
  static choice(...parts: ExprPart[]): Expr;
  /** Creates a concatenation; a model token may cross any number of operand boundaries, and an empty sequence is epsilon. */
  static sequence(...parts: ExprPart[]): Expr;
  /** Matches exact bytes, with no whitespace or ignore inserted; strings are UTF-8 encoded. */
  static literal(value: string | Uint8Array): Expr;
  /** An independent handle. */
  copy(): Expr;
  /** Releases the native handle. */
  dispose(): void;
  /** Releases the native handle when leaving a `using` block. */
  [Symbol.dispose](): void;
}

/** A reusable grammar with bindings and any retained compilation work. May still have unresolved externs. */
export declare class Module implements Disposable {
  private readonly __moduleBrand: void;
  /** Creates a module from a source grammar. */
  constructor(grammar: Grammar);
  /** A read-only map of the module's declared externs, in declaration order. */
  readonly externs: Externs;
  /** An independent snapshot sharing immutable source and retained work. */
  copy(): Module;
  /** Binds the extern `name` to `value`; fails if the extern is undeclared or already bound, and names containing `::` are rejected. */
  bind(name: string, value: ExprPart): void;
  /** Binds several externs atomically; if any binding fails, none is applied. */
  bind(bindings: Record<string, ExprPart>): void;
  /** Retains vocabulary-specific compilation artifacts for this module. */
  precompile(vocab: Vocab, options?: StrategyOptions | CompileOptions): Promise<void>;
  /** Retains vocabulary-specific compilation artifacts for this module. */
  precompileSync(vocab: Vocab, options?: StrategyOptions | CompileOptions): void;
  /** Compiles the module assembly against `vocab`; every extern must be bound. */
  compile(vocab: Vocab, options?: RootSettings | CompileOptions): Promise<Constraint>;
  /** Compiles the module assembly against `vocab`; every extern must be bound. */
  compileSync(vocab: Vocab, options?: RootSettings | CompileOptions): Constraint;
  /** Serializes the module assembly and any retained precompiled artifacts. */
  save(): Promise<Uint8Array>;
  /** Serializes the module assembly and any retained precompiled artifacts. */
  saveSync(): Uint8Array;
  /** Deserializes a module assembly, verifying it against `vocab` when supplied. */
  static load(data: Uint8Array, vocab?: Vocab): Promise<Module>;
  /** Deserializes a module assembly, verifying it against `vocab` when supplied. */
  static loadSync(data: Uint8Array, vocab?: Vocab): Module;
  /** Releases the native handle. */
  dispose(): void;
  /** Releases the native handle when leaving a `using` block. */
  [Symbol.dispose](): void;
}

/** A fully runnable grammar compiled against a specific vocabulary. Immutable and safe to share; each `start()` creates independent mutable state. */
export declare class Constraint implements Disposable {
  private constructor();
  private readonly __constraintBrand: void;
  /** Number of packed `u32` words in a mask. */
  readonly maskLength: number;
  /** An independent handle. */
  copy(): Constraint;
  /** An independent handle. */
  clone(): Constraint;
  /** Creates independent state for one generated sequence. */
  start(): ConstraintState;
  /** Sorted, distinct application marker IDs. */
  markerIds(): Uint32Array;
  /** Serializes the constraint, including its vocabulary and root stopping policy; `externalVocab` outputs an artifact that needs the exact original token-ID/byte mapping to load. */
  save(options?: SaveOptions): Promise<Uint8Array>;
  /** Serializes the constraint, including its vocabulary and root stopping policy; `externalVocab` outputs an artifact that needs the exact original token-ID/byte mapping to load. */
  saveSync(options?: SaveOptions): Uint8Array;
  /** Deserializes a constraint; `vocab` is required for external-vocabulary artifacts and must match the original mapping. */
  static load(data: Uint8Array, vocab?: Vocab): Promise<Constraint>;
  /** Deserializes a constraint; `vocab` is required for external-vocabulary artifacts and must match the original mapping. */
  static loadSync(data: Uint8Array, vocab?: Vocab): Constraint;
  /** Releases the native handle. */
  dispose(): void;
  /** Releases the native handle when leaving a `using` block. */
  [Symbol.dispose](): void;
}

/** The mutable state for a single generated sequence. */
export declare class ConstraintState implements Disposable {
  private constructor();
  private readonly __stateBrand: void;
  /** Number of packed `u32` words in a mask. */
  readonly maskLength: number;
  /** An independent snapshot at the current prefix. */
  copy(): ConstraintState;
  /** An independent snapshot at the current prefix. */
  clone(): ConstraintState;
  /** The current allowed-token mask as packed original-token-ID `u32` words. */
  mask(): Uint32Array;
  /** Writes the current mask into `output`, which must hold at least `maskLength` words; extra words are cleared. */
  fillMask(output: Uint32Array): void;
  /** Commits one model token event; unknown or forbidden IDs reject the state. */
  commitToken(id: number): void;
  /** Commits raw bytes rather than a model token event; does not substitute for committing exact-token or end-token events. */
  commitBytes(bytes: Uint8Array): void;
  /** Whether the current prefix is a complete grammar match; further tokens may still be allowed. */
  isAccepting(): boolean;
  /** Whether the prefix cannot be extended to a valid match. */
  isRejected(): boolean;
  /** A read-only prefix of library-canonical forced output. */
  forcedTokens(): Uint32Array;
  /** Initial or most recent commit's observations, ordered by byte offset then marker ID. */
  markerEvents(): MarkerEvent[];
  /** Releases the native handle. */
  dispose(): void;
  /** Releases the native handle when leaving a `using` block. */
  [Symbol.dispose](): void;
}
