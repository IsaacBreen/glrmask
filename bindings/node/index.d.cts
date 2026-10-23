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
}>;
/** A boundary parser strategy. */
export type BoundaryMaskStrategy = typeof BoundaryMaskStrategy[keyof typeof BoundaryMaskStrategy];

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
  /** Token IDs valid for token-set bindings that have no byte spelling. */
  specialTokenIds?: Iterable<number>;
}
/** The model vocabulary a constraint is compiled against. Entries map model token IDs to exact token bytes; token IDs may be sparse. */
export declare class Vocab implements Disposable {
  private readonly __vocabBrand: void;
  /** Creates a vocabulary from `[token_id, bytes]` entries. */
  constructor(entries: Iterable<readonly [number, Uint8Array]>, options?: VocabOptions);
  /** Number of byte-backed entries; special IDs are not counted. */
  readonly size: number;
  /** An independent handle. */
  clone(): Vocab;
  /** Releases the native handle. */
  dispose(): void;
  /** Releases the native handle when leaving a `using` block. */
  [Symbol.dispose](): void;
}

/** A set of model token IDs; an empty set matches nothing. IDs are validated against the compile vocabulary. */
export declare class TokenSet implements Disposable {
  private readonly __tokenSetBrand: void;
  /** Creates a set from token IDs; duplicates are ignored. */
  constructor(ids: Iterable<number>);
  /** Creates a set from half-open `ranges`; overlapping or adjacent ranges merge, and an empty or reversed range contributes nothing. */
  static fromRanges(ranges: Iterable<TokenRange>): TokenSet;
  /** Whether both handles hold the same token IDs. */
  equals(other: unknown): boolean;
  /** The normalized ranges, for example `TokenSet[0..=9, 20..=29]`. */
  toString(): string;
  /** An independent handle. */
  clone(): TokenSet;
  /** Releases the native handle. */
  dispose(): void;
  /** Releases the native handle when leaving a `using` block. */
  [Symbol.dispose](): void;
}

/** Compilation settings for a grammar or module assembly. */
export interface CompileOptions {
  /** The mask computation strategy for the whole compile. */
  maskStrategy?: MaskStrategy;
  /** The boundary parser strategy for the whole compile. */
  boundaryMaskStrategy?: BoundaryMaskStrategy;
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
  clone(): Grammar;
  /** Builds a source grammar from an expression. */
  static fromExpr(expr: Expr | Grammar | TokenSet): Grammar;
  /** Binds one declared extern. */
  bind(name: string, value: ExprPart): void;
  /** Retains vocabulary-specific compilation artifacts; equivalent to `new Module(this).precompile(vocab, ...)`. */
  precompile(vocab: Vocab, options?: CompileOptions): Promise<Module>;
  /** Retains vocabulary-specific compilation artifacts; equivalent to `new Module(this).precompileSync(vocab, ...)`. */
  precompileSync(vocab: Vocab, options?: CompileOptions): Module;
  /** Compiles the grammar against `vocab`; equivalent to `new Module(this).compile(vocab, ...)`. */
  compile(vocab: Vocab, options?: CompileOptions): Promise<Constraint>;
  /** Compiles the grammar against `vocab`; equivalent to `new Module(this).compileSync(vocab, ...)`. */
  compileSync(vocab: Vocab, options?: CompileOptions): Constraint;
  /** Releases the native handle. */
  dispose(): void;
  /** Releases the native handle when leaving a `using` block. */
  [Symbol.dispose](): void;
}


/** A value accepted as an `Expr` operand. */
export type ExprPart = Grammar | Module | Expr | TokenSet | number;

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
  clone(): Expr;
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
  clone(): Module;
  /** Binds the extern `name` to `value`; fails if the extern is undeclared or already bound, and names containing `::` are rejected. */
  bind(name: string, value: ExprPart): void;
  /** Binds several externs atomically; if any binding fails, none is applied. */
  bind(bindings: Record<string, ExprPart>): void;
  /** Retains vocabulary-specific compilation artifacts for this module. */
  precompile(vocab: Vocab, options?: CompileOptions): Promise<void>;
  /** Retains vocabulary-specific compilation artifacts for this module. */
  precompileSync(vocab: Vocab, options?: CompileOptions): void;
  /** Compiles the module assembly against `vocab`; every extern must be bound. */
  compile(vocab: Vocab, options?: CompileOptions): Promise<Constraint>;
  /** Compiles the module assembly against `vocab`; every extern must be bound. */
  compileSync(vocab: Vocab, options?: CompileOptions): Constraint;
  /** Serializes the module assembly and any retained precompiled artifacts. */
  save(): Promise<Uint8Array>;
  /** Serializes the module assembly and any retained precompiled artifacts. */
  saveSync(): Uint8Array;
  /** Deserializes a module assembly. Retained work is checked against the vocabulary supplied to `compile` or `precompile`. */
  static load(data: Uint8Array): Promise<Module>;
  /** Deserializes a module assembly. Retained work is checked against the vocabulary supplied to `compile` or `precompile`. */
  static loadSync(data: Uint8Array): Module;
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
  clone(): Constraint;
  /** Creates independent state for one generated sequence. */
  start(): ConstraintState;
  /** Sorted, distinct application marker IDs. */
  markerIds(): Uint32Array;
  /** Serializes the constraint without its vocabulary; load with the exact original token-ID/byte mapping. */
  save(): Promise<Uint8Array>;
  /** Serializes the constraint without its vocabulary; load with the exact original token-ID/byte mapping. */
  saveSync(): Uint8Array;
  /** Deserializes a constraint. `vocab` must match the original mapping. */
  static load(data: Uint8Array, vocab: Vocab): Promise<Constraint>;
  /** Deserializes a constraint. `vocab` must match the original mapping. */
  static loadSync(data: Uint8Array, vocab: Vocab): Constraint;
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
  clone(): ConstraintState;
  /** The current allowed-token mask as packed original-token-ID `u32` words. */
  mask(): Uint32Array;
  /** Writes the current mask into `output`, which must hold at least `maskLength` words; extra words are cleared. */
  fillMask(output: Uint32Array): void;
  /** Commits one model token event; false when it is disallowed and rejects the state. */
  commitToken(id: number): boolean;
  /** Commits model token events in order; false at the first disallowed event. */
  commitTokens(ids: Iterable<number> | Uint32Array): boolean;
  /** Commits raw bytes rather than a model token event; false when disallowed. */
  commitBytes(bytes: Uint8Array): boolean;
  /** Whether the input so far is a complete match. With end tokens configured, this becomes true only once an end token is committed. */
  isAccepting(): boolean;
  /** TODO(docs) */
  isTerminated(): boolean;
  /** A read-only prefix of library-canonical forced output. */
  forcedTokens(): Uint32Array;
  /** Initial or most recent commit's observations, ordered by byte offset then marker ID. */
  markerEvents(): MarkerEvent[];
  /** Releases the native handle. */
  dispose(): void;
  /** Releases the native handle when leaving a `using` block. */
  [Symbol.dispose](): void;
}
