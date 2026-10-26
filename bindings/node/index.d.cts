/// <reference lib="esnext.disposable" />
/** The GLRMask version, for example "0.4.0". */
export declare const version: string;

/** How masks are computed. */
export declare const MaskStrategy: Readonly<{
  /** Reuses any precompiled work; otherwise `ParserDwa`, or `TokenTrie` when that would be too large. The default. */
  Auto: 'auto';
  /** Builds a parser DWA for the grammar. Slowest to compile, with the fastest masks. */
  ParserDwa: 'parser-dwa';
  /** Computes each mask by walking the vocabulary. Fastest to compile, with slower masks. Pass a `TokenTrie` object instead to change its settings. */
  TokenTrie: 'token-trie';
  /** Same as `TokenTrie`. */
  FastCompile: 'token-trie';
  /** Same as `ParserDwa`. */
  FastMasks: 'parser-dwa';
}>;
/** A `MaskStrategy` value, or `TokenTrie` settings. */
export type MaskStrategy = typeof MaskStrategy[keyof typeof MaskStrategy] | TokenTrie;

/** How masks are computed at grammar boundaries. */
export declare const BoundaryMaskStrategy: Readonly<{
  /** `ParserDwa`, or `TokenTrie` when that would be too large. The default. */
  Auto: 'auto';
  /** Builds a parser DWA across each boundary. Slower to compile, with faster masks. Throws `GlrMaskError` if it would be too large. */
  ParserDwa: 'parser-dwa';
  /** Walks the vocabulary at each boundary as needed. Faster to compile, with slower masks. */
  TokenTrie: 'token-trie';
  /** Same as `TokenTrie`. */
  FastCompile: 'token-trie';
  /** Same as `ParserDwa`. */
  FastMasks: 'parser-dwa';
}>;
/** A `BoundaryMaskStrategy` value. */
export type BoundaryMaskStrategy = typeof BoundaryMaskStrategy[keyof typeof BoundaryMaskStrategy];

/** How the parser is represented and run when masks are computed with `MaskStrategy.TokenTrie`. */
export declare const ParserBackend: Readonly<{
  /** An LR table. Faster to compile, with slower masks. The default. */
  LrTable: 'lr-table';
  /** Template DFAs. Slower to compile, with faster masks. */
  TemplateDfa: 'template-dfa';
}>;
/** A `ParserBackend` value. */
export type ParserBackend = typeof ParserBackend[keyof typeof ParserBackend];

/** Settings for `MaskStrategy.TokenTrie`. Pass one as `maskStrategy`. */
export interface TokenTrie {
  /** Always `'token-trie'`. */
  readonly kind: 'token-trie';
  /** How the parser is represented. Defaults to `ParserBackend.LrTable`. */
  readonly parserBackend?: ParserBackend;
  /** Whether tokens that the grammar treats the same are merged, so masks walk fewer tokens. Defaults to `false`. Tokens are not merged in a composition that includes precompiled parts. */
  readonly mergeEquivalentTokens?: boolean;
}

/** The kinds of `GlrMaskError`. */
export type ErrorCode = 'GLRMASK_INVALID_ARGUMENT' | 'GLRMASK_ENGINE_ERROR' | 'GLRMASK_PANIC' | 'GLRMASK_BUFFER_TOO_SMALL' | 'GLRMASK_CLOSED';
/** A GLRMask error. */
export declare class GlrMaskError extends Error {
  /** What kind of error this is. */
  readonly code: ErrorCode;
  /** Creates an error. GLRMask throws these; you don't need to create them. */
  constructor(code: ErrorCode, message: string, options?: ErrorOptions);
}

/** A range of token IDs from `start` up to, but not including, `end`. */
export interface TokenRange {
  /** The first ID. */
  readonly start: number;
  /** The ID just past the range, at most 4294967296. */
  readonly end: number;
}
/** A named placeholder in a grammar. */
export interface Extern {
  /** The extern's name. */
  readonly name: string;
  /** What the extern can be bound to: `'token'` or `'grammar'`. */
  readonly kind: 'token' | 'grammar';
}
/** A grammar's or module's externs, in declaration order.
 *
 * A read-only map from name to `Extern`. */
export interface Externs extends ReadonlyMap<string, Extern> {
  /** The externs not yet bound. */
  readonly unbound: Externs;
}

/** Options for `new Vocab`. */
export interface VocabOptions {
  /** Tokens that have no bytes, such as end-of-sequence tokens. They can be used in token sets but never match text. */
  specialTokenIds?: Iterable<number>;
}
/** A model's tokens and their bytes. */
export declare class Vocab implements Disposable {
  private readonly __vocabBrand: void;
  /** Creates a vocabulary from `[id, bytes]` entries. */
  constructor(entries: Iterable<readonly [number, Uint8Array]>, options?: VocabOptions);
  /** A new handle to the same object. */
  clone(): Vocab;
  /** Frees the native handle now instead of at garbage collection. */
  dispose(): void;
  /** Same as `dispose()`, for `using` declarations. */
  [Symbol.dispose](): void;
}

/** A set of token IDs. */
export declare class TokenSet implements Disposable {
  private readonly __tokenSetBrand: void;
  /** Creates a set from `ids`. */
  constructor(ids: Iterable<number>);
  /** Creates a set from `ranges`. */
  static fromRanges(ranges: Iterable<TokenRange>): TokenSet;
  /** Whether `other` is a set with the same IDs. */
  equals(other: unknown): boolean;
  /** The set as ranges, for example `TokenSet[0..=9, 20..=29]`. */
  toString(): string;
  /** A new handle to the same object. */
  clone(): TokenSet;
  /** Frees the native handle now instead of at garbage collection. */
  dispose(): void;
  /** Same as `dispose()`, for `using` declarations. */
  [Symbol.dispose](): void;
}

/** Options for compiling a grammar or module. */
export interface CompileOptions {
  /** How masks are computed. Defaults to `MaskStrategy.Auto`. */
  maskStrategy?: MaskStrategy;
  /** How masks are computed at grammar boundaries. Defaults to `BoundaryMaskStrategy.Auto`. */
  boundaryMaskStrategy?: BoundaryMaskStrategy;
}

/** A grammar, written as source text or built from an `Expr`. */
export declare class Grammar implements Disposable {
  private constructor();
  private readonly __grammarBrand: void;
  /** Creates a grammar from GLRM source. */
  static fromGlrm(source: string): Grammar;
  /** Creates a grammar from a JSON Schema, given as JSON text or an object. */
  static fromJsonSchema(schema: unknown): Grammar;
  /** Creates a grammar from Lark source. */
  static fromLark(source: string): Grammar;
  /** The grammar's externs. */
  readonly externs: Externs;
  /** A copy of the grammar. Binding the copy does not change the original. */
  clone(): Grammar;
  /** Creates a grammar from an expression. */
  static fromExpr(expr: Expr | Grammar | TokenSet): Grammar;
  /** Binds the extern `name` to `value`. Throws if `name` is undeclared or already bound, or `value` is the wrong kind. */
  bind(name: string, value: ExprPart): void;
  /** Binds several externs at once; if one fails, none is bound. */
  bind(bindings: Record<string, ExprPart>): void;
  /** Like `precompileSync`, but returns a promise and doesn't block the event loop. */
  precompile(vocab: Vocab, options?: CompileOptions): Promise<Module>;
  /** Compiles as much as possible for `vocab` now, even with externs unbound, and returns a `Module` that keeps the work. */
  precompileSync(vocab: Vocab, options?: CompileOptions): Module;
  /** Like `compileSync`, but returns a promise and doesn't block the event loop. */
  compile(vocab: Vocab, options?: CompileOptions): Promise<Constraint>;
  /** Compiles the grammar for `vocab`. */
  compileSync(vocab: Vocab, options?: CompileOptions): Constraint;
  /** Frees the native handle now instead of at garbage collection. */
  dispose(): void;
  /** Same as `dispose()`, for `using` declarations. */
  [Symbol.dispose](): void;
}


/** A value that can be part of an `Expr` or bound to an extern. */
export type ExprPart = Grammar | Module | Expr | TokenSet | number;

/** A grammar expression. */
export declare class Expr implements Disposable {
  private constructor();
  private readonly __exprBrand: void;
  /** Matches any one of `parts`.
   *
   * Throws if a part has unbound externs. */
  static choice(...parts: ExprPart[]): Expr;
  /** Matches `parts` in order.
   *
   * Throws if a part has unbound externs. */
  static sequence(...parts: ExprPart[]): Expr;
  /** Matches exactly `value`. A string is UTF-8 encoded. */
  static literal(value: string | Uint8Array): Expr;
  /** A new handle to the same object. */
  clone(): Expr;
  /** Frees the native handle now instead of at garbage collection. */
  dispose(): void;
  /** Same as `dispose()`, for `using` declarations. */
  [Symbol.dispose](): void;
}

/** A grammar you can compose with others and partly compile before its externs are bound. */
export declare class Module implements Disposable {
  private readonly __moduleBrand: void;
  /** Creates a module from `grammar`. */
  constructor(grammar: Grammar);
  /** The module's externs. */
  readonly externs: Externs;
  /** A copy of the module. Binding the copy does not change the original. */
  clone(): Module;
  /** Binds the extern `name` to `value`. Throws if `name` is undeclared or already bound, or `value` is the wrong kind. */
  bind(name: string, value: ExprPart): void;
  /** Binds several externs at once; if one fails, none is bound. */
  bind(bindings: Record<string, ExprPart>): void;
  /** Like `precompileSync`, but returns a promise and doesn't block the event loop. */
  precompile(vocab: Vocab, options?: CompileOptions): Promise<void>;
  /** Compiles as much as possible for `vocab` now, even with externs unbound, so later compiles are faster. */
  precompileSync(vocab: Vocab, options?: CompileOptions): void;
  /** Like `compileSync`, but returns a promise and doesn't block the event loop. */
  compile(vocab: Vocab, options?: CompileOptions): Promise<Constraint>;
  /** Compiles the module for `vocab`, reusing any precompiled work. Every extern must be bound. */
  compileSync(vocab: Vocab, options?: CompileOptions): Constraint;
  /** Like `saveSync`, but returns a promise and doesn't block the event loop. */
  save(): Promise<Uint8Array>;
  /** Saves the module as bytes, with its precompiled work but without the vocabulary. Bytes saved by one GLRMask version may not load in another. */
  saveSync(): Uint8Array;
  /** Like `loadSync`, but returns a promise and doesn't block the event loop. */
  static load(data: Uint8Array): Promise<Module>;
  /** Loads a module saved with `save`. Its precompiled work only applies to the vocabulary it was built for; compiling with another is an error. */
  static loadSync(data: Uint8Array): Module;
  /** Frees the native handle now instead of at garbage collection. */
  dispose(): void;
  /** Same as `dispose()`, for `using` declarations. */
  [Symbol.dispose](): void;
}

/** A grammar compiled for a vocabulary. */
export declare class Constraint implements Disposable {
  private constructor();
  private readonly __constraintBrand: void;
  /** The number of `u32` words in a mask. */
  readonly maskLength: number;
  /** A new handle to the same object. */
  clone(): Constraint;
  /** A new state with nothing committed. */
  start(): ConstraintState;
  /** Like `saveSync`, but returns a promise and doesn't block the event loop. */
  save(): Promise<Uint8Array>;
  /** Saves the constraint as bytes, without the vocabulary.
   *
   * Bytes saved by one GLRMask version may not load in another. */
  saveSync(): Uint8Array;
  /** Like `loadSync`, but returns a promise and doesn't block the event loop. */
  static load(data: Uint8Array, vocab: Vocab): Promise<Constraint>;
  /** Loads a constraint saved with `save`. `vocab` must be the one it was compiled for, or loading fails. */
  static loadSync(data: Uint8Array, vocab: Vocab): Constraint;
  /** Frees the native handle now instead of at garbage collection. */
  dispose(): void;
  /** Same as `dispose()`, for `using` declarations. */
  [Symbol.dispose](): void;
}

/** A constraint's state during generation. */
export declare class ConstraintState implements Disposable {
  private constructor();
  private readonly __stateBrand: void;
  /** The number of `u32` words in a mask. */
  readonly maskLength: number;
  /** A copy of the state. */
  clone(): ConstraintState;
  /** The tokens allowed next, as a bitset. Token `id` is allowed when bit `id % 32` of element `Math.floor(id / 32)` is set. */
  mask(): Uint32Array;
  /** Writes the mask into `output`, zeroing any extra elements. Throws `GlrMaskError` with code `GLRMASK_BUFFER_TOO_SMALL` if it has fewer than `maskLength` elements. */
  fillMask(output: Uint32Array): void;
  /** Commits a token. Returns `false` if it is not allowed, which terminates the state. */
  commitToken(id: number): boolean;
  /** Commits each token in order. Returns `false` at the first one that is not allowed, which terminates the state. */
  commitTokens(ids: Iterable<number> | Uint32Array): boolean;
  /** Commits raw bytes. Returns `false` if they are not allowed, which terminates the state. */
  commitBytes(bytes: Uint8Array): boolean;
  /** Whether the input so far is a complete match. */
  isAccepting(): boolean;
  /** Whether no more input can be accepted. */
  isTerminated(): boolean;
  /** Tokens for what must come next, taking the longest token at each step. Stops before anything that could vary, or at 10,000 tokens or bytes. */
  forcedTokens(): Uint32Array;
  /** Frees the native handle now instead of at garbage collection. */
  dispose(): void;
  /** Same as `dispose()`, for `using` declarations. */
  [Symbol.dispose](): void;
}
