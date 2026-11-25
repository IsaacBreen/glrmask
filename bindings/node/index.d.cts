/// <reference lib="esnext.disposable" />
/** The GLRMask version, for example "1.0.0". */
export declare const version: string;

/** Cost preferences, not engine selections or performance guarantees. */
export declare const MaskStrategy: Readonly<{
  /** Let the library choose. The default. */
  Auto: 'auto';
  /** Prefer lower preparation cost, even if masks cost more. */
  FastCompile: 'fast-compile';
  /** Prefer lower mask cost, even if preparation costs more. */
  FastMasks: 'fast-masks';
}>;
export type MaskStrategy = typeof MaskStrategy[keyof typeof MaskStrategy];

/** Cost preferences for masks at grammar boundaries. */
export declare const BoundaryMaskStrategy: Readonly<{
  /** Let the library choose. The default. */
  Auto: 'auto';
  /** Prefer lower boundary preparation cost. */
  FastCompile: 'fast-compile';
  /** Prefer lower boundary mask cost. */
  FastMasks: 'fast-masks';
}>;
export type BoundaryMaskStrategy = typeof BoundaryMaskStrategy[keyof typeof BoundaryMaskStrategy];

/** The kinds of `GLRMaskError`. */
export type KnownErrorCode = 'GLRMASK_SYNTAX' | 'GLRMASK_INVALID_INPUT' | 'GLRMASK_COMPILATION' | 'GLRMASK_RESOURCE_LIMIT' | 'GLRMASK_STRATEGY_UNSUPPORTED' | 'GLRMASK_SERIALIZATION' | 'GLRMASK_INTERNAL' | 'GLRMASK_BUFFER_TOO_SMALL' | 'GLRMASK_CLOSED';
/** Error codes may grow in a minor release. */
export type ErrorCode = KnownErrorCode | (string & {});
/** A GLRMask error. */
export declare class GLRMaskError extends Error {
  /** What kind of error this is. */
  readonly code: ErrorCode;
  /** Creates an error. GLRMask throws these; you don't need to create them. */
  constructor(code: ErrorCode, message: string, options?: ErrorOptions);
}

/** A range of token IDs from `start` up to, but not including, `end`. */
export interface TokenRange {
  /** The first ID. */
  readonly start: number;
  /** The ID just past the range, at most 16777216. */
  readonly end: number;
}
/** A named placeholder in a grammar. */
export interface Extern {
  /** The extern's name. */
  readonly name: string;
  /** What the extern can be bound to: `'token'` or `'grammar'`. */
  readonly kind: 'token' | 'grammar' | (string & {});
}
/** A grammar's or module's externs, in declaration order.
 *
 * A read-only map from name to `Extern`. */
export interface Externs extends ReadonlyMap<string, Extern> {
  /** The externs not yet bound. */
  readonly unbound: Externs;
}

/** A model's tokens and their bytes. A token ID it doesn't contain never matches text, but a token set can still allow it. */
export declare class Vocab implements Disposable {
  private readonly __vocabBrand: void;
  /** Creates a vocabulary from `[id, bytes]` entries. */
  constructor(entries: Iterable<readonly [number, Uint8Array]> | Readonly<Record<number, Uint8Array>>);
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

/** A grammar, parsed from source or built from an `Expr`. */
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
  /** Creates a grammar from a portable expression (no Module or TokenSet, even nested). */
  static fromExpr(expr: Expr | Grammar): Grammar;
  /** Binds the extern `name` to `value`. Throws if `name` is undeclared or already bound, or `value` is the wrong kind. */
  bind(name: string, value: GrammarBindValue): void;
  /** Binds several externs at once; if one fails, none is bound. */
  bind(bindings: Record<string, GrammarBindValue>): void;
  /** Like `precompileSync`, but returns a promise and doesn't block the event loop. */
  precompile(vocab: Vocab, options?: CompileOptions): Promise<Module>;
  /** Compiles as much as possible for `vocab` now, even with externs unbound, and returns a `Module` that keeps the work. */
  precompileSync(vocab: Vocab, options?: CompileOptions): Module;
  /** Like `compileSync`, but returns a promise and doesn't block the event loop.
   *
   * Throws if any extern is unbound. */
  compile(vocab: Vocab, options?: CompileOptions): Promise<Constraint>;
  /** Compiles the grammar for `vocab`.
   *
   * Throws if any extern is unbound. */
  compileSync(vocab: Vocab, options?: CompileOptions): Constraint;
  /** Frees the native handle now instead of at garbage collection. */
  dispose(): void;
  /** Same as `dispose()`, for `using` declarations. */
  [Symbol.dispose](): void;
}


export type GrammarBindValue = Grammar | Expr;

/** A value that can be part of an `Expr`. */
export type ExprPart = Grammar | Module | Expr | TokenSet;

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
  /** Like `compileSync`, but returns a promise and doesn't block the event loop.
   *
   * Throws if any extern is unbound or the module was precompiled for another vocabulary. */
  compile(vocab: Vocab, options?: CompileOptions): Promise<Constraint>;
  /** Compiles the module for `vocab`, reusing any precompiled work. Throws if any extern is unbound or the module was precompiled for another vocabulary. */
  compileSync(vocab: Vocab, options?: CompileOptions): Constraint;
  /** Like `saveSync`, but returns a promise and doesn't block the event loop. */
  save(): Promise<Uint8Array>;
  /** Saves the module as bytes, with its precompiled work but without the vocabulary. Bytes saved by one GLRMask version may not load in another. */
  saveSync(): Uint8Array;
  /** Like `loadSync`, but returns a promise and doesn't block the event loop.
   *
   * Loads a module saved with `save()` or `saveSync()`. */
  static load(data: Uint8Array): Promise<Module>;
  /** Loads a module saved with `save()` or `saveSync()`. */
  static loadSync(data: Uint8Array): Module;
  /** Frees the native handle now instead of at garbage collection. */
  dispose(): void;
  /** Same as `dispose()`, for `using` declarations. */
  [Symbol.dispose](): void;
}

/** A grammar compiled for a vocabulary. Call `start()` for each sequence you generate. */
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
  /** Like `loadSync`, but returns a promise and doesn't block the event loop.
   *
   * Loads a constraint saved with `save()` or `saveSync()`. Throws if `vocab` isn't the one it was compiled for. */
  static load(data: Uint8Array, vocab: Vocab): Promise<Constraint>;
  /** Loads a constraint saved with `save()` or `saveSync()`. Throws if `vocab` isn't the one it was compiled for. */
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
  /** The tokens allowed next, as a bitmask. Token `id` is allowed when bit `id % 32` of element `Math.floor(id / 32)` is set. */
  mask(): Uint32Array;
  /** Writes the mask into `output`, zeroing any extra elements. Throws `GLRMaskError` with code `GLRMASK_BUFFER_TOO_SMALL` if `output` has fewer than `maskLength` elements. */
  fillMask(output: Uint32Array): void;
  /** Commits a token. If it isn't allowed, returns `false` and terminates the state. */
  commitToken(id: number): boolean;
  /** Returns the number of tokens committed. If fewer than requested, the state is terminated. To try tokens without changing this state, commit them on a clone. */
  commitTokens(ids: Iterable<number> | Uint32Array): number;
  /** Commits raw bytes. If they aren't allowed, returns `false` and terminates the state. */
  commitBytes(bytes: Uint8Array): boolean;
  /** Whether everything committed so far is a complete match. To allow EOS, add its token ID(s) to the mask when true, and stop without committing EOS. */
  isAccepting(): boolean;
  /** Whether nothing more can be committed. */
  isTerminated(): boolean;
  /** Commits the tokens for what must come next, taking the longest token at each step, and returns them. Stops before anything that could vary, or after 1,000 tokens. */
  commitForcedTokens(): Uint32Array;
  /** Frees the native handle now instead of at garbage collection. */
  dispose(): void;
  /** Same as `dispose()`, for `using` declarations. */
  [Symbol.dispose](): void;
}
