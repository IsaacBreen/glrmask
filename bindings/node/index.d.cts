/// <reference lib="esnext.disposable" />
/** The GLRMask version, for example "1.0.0". */
export declare const version: string;

/** A preference for faster compiling or faster masks. */
export declare const MaskStrategy: Readonly<{
  /** Let GLRMask choose. The default. */
  Auto: 'auto';
  /** Prefer faster compiling over faster masks. */
  FastCompile: 'fast-compile';
  /** Prefer faster masks over faster compiling. */
  FastMasks: 'fast-masks';
}>;
export type MaskStrategy = typeof MaskStrategy[keyof typeof MaskStrategy];

/** The kinds of `GLRMaskError`. */
export type KnownErrorCode = 'GLRMASK_SYNTAX' | 'GLRMASK_INVALID_INPUT' | 'GLRMASK_COMPILATION' | 'GLRMASK_RESOURCE_LIMIT' | 'GLRMASK_STRATEGY_UNSUPPORTED' | 'GLRMASK_SERIALIZATION' | 'GLRMASK_INTERNAL' | 'GLRMASK_BUFFER_TOO_SMALL' | 'GLRMASK_CLOSED';
/** Error codes may grow in a minor release. */
export type ErrorCode = KnownErrorCode | (string & {});
/** A GLRMask error. */
export declare class GLRMaskError extends Error {
  /** What kind of error this is. */
  readonly code: ErrorCode;
  constructor(code: ErrorCode, message: string, options?: ErrorOptions);
}

/** A range of token IDs from `start` up to, but not including, `end`. */
export interface TokenRange {
  readonly start: number;
  /** The ID just past the range, at most 16777216. */
  readonly end: number;
}
/** A named placeholder in a grammar. */
export interface Extern {
  readonly name: string;
  /** What the extern can be bound to: `'token'` or `'grammar'`. */
  readonly kind: 'token' | 'grammar' | (string & {});
}
/** Externs by name, in name order. */
export interface Externs extends ReadonlyMap<string, Extern> {
  /** Unbound names at the time of the snapshot. */
  readonly unbound: Externs;
}

/** Each token ID and its bytes. */
export declare class Vocab implements Disposable {
  private readonly __vocabBrand: void;

  constructor(entries: Iterable<readonly [number, Uint8Array]> | Readonly<Record<number, Uint8Array>>);

  clone(): Vocab;
  /** Frees the native handle now instead of at garbage collection. */
  dispose(): void;
  /** Same as `dispose()`, for `using` declarations. */
  [Symbol.dispose](): void;
}

/** A set of token IDs. */
export declare class TokenSet implements Disposable {
  private readonly __tokenSetBrand: void;

  constructor(ids: Iterable<number>);
  static fromRanges(ranges: Iterable<TokenRange>): TokenSet;
  /** Whether `other` is a set with the same IDs. */
  equals(other: unknown): boolean;
  /** The set as ranges, for example `TokenSet[0..=9, 20..=29]`. */
  toString(): string;

  clone(): TokenSet;
  /** Frees the native handle now instead of at garbage collection. */
  dispose(): void;
  /** Same as `dispose()`, for `using` declarations. */
  [Symbol.dispose](): void;
}

/** Options for compiling a grammar or module. */
export interface CompileOptions {
  /** Defaults to `MaskStrategy.Auto`. */
  maskStrategy?: MaskStrategy;
}

export declare class Grammar implements Disposable {
  private constructor();
  private readonly __grammarBrand: void;
  /** Parses GLRM source. */
  static fromGlrm(source: string): Grammar;
  /** Parses a JSON Schema given as JSON text or an object. */
  static fromJsonSchema(schema: unknown): Grammar;
  /** Parses Lark source. */
  static fromLark(source: string): Grammar;
  /** A snapshot; read it again after `bind()`. */
  readonly externs: Externs;

  clone(): Grammar;

  static fromExpr(expr: Expr | Grammar): Grammar;
  /** Binds the extern `name` to `value`. Throws if `name` is undeclared or already bound, or `value` is the wrong kind. */
  bind(name: string, value: GrammarBindValue): void;
  /** Binds several externs at once; if one fails, none is bound. */
  bind(bindings: Record<string, GrammarBindValue>): void;
  /** Like `compileSync`, but returns a promise and doesn't block the event loop.
   *
   * Throws if any extern is unbound. */
  compile(vocab: Vocab, options?: CompileOptions): Promise<Constraint>;
  /** Throws if any extern is unbound. */
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

  clone(): Expr;
  /** Frees the native handle now instead of at garbage collection. */
  dispose(): void;
  /** Same as `dispose()`, for `using` declarations. */
  [Symbol.dispose](): void;
}

/** A partially compiled grammar. */
export declare class Module implements Disposable {
  private readonly __moduleBrand: void;

  constructor(grammar: Grammar);
  /** A snapshot; read it again after `bind()`. */
  readonly externs: Externs;

  clone(): Module;
  /** Binds the extern `name` to `value`. Throws if `name` is undeclared or already bound, or `value` is the wrong kind. */
  bind(name: string, value: ExprPart): void;
  /** Binds several externs at once; if one fails, none is bound. */
  bind(bindings: Record<string, ExprPart>): void;
  /** Like `precompileSync`, but returns a promise and doesn't block the event loop. */
  precompile(vocab: Vocab, options?: CompileOptions): Promise<void>;
  /** Compiles what it can for `vocab` now, even with externs unbound. */
  precompileSync(vocab: Vocab, options?: CompileOptions): void;
  /** Like `compileSync`, but returns a promise and doesn't block the event loop.
   *
   * Throws if any extern is unbound or the module was precompiled for another vocabulary. */
  compile(vocab: Vocab, options?: CompileOptions): Promise<Constraint>;
  /** Reuses any precompiled work, which must be for the same vocabulary. */
  compileSync(vocab: Vocab, options?: CompileOptions): Constraint;
  /** Like `saveSync`, but returns a promise and doesn't block the event loop. */
  save(): Promise<Uint8Array>;
  /** Serializes the module with its precompiled work. Other GLRMask versions may not load it. */
  saveSync(): Uint8Array;
  /** Like `loadSync`, but asynchronous. */
  static load(data: Uint8Array): Promise<Module>;

  static loadSync(data: Uint8Array): Module;
  /** Frees the native handle now instead of at garbage collection. */
  dispose(): void;
  /** Same as `dispose()`, for `using` declarations. */
  [Symbol.dispose](): void;
}

/** A compiled grammar, ready to use. */
export declare class Constraint implements Disposable {
  private constructor();
  private readonly __constraintBrand: void;
  /** One more than the largest token ID. */
  readonly maskBitLength: number;

  clone(): Constraint;
  /** Starts a new sequence. */
  start(): ConstraintState;
  /** Like `saveSync`, but returns a promise and doesn't block the event loop. */
  save(): Promise<Uint8Array>;
  /** Serializes the constraint without its vocabulary. Other GLRMask versions may not load it. */
  saveSync(): Uint8Array;
  /** Like `loadSync`, but returns a promise and doesn't block the event loop.
   *
   * `vocab` must be the one it was compiled with. */
  static load(data: Uint8Array, vocab: Vocab): Promise<Constraint>;
  /** `vocab` must be the one it was compiled with. */
  static loadSync(data: Uint8Array, vocab: Vocab): Constraint;
  /** Frees the native handle now instead of at garbage collection. */
  dispose(): void;
  /** Same as `dispose()`, for `using` declarations. */
  [Symbol.dispose](): void;
}

/** The state of one generated sequence. */
export declare class ConstraintState implements Disposable {
  private constructor();
  private readonly __stateBrand: void;
  /** One more than the largest token ID. */
  readonly maskBitLength: number;

  clone(): ConstraintState;
  /** Allowed token IDs as a packed bitmask. Token `id` is allowed when bit `id % 32` of element `Math.floor(id / 32)` is set. */
  mask(): Uint32Array;
  /** Writes the mask into `output`, zeroing any extra elements. Throws `GLRMaskError` with code `GLRMASK_BUFFER_TOO_SMALL` if `output` has fewer than `Math.ceil(maskBitLength / 32)` elements. */
  fillMask(output: Uint32Array): void;
  /** Returns `false`, and ends the state, if the token isn't allowed. */
  commitToken(id: number): boolean;
  /** Returns how many were committed. Stops at the first token that isn't allowed, which ends the state. */
  commitTokens(ids: Iterable<number> | Uint32Array): number;
  /** Returns `false`, and ends the state, if the bytes aren't allowed. */
  commitBytes(bytes: Uint8Array): boolean;
  /** Whether what's committed is a complete match. If so, allow EOS (don't commit it). */
  isAccepting(): boolean;
  /** Whether nothing more can be committed. */
  isTerminated(): boolean;
  /** The bytes the grammar requires next. Stops where a token from a token set, the end of the grammar, or more than one byte could come next. May return only part of a long run; call again after committing. */
  forcedBytes(): Uint8Array;
  /** Frees the native handle now instead of at garbage collection. */
  dispose(): void;
  /** Same as `dispose()`, for `using` declarations. */
  [Symbol.dispose](): void;
}
