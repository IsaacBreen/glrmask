/// <reference lib="esnext.disposable" />
export declare const version: string;

/** Each token ID and its bytes. */
export declare class Vocab implements Disposable {
  private readonly __vocabBrand: void;

  constructor(entries: Iterable<readonly [number, Uint8Array]>);

  clone(): Vocab;
  /** Frees the native handle now instead of at garbage collection. */
  dispose(): void;
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
  /** Parses [GLRM](https://github.com/IsaacBreen/glrmask/blob/main/docs/glrm-format.md) source. */
  static fromGlrm(source: string): Grammar;
  /** Parses [Lark](https://github.com/IsaacBreen/glrmask/blob/main/docs/lark-dialect.md) source. */
  static fromLark(source: string): Grammar;
  /** Parses a JSON Schema given as JSON text or an object. */
  static fromJsonSchema(schema: string | boolean | object): Grammar;
  /** Builds a grammar from an `Expr`. */
  static fromExpr(expr: Expr): Grammar;
  readonly externs: Externs;


  /** Compiles the grammar into a constraint. */
  compile(vocab: Vocab, options?: CompileOptions): Promise<Constraint>;
  /** Compiles the grammar into a constraint. */
  compileSync(vocab: Vocab, options?: CompileOptions): Constraint;
  clone(): Grammar;
  /** Frees the native handle now instead of at garbage collection. */
  dispose(): void;
  [Symbol.dispose](): void;
}



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
  [Symbol.dispose](): void;
}

/** A range of token IDs from `start` up to, but not including, `end`. */
export interface TokenRange {
  readonly start: number;
  /** The ID just past the range, at most 16777216. */
  readonly end: number;
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
  [Symbol.dispose](): void;
}

/** A partially compiled grammar. */
export declare class Module implements Disposable {
  private readonly __moduleBrand: void;

  constructor(grammar: Grammar);
  readonly externs: Externs;

  /** Throws if the extern is undeclared, already bound or the wrong kind. */
  bind(name: string, value: ExprPart): void;
  precompile(vocab: Vocab, options?: CompileOptions): Promise<void>;
  /** Compiles what it can now, even with externs unbound. Further compilation must use the same `vocab`. */
  precompileSync(vocab: Vocab, options?: CompileOptions): void;
  /** Finishes compiling the module into a constraint. */
  compile(vocab: Vocab, options?: CompileOptions): Promise<Constraint>;
  /** Finishes compiling the module into a constraint. */
  compileSync(vocab: Vocab, options?: CompileOptions): Constraint;
  save(): Promise<Uint8Array>;
  saveSync(): Uint8Array;
  /** May fail if saved by a different GLRMask version. */
  static load(data: Uint8Array): Promise<Module>;

  /** May fail if saved by a different GLRMask version. */
  static loadSync(data: Uint8Array): Module;
  clone(): Module;
  /** Frees the native handle now instead of at garbage collection. */
  dispose(): void;
  [Symbol.dispose](): void;
}

export declare class Constraint implements Disposable {
  private constructor();
  private readonly __constraintBrand: void;
  /** Starts a new sequence. */
  start(): ConstraintState;
  /** The mask's length in bits (the largest token ID plus one). */
  readonly maskBitLength: number;

  save(): Promise<Uint8Array>;

  saveSync(): Uint8Array;
  /** May fail if saved by a different GLRMask version. Fails if `vocab` differs from the one it was compiled with. */
  static load(data: Uint8Array, vocab: Vocab): Promise<Constraint>;
  /** May fail if saved by a different GLRMask version. Fails if `vocab` differs from the one it was compiled with. */
  static loadSync(data: Uint8Array, vocab: Vocab): Constraint;
  clone(): Constraint;
  /** Frees the native handle now instead of at garbage collection. */
  dispose(): void;
  [Symbol.dispose](): void;
}

export declare class ConstraintState implements Disposable {
  private constructor();
  private readonly __stateBrand: void;

  /** Which tokens are allowed next, as a bitmask: token `i` is bit `i % 32` of word `Math.floor(i / 32)`. */
  mask(): Uint32Array;
  /** Like `mask()`, but writes the words into `output`. Throws `GLRMaskError` with code `GLRMASK_BUFFER_TOO_SMALL` if `output` has fewer than `Math.ceil(maskBitLength / 32)` elements. */
  fillMask(output: Uint32Array): void;
  /** The mask's length in bits (the largest token ID plus one). */
  readonly maskBitLength: number;
  /** Advances the state by the token. If it isn't allowed, returns `false` and terminates the state. */
  commitToken(id: number): boolean;

  /** Advances the state by the bytes. If they aren't allowed, returns `false` and terminates the state. */
  commitBytes(bytes: Uint8Array): boolean;
  /** The bytes that must come next. */
  forcedBytes(): Uint8Array;
  /** Whether the output so far could end here. */
  isAccepting(): boolean;
  /** Whether nothing more can be committed. */
  isTerminated(): boolean;
  clone(): ConstraintState;
  /** Frees the native handle now instead of at garbage collection. */
  dispose(): void;
  [Symbol.dispose](): void;
}

/** Controls how the constraint is compiled. */
export declare const MaskStrategy: Readonly<{
  /** The default; currently `FastMasks`. */
  Auto: 'auto';
  /** Faster compilation, slower masks. */
  FastCompile: 'fast-compile';
  /** Faster masks, slower compilation. */
  FastMasks: 'fast-masks';
}>;
export type MaskStrategy = typeof MaskStrategy[keyof typeof MaskStrategy];

/** A read-only mapping of extern names to `Extern`. */
export interface Externs extends ReadonlyMap<string, Extern> {
  readonly unbound: Externs;
}

/** A named placeholder in a grammar. */
export interface Extern {
  readonly name: string;
  readonly kind: 'token' | 'grammar' | (string & {});
}

export type KnownErrorCode = 'GLRMASK_SYNTAX' | 'GLRMASK_INVALID_INPUT' | 'GLRMASK_COMPILATION' | 'GLRMASK_RESOURCE_LIMIT' | 'GLRMASK_STRATEGY_UNSUPPORTED' | 'GLRMASK_SERIALIZATION' | 'GLRMASK_INTERNAL' | 'GLRMASK_BUFFER_TOO_SMALL' | 'GLRMASK_CLOSED' | 'GLRMASK_VOCAB_MISMATCH' | 'GLRMASK_PRECOMPILE_RACE';
export type ErrorCode = KnownErrorCode | (string & {});
export declare class GLRMaskError extends Error {
  /** One of `GLRMASK_SYNTAX`, `GLRMASK_INVALID_INPUT`, `GLRMASK_COMPILATION`, `GLRMASK_RESOURCE_LIMIT`, `GLRMASK_STRATEGY_UNSUPPORTED`, `GLRMASK_SERIALIZATION`, `GLRMASK_INTERNAL`, `GLRMASK_BUFFER_TOO_SMALL`, `GLRMASK_CLOSED`, `GLRMASK_VOCAB_MISMATCH`, `GLRMASK_PRECOMPILE_RACE`. */
  readonly code: ErrorCode;
  constructor(code: ErrorCode, message: string, options?: ErrorOptions);
}
