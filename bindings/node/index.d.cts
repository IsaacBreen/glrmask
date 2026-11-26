/// <reference lib="esnext.disposable" />
export declare const version: string;

/** Each token ID and its bytes. */
export declare class Vocab implements Disposable {
  private readonly __vocabBrand: void;

  constructor(entries: Iterable<readonly [number, Uint8Array]> | Readonly<Record<number, Uint8Array>>);

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
  static fromJsonSchema(schema: unknown): Grammar;
  /** Builds a grammar from an `Expr`. */
  static fromExpr(expr: Expr | Grammar): Grammar;
  readonly externs: Externs;


  /** Throws if the extern is undeclared, already bound or the wrong kind. */
  bind(name: string, value: GrammarBindValue): void;
  /** If one fails, none are bound. */
  bind(bindings: Record<string, GrammarBindValue>): void;
  /** Throws if any extern is unbound. */
  compile(vocab: Vocab, options?: CompileOptions): Promise<Constraint>;
  /** Throws if any extern is unbound. */
  compileSync(vocab: Vocab, options?: CompileOptions): Constraint;
  clone(): Grammar;
  /** Frees the native handle now instead of at garbage collection. */
  dispose(): void;
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
  /** If one fails, none are bound. */
  bind(bindings: Record<string, ExprPart>): void;
  precompile(vocab: Vocab, options?: CompileOptions): Promise<void>;
  /** Compiles what it can for `vocab` now, even with externs unbound. */
  precompileSync(vocab: Vocab, options?: CompileOptions): void;
  /** Reuses any precompiled work, which must be for the same vocabulary. Throws if an extern is unbound. */
  compile(vocab: Vocab, options?: CompileOptions): Promise<Constraint>;
  /** Reuses any precompiled work, which must be for the same vocabulary. */
  compileSync(vocab: Vocab, options?: CompileOptions): Constraint;
  /** Includes precompiled work. */
  save(): Promise<Uint8Array>;
  /** Includes precompiled work. */
  saveSync(): Uint8Array;
  /** Loads what `save()` wrote, in the same GLRMask version. */
  static load(data: Uint8Array): Promise<Module>;

  /** Loads what `saveSync()` wrote, in the same GLRMask version. */
  static loadSync(data: Uint8Array): Module;
  clone(): Module;
  /** Frees the native handle now instead of at garbage collection. */
  dispose(): void;
  [Symbol.dispose](): void;
}

/** A compiled grammar, ready to use. */
export declare class Constraint implements Disposable {
  private constructor();
  private readonly __constraintBrand: void;
  /** Starts a new sequence. */
  start(): ConstraintState;
  /** The mask's length in bits: one per token ID, from 0. */
  readonly maskBitLength: number;

  save(): Promise<Uint8Array>;

  saveSync(): Uint8Array;
  /** Loads what `save()` wrote, in the same GLRMask version and with the vocab it was compiled with. */
  static load(data: Uint8Array, vocab: Vocab): Promise<Constraint>;
  /** Loads what `saveSync()` wrote, in the same GLRMask version and with the vocab it was compiled with. */
  static loadSync(data: Uint8Array, vocab: Vocab): Constraint;
  clone(): Constraint;
  /** Frees the native handle now instead of at garbage collection. */
  dispose(): void;
  [Symbol.dispose](): void;
}

/** The state of one generated sequence. */
export declare class ConstraintState implements Disposable {
  private constructor();
  private readonly __stateBrand: void;

  /** Which tokens may come next: one bit per token ID in `Uint32Array` words (bit `i % 32` of word `Math.floor(i / 32)`). */
  mask(): Uint32Array;
  /** Like `mask()`, but writes the words into `output`. Extra words are set to 0. Throws `GLRMaskError` with code `GLRMASK_BUFFER_TOO_SMALL` if `output` has fewer than `Math.ceil(maskBitLength / 32)` elements. */
  fillMask(output: Uint32Array): void;
  /** The mask's length in bits: one per token ID, from 0. */
  readonly maskBitLength: number;
  /** If the token isn't allowed, returns `false` and the state ends. */
  commitToken(id: number): boolean;
  /** Commits tokens in order until one isn't allowed, which ends the state. Returns how many were committed. */
  commitTokens(ids: Iterable<number> | Uint32Array): number;
  /** If the bytes aren't allowed, returns `false` and the state ends. */
  commitBytes(bytes: Uint8Array): boolean;
  /** The bytes that must come next, if any. Commit them with `commitBytes()`. */
  forcedBytes(): Uint8Array;
  /** Whether the output so far is a complete match. */
  isAccepting(): boolean;
  /** Whether nothing more can be committed. */
  isTerminated(): boolean;
  clone(): ConstraintState;
  /** Frees the native handle now instead of at garbage collection. */
  dispose(): void;
  [Symbol.dispose](): void;
}

/** A preference for faster compiling or faster masks. */
export declare const MaskStrategy: Readonly<{
  /** Currently `FastMasks`, or whatever the module was precompiled with. */
  Auto: 'auto';
  /** Prefer faster compiling over faster masks. */
  FastCompile: 'fast-compile';
  /** Prefer faster masks over faster compiling. */
  FastMasks: 'fast-masks';
}>;
export type MaskStrategy = typeof MaskStrategy[keyof typeof MaskStrategy];

/** Externs by name, in name order. */
export interface Externs extends ReadonlyMap<string, Extern> {
  readonly unbound: Externs;
}

/** A named placeholder in a grammar. */
export interface Extern {
  readonly name: string;
  /** What the extern can be bound to: `'token'` or `'grammar'`. */
  readonly kind: 'token' | 'grammar' | (string & {});
}

export type KnownErrorCode = 'GLRMASK_SYNTAX' | 'GLRMASK_INVALID_INPUT' | 'GLRMASK_COMPILATION' | 'GLRMASK_RESOURCE_LIMIT' | 'GLRMASK_STRATEGY_UNSUPPORTED' | 'GLRMASK_SERIALIZATION' | 'GLRMASK_INTERNAL' | 'GLRMASK_BUFFER_TOO_SMALL' | 'GLRMASK_CLOSED';
/** Error codes may grow in a minor release. */
export type ErrorCode = KnownErrorCode | (string & {});
export declare class GLRMaskError extends Error {
  readonly code: ErrorCode;
  constructor(code: ErrorCode, message: string, options?: ErrorOptions);
}
