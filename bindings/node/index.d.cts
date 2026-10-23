/// <reference lib="esnext.disposable" />
/** GLRMask for Node.js. */
export declare const version: string;

/** How masks are computed, trading compile time against mask speed. */
export declare const MaskStrategy: Readonly<{
  /** `ParserDwa`, unless the module was precompiled with `TokenClassTrie`. The default. */
  Auto: 'auto';
  /** Precomputes masks: slowest compile, fastest masks. */
  ParserDwa: 'parser-dwa';
  /** Computes each mask by walking the vocabulary: fastest compile, slower masks. */
  TokenTrie: 'token-trie';
  /** Like `TokenTrie`, but treats tokens the grammar cannot tell apart as one. */
  TokenClassTrie: 'token-class-trie';
}>;
/** A `MaskStrategy` value. */
export type MaskStrategy = typeof MaskStrategy[keyof typeof MaskStrategy];

/** How masks are computed where one grammar meets another. */
export declare const BoundaryMaskStrategy: Readonly<{
  /** `ParserDwa`, falling back to `TokenTrie` when that would be too large. The default. */
  Auto: 'auto';
  /** Precomputes boundary masks: slower compile, faster masks.
   *
   * Throws `GlrMaskError` if they would be too large. */
  ParserDwa: 'parser-dwa';
  /** Computes boundary masks as needed: faster compile, slower masks. */
  TokenTrie: 'token-trie';
}>;
/** A `BoundaryMaskStrategy` value. */
export type BoundaryMaskStrategy = typeof BoundaryMaskStrategy[keyof typeof BoundaryMaskStrategy];

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
/** An extern declared by a grammar: a named slot to bind later. */
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
  /** Only the externs that are not yet bound. */
  readonly unbound: Externs;
}

/** A marker passed by a commit. */
export interface MarkerEvent {
  /** The marker's ID. */
  readonly id: number;
  /** Where the marker is, in bytes from the start of the sequence. */
  readonly byteOffset: bigint;
}
/** Options for `new Vocab`. */
export interface VocabOptions {
  /** Tokens that have no bytes, such as end-of-sequence or tool-call tokens. They can be used in token sets but never match text. */
  specialTokenIds?: Iterable<number>;
}
/** A model's vocabulary: each token ID and its bytes.
 *
 * Use each token's exact bytes, not its display form: `b" hello"`, not `"Ġhello"`. IDs need not be contiguous. */
export declare class Vocab implements Disposable {
  private readonly __vocabBrand: void;
  /** Creates a vocabulary from `[id, bytes]` entries. */
  constructor(entries: Iterable<readonly [number, Uint8Array]>, options?: VocabOptions);
  /** The number of tokens with bytes; special tokens are not counted. */
  readonly size: number;
  /** A new handle to the same object. */
  clone(): Vocab;
  /** Frees the native handle now instead of at garbage collection. */
  dispose(): void;
  /** Same as `dispose()`, for `using` declarations. */
  [Symbol.dispose](): void;
}

/** A set of token IDs.
 *
 * In a grammar, it matches any one of its tokens. Every ID must be in the vocabulary. */
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

/** Options for compiling a grammar or module.
 *
 * They apply to every part of the compile, including bound grammars. */
export interface CompileOptions {
  /** How masks are computed. Defaults to `MaskStrategy.Auto`.
   *
   * Precompiled parts keep their precompiled masks. */
  maskStrategy?: MaskStrategy;
  /** How masks are computed where one grammar meets another. Defaults to `BoundaryMaskStrategy.Auto`. */
  boundaryMaskStrategy?: BoundaryMaskStrategy;
}

/** A grammar, written as source text or built from an `Expr`. */
export declare class Grammar implements Disposable {
  private constructor();
  private readonly __grammarBrand: void;
  /** Creates a grammar from EBNF source. */
  static fromEbnf(source: string): Grammar;
  /** Creates a grammar from GLRM, GLRMask's native format. */
  static fromGlrm(source: string): Grammar;
  /** Creates a grammar from a JSON Schema, given as JSON text or an object.
   *
   * Standard keywords that GLRMask does not support are an error. Unknown keywords and `format` are ignored. */
  static fromJsonSchema(schema: unknown): Grammar;
  /** Creates a grammar from Lark source.
   *
   * A token name such as `<|end|>` declares a token extern, which must be bound before compiling. */
  static fromLark(source: string): Grammar;
  /** The grammar's externs, in declaration order. */
  readonly externs: Externs;
  /** A new handle to the same object. */
  clone(): Grammar;
  /** Creates a grammar from an expression.
   *
   * Use it to require an end token after a grammar: `Grammar.fromExpr(Expr.sequence(grammar, eosId))`.
   *
   * The end token must be in the vocabulary. */
  static fromExpr(expr: Expr | Grammar | TokenSet): Grammar;
  /** Binds the extern `name` to `value`: a grammar, expression, token set or token ID.
   *
   * Throws if `name` is undeclared or already bound, or if `value` is the wrong kind for it. */
  bind(name: string, value: ExprPart): void;
  /** Like `precompileSync`, but returns a promise and doesn't block the event loop. */
  precompile(vocab: Vocab, options?: CompileOptions): Promise<Module>;
  /** Compiles as much as possible for `vocab` now, and returns a `Module` that keeps the work.
   *
   * Externs may still be unbound. */
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

/** A choice or sequence of grammars, modules, token sets and literals.
 *
 * Each grammar keeps its own rules, including which whitespace it ignores. */
export declare class Expr implements Disposable {
  private constructor();
  private readonly __exprBrand: void;
  /** Matches any one of `parts`.
   *
   * Throws if a part has unbound externs. */
  static choice(...parts: ExprPart[]): Expr;
  /** Matches `parts` one after another.
   *
   * One token may span several parts. Throws if a part has unbound externs. */
  static sequence(...parts: ExprPart[]): Expr;
  /** Matches exactly `value`, with no whitespace skipped. A string is UTF-8 encoded. */
  static literal(value: string | Uint8Array): Expr;
  /** A new handle to the same object. */
  clone(): Expr;
  /** Frees the native handle now instead of at garbage collection. */
  dispose(): void;
  /** Same as `dispose()`, for `using` declarations. */
  [Symbol.dispose](): void;
}

/** A grammar plus its extern bindings and any precompiled work.
 *
 * Externs can stay unbound until `compile`. */
export declare class Module implements Disposable {
  private readonly __moduleBrand: void;
  /** Creates a module from `grammar`. */
  constructor(grammar: Grammar);
  /** The module's externs, in declaration order. */
  readonly externs: Externs;
  /** A copy of the module. Binding the copy does not change the original. */
  clone(): Module;
  /** Binds the extern `name` to `value`: a module, grammar, expression, token set or token ID.
   *
   * Throws if `name` is undeclared or already bound, or if `value` is the wrong kind for it. A token extern takes only a token set or token ID. */
  bind(name: string, value: ExprPart): void;
  /** Binds several externs at once; if one fails, none is bound. */
  bind(bindings: Record<string, ExprPart>): void;
  /** Like `precompileSync`, but returns a promise and doesn't block the event loop. */
  precompile(vocab: Vocab, options?: CompileOptions): Promise<void>;
  /** Compiles as much as possible for `vocab` now, so later compiles are fast.
   *
   * Externs may still be unbound. The work is kept in the module and included by `save`. */
  precompileSync(vocab: Vocab, options?: CompileOptions): void;
  /** Like `compileSync`, but returns a promise and doesn't block the event loop. */
  compile(vocab: Vocab, options?: CompileOptions): Promise<Constraint>;
  /** Compiles the module for `vocab`, reusing any precompiled work.
   *
   * Every extern must be bound. */
  compileSync(vocab: Vocab, options?: CompileOptions): Constraint;
  /** Like `saveSync`, but returns a promise and doesn't block the event loop. */
  save(): Promise<Uint8Array>;
  /** Saves the module, including precompiled work, as bytes.
   *
   * The vocabulary is not included. Bytes saved by one GLRMask version may not load in another. */
  saveSync(): Uint8Array;
  /** Like `loadSync`, but returns a promise and doesn't block the event loop. */
  static load(data: Uint8Array): Promise<Module>;
  /** Loads a module saved with `save`.
   *
   * Precompiled work applies only to the vocabulary it was built for; compiling with a different one is an error. */
  static loadSync(data: Uint8Array): Module;
  /** Frees the native handle now instead of at garbage collection. */
  dispose(): void;
  /** Same as `dispose()`, for `using` declarations. */
  [Symbol.dispose](): void;
}

/** A grammar compiled for a vocabulary, ready to constrain generation.
 *
 * Share one between threads, and call `start` for each sequence. */
export declare class Constraint implements Disposable {
  private constructor();
  private readonly __constraintBrand: void;
  /** The number of `u32` words in a mask. */
  readonly maskLength: number;
  /** A new handle to the same object. */
  clone(): Constraint;
  /** Starts a new sequence. */
  start(): ConstraintState;
  /** The IDs of the grammar's markers, in ascending order. */
  markerIds(): Uint32Array;
  /** Like `saveSync`, but returns a promise and doesn't block the event loop. */
  save(): Promise<Uint8Array>;
  /** Saves the constraint as bytes, without the vocabulary.
   *
   * Bytes saved by one GLRMask version may not load in another. */
  saveSync(): Uint8Array;
  /** Like `loadSync`, but returns a promise and doesn't block the event loop. */
  static load(data: Uint8Array, vocab: Vocab): Promise<Constraint>;
  /** Loads a constraint saved with `save`.
   *
   * `vocab` must be the vocabulary it was compiled for; a different one is an error. */
  static loadSync(data: Uint8Array, vocab: Vocab): Constraint;
  /** Frees the native handle now instead of at garbage collection. */
  dispose(): void;
  /** Same as `dispose()`, for `using` declarations. */
  [Symbol.dispose](): void;
}

/** The state of one generated sequence.
 *
 * Clone it to try tokens without changing the original. */
export declare class ConstraintState implements Disposable {
  private constructor();
  private readonly __stateBrand: void;
  /** The number of `u32` words in a mask. */
  readonly maskLength: number;
  /** A copy of the state, for trying tokens without changing this one. */
  clone(): ConstraintState;
  /** The tokens allowed next, as a bitset.
   *
   * Token `id` is allowed when bit `id % 32` of element `Math.floor(id / 32)` is set. No bits are set when nothing can follow: the sequence is complete, or a commit failed. */
  mask(): Uint32Array;
  /** Writes the mask into `output`, setting any extra elements to zero.
   *
   * Throws `GlrMaskError` with code `GLRMASK_BUFFER_TOO_SMALL` if it has fewer than `maskLength` elements. */
  fillMask(output: Uint32Array): void;
  /** Commits one token.
   *
   * Returns `false` if it is not allowed, leaving the state unusable. To try a token safely, commit it on a clone. */
  commitToken(id: number): boolean;
  /** Commits the tokens in order.
   *
   * Returns `false` at the first token that is not allowed, leaving the state unusable. */
  commitTokens(ids: Iterable<number> | Uint32Array): boolean;
  /** Commits raw bytes instead of a token.
   *
   * Returns `false` if they are not allowed, leaving the state unusable. Bytes never match a token set; commit those tokens with `commitToken`. */
  commitBytes(bytes: Uint8Array): boolean;
  /** Whether the input committed so far is a complete match for the grammar.
   *
   * More input may still be allowed; see {@link isTerminated}. */
  isAccepting(): boolean;
  /** Whether nothing more can be committed.
   *
   * This is true when the sequence is finished, for example after its end token, and after a failed commit. `isAccepting` tells the two apart.
   *
   * A complete match is not terminated if more input could follow, such as whitespace the grammar ignores. */
  isTerminated(): boolean;
  /** The tokens that must come next, without committing them.
   *
   * Stops where more than one token could follow, or after 10,000 tokens or bytes. When forced text could be tokenized in several ways, the longest token is chosen at each step. */
  forcedTokens(): Uint32Array;
  /** The markers passed by the most recent commit, or by `start`, in byte order.
   *
   * A marker means some parse reached that point. That parse may still fail later. */
  markerEvents(): MarkerEvent[];
  /** Frees the native handle now instead of at garbage collection. */
  dispose(): void;
  /** Same as `dispose()`, for `using` declarations. */
  [Symbol.dispose](): void;
}
