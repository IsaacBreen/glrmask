/// <reference lib="esnext.disposable" />
/** Native GLRMask bindings for Node.js. Masks use packed original-token-ID u32 words. */
export declare const version: string;

export declare const MaskStrategy: Readonly<{
  Auto: 'auto';
  ParserDwa: 'parser-dwa';
  TokenTrie: 'token-trie';
  TokenClassTrie: 'token-class-trie';
  FastBuild: 'token-trie';
  FastRuntime: 'parser-dwa';
}>;
export type MaskStrategy = typeof MaskStrategy[keyof typeof MaskStrategy];

export declare const BoundaryMaskStrategy: Readonly<{
  FastBuild: 'fast-build';
  FastRuntime: 'fast-runtime';
}>;
export type BoundaryMaskStrategy = typeof BoundaryMaskStrategy[keyof typeof BoundaryMaskStrategy];

export declare const LarkProfile: Readonly<{Recognition: 'recognition'; Llm: 'llm'}>;
export type LarkProfile = typeof LarkProfile[keyof typeof LarkProfile];

export type ErrorCode = 'GLRMASK_INVALID_ARGUMENT' | 'GLRMASK_ENGINE_ERROR' | 'GLRMASK_PANIC' | 'GLRMASK_BUFFER_TOO_SMALL' | 'GLRMASK_CLOSED';
export declare class GlrMaskError extends Error {
  readonly code: ErrorCode;
  constructor(code: ErrorCode, message: string, options?: ErrorOptions);
}

/** Half-open interval. end can be 4294967296 to include token ID 4294967295. */
export interface TokenRange { readonly start: number; readonly end: number; }
export interface SourceSpan {
  readonly startByte: number; readonly endByte: number;
  /** One-based line and Unicode-scalar column. */
  readonly line: number; readonly column: number;
}
export interface Extern {
  readonly name: string;
  readonly kind: 'token' | 'grammar';
  readonly span: SourceSpan | null;
}
export interface Externs extends ReadonlyMap<string, Extern> {
  readonly unbound: Externs;
}

export interface MarkerEvent { readonly id: number; readonly byteOffset: bigint; }
export interface VocabOptions { exactOnlyIds?: Iterable<number>; }
export interface SaveOptions { externalVocab?: boolean; }
export interface LarkOptions { profile?: LarkProfile; tokenNames?: Record<string, number | Iterable<number>>; }

/** A raw original ID without vocabulary identity; use vocab.tokens([id]) for a bound singleton. */
export declare class Token { readonly id: number; constructor(id: number); }

/** Exact tokenizer bytes, including NUL/partial UTF-8. Original IDs may be sparse. */
export declare class Vocab implements Disposable {
  private readonly __vocabBrand: void;
  constructor(entries: Iterable<readonly [number, Uint8Array]>, options?: VocabOptions);
  readonly size: number;
  tokens(ids: Iterable<number>): ExactTokens;
  tokenRanges(ranges: Iterable<TokenRange>): ExactTokens;
  copy(): Vocab;
  clone(): Vocab;
  dispose(): void;
  [Symbol.dispose](): void;
}

export declare class ExactTokens implements Disposable {
  private constructor();
  private readonly __exactTokensBrand: void;
  readonly size: number;
  readonly ids: Uint32Array;
  readonly ranges: TokenRange[];
  copy(): ExactTokens;
  clone(): ExactTokens;
  dispose(): void;
  [Symbol.dispose](): void;
}

export interface GrammarCompileOptions {
  endTokens?: Iterable<number>;
  maskStrategy?: MaskStrategy;
  boundaryMaskStrategy?: BoundaryMaskStrategy;
}

export declare class Grammar implements Disposable {
  private constructor();
  private readonly __grammarBrand: void;
  static fromEbnf(source: string): Grammar;
  static fromGlrm(source: string): Grammar;
  static fromJsonSchema(schema: unknown): Grammar;
  static fromLark(source: string, options?: LarkOptions): Grammar;
  copy(): Grammar;
  compile(vocab: Vocab, options?: GrammarCompileOptions): Promise<Constraint>;
  compileSync(vocab: Vocab, options?: GrammarCompileOptions): Constraint;
  dispose(): void;
  [Symbol.dispose](): void;
}

export interface ModuleSettings {
  maskStrategy?: MaskStrategy;
  boundaryMaskStrategy?: BoundaryMaskStrategy;
}
export interface RootSettings {
  endTokens?: Iterable<number>;
}
export declare class CompileOptions {
  constructor(options?: RootSettings);
  readonly endTokens: readonly number[];
}

export type ExprPart = Grammar | Module | Expr | Token | ExactTokens;

export declare class Expr implements Disposable {
  private constructor();
  private readonly __exprBrand: void;
  static choice(...parts: ExprPart[]): Expr;
  static sequence(...parts: ExprPart[]): Expr;
  static literal(value: string | Uint8Array): Expr;
  copy(): Expr;
  dispose(): void;
  [Symbol.dispose](): void;
}

export declare class Module implements Disposable {
  private readonly __moduleBrand: void;
  constructor(grammar: Grammar, options?: ModuleSettings);
  maskStrategy: MaskStrategy;
  boundaryMaskStrategy: BoundaryMaskStrategy;
  readonly externs: Externs;
  copy(): Module;
  bind(name: string, value: ExprPart): void;
  bind(bindings: Record<string, ExprPart>): void;
  precompile(vocab: Vocab): Promise<void>;
  precompileSync(vocab: Vocab): void;
  compile(vocab: Vocab, options?: RootSettings | CompileOptions): Promise<Constraint>;
  compileSync(vocab: Vocab, options?: RootSettings | CompileOptions): Constraint;
  save(): Promise<Uint8Array>;
  saveSync(): Uint8Array;
  static load(data: Uint8Array, vocab?: Vocab): Promise<Module>;
  static loadSync(data: Uint8Array, vocab?: Vocab): Module;
  dispose(): void;
  [Symbol.dispose](): void;
}

export declare class Constraint implements Disposable {
  private constructor();
  private readonly __constraintBrand: void;
  readonly maskLength: number;
  copy(): Constraint;
  clone(): Constraint;
  start(): ConstraintState;
  markerIds(): Uint32Array;
  save(options?: SaveOptions): Promise<Uint8Array>;
  saveSync(options?: SaveOptions): Uint8Array;
  static load(data: Uint8Array, vocab?: Vocab): Promise<Constraint>;
  static loadSync(data: Uint8Array, vocab?: Vocab): Constraint;
  dispose(): void;
  [Symbol.dispose](): void;
}

export declare class ConstraintState implements Disposable {
  private constructor();
  private readonly __stateBrand: void;
  readonly maskLength: number;
  copy(): ConstraintState;
  clone(): ConstraintState;
  mask(): Uint32Array;
  fillMask(output: Uint32Array): void;
  commitToken(id: number): void;
  commitBytes(bytes: Uint8Array): void;
  isAccepting(): boolean;
  isRejected(): boolean;
  forcedTokens(): Uint32Array;
  markerEvents(): MarkerEvent[];
  dispose(): void;
  [Symbol.dispose](): void;
}
