/// <reference lib="esnext.disposable" />
/** Native GLRMask bindings for Node.js. Masks use packed original-token-ID u32 words. */
export declare const version: string;
export declare const Optimization: Readonly<{
  Auto: 'auto'; FastBuild: 'fast-build'; Balanced: 'balanced'; FastRuntime: 'fast-runtime';
}>;
export type Optimization = typeof Optimization[keyof typeof Optimization];
export declare const BoundaryTriggerDetail: Readonly<{None:'none'; Tokens:'tokens'; Exact:'exact'}>;
export type BoundaryTriggerDetail = typeof BoundaryTriggerDetail[keyof typeof BoundaryTriggerDetail];
export declare const LarkProfile: Readonly<{Recognition:'recognition'; Llm:'llm'}>;
export type LarkProfile = typeof LarkProfile[keyof typeof LarkProfile];
export type ErrorCode = 'GLRMASK_INVALID_ARGUMENT' | 'GLRMASK_ENGINE_ERROR' | 'GLRMASK_PANIC' | 'GLRMASK_BUFFER_TOO_SMALL' | 'GLRMASK_CLOSED';
export declare class GlrMaskError extends Error {
  readonly code: ErrorCode;
  constructor(code: ErrorCode, message: string, options?: ErrorOptions);
}
/** Half-open interval. end can be 4294967296 to include token ID 4294967295. */
export interface TokenRange {readonly start:number; readonly end:number}
export interface SourceSpan {
  readonly startByte:number; readonly endByte:number;
  /** One-based line and Unicode-scalar column. */
  readonly line:number; readonly column:number;
}
export interface Requirement {
  readonly name:string; readonly path:readonly string[];
  readonly kind:'token'|'grammar'; readonly sourceSpan:SourceSpan|null;
}
export interface MarkerEvent {readonly id:number; readonly byteOffset:bigint}
export interface VocabOptions {exactOnlyIds?:Iterable<number>}
export interface PrepareOptions {boundaryTrigger?:BoundaryTriggerDetail}
export interface BuildOptions extends PrepareOptions {optimization?:Optimization; endTokens?:Iterable<number>}
export interface SaveOptions {externalVocab?:boolean}
export interface LarkOptions {profile?:LarkProfile; tokenNames?:Record<string,number|Iterable<number>>}
/** A raw original ID without vocabulary identity; use vocab.tokens([id]) for a bound singleton. */
export declare class Token {readonly id:number; constructor(id:number)}
/** Exact tokenizer bytes, including NUL/partial UTF-8. Original IDs may be sparse. */
export declare class Vocab implements Disposable {
  private readonly __vocabBrand: void;
  constructor(entries:Iterable<readonly [number,Uint8Array]>, options?:VocabOptions);
  /** Byte-backed entry count, excluding exact-only controls. */
  readonly size:number;
  tokens(ids:Iterable<number>):ExactTokens;
  /** Overlapping/adjacent ranges merge without expanding all IDs. */
  tokenRanges(ranges:Iterable<TokenRange>):ExactTokens;
  clone():Vocab;
  dispose():void;
  [Symbol.dispose]():void;
}
/** One vocabulary-bound token event selected from a compact set. Empty matches nothing. */
export declare class ExactTokens implements Disposable {
  private constructor();
  private readonly __exactTokensBrand: void;
  readonly size:number;
  /** Materializes IDs; inspect ranges to keep large sets compact. */
  readonly ids:Uint32Array;
  readonly ranges:TokenRange[];
  clone():ExactTokens;
  dispose():void;
  [Symbol.dispose]():void;
}
/** Mutable source bindings. Attached children and clones retain independent snapshots. */
export declare class Grammar implements Disposable {
  private constructor();
  private readonly __grammarBrand: void;
  static fromEbnf(source:string):Grammar;
  static fromGlrm(source:string):Grammar;
  /** Accepts JSON text or JSON-serializable data. The core importer is not a full JSON Schema validator. */
  static fromJsonSchema(schema:unknown):Grammar;
  static fromLark(source:string,options?:LarkOptions):Grammar;
  static exact(tokens:ExactTokens):Grammar;
  readonly requirements:Requirement[];
  /** Mutates this handle transactionally; qualified paths use ::. */
  bind(name:string,value:Grammar|Token|ExactTokens):void;
  clone():Grammar;
  /** Snapshots inputs immediately and compiles on a native worker, without blocking the JS event loop. */
  compile(vocab:Vocab,options?:BuildOptions):Promise<Constraint>;
  /** Explicitly blocks the caller; use compile() in request-handling paths. */
  compileSync(vocab:Vocab,options?:BuildOptions):Constraint;
  prepare(vocab:Vocab,options?:PrepareOptions):Promise<PreparedGrammar>;
  prepareSync(vocab:Vocab,options?:PrepareOptions):PreparedGrammar;
  dispose():void;
  [Symbol.dispose]():void;
}
/** Reusable compiled machinery that can retain open slots. */
export declare class PreparedGrammar implements Disposable {
  private constructor();
  private readonly __preparedGrammarBrand: void;
  readonly requirements:Requirement[];
  bind(name:string,value:PreparedGrammar|Token|ExactTokens):void;
  clone():PreparedGrammar;
  link(options?:BuildOptions):Promise<Constraint>;
  linkSync(options?:BuildOptions):Constraint;
  save():Promise<Uint8Array>;
  saveSync():Uint8Array;
  static load(data:Uint8Array,vocab?:Vocab):Promise<PreparedGrammar>;
  static loadSync(data:Uint8Array,vocab?:Vocab):PreparedGrammar;
  dispose():void;
  [Symbol.dispose]():void;
}
/** Immutable runnable constraint. States and queued tasks keep their native owners alive. */
export declare class Constraint implements Disposable {
  private constructor();
  private readonly __constraintBrand: void;
  /** Number of packed uint32 words, not the vocabulary size. */
  readonly maskLength:number;
  clone():Constraint;
  start():ConstraintState;
  markerIds():Uint32Array;
  save(options?:SaveOptions):Promise<Uint8Array>;
  saveSync(options?:SaveOptions):Uint8Array;
  static load(data:Uint8Array,vocab?:Vocab):Promise<Constraint>;
  static loadSync(data:Uint8Array,vocab?:Vocab):Constraint;
  dispose():void;
  [Symbol.dispose]():void;
}
/** One generated sequence. All decoding methods are synchronous native operations. */
export declare class ConstraintState implements Disposable {
  private constructor();
  private readonly __stateBrand: void;
  readonly maskLength:number;
  clone():ConstraintState;
  /** A newly allocated packed mask; bit (id % 32) of word floor(id / 32). */
  mask():Uint32Array;
  /** Fills caller memory directly and zeroes extra words. Shared/detached backing is rejected. */
  fillMask(output:Uint32Array):void;
  commitToken(id:number):void;
  /** Byte input does not stand in for exact-token or end-token events. */
  commitBytes(bytes:Uint8Array):void;
  /** Current prefix is a complete match; more tokens may still be accepted. */
  isAccepting():boolean;
  isRejected():boolean;
  /** Read-only lookahead, bounded by the core; no tokens are committed. */
  forcedTokens():Uint32Array;
  /** Initial or latest commit's possible marker observations, not final parse captures. */
  markerEvents():MarkerEvent[];
  dispose():void;
  [Symbol.dispose]():void;
}
