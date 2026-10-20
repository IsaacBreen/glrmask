'use strict';
const {types} = require('node:util');
const native = require('./loader.cjs');
const key = Symbol('GLRMask owning native handle');
const handles = new WeakMap();

class GlrMaskError extends Error {
  constructor(code, message, options) { super(message, options); this.name = 'GlrMaskError'; this.code = code; }
}
function normalize(error) {
  const match = error instanceof Error && /^(GLRMASK_[A-Z_]+): ([\s\S]*)$/.exec(error.message);
  return match ? new GlrMaskError(match[1], match[2], {cause: error}) : error;
}
function invoke(fn) { try { return fn(); } catch (error) { throw normalize(error); } }
async function invokeAsync(fn) {
  try { return await fn(); } catch (error) { throw normalize(error); }
}
function handle(value, Type) {
  if (!(value instanceof Type) || !handles.has(value)) throw new TypeError(`Expected ${Type.name}`);
  return handles.get(value);
}
function wrap(Type, value) { return new Type(key, value); }
function string(value, name = 'source') {
  if (typeof value !== 'string') throw new TypeError(`${name} must be a string`);
  return value;
}
function tokenId(value) {
  if (typeof value !== 'number' || !Number.isInteger(value) || value < 0 || value > 0xffffffff) {
    throw new RangeError('Token IDs must be integers in 0..4294967295');
  }
  return value;
}
function iterable(value, name) {
  if (value == null || typeof value[Symbol.iterator] !== 'function') throw new TypeError(`${name} must be iterable`);
  return Array.from(value);
}
function ids(value) { return iterable(value, 'token IDs').map(tokenId); }
function bytes(value) {
  if (!types.isUint8Array(value)) throw new TypeError('Expected Uint8Array or Buffer containing exact token bytes');
  return value; // Native code validates backing memory before borrowing it.
}
function checkedOptions(value, allowed) {
  if (value === undefined) return {};
  if (value === null || typeof value !== 'object' || Array.isArray(value)) throw new TypeError('Options must be an object');
  for (const name of Object.keys(value)) if (!allowed.includes(name)) throw new TypeError(`Unknown option: ${name}`);
  return value;
}
const Optimization = Object.freeze({Auto:'auto', FastBuild:'fast-build', Balanced:'balanced', FastRuntime:'fast-runtime'});
const BoundaryTriggerDetail = Object.freeze({None:'none', Tokens:'tokens', Exact:'exact'});
const LarkProfile = Object.freeze({Recognition:'recognition', Llm:'llm'});
const modes = new Map([['auto',0],['fast-build',1],['balanced',2],['fast-runtime',3]]);
const boundaries = new Map([['none',0],['tokens',1],['exact',2]]);
function boundary(value) {
  if (value === undefined) return undefined;
  if (!boundaries.has(value)) throw new TypeError('boundaryTrigger must be none, tokens or exact');
  return boundaries.get(value);
}
function buildOptions(value) {
  const o = checkedOptions(value, ['optimization','endTokens','boundaryTrigger']);
  const mode = o.optimization === undefined ? 'auto' : o.optimization;
  if (!modes.has(mode)) throw new TypeError('Unknown optimization');
  return [modes.get(mode), o.endTokens === undefined ? [] : ids(o.endTokens), boundary(o.boundaryTrigger)];
}
function prepareOptions(value) { return boundary(checkedOptions(value,['boundaryTrigger']).boundaryTrigger); }
function saveOptions(value) {
  const flag = checkedOptions(value,['externalVocab']).externalVocab;
  if (flag !== undefined && typeof flag !== 'boolean') throw new TypeError('externalVocab must be a Boolean');
  return flag === true;
}
function requirements(values) {
  return values.map(r => Object.freeze({name:r.name, path:Object.freeze(r.path.slice()), kind:r.kind,
    sourceSpan:r.sourceSpan == null ? null : Object.freeze({startByte:r.sourceSpan[0],endByte:r.sourceSpan[1],line:r.sourceSpan[2],column:r.sourceSpan[3]})}));
}
class Owned {
  constructor(secret, value) {
    if (secret !== key) throw new TypeError('Create this object through its documented factory');
    handles.set(this, value);
  }
  dispose() { invoke(() => handles.get(this).dispose()); }
  [Symbol.dispose]() { this.dispose(); }
}
class Token {
  constructor(id) { this.id = tokenId(id); Object.freeze(this); }
}
class Vocab extends Owned {
  constructor(entries, options) {
    if (entries === key) { super(key, options); return; }
    const o = checkedOptions(options,['exactOnlyIds']);
    const input = iterable(entries,'vocabulary entries');
    const tokenIds = [], tokenBytes = [];
    for (const entry of input) {
      if (!Array.isArray(entry) || entry.length !== 2) throw new TypeError('Each vocabulary entry must be [id, Uint8Array]');
      tokenIds.push(tokenId(entry[0])); tokenBytes.push(bytes(entry[1]));
    }
    const exact = o.exactOnlyIds === undefined ? [] : ids(o.exactOnlyIds);
    super(key, invoke(() => native.NativeVocab.fromEntries(tokenIds, tokenBytes, exact)));
  }
  get size() { return invoke(() => handle(this,Vocab).size); }
  clone() { return wrap(Vocab,invoke(() => handle(this,Vocab).clone())); }
  tokens(values) { return wrap(ExactTokens,invoke(() => handle(this,Vocab).tokens(ids(values)))); }
  tokenRanges(values) {
    const ranges = iterable(values,'token ranges').map(r => {
      if (r === null || typeof r !== 'object') throw new TypeError('A token range must have start and end');
      const start = tokenId(r.start), end = r.end;
      if (!Number.isInteger(end) || end <= start || end > 4294967296) throw new RangeError('Invalid half-open token range');
      return {start, end};
    });
    return wrap(ExactTokens,invoke(() => handle(this,Vocab).tokenRanges(ranges)));
  }
}
class ExactTokens extends Owned {
  get size() { return invoke(() => handle(this,ExactTokens).size); }
  get ids() { return invoke(() => handle(this,ExactTokens).ids()); }
  get ranges() { return invoke(() => handle(this,ExactTokens).ranges()); }
  clone() { return wrap(ExactTokens,invoke(() => handle(this,ExactTokens).clone())); }
}
class Grammar extends Owned {
  static fromEbnf(source) { return wrap(Grammar,invoke(() => native.NativeGrammar.fromSource('ebnf',string(source)))); }
  static fromGlrm(source) { return wrap(Grammar,invoke(() => native.NativeGrammar.fromSource('glrm',string(source)))); }
  static fromJsonSchema(schema) {
    const source = typeof schema === 'string' ? schema : JSON.stringify(schema);
    return wrap(Grammar,invoke(() => native.NativeGrammar.fromSource('json-schema',string(source,'JSON Schema'))));
  }
  static fromLark(source, options) {
    const o = checkedOptions(options,['profile','tokenNames']);
    const profile = o.profile === undefined ? 'recognition' : o.profile;
    if (profile !== 'recognition' && profile !== 'llm') throw new TypeError('Lark profile must be recognition or llm');
    const names = Object.create(null);
    if (o.tokenNames !== undefined) {
      if (o.tokenNames === null || typeof o.tokenNames !== 'object' || Array.isArray(o.tokenNames)) throw new TypeError('tokenNames must be an object');
      for (const [name,value] of Object.entries(o.tokenNames)) names[name] = typeof value === 'number' ? [tokenId(value)] : ids(value);
    }
    return wrap(Grammar,invoke(() => native.NativeGrammar.fromSource('lark',string(source),profile,names)));
  }
  static fromExpr(expr) { return wrap(Grammar,invoke(() => native.NativeGrammar.fromExpr(handle(expr,Expr)))); }
  module() { return wrap(Module,invoke(() => handle(this,Grammar).module())); }
  get unbound() { return requirements(invoke(() => handle(this,Grammar).unbound())); }
  get externs() { return requirements(invoke(() => handle(this,Grammar).externs())); }
  static exact(tokens) { return wrap(Grammar,invoke(() => native.NativeGrammar.exact(handle(tokens,ExactTokens)))); }
  get requirements() { return requirements(invoke(() => handle(this,Grammar).requirements())); }
  clone() { return wrap(Grammar,invoke(() => handle(this,Grammar).clone())); }
  bind(name, value) {
    string(name,'binding name');
    invoke(() => {
      const target = handle(this,Grammar);
      if (value instanceof Grammar) target.bindGrammar(name,handle(value,Grammar));
      else if (value instanceof Token) target.bindToken(name,tokenId(value.id));
      else if (value instanceof ExactTokens) target.bindTokens(name,handle(value,ExactTokens));
      else if (value instanceof Expr) target.bindExpr(name,handle(value,Expr));
      else throw new TypeError('A source binding requires Grammar, Expr, Token or ExactTokens');
    });
  }
  compileSync(vocab, options) {
    const args = buildOptions(options);
    return wrap(Constraint,invoke(() => handle(this,Grammar).compileSync(handle(vocab,Vocab),...args)));
  }
  async compile(vocab, options) {
    const args = buildOptions(options);
    return wrap(Constraint,await invokeAsync(() => handle(this,Grammar).compile(handle(vocab,Vocab),...args)));
  }
  prepareSync(vocab, options) {
    const b = prepareOptions(options);
    return wrap(PreparedGrammar,invoke(() => handle(this,Grammar).prepareSync(handle(vocab,Vocab),b)));
  }
  async prepare(vocab, options) {
    const b = prepareOptions(options);
    return wrap(PreparedGrammar,await invokeAsync(() => handle(this,Grammar).prepare(handle(vocab,Vocab),b)));
  }
}
class ModuleOptions {
  constructor(options) {
    const o = checkedOptions(options,['optimization','boundaryTrigger']);
    this.optimization = o.optimization === undefined ? Optimization.Auto : o.optimization;
    this.boundaryTrigger = o.boundaryTrigger === undefined ? BoundaryTriggerDetail.None : o.boundaryTrigger;
    if (!modes.has(this.optimization)) throw new TypeError('Unknown optimization');
    boundary(this.boundaryTrigger);
    Object.freeze(this);
  }
}
class CompileOptions {
  constructor(options) {
    const o = checkedOptions(options,['endTokens']);
    this.endTokens = Object.freeze(o.endTokens === undefined ? [] : ids(o.endTokens));
    Object.freeze(this);
  }
}
function localOptions(options) {
  const o = new ModuleOptions(options);
  return [modes.get(o.optimization),boundary(o.boundaryTrigger)];
}
function rootOptions(options) { return new CompileOptions(options).endTokens; }
class Expr extends Owned {
  static _make(values, operator) {
    // Materialize the iterator before borrowing any native handle. Getters and
    // generators may run JS; every native operand is an immediate snapshot.
    if (values == null || typeof values[Symbol.iterator] !== 'function') throw new TypeError('Expression operands must be iterable');
    const parts = [];
    try {
      for (const value of values) parts.push(invoke(() => {
        if (value instanceof Grammar) return native.NativeExpr.fromGrammar(handle(value,Grammar));
        if (value instanceof Module) return native.NativeExpr.fromModule(handle(value,Module));
        if (value instanceof Expr) return handle(value,Expr).clone();
        throw new TypeError('Expr operands must be Grammar, Module or Expr');
      }));
      return wrap(Expr,invoke(() => native.NativeExpr[operator](parts)));
    } finally { for (const part of parts) part.dispose(); }
  }
  static choice(values) { return Expr._make(values,'choice'); }
  static sequence(values) { return Expr._make(values,'sequence'); }
  clone() { return wrap(Expr,invoke(() => handle(this,Expr).clone())); }
}
class Module extends Owned {
  constructor(value, options) {
    if (value === key) { super(key,options); return; }
    const args = localOptions(options);
    let inner;
    if (value instanceof Grammar) inner = invoke(() => native.NativeModule.fromGrammar(handle(value,Grammar),...args));
    else if (value instanceof Expr) inner = invoke(() => native.NativeModule.fromExpr(handle(value,Expr),...args));
    else throw new TypeError('Module expects Grammar or Expr');
    super(key,inner);
  }
  get options() {
    const o = invoke(() => handle(this,Module).options);
    return new ModuleOptions({optimization:[...modes.keys()][o.optimization],boundaryTrigger:[...boundaries.keys()][o.boundaryTrigger]});
  }
  withOptions(options) { const args = localOptions(options); return wrap(Module,invoke(() => handle(this,Module).withOptions(...args))); }
  setOptions(options) { const args = localOptions(options); invoke(() => handle(this,Module).setOptions(...args)); }
  get unbound() { return requirements(invoke(() => handle(this,Module).unbound())); }
  get externs() { return requirements(invoke(() => handle(this,Module).externs())); }
  clone() { return wrap(Module,invoke(() => handle(this,Module).clone())); }
  bind(name, value) {
    string(name,'binding path');
    invoke(() => {
      const target = handle(this,Module);
      if (value instanceof Grammar) target.bindGrammar(name,handle(value,Grammar));
      else if (value instanceof Module) target.bindModule(name,handle(value,Module));
      else if (value instanceof Expr) target.bindExpr(name,handle(value,Expr));
      else if (value instanceof Token) target.bindToken(name,tokenId(value.id));
      else if (value instanceof ExactTokens) target.bindTokens(name,handle(value,ExactTokens));
      else throw new TypeError('Module.bind requires Grammar, Module, Expr, Token or ExactTokens');
    });
  }
  precompileSync(vocab) { invoke(() => handle(this,Module).precompileSync(handle(vocab,Vocab))); }
  // A concurrent mutation rejects completion instead of overwriting newer bindings.
  async precompile(vocab) { await invokeAsync(() => handle(this,Module).precompile(handle(vocab,Vocab))); }
  compileSync(vocab, options) {
    const end = rootOptions(options);
    return wrap(Constraint,invoke(() => handle(this,Module).compileSync(handle(vocab,Vocab),end)));
  }
  async compile(vocab, options) {
    const end = rootOptions(options);
    return wrap(Constraint,await invokeAsync(() => handle(this,Module).compile(handle(vocab,Vocab),end)));
  }
  saveSync() { return invoke(() => handle(this,Module).saveSync()); }
  async save() { return invokeAsync(() => handle(this,Module).save()); }
  static loadSync(data, vocab) {
    bytes(data); return wrap(Module,invoke(() => native.loadModuleSync(data,vocab === undefined ? undefined : handle(vocab,Vocab))));
  }
  static async load(data, vocab) {
    bytes(data); return wrap(Module,await invokeAsync(() => native.loadModule(data,vocab === undefined ? undefined : handle(vocab,Vocab))));
  }
}
class PreparedGrammar extends Owned {
  get requirements() { return requirements(invoke(() => handle(this,PreparedGrammar).requirements())); }
  clone() { return wrap(PreparedGrammar,invoke(() => handle(this,PreparedGrammar).clone())); }
  bind(name, value) {
    string(name,'binding name');
    invoke(() => {
      const target = handle(this,PreparedGrammar);
      if (value instanceof PreparedGrammar) target.bindPrepared(name,handle(value,PreparedGrammar));
      else if (value instanceof Token) target.bindToken(name,tokenId(value.id));
      else if (value instanceof ExactTokens) target.bindTokens(name,handle(value,ExactTokens));
      else throw new TypeError('A prepared binding requires PreparedGrammar, Token or ExactTokens');
    });
  }
  linkSync(options) {
    const args = buildOptions(options);
    return wrap(Constraint,invoke(() => handle(this,PreparedGrammar).linkSync(...args)));
  }
  async link(options) {
    const args = buildOptions(options);
    return wrap(Constraint,await invokeAsync(() => handle(this,PreparedGrammar).link(...args)));
  }
  saveSync() { return invoke(() => handle(this,PreparedGrammar).saveSync()); }
  async save() { return invokeAsync(() => handle(this,PreparedGrammar).save()); }
  static loadSync(data, vocab) {
    bytes(data);
    return wrap(PreparedGrammar,invoke(() => native.loadPreparedSync(data,vocab === undefined ? undefined : handle(vocab,Vocab))));
  }
  static async load(data, vocab) {
    bytes(data);
    return wrap(PreparedGrammar,await invokeAsync(() => native.loadPrepared(data,vocab === undefined ? undefined : handle(vocab,Vocab))));
  }
}
class Constraint extends Owned {
  get maskLength() { return invoke(() => handle(this,Constraint).maskLen); }
  clone() { return wrap(Constraint,invoke(() => handle(this,Constraint).clone())); }
  start() { return wrap(ConstraintState,invoke(() => handle(this,Constraint).start())); }
  markerIds() { return invoke(() => handle(this,Constraint).markerIds()); }
  saveSync(options) { const external = saveOptions(options); return invoke(() => handle(this,Constraint).saveSync(external)); }
  async save(options) { const external = saveOptions(options); return invokeAsync(() => handle(this,Constraint).save(external)); }
  static loadSync(data, vocab) {
    bytes(data);
    return wrap(Constraint,invoke(() => native.loadConstraintSync(data,vocab === undefined ? undefined : handle(vocab,Vocab))));
  }
  static async load(data, vocab) {
    bytes(data);
    return wrap(Constraint,await invokeAsync(() => native.loadConstraint(data,vocab === undefined ? undefined : handle(vocab,Vocab))));
  }
}
class ConstraintState extends Owned {
  get maskLength() { return invoke(() => handle(this,ConstraintState).maskLen); }
  clone() { return wrap(ConstraintState,invoke(() => handle(this,ConstraintState).clone())); }
  mask() { return invoke(() => handle(this,ConstraintState).mask()); }
  fillMask(output) {
    if (!types.isUint32Array(output)) throw new TypeError('fillMask requires a Uint32Array');
    invoke(() => handle(this,ConstraintState).fillMask(output));
  }
  commitToken(id) { tokenId(id); invoke(() => handle(this,ConstraintState).commitToken(id)); }
  commitBytes(data) { bytes(data); invoke(() => handle(this,ConstraintState).commitBytes(data)); }
  isAccepting() { return invoke(() => handle(this,ConstraintState).isAccepting()); }
  isRejected() { return invoke(() => handle(this,ConstraintState).isRejected()); }
  forcedTokens() { return invoke(() => handle(this,ConstraintState).forcedTokens()); }
  markerEvents() { return invoke(() => handle(this,ConstraintState).markerEvents()); }
}
module.exports = Object.freeze({
  Vocab, Token, ExactTokens, Grammar, Module, ModuleOptions, CompileOptions, Expr, PreparedGrammar, Constraint, ConstraintState,
  Optimization, BoundaryTriggerDetail, LarkProfile, GlrMaskError, version:native.version(),
});
