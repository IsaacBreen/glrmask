'use strict';
const {types, inspect} = require('node:util');
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
  return value;
}
function checkedOptions(value, allowed) {
  if (value === undefined) return {};
  if (value === null || typeof value !== 'object' || Array.isArray(value)) throw new TypeError('Options must be an object');
  for (const name of Object.keys(value)) if (!allowed.includes(name)) throw new TypeError(`Unknown option: ${name}`);
  return value;
}

const MaskStrategy = Object.freeze({
  Auto: 'auto',
  ParserDwa: 'parser-dwa',
  TokenTrie: 'token-trie',
  TokenClassTrie: 'token-class-trie',
});

const BoundaryMaskStrategy = Object.freeze({
  Auto: 'auto',
  ParserDwa: 'parser-dwa',
  TokenTrie: 'token-trie',
});

function parseMaskStrategy(value) {
  if (value === undefined) return 0;
  if (typeof value !== 'string') throw new TypeError('maskStrategy must be a string');
  switch (value.toLowerCase()) {
    case 'auto': return 0;
    case 'parser-dwa':
    case 'parser_dwa': return 1;
    case 'token-trie':
    case 'token_trie': return 2;
    case 'token-class-trie':
    case 'token_class_trie': return 3;
    default: throw new TypeError(`Unknown maskStrategy: ${value}`);
  }
}
function formatMaskStrategy(num) {
  switch (num) {
    case 0: return 'auto';
    case 1: return 'parser-dwa';
    case 2: return 'token-trie';
    case 3: return 'token-class-trie';
    default: return 'auto';
  }
}

function parseBoundaryMaskStrategy(value) {
  if (value === undefined) return 0;
  if (typeof value !== 'string') throw new TypeError('boundaryMaskStrategy must be a string');
  switch (value.toLowerCase()) {
    case 'auto': return 0;
    case 'parser-dwa':
    case 'parser_dwa': return 1;
    case 'token-trie':
    case 'token_trie': return 2;
    default: throw new TypeError(`Unknown boundaryMaskStrategy: ${value}`);
  }
}
function formatBoundaryMaskStrategy(num) {
  switch (num) {
    case 0: return 'auto';
    case 1: return 'parser-dwa';
    case 2: return 'token-trie';
    default: return 'auto';
  }
}

class ExternsMap {
  constructor(items, unboundFn) {
    this._items = items;
    this._map = new Map(items.map(item => [item.name, item]));
    this._unboundFn = unboundFn;
    Object.freeze(this);
  }
  get(name) { return this._map.get(name); }
  has(name) { return this._map.has(name); }
  get size() { return this._map.size; }
  keys() { return this._map.keys(); }
  values() { return this._map.values(); }
  entries() { return this._map.entries(); }
  [Symbol.iterator]() { return this._map.entries(); }
  forEach(callback, thisArg) { this._map.forEach((v, k) => callback.call(thisArg, v, k, this)); }
  get unbound() { return this._unboundFn ? this._unboundFn() : this; }
}

function formatExterns(raw, unboundSupplier) {
  const items = raw.map(e => Object.freeze({name: e.name, kind: e.kind}));
  return new ExternsMap(items, unboundSupplier);
}

class Owned {
  constructor(secret, value) {
    if (secret !== key) throw new TypeError('Create this object through its documented factory');
    handles.set(this, value);
  }
  dispose() { invoke(() => handles.get(this).dispose()); }
  [Symbol.dispose]() { this.dispose(); }
}

class Vocab extends Owned {
  constructor(entries, options) {
    if (entries === key) { super(key, options); return; }
    const o = checkedOptions(options, ['specialTokenIds']);
    const input = iterable(entries, 'vocabulary entries');
    const tokenIds = [], tokenBytes = [];
    for (const entry of input) {
      if (!Array.isArray(entry) || entry.length !== 2) throw new TypeError('Each vocabulary entry must be [id, Uint8Array]');
      tokenIds.push(tokenId(entry[0])); tokenBytes.push(bytes(entry[1]));
    }
    const special = o.specialTokenIds === undefined ? [] : ids(o.specialTokenIds);
    super(key, invoke(() => native.NativeVocab.fromEntries(tokenIds, tokenBytes, special)));
  }
  get size() { return invoke(() => handle(this, Vocab).size); }
  clone() { return wrap(Vocab, invoke(() => handle(this, Vocab).clone())); }
}

class TokenSet extends Owned {
  constructor(values) {
    if (values === key) { super(key, arguments[1]); return; }
    super(key, invoke(() => native.NativeTokenSet.fromIds(ids(values))));
  }
  static fromRanges(values) {
    const ranges = iterable(values, 'token ranges').map(r => {
      if (r === null || typeof r !== 'object') throw new TypeError('A token range must have start and end');
      const start = tokenId(r.start), end = r.end;
      if (!Number.isInteger(end) || end <= start || end > 4294967296) throw new RangeError('Invalid half-open token range');
      return {start, end};
    });
    return wrap(TokenSet, invoke(() => native.NativeTokenSet.fromRanges(ranges)));
  }
  clone() { return wrap(TokenSet, invoke(() => handle(this, TokenSet).clone())); }
  equals(other) {
    if (!(other instanceof TokenSet) || !handles.has(other)) return false;
    return invoke(() => handle(this, TokenSet).equals(handle(other, TokenSet)));
  }
  toString() { return invoke(() => handle(this, TokenSet).toString()); }
  [inspect.custom]() { return this.toString(); }
}

class Grammar extends Owned {
  static fromEbnf(source) { return wrap(Grammar, invoke(() => native.NativeGrammar.fromSource('ebnf', string(source)))); }
  static fromGlrm(source) { return wrap(Grammar, invoke(() => native.NativeGrammar.fromSource('glrm', string(source)))); }
  static fromJsonSchema(schema) {
    const source = typeof schema === 'string' ? schema : JSON.stringify(schema);
    return wrap(Grammar, invoke(() => native.NativeGrammar.fromSource('json-schema', string(source, 'JSON Schema'))));
  }
  static fromLark(source) {
    return wrap(Grammar, invoke(() => native.NativeGrammar.fromSource('lark', string(source))));
  }
  get externs() { return formatExterns(invoke(() => handle(this, Grammar).externs()), null); }
  clone() { return wrap(Grammar, invoke(() => handle(this, Grammar).clone())); }
  static fromExpr(value) {
    let expr;
    if (value instanceof Expr) expr = handle(value, Expr);
    else if (value instanceof Grammar) expr = invoke(() => native.NativeExpr.fromGrammar(handle(value, Grammar)));
    else if (value instanceof TokenSet) expr = invoke(() => native.NativeExpr.fromTokenSet(handle(value, TokenSet)));
    else throw new TypeError('Grammar.fromExpr requires Expr, Grammar, or TokenSet');
    return wrap(Grammar, invoke(() => native.NativeGrammar.fromExpr(expr)));
  }
  bind(name, value) {
    const target = handle(this, Grammar);
    if (value instanceof Grammar) return invoke(() => target.bindGrammar(name, handle(value, Grammar)));
    if (value instanceof Expr) return invoke(() => target.bindExpr(name, handle(value, Expr)));
    if (value instanceof TokenSet) return invoke(() => target.bindTokens(name, handle(value, TokenSet)));
    if (typeof value === 'number') return invoke(() => target.bindToken(name, tokenId(value)));
    throw new TypeError('Grammar.bind requires Grammar, Expr, Token or TokenSet');
  }
  precompileSync(vocab, options) {
    const s = compileSettings(options);
    return wrap(Module, invoke(() => handle(this, Grammar).precompileSync(handle(vocab, Vocab), s.mode, s.boundary)));
  }
  async precompile(vocab, options) {
    const s = compileSettings(options);
    return wrap(Module, await invokeAsync(() => handle(this, Grammar).precompile(handle(vocab, Vocab), s.mode, s.boundary)));
  }
  compileSync(vocab, options) {
    const s = compileSettings(options);
    const mod = new Module(this);
    try {
      return wrap(Constraint, invoke(() => handle(mod, Module).compileSync(handle(vocab, Vocab), s.mode, s.boundary)));
    } finally {
      mod.dispose();
    }
  }
  async compile(vocab, options) {
    const s = compileSettings(options);
    const mod = new Module(this);
    try {
      return wrap(Constraint, await invokeAsync(() => handle(mod, Module).compile(handle(vocab, Vocab), s.mode, s.boundary)));
    } finally {
      mod.dispose();
    }
  }
}

function compileSettings(options) {
  const o = checkedOptions(options, ['maskStrategy', 'boundaryMaskStrategy']);
  return {
    mode: parseMaskStrategy(o.maskStrategy),
    boundary: parseBoundaryMaskStrategy(o.boundaryMaskStrategy),
  };
}

class Expr extends Owned {
  static _make(values, operator) {
    const parts = [];
    try {
      for (const value of values) {
        if (typeof value === 'string') {
          throw new TypeError('Plain strings are not supported as Expr operands; use Expr.literal(stringOrBytes)');
        }
        parts.push(invoke(() => {
          if (value instanceof Grammar) return native.NativeExpr.fromGrammar(handle(value, Grammar));
          if (value instanceof Module) return native.NativeExpr.fromModule(handle(value, Module));
          if (value instanceof Expr) return handle(value, Expr).clone();
          if (typeof value === 'number') return native.NativeExpr.fromToken(tokenId(value));
          if (value instanceof TokenSet) return native.NativeExpr.fromTokenSet(handle(value, TokenSet));
          throw new TypeError('Expr operands must be Grammar, Module, Expr, Token or TokenSet');
        }));
      }
      return wrap(Expr, invoke(() => native.NativeExpr[operator](parts)));
    } finally {
      for (const part of parts) part.dispose();
    }
  }
  static choice(...parts) {
    return Expr._make(parts, 'choice');
  }
  static sequence(...parts) {
    return Expr._make(parts, 'sequence');
  }
  static literal(stringOrBytes) {
    let buf;
    if (typeof stringOrBytes === 'string') {
      buf = Buffer.from(stringOrBytes, 'utf8');
    } else {
      buf = bytes(stringOrBytes);
    }
    return wrap(Expr, invoke(() => native.NativeExpr.literal(buf)));
  }
  clone() { return wrap(Expr, invoke(() => handle(this, Expr).clone())); }
}

class Module extends Owned {
  constructor(value, nativeHandle) {
    if (value === key) { super(key, nativeHandle); return; }
    if (!(value instanceof Grammar)) throw new TypeError('Module expects a Grammar');
    super(key, invoke(() => native.NativeModule.fromGrammar(handle(value, Grammar))));
  }

  clone() { return wrap(Module, invoke(() => handle(this, Module).clone())); }

  _bindSingle(name, value) {
    string(name, 'binding name');
    if (name.includes('::')) throw new GlrMaskError('GLRMASK_ENGINE_ERROR', 'qualified paths are not supported');
    invoke(() => {
      const target = handle(this, Module);
      if (value instanceof Grammar) target.bindGrammar(name, handle(value, Grammar));
      else if (value instanceof Module) {
        if (value.externs.unbound.size > 0) {
          const names = Array.from(value.externs.unbound.keys()).map(n => `"${n}"`).join(', ');
          throw new GlrMaskError('GLRMASK_ENGINE_ERROR', `cannot bind "${name}": the value has unbound externs [${names}]`);
        }
        target.bindModule(name, handle(value, Module));
      }
      else if (value instanceof Expr) target.bindExpr(name, handle(value, Expr));
      else if (typeof value === 'number') target.bindToken(name, tokenId(value));
      else if (value instanceof TokenSet) target.bindTokens(name, handle(value, TokenSet));
      else throw new TypeError('Module.bind requires Grammar, Module, Expr, Token or TokenSet');
    });
  }

  bind(nameOrBindings, value) {
    if (arguments.length === 1 && typeof nameOrBindings === 'object' && nameOrBindings !== null && !(nameOrBindings instanceof String)) {
      const entries = Object.entries(nameOrBindings);
      const snapshot = this.clone();
      for (const [k, v] of entries) {
        snapshot._bindSingle(k, v);
      }
      handles.set(this, handles.get(snapshot));
      return;
    }
    this._bindSingle(nameOrBindings, value);
  }

  get externs() {
    const target = handle(this, Module);
    const rawAll = invoke(() => target.externs());
    const unboundSupplier = () => {
      const rawUnbound = invoke(() => target.unboundExterns());
      return formatExterns(rawUnbound, null);
    };
    return formatExterns(rawAll, unboundSupplier);
  }

  precompileSync(vocab, options) {
    const s = compileSettings(options);
    invoke(() => handle(this, Module).precompileSync(handle(vocab, Vocab), s.mode, s.boundary));
  }
  async precompile(vocab, options) {
    const s = compileSettings(options);
    await invokeAsync(() => handle(this, Module).precompile(handle(vocab, Vocab), s.mode, s.boundary));
  }

  compileSync(vocab, options) {
    const s = compileSettings(options);
    return wrap(Constraint, invoke(() => handle(this, Module).compileSync(handle(vocab, Vocab), s.mode, s.boundary)));
  }
  async compile(vocab, options) {
    const s = compileSettings(options);
    return wrap(Constraint, await invokeAsync(() => handle(this, Module).compile(handle(vocab, Vocab), s.mode, s.boundary)));
  }

  saveSync() { return invoke(() => handle(this, Module).saveSync()); }
  async save() { return invokeAsync(() => handle(this, Module).save()); }

  static loadSync(data) {
    bytes(data); return wrap(Module, invoke(() => native.loadModuleSync(data)));
  }
  static async load(data) {
    bytes(data); return wrap(Module, await invokeAsync(() => native.loadModule(data)));
  }
}

class Constraint extends Owned {
  get maskLength() { return invoke(() => handle(this, Constraint).maskLen); }
  clone() { return wrap(Constraint, invoke(() => handle(this, Constraint).clone())); }
  start() { return wrap(ConstraintState, invoke(() => handle(this, Constraint).start())); }
  markerIds() { return invoke(() => handle(this, Constraint).markerIds()); }
  saveSync() { return invoke(() => handle(this, Constraint).saveSync()); }
  async save() { return invokeAsync(() => handle(this, Constraint).save()); }
  static loadSync(data, vocab) {
    bytes(data); return wrap(Constraint, invoke(() => native.loadConstraintSync(data, handle(vocab, Vocab))));
  }
  static async load(data, vocab) {
    bytes(data); const copy = Uint8Array.from(data);
    return wrap(Constraint, await invokeAsync(() => native.loadConstraint(copy, handle(vocab, Vocab))));
  }
}

class ConstraintState extends Owned {
  get maskLength() { return invoke(() => handle(this, ConstraintState).maskLen); }
  clone() { return wrap(ConstraintState, invoke(() => handle(this, ConstraintState).clone())); }
  mask() { return invoke(() => handle(this, ConstraintState).mask()); }
  fillMask(output) {
    if (!types.isUint32Array(output)) throw new TypeError('fillMask requires a Uint32Array');
    invoke(() => handle(this, ConstraintState).fillMask(output));
  }
  commitToken(id) { tokenId(id); return invoke(() => handle(this, ConstraintState).commitToken(id)); }
  commitTokens(ids) {
    const words = ids instanceof Uint32Array ? ids : Uint32Array.from(ids, tokenId);
    return invoke(() => handle(this, ConstraintState).commitTokens(words));
  }
  commitBytes(data) { bytes(data); return invoke(() => handle(this, ConstraintState).commitBytes(data)); }
  isAccepting() { return invoke(() => handle(this, ConstraintState).isAccepting()); }
  forcedTokens() { return invoke(() => handle(this, ConstraintState).forcedTokens()); }
  markerEvents() { return invoke(() => handle(this, ConstraintState).markerEvents()); }
}

module.exports = Object.freeze({
  Vocab, TokenSet, Grammar, Module, Expr, Constraint, ConstraintState,
  MaskStrategy, BoundaryMaskStrategy, GlrMaskError, version: native.version(),
});
