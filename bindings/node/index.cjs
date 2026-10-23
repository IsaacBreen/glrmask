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
  FastBuild: 'token-trie',
  FastRuntime: 'parser-dwa',
});

const BoundaryMaskStrategy = Object.freeze({
  Auto: 'auto',
  ParserDwa: 'parser-dwa',
  TokenTrie: 'token-trie',
  FastBuild: 'token-trie',
  FastRuntime: 'parser-dwa',
});

const LarkProfile = Object.freeze({Recognition: 'recognition', Llm: 'llm'});

function parseMaskStrategy(value) {
  if (value === undefined) return 0;
  if (typeof value !== 'string') throw new TypeError('maskStrategy must be a string');
  switch (value.toLowerCase()) {
    case 'auto': return 0;
    case 'parser-dwa':
    case 'parser_dwa':
    case 'fast-runtime':
    case 'fast_runtime': return 1;
    case 'token-trie':
    case 'token_trie':
    case 'fast-build':
    case 'fast_build': return 2;
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
    case 'parser_dwa':
    case 'fast-runtime':
    case 'fast_runtime': return 1;
    case 'token-trie':
    case 'token_trie':
    case 'fast-build':
    case 'fast_build': return 2;
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
  forEach(callback, thisArg) { this._map.forEach(callback, thisArg); }
  get unbound() { return this._unboundFn ? this._unboundFn() : this; }
}

function formatExterns(raw, unboundSupplier) {
  const items = raw.map(e => Object.freeze({
    name: e.name,
    kind: e.kind,
    span: e.sourceSpan ? Object.freeze({
      startByte: e.sourceSpan[0],
      endByte: e.sourceSpan[1],
      line: e.sourceSpan[2],
      column: e.sourceSpan[3],
    }) : null,
  }));
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

class Token {
  constructor(id) { this.id = tokenId(id); Object.freeze(this); }
}

class Vocab extends Owned {
  constructor(entries, options) {
    if (entries === key) { super(key, options); return; }
    const o = checkedOptions(options, ['exactOnlyIds']);
    const input = iterable(entries, 'vocabulary entries');
    const tokenIds = [], tokenBytes = [];
    for (const entry of input) {
      if (!Array.isArray(entry) || entry.length !== 2) throw new TypeError('Each vocabulary entry must be [id, Uint8Array]');
      tokenIds.push(tokenId(entry[0])); tokenBytes.push(bytes(entry[1]));
    }
    const exact = o.exactOnlyIds === undefined ? [] : ids(o.exactOnlyIds);
    super(key, invoke(() => native.NativeVocab.fromEntries(tokenIds, tokenBytes, exact)));
  }
  get size() { return invoke(() => handle(this, Vocab).size); }
  copy() { return wrap(Vocab, invoke(() => handle(this, Vocab).clone())); }
  clone() { return this.copy(); }
  tokens(values) { return wrap(ExactTokens, invoke(() => handle(this, Vocab).tokens(ids(values)))); }
  tokenRanges(values) {
    const ranges = iterable(values, 'token ranges').map(r => {
      if (r === null || typeof r !== 'object') throw new TypeError('A token range must have start and end');
      const start = tokenId(r.start), end = r.end;
      if (!Number.isInteger(end) || end <= start || end > 4294967296) throw new RangeError('Invalid half-open token range');
      return {start, end};
    });
    return wrap(ExactTokens, invoke(() => handle(this, Vocab).tokenRanges(ranges)));
  }
}

class ExactTokens extends Owned {
  get size() { return invoke(() => handle(this, ExactTokens).size); }
  get ids() { return invoke(() => handle(this, ExactTokens).ids()); }
  get ranges() { return invoke(() => handle(this, ExactTokens).ranges()); }
  copy() { return wrap(ExactTokens, invoke(() => handle(this, ExactTokens).clone())); }
  clone() { return this.copy(); }
}

class Grammar extends Owned {
  static fromEbnf(source) { return wrap(Grammar, invoke(() => native.NativeGrammar.fromSource('ebnf', string(source)))); }
  static fromGlrm(source) { return wrap(Grammar, invoke(() => native.NativeGrammar.fromSource('glrm', string(source)))); }
  static fromJsonSchema(schema) {
    const source = typeof schema === 'string' ? schema : JSON.stringify(schema);
    return wrap(Grammar, invoke(() => native.NativeGrammar.fromSource('json-schema', string(source, 'JSON Schema'))));
  }
  static fromLark(source, options) {
    const o = checkedOptions(options, ['profile', 'tokenNames']);
    const profile = o.profile === undefined ? 'recognition' : o.profile;
    if (profile !== 'recognition' && profile !== 'llm') throw new TypeError('Lark profile must be recognition or llm');
    const names = Object.create(null);
    if (o.tokenNames !== undefined) {
      if (o.tokenNames === null || typeof o.tokenNames !== 'object' || Array.isArray(o.tokenNames)) throw new TypeError('tokenNames must be an object');
      for (const [name, value] of Object.entries(o.tokenNames)) names[name] = typeof value === 'number' ? [tokenId(value)] : ids(value);
    }
    return wrap(Grammar, invoke(() => native.NativeGrammar.fromSource('lark', string(source), profile, names)));
  }
  copy() { return wrap(Grammar, invoke(() => handle(this, Grammar).clone())); }
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
      return wrap(Constraint, invoke(() => handle(mod, Module).compileSync(handle(vocab, Vocab), s.end, s.mode, s.boundary)));
    } finally {
      mod.dispose();
    }
  }
  async compile(vocab, options) {
    const s = compileSettings(options);
    const mod = new Module(this);
    try {
      return wrap(Constraint, await invokeAsync(() => handle(mod, Module).compile(handle(vocab, Vocab), s.end, s.mode, s.boundary)));
    } finally {
      mod.dispose();
    }
  }
}

class CompileOptions {
  constructor(options) {
    const o = checkedOptions(options, ['endTokens', 'maskStrategy', 'boundaryMaskStrategy']);
    this.endTokens = Object.freeze(o.endTokens === undefined ? [] : ids(o.endTokens));
    this.maskStrategy = formatMaskStrategy(parseMaskStrategy(o.maskStrategy));
    this.boundaryMaskStrategy = formatBoundaryMaskStrategy(parseBoundaryMaskStrategy(o.boundaryMaskStrategy));
    Object.freeze(this);
  }
  _settings() {
    return {
      end: this.endTokens,
      mode: parseMaskStrategy(this.maskStrategy),
      boundary: parseBoundaryMaskStrategy(this.boundaryMaskStrategy),
    };
  }
}
function compileSettings(options) {
  if (options instanceof CompileOptions) return options._settings();
  const o = checkedOptions(options, ['endTokens', 'maskStrategy', 'boundaryMaskStrategy']);
  return {
    end: o.endTokens === undefined ? [] : ids(o.endTokens),
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
          if (value instanceof Token) return native.NativeExpr.fromToken(tokenId(value.id));
          if (value instanceof ExactTokens) return native.NativeExpr.fromExactTokens(handle(value, ExactTokens));
          throw new TypeError('Expr operands must be Grammar, Module, Expr, Token or ExactTokens');
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
  copy() { return wrap(Expr, invoke(() => handle(this, Expr).clone())); }
}

class Module extends Owned {
  constructor(value, nativeHandle) {
    if (value === key) { super(key, nativeHandle); return; }
    if (!(value instanceof Grammar)) throw new TypeError('Module expects a Grammar');
    super(key, invoke(() => native.NativeModule.fromGrammar(handle(value, Grammar))));
  }

  copy() { return wrap(Module, invoke(() => handle(this, Module).clone())); }

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
      else if (value instanceof Token) target.bindToken(name, tokenId(value.id));
      else if (value instanceof ExactTokens) target.bindTokens(name, handle(value, ExactTokens));
      else throw new TypeError('Module.bind requires Grammar, Module, Expr, Token or ExactTokens');
    });
  }

  bind(nameOrBindings, value) {
    if (arguments.length === 1 && typeof nameOrBindings === 'object' && nameOrBindings !== null && !(nameOrBindings instanceof String)) {
      const entries = Object.entries(nameOrBindings);
      const snapshot = this.copy();
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
    return wrap(Constraint, invoke(() => handle(this, Module).compileSync(handle(vocab, Vocab), s.end, s.mode, s.boundary)));
  }
  async compile(vocab, options) {
    const s = compileSettings(options);
    return wrap(Constraint, await invokeAsync(() => handle(this, Module).compile(handle(vocab, Vocab), s.end, s.mode, s.boundary)));
  }

  saveSync() { return invoke(() => handle(this, Module).saveSync()); }
  async save() { return invokeAsync(() => handle(this, Module).save()); }

  static loadSync(data, vocab) {
    bytes(data); return wrap(Module, invoke(() => native.loadModuleSync(data, vocab === undefined ? undefined : handle(vocab, Vocab))));
  }
  static async load(data, vocab) {
    bytes(data); return wrap(Module, await invokeAsync(() => native.loadModule(data, vocab === undefined ? undefined : handle(vocab, Vocab))));
  }
}

class Constraint extends Owned {
  get maskLength() { return invoke(() => handle(this, Constraint).maskLen); }
  copy() { return wrap(Constraint, invoke(() => handle(this, Constraint).clone())); }
  clone() { return this.copy(); }
  start() { return wrap(ConstraintState, invoke(() => handle(this, Constraint).start())); }
  markerIds() { return invoke(() => handle(this, Constraint).markerIds()); }
  saveSync(options) {
    const flag = checkedOptions(options, ['externalVocab']).externalVocab;
    return invoke(() => handle(this, Constraint).saveSync(flag === true));
  }
  async save(options) {
    const flag = checkedOptions(options, ['externalVocab']).externalVocab;
    return invokeAsync(() => handle(this, Constraint).save(flag === true));
  }
  static loadSync(data, vocab) {
    bytes(data); return wrap(Constraint, invoke(() => native.loadConstraintSync(data, vocab === undefined ? undefined : handle(vocab, Vocab))));
  }
  static async load(data, vocab) {
    bytes(data); return wrap(Constraint, await invokeAsync(() => native.loadConstraint(data, vocab === undefined ? undefined : handle(vocab, Vocab))));
  }
}

class ConstraintState extends Owned {
  get maskLength() { return invoke(() => handle(this, ConstraintState).maskLen); }
  copy() { return wrap(ConstraintState, invoke(() => handle(this, ConstraintState).clone())); }
  clone() { return this.copy(); }
  mask() { return invoke(() => handle(this, ConstraintState).mask()); }
  fillMask(output) {
    if (!types.isUint32Array(output)) throw new TypeError('fillMask requires a Uint32Array');
    invoke(() => handle(this, ConstraintState).fillMask(output));
  }
  commitToken(id) { tokenId(id); invoke(() => handle(this, ConstraintState).commitToken(id)); }
  commitBytes(data) { bytes(data); invoke(() => handle(this, ConstraintState).commitBytes(data)); }
  isAccepting() { return invoke(() => handle(this, ConstraintState).isAccepting()); }
  isRejected() { return invoke(() => handle(this, ConstraintState).isRejected()); }
  forcedTokens() { return invoke(() => handle(this, ConstraintState).forcedTokens()); }
  markerEvents() { return invoke(() => handle(this, ConstraintState).markerEvents()); }
}

module.exports = Object.freeze({
  Vocab, Token, ExactTokens, Grammar, Module, CompileOptions, Expr, Constraint, ConstraintState,
  MaskStrategy, BoundaryMaskStrategy, LarkProfile, GlrMaskError, version: native.version(),
});
