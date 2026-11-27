'use strict';
const {types, inspect} = require('node:util');
const native = require('./loader.cjs');
const key = Symbol('GLRMask owning native handle');
const handles = new WeakMap();

class GLRMaskError extends Error {
  constructor(code, message, options) { super(message, options); this.name = 'GLRMaskError'; this.code = code; }
}
function normalize(error) {
  const match = error instanceof Error && /^(GLRMASK_[A-Z_]+): ([\s\S]*)$/.exec(error.message);
  return match ? new GLRMaskError(match[1], match[2], {cause: error}) : error;
}
function invoke(fn) { try { return fn(); } catch (error) { throw normalize(error); } }
async function invokeAsync(fn) {
  try { return await fn(); } catch (error) { throw normalize(error); }
}
function handle(value, Type) {
  if (!(value instanceof Type) || !handles.has(value)) throw new TypeError(`Expected a ${Type.name}`);
  return handles.get(value);
}
function wrap(Type, value) { return new Type(key, value); }
function string(value, name = 'source') {
  if (typeof value !== 'string') throw new TypeError(`${name} must be a string`);
  return value;
}
// The largest token ID GLRMask accepts, 2 ** 24 - 1, as in every language.
const MAX_TOKEN_ID = 0xffffff;
function tokenId(value) {
  if (typeof value !== 'number' || !Number.isInteger(value) || value < 0 || value > MAX_TOKEN_ID) {
    throw new RangeError(`Token IDs must be integers from 0 to ${MAX_TOKEN_ID}`);
  }
  return value;
}
function iterable(value, name) {
  if (value == null || typeof value[Symbol.iterator] !== 'function') throw new TypeError(`${name} must be iterable`);
  return Array.from(value);
}
function ids(value) { return iterable(value, 'token IDs').map(tokenId); }
function bytes(value) {
  if (!types.isUint8Array(value)) throw new TypeError('Expected a Uint8Array or Buffer');
  return value;
}
function checkedOptions(value, allowed) {
  if (value === undefined) return {};
  if (value === null || typeof value !== 'object' || Array.isArray(value)) throw new TypeError('Options must be an object');
  for (const name of Object.keys(value)) if (!allowed.includes(name)) throw new TypeError(`Unknown option: ${name}`);
  return value;
}

const MaskStrategy = Object.freeze({
  Auto: 'auto', FastCompile: 'fast-compile', FastMasks: 'fast-masks',
});
function parseStrategy(value, field) {
  if (value === undefined || value === 'auto') return 0;
  if (value === 'fast-compile') return 3;
  if (value === 'fast-masks') return 4;
  throw new TypeError(field + " must be 'auto', 'fast-compile' or 'fast-masks'");
}

class ExternsMap {
  #map;
  #unbound;
  constructor(items, unbound) {
    this.#map = new Map(items.map(item => [item.name, item]));
    this.#unbound = unbound;
    Object.freeze(this);
  }
  get(name) { return this.#map.get(name); }
  has(name) { return this.#map.has(name); }
  get size() { return this.#map.size; }
  keys() { return this.#map.keys(); }
  values() { return this.#map.values(); }
  entries() { return this.#map.entries(); }
  [Symbol.iterator]() { return this.#map.entries(); }
  forEach(callback, thisArg) { this.#map.forEach((v, k) => callback.call(thisArg, v, k, this)); }
  get unbound() { return this.#unbound ?? this; }
  [inspect.custom](depth, options) {
    return `Externs(${inspect(this.#map, {...options, depth: depth - 1})}, unbound=${inspect([...this.unbound.keys()], options)})`;
  }
}

function formatExterns(raw, rawUnbound) {
  const items = raw.map(e => Object.freeze({name: e.name, kind: e.kind}));
  return new ExternsMap(items, rawUnbound === null ? null : formatExterns(rawUnbound, null));
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
  constructor(entries) {
    if (entries === key) { super(key, arguments[1]); return; }
    const plain = entries != null && typeof entries === 'object' &&
      (Object.getPrototypeOf(entries) === Object.prototype || Object.getPrototypeOf(entries) === null);
    const input = plain ? Object.entries(entries).map(([id, value]) => {
      if (!/^(0|[1-9][0-9]*)$/.test(id)) throw new TypeError('Vocabulary keys must be integer strings');
      return [Number(id), value];
    }) : iterable(entries, 'vocabulary entries');
    const tokenIds = [], tokenBytes = [];
    for (const entry of input) {
      if (!Array.isArray(entry) || entry.length !== 2) throw new TypeError('Each vocabulary entry must be [id, Uint8Array]');
      tokenIds.push(tokenId(entry[0])); tokenBytes.push(bytes(entry[1]));
    }
    super(key, invoke(() => native.NativeVocab.fromEntries(tokenIds, tokenBytes)));
  }
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
      if (!Number.isInteger(end) || end <= start || end > MAX_TOKEN_ID + 1) throw new RangeError('Invalid half-open token range');
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
  static fromGlrm(source) { return wrap(Grammar, invoke(() => native.NativeGrammar.fromSource('glrm', string(source)))); }
  static fromJsonSchema(schema) {
    const source = typeof schema === 'string' ? schema : JSON.stringify(schema);
    return wrap(Grammar, invoke(() => native.NativeGrammar.fromSource('json-schema', string(source, 'JSON Schema'))));
  }
  static fromLark(source) {
    return wrap(Grammar, invoke(() => native.NativeGrammar.fromSource('lark', string(source))));
  }
  get externs() {
    const target = handle(this, Grammar);
    const rawAll = invoke(() => target.externs());
    return formatExterns(rawAll, invoke(() => target.unboundExterns()));
  }
  clone() { return wrap(Grammar, invoke(() => handle(this, Grammar).clone())); }
  static fromExpr(value) {
    let expr;
    if (value instanceof Expr) expr = handle(value, Expr);
    else if (value instanceof Module) expr = invoke(() => native.NativeExpr.fromModule(handle(value, Module)));
    else if (value instanceof Grammar) expr = invoke(() => native.NativeExpr.fromGrammar(handle(value, Grammar)));
    else if (value instanceof TokenSet) expr = invoke(() => native.NativeExpr.fromTokenSet(handle(value, TokenSet)));
    else throw new TypeError('Grammar.fromExpr requires an Expr or Grammar');
    return wrap(Grammar, invoke(() => native.NativeGrammar.fromExpr(expr)));
  }
  bind(name, value) {
    string(name, 'binding name');
    const target = handle(this, Grammar);
    if (value instanceof Grammar) return invoke(() => target.bindGrammar(name, handle(value, Grammar)));
    if (value instanceof Expr) return invoke(() => target.bindExpr(name, handle(value, Expr)));
    if (value instanceof TokenSet) return invoke(() => target.bindTokens(name, handle(value, TokenSet)));
    throw new TypeError('Grammar.bind requires a Grammar or Expr');
  }
  compileSync(vocab, options) {
    const s = compileSettings(options);
    const mod = new Module(this);
    try {
      return wrap(Constraint, invoke(() => handle(mod, Module).compileSync(handle(vocab, Vocab), s.mode)));
    } finally {
      mod.dispose();
    }
  }
  async compile(vocab, options) {
    const s = compileSettings(options);
    const mod = new Module(this);
    try {
      return wrap(Constraint, await invokeAsync(() => handle(mod, Module).compile(handle(vocab, Vocab), s.mode)));
    } finally {
      mod.dispose();
    }
  }
}

function compileSettings(options) {
  const o = checkedOptions(options, ['maskStrategy']);
  return {
    mode: parseStrategy(o.maskStrategy, 'maskStrategy'),
  };
}

class Expr extends Owned {
  static _make(values, operator) {
    const parts = [];
    try {
      for (const value of values) {
        if (typeof value === 'string') {
          throw new TypeError('plain strings are not Expr operands; wrap them in a literal expression');
        }
        parts.push(invoke(() => {
          if (value instanceof Grammar) return native.NativeExpr.fromGrammar(handle(value, Grammar));
          if (value instanceof Module) return native.NativeExpr.fromModule(handle(value, Module));
          if (value instanceof Expr) return handle(value, Expr).clone();
          if (value instanceof TokenSet) return native.NativeExpr.fromTokenSet(handle(value, TokenSet));
          throw new TypeError('Expr operands must be a Grammar, Module, Expr or TokenSet');
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

  bind(name, value) {
    string(name, 'binding name');
    if (name.includes('::')) throw new GLRMaskError('GLRMASK_INVALID_INPUT', `there is no extern named ${JSON.stringify(name)}`);
    invoke(() => {
      const target = handle(this, Module);
      if (value instanceof Grammar) target.bindGrammar(name, handle(value, Grammar));
      else if (value instanceof Module) {
        if (value.externs.unbound.size > 0) {
          const names = Array.from(value.externs.unbound.keys()).map(n => `"${n}"`).join(', ');
          throw new GLRMaskError('GLRMASK_INVALID_INPUT', `cannot bind "${name}": the value has unbound externs [${names}]`);
        }
        target.bindModule(name, handle(value, Module));
      }
      else if (value instanceof Expr) target.bindExpr(name, handle(value, Expr));
      else if (value instanceof TokenSet) target.bindTokens(name, handle(value, TokenSet));
      else throw new TypeError('Module.bind requires a Grammar, Module, Expr, or TokenSet');
    });
  }


  get externs() {
    const target = handle(this, Module);
    const rawAll = invoke(() => target.externs());
    return formatExterns(rawAll, invoke(() => target.unboundExterns()));
  }

  precompileSync(vocab, options) {
    const s = compileSettings(options);
    invoke(() => handle(this, Module).precompileSync(handle(vocab, Vocab), s.mode));
  }
  async precompile(vocab, options) {
    const s = compileSettings(options);
    await invokeAsync(() => handle(this, Module).precompile(handle(vocab, Vocab), s.mode));
  }

  compileSync(vocab, options) {
    const s = compileSettings(options);
    return wrap(Constraint, invoke(() => handle(this, Module).compileSync(handle(vocab, Vocab), s.mode)));
  }
  async compile(vocab, options) {
    const s = compileSettings(options);
    return wrap(Constraint, await invokeAsync(() => handle(this, Module).compile(handle(vocab, Vocab), s.mode)));
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
  [inspect.custom]() { return `<glrmask.Constraint mask_bit_len=${this.maskBitLength}>`; }
  get maskBitLength() { return invoke(() => handle(this, Constraint).maskBitLen); }
  clone() { return wrap(Constraint, invoke(() => handle(this, Constraint).clone())); }
  start() { return wrap(ConstraintState, invoke(() => handle(this, Constraint).start())); }
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
  [inspect.custom]() { return `<glrmask.ConstraintState accepting=${this.isAccepting()} terminated=${this.isTerminated()}>`; }
  get maskBitLength() { return invoke(() => handle(this, ConstraintState).maskBitLen); }
  clone() { return wrap(ConstraintState, invoke(() => handle(this, ConstraintState).clone())); }
  mask() { return invoke(() => handle(this, ConstraintState).mask()); }
  fillMask(output) {
    if (!types.isUint32Array(output)) throw new TypeError('fillMask needs a Uint32Array; for an Int32Array pass new Uint32Array(a.buffer, a.byteOffset, a.length)');
    invoke(() => handle(this, ConstraintState).fillMask(output));
  }
  commitToken(id) {
    if (typeof id !== 'number') throw new TypeError('Token IDs must be numbers');
    return invoke(() => handle(this, ConstraintState).commitToken(id));
  }
  commitBytes(data) { bytes(data); return invoke(() => handle(this, ConstraintState).commitBytes(data)); }
  isAccepting() { return invoke(() => handle(this, ConstraintState).isAccepting()); }
  isTerminated() { return invoke(() => handle(this, ConstraintState).isTerminated()); }
  forcedBytes() {
    return invoke(() => handle(this, ConstraintState).forcedBytes());
  }
}

module.exports = Object.freeze({
  Vocab, TokenSet, Grammar, Module, Expr, Constraint, ConstraintState,
  MaskStrategy, GLRMaskError, version: native.version(),
});
