import api = require('@glrmask/node');
const v = new api.Vocab([[0,Buffer.from('a')],[1,Buffer.from('b')]], {specialTokenIds:[99]});
const exact: api.TokenSet = api.TokenSet.fromRanges([{start:0,end:2}]);
const range: api.TokenRange = {start:0,end:2};
const g: api.Grammar = api.Grammar.fromEbnf('start ::= "a"');
const c = api.Grammar.fromExpr(api.Expr.sequence(g, new api.TokenSet([99]))).compileSync(v, {maskStrategy: api.MaskStrategy.TokenTrie});
const s = c.start(); const words: Uint32Array = s.mask(); s.fillMask(words); s.commitToken(0);
const accepted: boolean = s.isAccepting(); const markers: api.MarkerEvent[] = s.markerEvents();
const offset: bigint|undefined = markers[0]?.byteOffset;
const saved: Uint8Array = c.saveSync(); api.Constraint.loadSync(saved, v);

// @ts-expect-error Mask words must be unsigned 32-bit values.
s.fillMask(new Int32Array(1));
// @ts-expect-error Token sets are constructed from an iterable of IDs.
new api.TokenSet('invalid');
// @ts-expect-error Invalid option spelling must not be accepted by declarations.
g.compileSync(v, {optimisation: 'auto'});
// @ts-expect-error States are obtained from a compiled Constraint, not constructed.
new api.ConstraintState();
s[Symbol.dispose](); c.dispose();
void [range, accepted, offset];

const precompiled: api.Module = g.precompileSync(v, {maskStrategy: api.MaskStrategy.ParserDwa});
const precompiledAsync: Promise<api.Module> = g.precompile(v);
void [precompiled, precompiledAsync];

const module = new api.Module(g);
const expr = api.Expr.sequence(g, module, api.Expr.choice(g));
const entry: api.Extern|undefined = module.externs.get('slot');
module.precompileSync(v, {maskStrategy: api.MaskStrategy.TokenTrie});
const independent: api.Module = module.clone();
const result: api.Constraint = independent.compileSync(v, {maskStrategy: api.MaskStrategy.TokenTrie, boundaryMaskStrategy: api.BoundaryMaskStrategy.ParserDwa});
const savedModule: Uint8Array = module.saveSync(); api.Module.loadSync(savedModule);
// @ts-expect-error Strings are not valid operands in Expr.choice.
api.Expr.choice('invalid');
void [entry, result, expr];
