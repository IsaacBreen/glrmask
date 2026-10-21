import api = require('@glrmask/node');
const v = new api.Vocab([[0,Buffer.from('a')],[1,Buffer.from('b')]], {exactOnlyIds:[99]});
const exact: api.ExactTokens = v.tokenRanges([{start:0,end:2}]);
const range: api.TokenRange|undefined = exact.ranges[0];
const g: api.Grammar = api.Grammar.fromEbnf('start ::= "a"');
const c = g.compileSync(v, {maskStrategy: api.MaskStrategy.TokenTrie, endTokens: [99]});
const s = c.start(); const words: Uint32Array = s.mask(); s.fillMask(words); s.commitToken(0);
const accepted: boolean = s.isAccepting(); const markers: api.MarkerEvent[] = s.markerEvents();
const offset: bigint|undefined = markers[0]?.byteOffset;
const saved: Uint8Array = c.saveSync({externalVocab: true}); api.Constraint.loadSync(saved, v);

// @ts-expect-error Mask words must be unsigned 32-bit values.
s.fillMask(new Int32Array(1));
// @ts-expect-error Raw numeric values need an explicit Token wrapper.
new api.Token('invalid');
// @ts-expect-error Invalid option spelling must not be accepted by declarations.
g.compileSync(v, {optimisation: 'auto'});
// @ts-expect-error States are obtained from a compiled Constraint, not constructed.
new api.ConstraintState();
s[Symbol.dispose](); c.dispose();
void [range, accepted, offset];

const module = new api.Module(g, {maskStrategy: api.MaskStrategy.TokenClassTrie});
const expr = api.Expr.sequence(g, module, api.Expr.choice(g));
const entry: api.Extern|undefined = module.externs.get('slot');
module.precompileSync(v);
const independent: api.Module = module.copy();
const result: api.Constraint = independent.compileSync(v, new api.CompileOptions({endTokens:[99]}));
const savedModule: Uint8Array = module.saveSync(); api.Module.loadSync(savedModule, v);
// @ts-expect-error Strings are not valid operands in Expr.choice.
api.Expr.choice('invalid');
void [entry, result, expr];
