import api = require('@glrmask/node');
const v = new api.Vocab([[0,Buffer.from('a')],[1,Buffer.from('b')]]);
const mapVocab = new api.Vocab(new Map([[0,Buffer.from("a")]]));
// @ts-expect-error Vocab requires iterable pairs, not a record.
new api.Vocab({0:Buffer.from("a")});
const exact: api.TokenSet = api.TokenSet.fromRanges([{start:0,end:2}]);
const range: api.TokenRange = {start:0,end:2};
const g: api.Grammar = api.Grammar.fromLark('start: "a"');
const trie: api.MaskStrategy = api.MaskStrategy.FastCompile;
const endModule = new api.Module(api.Grammar.fromGlrm('extern grammar VALUE start root nt root = VALUE'));
endModule.bind('VALUE',api.Expr.sequence(g, new api.TokenSet([99])));
const c = endModule.compileSync(v, {maskStrategy: trie});
const direct = new api.Module(api.Grammar.fromGlrm('extern grammar VALUE start root nt root = VALUE'));
direct.bind('VALUE', c);
const s = c.start(); const words: Uint32Array = s.mask(); s.fillMask(words); s.commitToken(0);
const accepted: boolean = s.isAccepting();
const saved: Uint8Array = c.saveSync(); api.Constraint.loadSync(saved, v);

// @ts-expect-error Mask words must be unsigned 32-bit values.
s.fillMask(new Int32Array(1));
// @ts-expect-error Mask words must be unsigned 32-bit values.
s.fillMask(new Float32Array(1));
// @ts-expect-error Grammar has no bind method.
g.bind("slot", exact);
// @ts-expect-error Token sets are constructed from an iterable of IDs.
new api.TokenSet('invalid');
// @ts-expect-error Invalid option spelling must not be accepted by declarations.
g.compileSync(v, {optimisation: 'auto'});
// @ts-expect-error States are obtained from a compiled Constraint, not constructed.
new api.ConstraintState();
s[Symbol.dispose](); c.dispose();
void [range, accepted];

const precompiled = new api.Module(g); precompiled.precompileSync(v, {maskStrategy: api.MaskStrategy.FastMasks});
const precompiledAsync: Promise<void> = precompiled.precompile(v);
// @ts-expect-error Grammar no longer has precompile.
g.precompile(v);
void [precompiled, precompiledAsync];

const module = new api.Module(g);
const expr = api.Expr.sequence(g, module, api.Expr.choice(g));
const entry: api.Extern|undefined = module.externs.get('slot');
module.precompileSync(v, {maskStrategy: api.MaskStrategy.FastCompile});
const independent: api.Module = module.clone();
const result: api.Constraint = independent.compileSync(v, {maskStrategy: api.MaskStrategy.FastCompile});
const savedModule: Uint8Array = module.saveSync(); api.Module.loadSync(savedModule);
// @ts-expect-error Strings are not valid operands in Expr.choice.
api.Expr.choice('invalid');
// @ts-expect-error Token IDs must be wrapped in TokenSet.
api.Expr.choice(5);
void [entry, result, expr];

// @ts-expect-error Plain token IDs are not valid bind values.
g.bind("slot", 1);
// @ts-expect-error Plain token IDs are not valid bind values.
module.bind("slot", 1);

// @ts-expect-error Removed hard algorithm selector.
g.compileSync(v, {maskStrategy: 'parser-dwa'});
// @ts-expect-error No tuning object in stable API.
g.compileSync(v, {maskStrategy: {kind:'token-trie'}});


// @ts-expect-error Removed batch commitment.
s.commitTokens([1]);
// @ts-expect-error Removed record binding.
module.bind({slot:g});
// @ts-expect-error Removed record binding.
g.bind({slot:g});
// @ts-expect-error null is not a mask strategy.
g.compileSync(v,{maskStrategy:null});
