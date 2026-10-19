import api = require('@glrmask/node');
const v=new api.Vocab([[0,Buffer.from('a')],[1,Buffer.from('b')]],{exactOnlyIds:[99]});
const exact:api.ExactTokens=v.tokenRanges([{start:0,end:2}]);
const range:api.TokenRange|undefined=exact.ranges[0];
const g:api.Grammar=api.Grammar.fromEbnf('start ::= "a"');
const c=g.compileSync(v,{optimization:api.Optimization.FastBuild,endTokens:[99]});
const s=c.start();const words:Uint32Array=s.mask();s.fillMask(words);s.commitToken(0);
const accepted:boolean=s.isAccepting();const markers:api.MarkerEvent[]=s.markerEvents();
const offset:bigint|undefined=markers[0]?.byteOffset;
const saved:Uint8Array=c.saveSync({externalVocab:true});api.Constraint.loadSync(saved,v);
const prepared=g.prepareSync(v,{boundaryTrigger:api.BoundaryTriggerDetail.None});
prepared.bind('child',prepared.clone());
// @ts-expect-error Source-stage values cannot be bound into a prepared module.
prepared.bind('child',g);
// @ts-expect-error A runnable constraint is not a source grammar.
g.bind('child',c);
// @ts-expect-error Mask words must be unsigned 32-bit values.
s.fillMask(new Int32Array(1));
// @ts-expect-error Raw numeric values need an explicit Token wrapper.
g.bind('TOKEN',0);
// @ts-expect-error Invalid option spelling must not be accepted by declarations.
g.compileSync(v,{optimisation:'auto'});
// @ts-expect-error States are obtained from a compiled Constraint, not constructed.
new api.ConstraintState();
s[Symbol.dispose]();c.dispose();
void [range,accepted,offset];

const local=new api.ModuleOptions({optimization:api.Optimization.Balanced});
const module=new api.Module(g,local);
const expr=api.Expr.sequence([g,module,api.Expr.choice([g])]);
const composed=new api.Module(expr);
const entry:api.UnboundExtern|undefined=composed.unbound[0];
module.precompileSync(v);
const independent:api.Module=module.clone();
const result:api.Constraint=independent.compileSync(v,new api.CompileOptions({endTokens:[99]}));
const savedModule:Uint8Array=module.saveSync();api.Module.loadSync(savedModule,v);
// @ts-expect-error Root compilation cannot override local component optimization.
module.compileSync(v,{optimization:api.Optimization.Auto});
// @ts-expect-error Options are frozen snapshots.
local.optimization=api.Optimization.FastBuild;
// @ts-expect-error Token events are not whole expression leaves.
api.Expr.choice([new api.Token(0)]);
void [entry,result];
