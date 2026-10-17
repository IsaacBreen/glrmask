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
