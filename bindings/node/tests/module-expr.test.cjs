'use strict';
const test=require('node:test');
const assert=require('node:assert/strict');
const {Vocab,TokenSet,Grammar,Module,Expr,MaskStrategy,BoundaryMaskStrategy,ParserBackend,GlrMaskError}=require('../index.cjs');
const vocab=(different=false)=>new Vocab([[0,Buffer.from('x')],[1,Buffer.from('a')],[2,Buffer.from('b')],[3,Buffer.from('y')],[4,Buffer.from('xay')],[5,Buffer.from('ab')],[8,Buffer.from(different?'z':'a')]]);
const source=(text='a')=>Grammar.fromLark(`start: "${text}"`);
const host=()=>Grammar.fromGlrm('glrm 1; start start; extern grammar child; nt start = "x" child "y";');
const mark=()=>Grammar.fromGlrm('glrm 1; start start; extern token MARK; nt start = MARK;');
const allowed=(s,id)=>(s.mask()[id>>>5] & (1<<(id&31)))!==0;
function accept(module,v,ids){const s=module.compileSync(v).start();for(const id of ids){assert.ok(allowed(s,id));assert.ok(s.commitToken(id));}assert.ok(s.isAccepting());}

test('Module: copy and root-only final options',()=>{
  const v=vocab(),original=new Module(host());

  const configured=original.clone();
  const child=new Module(source());
  assert.equal(child.precompileSync(v,{maskStrategy:{kind:'token-trie',parserBackend:ParserBackend.TemplateDfa,mergeEquivalentTokens:true},boundaryMaskStrategy:BoundaryMaskStrategy.TokenTrie}),undefined);
  accept(child,v,[1]);

  assert.equal(configured.bind('child',child),undefined);
  assert.equal(configured.externs.unbound.size,0);
  assert.equal(original.externs.unbound.size,1);
  assert.equal(original.externs.get('child').name,'child');

  const options={maskStrategy:{kind:'token-trie',parserBackend:ParserBackend.TemplateDfa,mergeEquivalentTokens:true},boundaryMaskStrategy:BoundaryMaskStrategy.TokenTrie};
  const root=Grammar.fromExpr(Expr.sequence(configured,new TokenSet([63])));
  const state=root.compileSync(v,options).start();
  assert.ok(!allowed(state,63));assert.ok(state.commitToken(4));assert.ok(!state.isAccepting());assert.ok(allowed(state,63));assert.ok(state.commitToken(63));assert.ok(state.isAccepting());assert.ok(state.mask().every(w=>w===0));
  assert.throws(()=>configured.compileSync(v,{bogus:true}),TypeError);
});

test('Grammar: precompile returns a bindable Module with retained work',async()=>{
  const v=vocab();
  const child=source();
  const parent=host().precompileSync(v,{maskStrategy:MaskStrategy.ParserDwa});
  assert.ok(parent instanceof Module);
  parent.bind('child',child.precompileSync(v,{maskStrategy:MaskStrategy.ParserDwa}));
  const state=parent.compileSync(v).start();
  assert.ok(allowed(state,4));assert.ok(state.commitToken(4));assert.ok(state.isAccepting());

  const reference=new Module(host());reference.bind('child',new Module(source()));reference.precompileSync(v);
  assert.deepEqual(Array.from(parent.compileSync(v).start().mask()),Array.from(reference.compileSync(v).start().mask()));
  assert.equal(parent.compileSync(v).start().isAccepting(),reference.compileSync(v).start().isAccepting());

  // The asynchronous spelling returns the same bindable Module.
  const asyncParent=await host().precompile(v,{boundaryMaskStrategy:BoundaryMaskStrategy.TokenTrie});
  assert.ok(asyncParent instanceof Module);
  asyncParent.bind('child',await child.precompile(v));
  accept(asyncParent,v,[4]);

  // Retained work is serialized, survives a round-trip, and a TokenTrie
  // compile keeps it instead of rebuilding dynamically.
  const retained=source().precompileSync(v,{maskStrategy:MaskStrategy.ParserDwa});
  const bytes=retained.saveSync();
  assert.ok(bytes.length>new Module(source()).saveSync().length);
  const restored=Module.loadSync(bytes,v);
  assert.deepEqual(Array.from(restored.compileSync(v,{maskStrategy:MaskStrategy.TokenTrie}).start().mask()),
    Array.from(new Module(source()).compileSync(v).start().mask()));
});

test('Module: retained work round-trips without stored settings',()=>{
  const v=vocab();
  const m=new Module(source());
  m.precompileSync(v,{maskStrategy:MaskStrategy.TokenTrie,boundaryMaskStrategy:BoundaryMaskStrategy.ParserDwa});
  const bytes=m.saveSync();
  const loaded=Module.loadSync(bytes,v);
  accept(loaded,v,[1]);
});

test('Module: legacy boundary strategy strings are rejected',()=>{
  const v=vocab();
  for (const input of ['fast-runtime','fast-build','dynamic']) {
    assert.throws(()=>new Module(source()).precompileSync(v,{boundaryMaskStrategy:input}),TypeError);
  }
});

test('Module: rebinding error, :: error and open-child error',()=>{
  const h=new Module(host());
  h.bind('child',source());
  // Rebinding error
  assert.throws(()=>h.bind('child',source('b')),/already bound/);

  // Qualified paths rejected
  const p2=new Module(host());
  assert.throws(()=>p2.bind('foo::bar',source()),e=>e instanceof GlrMaskError && e.code==='GLRMASK_ENGINE_ERROR' && e.message==='there is no extern named "foo::bar"');

  // Open-child error
  const openChild=new Module(Grammar.fromGlrm('glrm 1; start root; extern grammar inner; nt root = inner;'));
  const p3=new Module(host());
  assert.throws(()=>p3.bind('child',openChild),/unbound externs/);
});

test('Module: externs order and unbound Map-like snapshot',()=>{
  const multi=Grammar.fromGlrm('glrm 1; start root; extern token first; extern grammar second; extern token third; nt root = first second third;');
  const mm=new Module(multi);
  assert.equal(mm.externs.size,3);
  assert.equal(mm.externs.unbound.size,3);
  const names=Array.from(mm.externs.keys());
  assert.deepEqual(names,['first','second','third']);
  assert.ok(mm.externs.has('first'));
  assert.ok(mm.externs.has('second'));
  assert.ok(mm.externs.has('third'));
  assert.equal(mm.externs.get('first').kind,'token');
  assert.equal(mm.externs.get('second').kind,'grammar');

  // Bind second
  mm.bind('second',source());
  assert.equal(mm.externs.size,3);
  assert.equal(mm.externs.unbound.size,2);
  assert.deepEqual(Array.from(mm.externs.unbound.keys()),['first','third']);
  assert.ok(mm.externs.unbound.has('first'));
  assert.ok(!mm.externs.unbound.has('second'));
  assert.ok(mm.externs.unbound.has('third'));
});

test('Module: atomic multi-bind',()=>{
  const multi=Grammar.fromGlrm('glrm 1; start root; extern token a; extern token b; nt root = a b;');
  const m=new Module(multi);
  assert.equal(m.externs.unbound.size,2);

  // Failing multi-bind leaves m completely untouched
  assert.throws(()=>m.bind({a:new TokenSet([1]),b:'not_a_valid_binding'}));
  assert.equal(m.externs.unbound.size,2);
  assert.ok(m.externs.unbound.has('a'));
  assert.ok(m.externs.unbound.has('b'));

  // Successful multi-bind
  m.bind({a:new TokenSet([1]),b:new TokenSet([2])});
  assert.equal(m.externs.unbound.size,0);
});

test('Grammar: clone is independent and multi-bind is atomic',()=>{
  const v=vocab();
  const g=Grammar.fromGlrm('glrm 1; start root; extern token a; extern token b; nt root = a b;');
  const copy=g.clone();
  copy.bind('a',1);
  assert.equal(g.externs.unbound.size,2);
  assert.equal(copy.externs.unbound.size,1);
  assert.ok(copy.externs.unbound.has('b'));
  assert.throws(()=>g.compileSync(v),/extern "a" is unbound/);
  assert.throws(()=>copy.compileSync(v),/extern "b" is unbound/);

  const m=Grammar.fromGlrm('glrm 1; start root; extern token a; extern token b; nt root = a b;');
  assert.throws(()=>m.bind({a:new TokenSet([1]),b:'not_a_valid_binding'}));
  assert.equal(m.externs.unbound.size,2);
  assert.ok(m.externs.unbound.has('a'));
  assert.ok(m.externs.unbound.has('b'));
  // The failed batch left both externs unbound; 'a' would be named first if it had been bound.
  assert.throws(()=>m.compileSync(v),/extern "a" is unbound/);

  m.bind({a:new TokenSet([1]),b:new TokenSet([2])});
  const state=m.compileSync(v).start();
  assert.ok(state.commitToken(1));assert.ok(state.commitToken(2));assert.ok(state.isAccepting());
});

for(const parentPrepared of [false,true])for(const childPrepared of [false,true]){
  test(`Module: saved mixed graph parent=${parentPrepared}, child=${childPrepared}`,async()=>{
    const v=vocab(),parent=new Module(host()),child=new Module(mark());
    if(parentPrepared)assert.equal(await parent.precompile(v),undefined);
    if(childPrepared)assert.equal(await child.precompile(v,{maskStrategy:{kind:'token-trie',parserBackend:ParserBackend.TemplateDfa,mergeEquivalentTokens:true}}),undefined);
    child.bind('MARK',new TokenSet([8]));
    parent.bind('child',child);const bytes=await parent.save();
    const loaded=await Module.load(bytes);
    assert.equal(loaded.externs.unbound.size,0);
    accept(loaded,v,[0,8,3]);
  });
}

test('Module: raw source attachment, failure survival and target identity',()=>{
  const v=vocab(),parent=new Module(host());parent.precompileSync(v);const bytes=parent.saveSync();
  assert.throws(()=>parent.bind('missing',source()));assert.deepEqual(parent.saveSync(),bytes);
  const wrong=new Module(source());wrong.precompileSync(vocab(true));
  assert.throws(()=>parent.bind('child',wrong));assert.deepEqual(parent.saveSync(),bytes);
  assert.throws(()=>parent.precompileSync(vocab(true)));assert.deepEqual(parent.saveSync(),bytes);
  Module.loadSync(bytes,vocab(true));
  const loaded=Module.loadSync(bytes);loaded.bind('child',source());accept(loaded,v,[4]);
  const snapshot=loaded.saveSync();assert.throws(()=>loaded.bind('child',source('b')));assert.deepEqual(loaded.saveSync(),snapshot);
  const token=new Module(mark());assert.throws(()=>token.bind('MARK',source()));assert.equal(token.externs.unbound.get('MARK').name,'MARK');
});

test('Expr: choice, sequence, literal and plain string errors',()=>{
  const v=vocab();
  // Plain string error pointing to Expr.literal
  assert.throws(()=>Expr.choice('plain string'),/Plain strings are not supported/);
  assert.throws(()=>Expr.sequence('plain string'),/Plain strings are not supported/);

  // Literal
  const lit=Expr.literal('xay');
  const mLit=new Module(Grammar.fromGlrm('glrm 1; start root; extern grammar child; nt root = child;'));
  mLit.bind('child',lit);
  accept(mLit,v,[4]);

  // Choice & sequence
  const expr=Expr.sequence(source('x'),Expr.choice(source(),source('b')),source('y'));
  const mExpr=new Module(host());
  mExpr.bind('child',Expr.choice(source(),source('b')));
  accept(mExpr,v,[4]);
});

test('Module: asynchronous precompile rejects a stale completion atomically',async()=>{
  const v=vocab(),parent=new Module(host());const pending=parent.precompile(v);
  parent.bind('child',source());
  await assert.rejects(pending,/changed during precompile/);accept(parent,v,[4]);
});

test('Grammar.clone and Module.clone return independent copies', () => {
  const g1 = Grammar.fromGlrm('glrm 1; start root; extern token a; nt root = a;');
  const g2 = g1.clone();
  g2.bind('a', 1);
  assert.equal(g1.externs.unbound.size, 1);
  assert.equal(g2.externs.unbound.size, 0);

  const m1 = new Module(Grammar.fromGlrm('glrm 1; start root; extern token a; nt root = a;'));
  const m2 = m1.clone();
  m2.bind('a', 1);
  assert.equal(m1.externs.unbound.size, 1);
  assert.equal(m2.externs.unbound.size, 0);
});
