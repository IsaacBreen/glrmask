'use strict';
const test=require('node:test');
const assert=require('node:assert/strict');
const {Vocab,TokenSet,Grammar,Module,Expr,MaskStrategy,GLRMaskError}=require('../index.cjs');
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
  assert.equal(child.precompileSync(v,{maskStrategy:MaskStrategy.FastMasks}),undefined);
  accept(child,v,[1]);

  assert.equal(configured.bind('child',child),undefined);
  assert.equal(configured.externs.unbound.size,0);
  assert.equal(original.externs.unbound.size,1);
  assert.equal(original.externs.get('child').name,'child');

  const options={maskStrategy:MaskStrategy.FastMasks};
  const root=new Module(Grammar.fromGlrm('start s extern grammar item nt s = item'));
  root.bind('item',Expr.sequence(configured,new TokenSet([63])));
  const state=root.compileSync(v,options).start();
  assert.ok(!allowed(state,63));assert.ok(state.commitToken(4));assert.ok(!state.isAccepting());assert.ok(allowed(state,63));assert.ok(state.commitToken(63));assert.ok(state.isAccepting());assert.ok(state.mask().every(w=>w===0));
  assert.throws(()=>configured.compileSync(v,{bogus:true}),TypeError);
});

test('Module: precompile requires the same vocabulary in sync and async calls', async()=>{
  const v=vocab(),wrong=vocab(true);
  for (const maskStrategy of Object.values(MaskStrategy)) {
    const m=new Module(host()), options={maskStrategy};
    await m.precompile(v,options);
    assert.throws(()=>m.precompileSync(wrong,options),/vocabular/);
    await assert.rejects(m.precompile(wrong,options),/vocabular/);
    m.bind('child',source());
    assert.throws(()=>m.compileSync(wrong,options),/vocabular/);
    await assert.rejects(m.compile(wrong,options),/vocabular/);
    accept(m,v,[4]);
  }
});

test('Module: precompile retains work',async()=>{
  const v=vocab();
  const child=source();
  const parent=new Module(host());parent.precompileSync(v,{maskStrategy:MaskStrategy.FastMasks});
  assert.ok(parent instanceof Module);
  const childModule=new Module(child);childModule.precompileSync(v,{maskStrategy:MaskStrategy.FastMasks});parent.bind('child',childModule);
  const state=parent.compileSync(v).start();
  assert.ok(allowed(state,4));assert.ok(state.commitToken(4));assert.ok(state.isAccepting());

  const reference=new Module(host());reference.bind('child',new Module(source()));reference.precompileSync(v);
  assert.deepEqual(Array.from(parent.compileSync(v).start().mask()),Array.from(reference.compileSync(v).start().mask()));
  assert.equal(parent.compileSync(v).start().isAccepting(),reference.compileSync(v).start().isAccepting());

  // The asynchronous spelling returns the same bindable Module.
  const asyncParent=new Module(host());await asyncParent.precompile(v,{maskStrategy:MaskStrategy.FastCompile});
  assert.ok(asyncParent instanceof Module);
  const asyncChild=new Module(child);await asyncChild.precompile(v);asyncParent.bind('child',asyncChild);
  accept(asyncParent,v,[4]);

  // Retained work is serialized, survives a round-trip, and a TokenTrie
  // compile keeps it instead of rebuilding dynamically.
  const retained=new Module(source());retained.precompileSync(v,{maskStrategy:MaskStrategy.FastMasks});
  const bytes=retained.saveSync();
  assert.ok(bytes.length>new Module(source()).saveSync().length);
  const restored=Module.loadSync(bytes,v);
  assert.deepEqual(Array.from(restored.compileSync(v,{maskStrategy:MaskStrategy.FastCompile}).start().mask()),
    Array.from(new Module(source()).compileSync(v).start().mask()));
});

test('Module: retained work round-trips without stored settings',()=>{
  const v=vocab();
  const m=new Module(source());
  m.precompileSync(v,{maskStrategy:MaskStrategy.FastCompile});
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
  assert.throws(()=>p2.bind('foo::bar',source()),e=>e instanceof GLRMaskError && e.code==='GLRMASK_INVALID_INPUT' && e.message==='there is no extern named "foo::bar"');

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
  const g=Grammar.fromGlrm('glrm 1; start root; extern grammar a; extern grammar b; nt root = a b;');
  const copy=g.clone();
  copy.bind('a',source());
  assert.equal(g.externs.unbound.size,2);
  assert.equal(copy.externs.unbound.size,1);
  assert.ok(copy.externs.unbound.has('b'));
  assert.throws(()=>g.compileSync(v),/extern "a" is unbound/);
  assert.throws(()=>copy.compileSync(v),/extern "b" is unbound/);

  const m=Grammar.fromGlrm('glrm 1; start root; extern grammar a; extern grammar b; nt root = a b;');
  assert.throws(()=>m.bind({a:source(),b:'not_a_valid_binding'}));
  assert.equal(m.externs.unbound.size,2);
  assert.ok(m.externs.unbound.has('a'));
  assert.ok(m.externs.unbound.has('b'));
  // The failed batch left both externs unbound; 'a' would be named first if it had been bound.
  assert.throws(()=>m.compileSync(v),/extern "a" is unbound/);

  m.bind({a:source(),b:source('b')});
  const state=m.compileSync(v).start();
  assert.ok(state.commitToken(1));assert.ok(state.commitToken(2));assert.ok(state.isAccepting());
});

for(const parentPrepared of [false,true])for(const childPrepared of [false,true]){
  test(`Module: saved mixed graph parent=${parentPrepared}, child=${childPrepared}`,async()=>{
    const v=vocab(),parent=new Module(host()),child=new Module(mark());
    if(parentPrepared)assert.equal(await parent.precompile(v),undefined);
    if(childPrepared)assert.equal(await child.precompile(v,{maskStrategy:MaskStrategy.FastMasks}),undefined);
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
  assert.throws(()=>Expr.choice(5), TypeError);
  assert.throws(()=>Expr.sequence(5), TypeError);
  assert.throws(()=>Expr.choice('plain string'),/plain strings are not Expr operands/);
  assert.throws(()=>Expr.sequence('plain string'),/plain strings are not Expr operands/);

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
  const g1 = Grammar.fromGlrm('glrm 1; start root; extern grammar a; nt root = a;');
  const g2 = g1.clone();
  assert.throws(() => g2.bind('a', 1), TypeError);
  g2.bind('a', source());
  assert.equal(g1.externs.unbound.size, 1);
  assert.equal(g2.externs.unbound.size, 0);

  const m1 = new Module(Grammar.fromGlrm('glrm 1; start root; extern token a; nt root = a;'));
  const m2 = m1.clone();
  assert.throws(() => m2.bind('a', 1), TypeError);
  m2.bind('a', new TokenSet([1]));
  assert.equal(m1.externs.unbound.size, 1);
  assert.equal(m2.externs.unbound.size, 0);
});

test('Externs snapshots survive bind and owner disposal',()=>{
  for(const owner of [host(),new Module(host())]) {
    const before=owner.externs;
    assert.equal(before.unbound.size,1);
    owner.bind('child',source());
    const after=owner.externs;
    owner.dispose();
    assert.equal(before.unbound.size,1);
    assert.equal(after.unbound.size,0);
    assert.equal(before.get('child').kind,'grammar');
    assert.equal(before._map,undefined);
  }
});

test('Grammar rejects direct and nested module expressions; Module.bind still accepts',()=>{
  const v=vocab(), parent=()=>Grammar.fromGlrm('start s extern grammar item nt s = item');
  const prepared=new Module(source());prepared.precompileSync(v);
  const error=e=>e.code==='GLRMASK_INVALID_INPUT' && e.message==="a Module can't be part of a Grammar; bind it to a Module instead";
  for(const m of [new Module(source()),prepared]){
    assert.throws(()=>Grammar.fromExpr(m),error);
    for(const expr of [Expr.sequence(m),Expr.sequence(Expr.choice(m),Expr.literal(''))]){
      assert.throws(()=>Grammar.fromExpr(expr),error);
      const g=parent();
      assert.throws(()=>g.bind('item',expr),error);
      assert.equal(g.externs.unbound.size,1);
      const outer=new Module(parent());outer.bind('item',expr);accept(outer,v,[1]);
    }
  }
});

test('Grammar: reject direct and nested TokenSet; Module accepts',()=>{
  const v=new Vocab([[0,Buffer.from('x')]]);
  const g=Grammar.fromGlrm('extern grammar VALUE start root nt root = VALUE');
  const token=new TokenSet([5]);
  for(const value of [token,Expr.choice(token,Expr.literal('x')),Expr.sequence(Expr.literal(''),token),Expr.sequence(Expr.choice(token,Expr.literal('x')),Expr.literal(''))]){
    for(const call of [()=>g.bind('VALUE',value),()=>Grammar.fromExpr(value)]){
      assert.throws(call, e=>e.code==='GLRMASK_INVALID_INPUT' && /token sets can only be bound to a Module/.test(e.message));
    }
    assert.equal(g.externs.unbound.size,1);
    const m=new Module(g);m.bind('VALUE',value);
    const state=m.compileSync(v).start();assert.ok(state.commitToken(5));assert.ok(state.isAccepting());
  }
  const t=Grammar.fromGlrm('extern token T start root nt root=T');
  assert.throws(()=>t.bind('T',token),/token sets can only be bound to a Module/);
  const m=new Module(t);m.bind('T',token);assert.ok(m.compileSync(v).start().commitToken(5));
});

test('Externs inspect includes unbound names',()=>{
  const m=new Module(host()), before=m.externs;
  assert.match(require('node:util').inspect(before),/unbound=.*child/);
  m.bind('child',source());
  assert.equal(before.unbound.size,1);
  assert.equal(m.externs.unbound.size,0);
  assert.match(require('node:util').inspect(m.externs),/unbound=\[\]/);
});
