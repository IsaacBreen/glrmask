'use strict';
const test=require('node:test');
const assert=require('node:assert/strict');
const {Vocab,TokenSet,Grammar,Module,Expr,MaskStrategy,GLRMaskError}=require('../index.cjs');
const vocab=(different=false)=>new Vocab([[0,Buffer.from('x')],[1,Buffer.from('a')],[2,Buffer.from('b')],[3,Buffer.from('y')],[4,Buffer.from('xay')],[5,Buffer.from('ab')],[8,Buffer.from(different?'z':'a')]]);
const source=(text='a')=>Grammar.fromLark(`start: "${text}"`);
const host=()=>Grammar.fromGlrm('glrm 1; start start; extern grammar child; nt start = "x" child "y";');
const mark=()=>Grammar.fromGlrm('glrm 1; start start; extern token MARK; nt start = MARK;');
const allowed=(s,id)=>(s.mask()[id>>>5] & (1<<(id&31)))!==0;
async function accept(module,v,ids){const s=(await module.compile(v)).start();for(const id of ids){assert.ok(allowed(s,id));assert.ok(s.commitToken(id));}assert.ok(s.isAccepting());}

test('Module: copy and root-only final options',async()=>{
  const v=vocab(),original=new Module(host());

  const configured=original.clone();
  const child=new Module(source());
  assert.equal((await child.precompile(v,{maskStrategy:MaskStrategy.FastMasks})),undefined);
  await accept(child,v,[1]);

  assert.equal(configured.bind('child',child),undefined);
  assert.equal(configured.externs.unbound.size,0);
  assert.equal(original.externs.unbound.size,1);
  assert.equal(original.externs.get('child').name,'child');

  const options={maskStrategy:MaskStrategy.FastMasks};
  const root=new Module(Grammar.fromGlrm('start s extern grammar item nt s = item'));
  root.bind('item',Expr.sequence(configured,new TokenSet([63])));
  const state=(await root.compile(v,options)).start();
  assert.ok(!allowed(state,63));assert.ok(state.commitToken(4));assert.ok(!state.isAccepting());assert.ok(allowed(state,63));assert.ok(state.commitToken(63));assert.ok(state.isAccepting());assert.ok(state.mask().every(w=>w===0));
  await assert.rejects(async()=>(await configured.compile(v,{bogus:true})),TypeError);
});

test('Module: precompile requires the same vocabulary in async calls', async()=>{
  const v=vocab(),wrong=vocab(true);
  for (const maskStrategy of Object.values(MaskStrategy)) {
    const m=new Module(host()), options={maskStrategy};
    await m.precompile(v,options);
    await assert.rejects(m.precompile(wrong,options),/vocabular/);
    m.bind('child',source());
    await assert.rejects(m.compile(wrong,options),/vocabular/);
    await accept(m,v,[4]);
  }
});

test('Module: vocabulary lock survives end-token delegation and roundtrip',async()=>{
  const v=new Vocab([[0,Buffer.from('a')],[1,Buffer.from('')],[2,Buffer.from('b')]]);
  const wrong=new Vocab([[0,Buffer.from('b')],[1,Buffer.from('')],[2,Buffer.from('a')]]);
  for (const maskStrategy of Object.values(MaskStrategy)) {
    const options={maskStrategy};
    const root=new Module(Grammar.fromGlrm('start s extern grammar BODY extern token END nt s = BODY END'));
    await root.precompile(v,options);
    for (const m of [root.clone(),(await Module.load((await root.save())))]) {
      await assert.rejects(m.precompile(wrong,options),/vocabular/);
      m.bind('BODY',source());m.bind('END',new TokenSet([1]));
      const outer=new Module(Grammar.fromGlrm('start s extern grammar INNER nt s = INNER'));
      outer.bind('INNER',m);
      for (const nested of [m,(await Module.load((await m.save()))),outer,(await Module.load((await outer.save())))]) {
        await assert.rejects(nested.compile(wrong,options),/vocabular/);
        await assert.rejects(nested.precompile(wrong,options),/vocabular/);
        const state=(await nested.compile(v,options)).start();
        assert.ok(state.commitToken(0));assert.ok(state.commitToken(1));assert.ok(state.isAccepting());
      }
    }
  }
});

test('Module: precompile retains work',async()=>{
  const v=vocab();
  const child=source();
  const parent=new Module(host());(await parent.precompile(v,{maskStrategy:MaskStrategy.FastMasks}));
  assert.ok(parent instanceof Module);
  const childModule=new Module(child);(await childModule.precompile(v,{maskStrategy:MaskStrategy.FastMasks}));parent.bind('child',childModule);
  const state=(await parent.compile(v)).start();
  assert.ok(allowed(state,4));assert.ok(state.commitToken(4));assert.ok(state.isAccepting());

  const reference=new Module(host());reference.bind('child',new Module(source()));(await reference.precompile(v));
  assert.deepEqual(Array.from((await parent.compile(v)).start().mask()),Array.from((await reference.compile(v)).start().mask()));
  assert.equal((await parent.compile(v)).start().isAccepting(),(await reference.compile(v)).start().isAccepting());

  // The asynchronous spelling returns the same bindable Module.
  const asyncParent=new Module(host());await asyncParent.precompile(v,{maskStrategy:MaskStrategy.FastCompile});
  assert.ok(asyncParent instanceof Module);
  const asyncChild=new Module(child);await asyncChild.precompile(v);asyncParent.bind('child',asyncChild);
  await accept(asyncParent,v,[4]);

  // Retained work is serialized, survives a round-trip, and a TokenTrie
  // compile keeps it instead of rebuilding dynamically.
  const retained=new Module(source());(await retained.precompile(v,{maskStrategy:MaskStrategy.FastMasks}));
  const bytes=(await retained.save());
  assert.ok(bytes.length>(await new Module(source()).save()).length);
  const restored=(await Module.load(bytes,v));
  assert.deepEqual(Array.from((await restored.compile(v,{maskStrategy:MaskStrategy.FastCompile})).start().mask()),
    Array.from((await new Module(source()).compile(v)).start().mask()));
});

test('Module: retained work round-trips without stored settings',async()=>{
  const v=vocab();
  const m=new Module(source());
  (await m.precompile(v,{maskStrategy:MaskStrategy.FastCompile}));
  const bytes=(await m.save());
  const loaded=(await Module.load(bytes,v));
  await accept(loaded,v,[1]);
});

test('Module: legacy boundary strategy strings are rejected',async()=>{
  const v=vocab();
  for (const input of ['fast-runtime','fast-build','dynamic']) {
    await assert.rejects(async()=>(await new Module(source()).precompile(v,{boundaryMaskStrategy:input})),TypeError);
  }
});

test('Module: rebinding error, :: error and open-child error',async()=>{
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

test('Module: externs order and unbound Map-like snapshot',async()=>{
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

test('Module: clone is independent and single binds work',async()=>{
  const v=vocab();
  const g=new Module(Grammar.fromGlrm('glrm 1; start root; extern grammar a; extern grammar b; nt root = a b;'));
  const copy=g.clone();
  copy.bind('a',source());
  assert.equal(g.externs.unbound.size,2);
  assert.equal(copy.externs.unbound.size,1);
  assert.ok(copy.externs.unbound.has('b'));
  await assert.rejects(async()=>(await g.compile(v)),/extern "a" is unbound/);
  await assert.rejects(async()=>(await copy.compile(v)),/extern "b" is unbound/);

  const m=new Module(Grammar.fromGlrm('glrm 1; start root; extern grammar a; extern grammar b; nt root = a b;'));
  assert.throws(()=>m.bind({a:source(),b:source('b')}),TypeError);
  assert.equal(m.externs.unbound.size,2);
  m.bind('a',source());
  m.bind('b',source('b'));
  const state=(await m.compile(v)).start();
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
    await accept(loaded,v,[0,8,3]);
  });
}

test('Module: raw source attachment, failure survival and target identity',async()=>{
  const v=vocab(),parent=new Module(host());(await parent.precompile(v));const bytes=(await parent.save());
  assert.throws(()=>parent.bind('missing',source()));assert.deepEqual((await parent.save()),bytes);
  const wrong=new Module(source());(await wrong.precompile(vocab(true)));
  assert.throws(()=>parent.bind('child',wrong));assert.deepEqual((await parent.save()),bytes);
  await assert.rejects(async()=>(await parent.precompile(vocab(true))));assert.deepEqual((await parent.save()),bytes);
  (await Module.load(bytes,vocab(true)));
  const loaded=(await Module.load(bytes));loaded.bind('child',source());await accept(loaded,v,[4]);
  const snapshot=(await loaded.save());assert.throws(()=>loaded.bind('child',source('b')));assert.deepEqual((await loaded.save()),snapshot);
  const token=new Module(mark());assert.throws(()=>token.bind('MARK',source()));assert.equal(token.externs.unbound.get('MARK').name,'MARK');
});

test('Expr: choice, sequence, literal and plain string errors',async()=>{
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
  await accept(mLit,v,[4]);

  // Choice & sequence
  const expr=Expr.sequence(source('x'),Expr.choice(source(),source('b')),source('y'));
  const mExpr=new Module(host());
  mExpr.bind('child',Expr.choice(source(),source('b')));
  await accept(mExpr,v,[4]);
});

test('Module: asynchronous precompile rejects a stale completion atomically',async()=>{
  const v=vocab(),parent=new Module(host());const pending=parent.precompile(v);
  parent.bind('child',source());
  await assert.rejects(pending,e=>e.code==="GLRMASK_INVALID_INPUT" && /changed during precompile/.test(e.message));await accept(parent,v,[4]);
});

test('Grammar.clone and Module.clone return independent copies', async() => {
  const g1 = Grammar.fromGlrm('glrm 1; start root; extern grammar a; nt root = a;');
  const g2 = g1.clone();
  assert.equal(g2.bind,undefined);
  assert.equal(g1.externs.unbound.size, 1);
  assert.equal(g2.externs.unbound.size, 1);

  const m1 = new Module(Grammar.fromGlrm('glrm 1; start root; extern token a; nt root = a;'));
  const m2 = m1.clone();
  assert.throws(() => m2.bind('a', 1), TypeError);
  m2.bind('a', new TokenSet([1]));
  assert.equal(m1.externs.unbound.size, 1);
  assert.equal(m2.externs.unbound.size, 0);
});

test('Externs snapshots survive bind and owner disposal',async()=>{
  const grammar=host(), snapshot=grammar.externs; grammar.dispose();
  assert.equal(snapshot.unbound.size,1);
  for(const owner of [new Module(host())]) {
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

test('Grammar rejects direct and nested module expressions; Module.bind still accepts',async()=>{
  const v=vocab(), parent=()=>Grammar.fromGlrm('start s extern grammar item nt s = item');
  const prepared=new Module(source());(await prepared.precompile(v));
  const error=e=>e.code==='GLRMASK_INVALID_INPUT' && e.message==="a Grammar expression cannot contain a Module";
  for(const m of [new Module(source()),prepared]){
    assert.throws(()=>Grammar.fromExpr(m),TypeError);
    for(const expr of [Expr.sequence(m),Expr.sequence(Expr.choice(m),Expr.literal(''))]){
      assert.throws(()=>Grammar.fromExpr(expr),error);
      const g=parent();
      assert.equal(g.bind,undefined);
      assert.equal(g.externs.unbound.size,1);
      const outer=new Module(parent());outer.bind('item',expr);await accept(outer,v,[1]);
    }
  }
});

test('Grammar: reject direct and nested TokenSet; Module accepts',async()=>{
  const v=new Vocab([[0,Buffer.from('x')]]);
  const g=Grammar.fromGlrm('extern grammar VALUE start root nt root = VALUE');
  const token=new TokenSet([5]);
  for(const value of [token,Expr.choice(token,Expr.literal('x')),Expr.sequence(Expr.literal(''),token),Expr.sequence(Expr.choice(token,Expr.literal('x')),Expr.literal(''))]){
    if(value instanceof TokenSet) assert.throws(()=>Grammar.fromExpr(value),TypeError);
    for(const call of value instanceof Expr ? [()=>Grammar.fromExpr(value)] : []){
      assert.throws(call, e=>e.code==='GLRMASK_INVALID_INPUT' && /a Grammar expression cannot contain a TokenSet/.test(e.message));
    }
    assert.equal(g.externs.unbound.size,1);
    const m=new Module(g);m.bind('VALUE',value);
    const state=(await m.compile(v)).start();assert.ok(state.commitToken(5));assert.ok(state.isAccepting());
  }
  const t=Grammar.fromGlrm('extern token T start root nt root=T');
  assert.equal(t.bind,undefined);
  const m=new Module(t);m.bind('T',token);assert.ok((await m.compile(v)).start().commitToken(5));
});

test('Module.bind accepts compiled Constraint with trailing token set',async()=>{
  const v=vocab(), body=new Module(Grammar.fromGlrm('start s extern grammar item nt s = item'));
  body.bind('item',Expr.sequence(source(),new TokenSet([63])));
  for(const childMode of Object.values(MaskStrategy)){
    const child=(await body.compile(v,{maskStrategy:childMode}));
    const direct=new Module(host()), wrapped=new Module(host());
    direct.bind('child',child); wrapped.bind('child',Expr.sequence(child));
    for(const maskStrategy of Object.values(MaskStrategy)){
      const a=(await direct.compile(v,{maskStrategy})).start(), b=(await wrapped.compile(v,{maskStrategy})).start();
      for(const id of [0,1,63,3]){
        assert.deepEqual(a.mask(),b.mask());assert.ok(allowed(a,id));
        assert.ok(a.commitToken(id));assert.ok(b.commitToken(id));
      }
      assert.deepEqual(a.mask(),b.mask());assert.ok(a.isAccepting()&&b.isAccepting());
    }
  }
});

test('Externs inspect includes unbound names',async()=>{
  const m=new Module(host()), before=m.externs;
  assert.match(require('node:util').inspect(before),/unbound=.*child/);
  m.bind('child',source());
  assert.equal(before.unbound.size,1);
  assert.equal(m.externs.unbound.size,0);
  assert.match(require('node:util').inspect(m.externs),/unbound=\[\]/);
});


test('async compile and save capture the call-time module and grammar snapshot',async()=>{
  const v=vocab(), g=host(), m=new Module(host());
  const gc=g.compile(v), mc=m.compile(v), ms=m.save();
  // The work has already captured an unbound source; later binding cannot change it.
  const checks=Promise.all([assert.rejects(gc,/unbound/),assert.rejects(mc,/unbound/)]);
  m.bind('child',source());
  await checks;
  const bytes=await ms, pending=Module.load(bytes);
  bytes.fill(0);
  const restored=await pending;
  assert.equal(restored.externs.unbound.size,1);
  await accept(m,v,[4]);
  await assert.rejects(async()=>(await g.compile(v)),/unbound/);
});

test('singleton expression binding matches direct values',async()=>{
  const v=vocab(),g=source(),m=new Module(g),c=await m.compile(v);
  for(const [value,kind,id] of [[g,'grammar',1],[m,'grammar',1],[c,'grammar',1],
        [Expr.literal('a'),'grammar',1],[new TokenSet([99]),'token',99]]) {
    const h=Grammar.fromGlrm('extern '+kind+' T start root nt root=T');
    const direct=new Module(h);direct.bind('T',value);
    for(const expr of [Expr.choice(value),Expr.sequence(value)]) {
      const wrapped=new Module(h);wrapped.bind('T',expr);
      const a=(await direct.compile(v)).start(),b=(await wrapped.compile(v)).start();
      assert.deepEqual(a.mask(),b.mask());
      assert.ok(a.commitToken(id)&&b.commitToken(id));
      assert.deepEqual(a.mask(),b.mask());
      assert.ok(a.isAccepting()&&b.isAccepting());
    }
  }
});
