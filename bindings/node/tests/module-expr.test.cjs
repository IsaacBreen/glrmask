'use strict';
const test=require('node:test');
const assert=require('node:assert/strict');
const {Vocab,Token,Grammar,Module,ModuleOptions,CompileOptions,Expr,Optimization,BoundaryTriggerDetail}=require('../index.cjs');
const vocab=(different=false)=>new Vocab([[0,Buffer.from('x')],[1,Buffer.from('a')],[2,Buffer.from('b')],[3,Buffer.from('y')],[4,Buffer.from('xay')],[5,Buffer.from('ab')],[8,Buffer.from(different?'z':'a')]],{exactOnlyIds:[63]});
const source=(text='a')=>Grammar.fromEbnf(`start ::= "${text}"`);
const host=()=>Grammar.fromGlrm('glrm 1; start start; extern grammar child; nt start = "x" child "y";');
const mark=()=>Grammar.fromGlrm('glrm 1; start start; extern token MARK; nt start = MARK;');
const allowed=(s,id)=>(s.mask()[id>>>5] & (1<<(id&31)))!==0;
function accept(module,v,ids){const s=module.compileSync(v).start();for(const id of ids){assert.ok(allowed(s,id));s.commitToken(id);}assert.ok(s.isAccepting());}

test('Module: frozen local settings, independent copies and root-only final options',()=>{
  const v=vocab(),original=host().module(),configured=original.withOptions({optimization:Optimization.Balanced});
  assert.ok(Object.isFrozen(configured.options));assert.equal(configured.options.optimization,Optimization.Balanced);
  assert.equal(original.options.optimization,Optimization.Auto);
  const options=new ModuleOptions({optimization:Optimization.Balanced,boundaryTrigger:BoundaryTriggerDetail.Tokens});
  const child=new Module(source(),options);assert.equal(child.precompileSync(v),undefined);
  const bytes=child.saveSync();assert.throws(()=>child.setOptions({optimization:Optimization.FastRuntime}));
  assert.deepEqual(child.saveSync(),bytes);accept(child,v,[1]);
  assert.equal(configured.bind('child',child),undefined);assert.equal(configured.unbound.length,0);
  assert.equal(original.unbound[0].name,'child');
  assert.deepEqual(configured.externs[0].path,['child']);
  const state=configured.compileSync(v,new CompileOptions({endTokens:[63]})).start();
  assert.ok(!allowed(state,63));state.commitToken(4);assert.ok(state.isAccepting());assert.ok(allowed(state,63));state.commitToken(63);assert.ok(state.mask().every(w=>w===0));
  assert.throws(()=>configured.compileSync(v,{optimization:Optimization.Auto}),TypeError);
  assert.throws(()=>new Module(source(),{endTokens:[63]}),TypeError);
});

for(const parentPrepared of [false,true])for(const childPrepared of [false,true]){
  test(`Module: saved mixed graph parent=${parentPrepared}, child=${childPrepared}`,async()=>{
    const v=vocab(),parent=host().module(),child=new Module(mark(),{optimization:Optimization.Balanced});
    if(parentPrepared)assert.equal(await parent.precompile(v),undefined);
    if(childPrepared)assert.equal(await child.precompile(v),undefined);
    parent.bind('child',child);const bytes=await parent.save();
    const loaded=await Module.load(bytes,v);
    assert.equal(loaded.unbound[0].name,'child::MARK');
    const value=v.tokens([8]);loaded.bind('child::MARK',value);value.dispose();parent.dispose();child.dispose();
    accept(loaded,v,[0,8,3]);assert.ok(!allowed(loaded.compileSync(v).start(),4));
    const queued=loaded.compile(v);loaded.dispose();v.dispose();const constraint=await queued;const state=constraint.start();constraint.dispose();
    for(const id of [0,8,3])state.commitToken(id);assert.ok(state.isAccepting());
  });
}

test('Module: raw source attachment, failure survival and target identity',()=>{
  const v=vocab(),parent=host().module();parent.precompileSync(v);const bytes=parent.saveSync();
  assert.throws(()=>parent.bind('missing',source()));assert.deepEqual(parent.saveSync(),bytes);
  const wrong=source().module();wrong.precompileSync(vocab(true));
  assert.throws(()=>parent.bind('child',wrong));assert.deepEqual(parent.saveSync(),bytes);
  assert.throws(()=>parent.precompileSync(vocab(true)));assert.deepEqual(parent.saveSync(),bytes);
  assert.throws(()=>Module.loadSync(bytes,vocab(true)));
  const loaded=Module.loadSync(bytes);loaded.bind('child',source());accept(loaded,v,[4]);
  const snapshot=loaded.saveSync();assert.throws(()=>loaded.bind('child',source('b')));assert.deepEqual(loaded.saveSync(),snapshot);
  const token=mark().module();assert.throws(()=>token.bind('MARK',source()));assert.equal(token.unbound[0].name,'MARK');
});

test('Expr: deterministic leaf paths, source conversion and crossing tokens',()=>{
  const v=vocab(),expr=Expr.sequence([source('x'),Expr.choice([source(),source('b')]),source('y')]);
  accept(new Module(expr),v,[4]);assert.ok(allowed(Grammar.fromExpr(expr).compileSync(v).start(),4));
  const open=Expr.choice([Expr.sequence([source('x'),mark()]),source('b')]);
  const module=new Module(open);assert.deepEqual(module.unbound[0].path,['leaf1','MARK']);
  module.bind('leaf1::MARK',new Token(8));accept(module,v,[0,8]);
  const parent=host();parent.bind('child',Expr.choice([source(),source('b')]));assert.ok(allowed(parent.compileSync(v).start(),4));
  const configured=new Module(source(),{optimization:Optimization.Balanced});
  assert.throws(()=>Grammar.fromExpr(Expr.choice([configured])));
  const before=host();assert.throws(()=>before.bind('child',Expr.choice([configured])));assert.equal(before.unbound[0].name,'child');
  accept(new Module(Expr.choice([configured])),v,[1]);
  assert.ok(new Module(Expr.sequence([])).compileSync(v).start().isAccepting());
  assert.throws(()=>new Module(Expr.choice([])));
  accept(new Module(Expr.choice([source()])),v,[1]);
});

test('Expr: iteration snapshots, self attachment and ownership after disposal',()=>{
  const v=vocab(),leaf=mark().module();
  function* values(){yield leaf;leaf.bind('MARK',new Token(1));yield source('b');}
  const expr=Expr.sequence(values()),module=new Module(expr);expr.dispose();leaf.dispose();
  assert.equal(module.unbound[0].name,'leaf0::MARK');module.bind('leaf0::MARK',new Token(1));assert.ok(!allowed(module.compileSync(v).start(),5));accept(module,v,[1,2]);
  const parent=host().module();parent.bind('child',parent);assert.equal(parent.unbound[0].name,'child::child');
  parent.bind('child::child',source());accept(parent,v,[0,4,3]);
});

test('Module: asynchronous precompile rejects a stale completion atomically',async()=>{
  const v=vocab(),parent=host().module();const pending=parent.precompile(v);
  parent.bind('child',source());
  await assert.rejects(pending,/changed during precompile/);accept(parent,v,[4]);
});
