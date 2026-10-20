'use strict';
const test=require('node:test');
const assert=require('node:assert/strict');
const {Vocab,Token,Grammar,Module,CompileOptions,Expr,MaskStrategy,BoundaryMaskStrategy}=require('../index.cjs');
const vocab=(different=false)=>new Vocab([[0,Buffer.from('x')],[1,Buffer.from('a')],[2,Buffer.from('b')],[3,Buffer.from('y')],[4,Buffer.from('xay')],[5,Buffer.from('ab')],[8,Buffer.from(different?'z':'a')]],{exactOnlyIds:[63]});
const source=(text='a')=>Grammar.fromEbnf(`start ::= "${text}"`);
const host=()=>Grammar.fromGlrm('glrm 1; start start; extern grammar child; nt start = "x" child "y";');
const mark=()=>Grammar.fromGlrm('glrm 1; start start; extern token MARK; nt start = MARK;');
const allowed=(s,id)=>(s.mask()[id>>>5] & (1<<(id&31)))!==0;
function accept(module,v,ids){const s=module.compileSync(v).start();for(const id of ids){assert.ok(allowed(s,id));s.commitToken(id);}assert.ok(s.isAccepting());}

test('Module: properties, setters, copy and root-only final options',()=>{
  const v=vocab(),original=new Module(host());
  assert.equal(original.maskStrategy,MaskStrategy.Auto);
  assert.equal(original.boundaryMaskStrategy,BoundaryMaskStrategy.FastBuild);

  original.maskStrategy=MaskStrategy.TokenClassTrie;
  assert.equal(original.maskStrategy,MaskStrategy.TokenClassTrie);
  original.boundaryMaskStrategy=BoundaryMaskStrategy.FastRuntime;
  assert.equal(original.boundaryMaskStrategy,BoundaryMaskStrategy.FastRuntime);

  const configured=original.copy();
  assert.equal(configured.maskStrategy,MaskStrategy.TokenClassTrie);
  assert.equal(configured.boundaryMaskStrategy,BoundaryMaskStrategy.FastRuntime);

  const child=new Module(source(),{maskStrategy:MaskStrategy.TokenClassTrie,boundaryMaskStrategy:BoundaryMaskStrategy.FastBuild});
  assert.equal(child.precompileSync(v),undefined);
  accept(child,v,[1]);

  assert.equal(configured.bind('child',child),undefined);
  assert.equal(configured.externs.unbound.size,0);
  assert.equal(original.externs.unbound.size,1);
  assert.equal(original.externs.get('child').name,'child');

  const state=configured.compileSync(v,new CompileOptions({endTokens:[63]})).start();
  assert.ok(!allowed(state,63));state.commitToken(4);assert.ok(state.isAccepting());assert.ok(allowed(state,63));state.commitToken(63);assert.ok(state.mask().every(w=>w===0));
  assert.throws(()=>new Module(source(),{endTokens:[63]}),TypeError);
});

test('Module: settings round-trip through save/load',()=>{
  const v=vocab();
  const m=new Module(source(),{maskStrategy:MaskStrategy.TokenTrie,boundaryMaskStrategy:BoundaryMaskStrategy.FastRuntime});
  m.precompileSync(v);
  const bytes=m.saveSync();
  const loaded=Module.loadSync(bytes,v);
  assert.equal(loaded.maskStrategy,MaskStrategy.TokenTrie);
  assert.equal(loaded.boundaryMaskStrategy,BoundaryMaskStrategy.FastRuntime);
  accept(loaded,v,[1]);
});

test('Module: rebinding error, :: error and open-child error',()=>{
  const h=new Module(host());
  h.bind('child',source());
  // Rebinding error
  assert.throws(()=>h.bind('child',source('b')),/already bound/);

  // Qualified paths rejected
  const p2=new Module(host());
  assert.throws(()=>p2.bind('foo::bar',source()),/qualified paths are not supported/);

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
  assert.throws(()=>m.bind({a:new Token(1),b:'not_a_valid_binding'}));
  assert.equal(m.externs.unbound.size,2);
  assert.ok(m.externs.unbound.has('a'));
  assert.ok(m.externs.unbound.has('b'));

  // Successful multi-bind
  m.bind({a:new Token(1),b:new Token(2)});
  assert.equal(m.externs.unbound.size,0);
});

for(const parentPrepared of [false,true])for(const childPrepared of [false,true]){
  test(`Module: saved mixed graph parent=${parentPrepared}, child=${childPrepared}`,async()=>{
    const v=vocab(),parent=new Module(host()),child=new Module(mark(),{maskStrategy:MaskStrategy.TokenClassTrie});
    if(parentPrepared)assert.equal(await parent.precompile(v),undefined);
    if(childPrepared)assert.equal(await child.precompile(v),undefined);
    child.bind('MARK',new Token(8));
    parent.bind('child',child);const bytes=await parent.save();
    const loaded=await Module.load(bytes,v);
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
  assert.throws(()=>Module.loadSync(bytes,vocab(true)));
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
