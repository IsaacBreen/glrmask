'use strict';
const test=require('node:test');
const assert=require('node:assert/strict');
const {types}=require('node:util');
const {Vocab,Token,Grammar,PreparedGrammar,Constraint,ConstraintState,ExactTokens,Optimization,LarkProfile,BoundaryTriggerDetail,GlrMaskError,version}=require('../index.cjs');
const raw=require('../loader.cjs');
const modes=Object.values(Optimization);
const b=s=>Buffer.from(s);
const vocab=()=>new Vocab([[0,b('a')],[1,b('b')],[31,b('ab')],[32,b('a')],[65,b('z')]]);
const simple=()=>Grammar.fromEbnf('start ::= "a" "b"');
const allowed=(m,id)=>(m[Math.floor(id/32)] & (1<<(id%32)))!==0;
const closed=fn=>assert.throws(fn,e=>e instanceof GlrMaskError && e.code==='GLRMASK_CLOSED');

for(const optimization of modes) {
  test(`${optimization}: packed masks, sparse IDs, aliases, forcing, clone and owner disposal`,()=>{
    const v=vocab(),g=simple(),c=g.compileSync(v,{optimization}),s=c.start();
    assert.equal(version,'0.3.0');assert.equal(c.maskLength,3);assert.equal(s.maskLength,3);
    const initial=s.mask();assert.ok(types.isUint32Array(initial));
    assert.ok(allowed(initial,0));assert.ok(allowed(initial,31));assert.ok(allowed(initial,32));
    assert.ok(!allowed(initial,1));assert.ok(!allowed(initial,65));
    const backing=new Uint32Array(7).fill(123),view=backing.subarray(1,6);
    assert.equal(s.fillMask(view),undefined);assert.deepEqual(view.slice(0,3),initial);
    assert.deepEqual(Array.from(backing),[123,...initial,0,0,123]);
    const short=new Uint32Array([123]);
    assert.throws(()=>s.fillMask(short),e=>e.code==='GLRMASK_BUFFER_TOO_SMALL');assert.equal(short[0],123);
    const copy=s.clone();c.dispose();g.dispose();v.dispose();closed(()=>c.start());
    s.commitToken(31);assert.ok(s.isAccepting());assert.ok(!copy.isAccepting());
    copy.commitToken(0);assert.deepEqual(Array.from(copy.forcedTokens()),[1]);assert.ok(!copy.isAccepting());
    copy.commitToken(1);assert.ok(copy.isAccepting());assert.ok(!copy.isRejected());
    s.dispose();s.dispose();closed(()=>s.mask());closed(()=>s.commitToken(0));
    copy[Symbol.dispose]();closed(()=>copy.isAccepting());
  });
  test(`${optimization}: raw bytes, partial UTF-8 and exact-only events`,()=>{
    const v=new Vocab([[0,Buffer.from([0xc3])],[1,Buffer.from([0xa9])],[31,b('é')],[32,b('é')],[40,Buffer.from([0,255])]],{exactOnlyIds:[99]});
    const c=Grammar.fromEbnf('start ::= "é"').compileSync(v,{optimization}),s=c.start();
    assert.ok(allowed(s.mask(),0));assert.ok(allowed(s.mask(),31));s.commitBytes(Buffer.from([0xc3]));
    assert.ok(!s.isAccepting());assert.ok(allowed(s.mask(),1));s.commitBytes(Buffer.from([0xa9]));assert.ok(s.isAccepting());
    for(const id of [31,40,99]) {
      const e=Grammar.exact(v.tokens([id])).compileSync(v,{optimization}).start();
      assert.ok(allowed(e.mask(),id));assert.ok(!allowed(e.mask(),32));e.commitToken(id);assert.ok(e.isAccepting());
    }
  });
  test(`${optimization}: end-token policy is separate from acceptance`,()=>{
    const v=new Vocab([[0,b('a')]],{exactOnlyIds:[99]});
    const s=Grammar.fromEbnf('start ::= "a"').compileSync(v,{optimization,endTokens:[99]}).start();
    assert.ok(!allowed(s.mask(),99));s.commitToken(0);assert.ok(s.isAccepting());assert.ok(allowed(s.mask(),99));
    s.commitToken(99);assert.ok(s.isAccepting());assert.ok(!s.isRejected());assert.ok(s.mask().every(w=>w===0));
    assert.ok(s.clone().isAccepting());assert.throws(()=>s.commitToken(99));
  });
  test(`${optimization}: synchronous and asynchronous persistence`,async()=>{
    const v=vocab(),c=await simple().compile(v,{optimization});
    const expected=c.start().mask();
    const bytes=await c.save();assert.ok(types.isUint8Array(bytes));
    const copied=Buffer.from(bytes);const pending=Constraint.load(bytes);bytes.fill(0); // queued load already owns a copy
    assert.deepEqual((await pending).start().mask(),expected);
    assert.deepEqual(Constraint.loadSync(copied).start().mask(),expected);
    const external=await c.save({externalVocab:true});await assert.rejects(Constraint.load(external));
    assert.deepEqual((await Constraint.load(external,v)).start().mask(),expected);
    assert.deepEqual(Constraint.loadSync(external,v).start().mask(),expected);
    const wrong=new Vocab([[0,b('WRONG')],[1,b('b')],[31,b('ab')],[32,b('a')],[65,b('z')]]);
    await assert.rejects(Constraint.load(external,wrong));
    const saving=c.save();c.dispose();assert.ok((await saving).length>0);
    assert.throws(()=>Constraint.loadSync(Buffer.from('not an artifact')));
  });
  test(`${optimization}: markers use exact BigInt byte offsets and survive snapshots`,async()=>{
    const v=vocab(),g=Grammar.fromGlrm('glrm 1; start root; nt root = @mark(10) "a" @mark(20) "b" @mark(30);');
    const c=await g.compile(v,{optimization});assert.deepEqual(Array.from(c.markerIds()),[10,20,30]);
    const s=c.start();assert.deepEqual(s.markerEvents(),[{id:10,byteOffset:0n}]);
    s.commitToken(31);const expected=[{id:20,byteOffset:1n},{id:30,byteOffset:2n}];
    assert.deepEqual(s.markerEvents(),expected);assert.deepEqual(s.clone().markerEvents(),expected);
    assert.deepEqual((await Constraint.load(await c.save())).markerIds(),c.markerIds());
  });
}

test('compact ranges and full-width numeric IDs without accidental expansion',()=>{
  const v=vocab(),t=v.tokenRanges([{start:31,end:33},{start:32,end:33}]);
  assert.equal(t.size,2);assert.deepEqual(t.ranges,[{start:31,end:33}]);assert.deepEqual(Array.from(t.ids),[31,32]);
  const copy=t.clone(),kept=t.ids;t.dispose();assert.equal(copy.size,2);assert.deepEqual(Array.from(kept),[31,32]);
  const empty=v.tokens([]);assert.equal(empty.size,0);assert.deepEqual(empty.ranges,[]);
  assert.throws(()=>v.tokens([31,31]));assert.throws(()=>v.tokenRanges([{start:32,end:31}]));
  const high=new Vocab([],{exactOnlyIds:[0xffffffff]});
  assert.equal(high.size,0);assert.deepEqual(high.tokenRanges([{start:0xffffffff,end:4294967296}]).ranges,[{start:0xffffffff,end:4294967296}]);
  assert.equal(new Token(0xffffffff).id,0xffffffff);
});

test('source/prepared transactional binding, qualified paths and independent snapshots',async()=>{
  const v=new Vocab([[0,b('x')],[1,b('a')],[2,b('b')],[3,b('y')],[4,b('xay')],[5,b('xby')]]);
  const host=Grammar.fromGlrm('glrm 1; start root; extern grammar child; nt root = "x" child "y";');
  assert.equal(host.requirements.length,1);assert.equal(host.requirements[0].name,'child');
  assert.equal(host.requirements[0].kind,'grammar');assert.ok(host.requirements[0].sourceSpan.line>=1);
  assert.ok(Object.isFrozen(host.requirements[0]));
  const a=Grammar.fromEbnf('start ::= "a"'),bb=Grammar.fromEbnf('start ::= "b"');
  const left=host.clone(),right=host.clone();assert.equal(left.bind('child',a),undefined);right.bind('child',bb);
  assert.equal(host.requirements.length,1);assert.deepEqual(left.requirements,[]);
  assert.throws(()=>left.bind('child',bb));assert.deepEqual(left.requirements,[]);
  const prepared=await host.prepare(v),child=await a.prepare(v);prepared.bind('child',child);child.dispose();
  const pendingSave=prepared.save();prepared.dispose();
  const restored=await PreparedGrammar.load(await pendingSave,v);assert.deepEqual(restored.requirements,[]);
  assert.deepEqual(PreparedGrammar.loadSync(restored.saveSync()).requirements,[]);
  for(const optimization of modes) {
    const lc=await left.compile(v,{optimization}),rc=right.compileSync(v,{optimization}),pc=await restored.link({optimization});
    assert.ok(allowed(lc.start().mask(),4));assert.ok(!allowed(lc.start().mask(),5));
    assert.ok(allowed(rc.start().mask(),5));assert.ok(!allowed(rc.start().mask(),4));
    assert.deepEqual(pc.start().mask(),lc.start().mask());assert.deepEqual(restored.linkSync({optimization}).start().mask(),lc.start().mask());
  }
  const leaf=Grammar.fromGlrm('glrm 1; start root; extern token MARK; nt root = MARK;');
  const nested=host.clone();nested.bind('child',leaf);leaf.bind('MARK',new Token(2));
  assert.equal(nested.requirements[0].name,'child::MARK');assert.deepEqual(nested.requirements[0].path,['child','MARK']);
  nested.bind('child::MARK',new Token(1));assert.deepEqual(nested.requirements,[]);
  const s=(await nested.compile(v)).start();assert.ok(!allowed(s.mask(),4));s.commitToken(0);s.commitToken(1);s.commitToken(3);assert.ok(s.isAccepting());
  assert.throws(()=>host.bind('missing',a));assert.equal(host.requirements.length,1);
  assert.throws(()=>restored.bind('child',a),TypeError);
});

test('async work snapshots sources and vocabulary before disposal',async()=>{
  const v=vocab(),g=simple();const p=g.compile(v);g.dispose();v.dispose();
  const c=await p;assert.ok(allowed(c.start().mask(),31));
  const v2=vocab(),g2=simple(),p2=g2.prepare(v2);g2.dispose();v2.dispose();
  const prepared=await p2,linked=prepared.link();prepared.dispose();assert.ok(allowed((await linked).start().mask(),31));
});

test('an active JS timer continues during native compilation',{timeout:60000},async()=>{
  const values=Array.from({length:2000},(_,i)=>JSON.stringify(`prefix_${i.toString().padStart(4,'0')}_suffix`));
  const g=Grammar.fromEbnf('start ::= '+values.join(' | '));
  const v=new Vocab(Array.from({length:128},(_,i)=>[i,Buffer.from([i])]));
  let beats=0;const interval=setInterval(()=>beats++,1);
  try {await g.compile(v,{optimization:Optimization.FastRuntime});} finally {clearInterval(interval);}
  assert.ok(beats>0,'native compilation blocked the JavaScript event loop');
});

test('frontends and input validation preserve the public contracts',async()=>{
  const v=vocab();
  for(const profile of Object.values(LarkProfile)) {
    const c=await Grammar.fromLark('start: "a" "b"',{profile}).compile(v);const s=c.start();s.commitToken(31);assert.ok(s.isAccepting());
  }
  const j=new Vocab([[0,b('null')],[1,b('true')]]);
  const s=(await Grammar.fromJsonSchema({type:'null'}).compile(j)).start();assert.ok(allowed(s.mask(),0));assert.ok(!allowed(s.mask(),1));
  assert.throws(()=>new Vocab([[0,'a']]),TypeError);assert.throws(()=>new Vocab([[0,b('a')],[0,b('b')]]),GlrMaskError);
  assert.throws(()=>new Vocab([[0,b('a')]],{exactOnlyIds:[0]}),GlrMaskError);
  for(const bad of [-1,0.5,NaN,Infinity,4294967296,'1',1n,null]) {
    assert.throws(()=>new Token(bad),RangeError);assert.throws(()=>v.tokens([bad]),RangeError);
    assert.throws(()=>s.commitToken(bad),RangeError);
  }
  for(const Class of [Constraint,ConstraintState,Grammar,PreparedGrammar,ExactTokens]) assert.throws(()=>new Class(),TypeError);
  assert.throws(()=>simple().compileSync(v,{optimisation:'auto'}),TypeError);
  assert.throws(()=>simple().compileSync(v,{optimization:'bogus'}),TypeError);
  assert.throws(()=>simple().compileSync(v,{boundaryTrigger:'bogus'}),TypeError);
  assert.throws(()=>simple().prepareSync(v,{optimization:Optimization.FastBuild}),TypeError);
  await assert.rejects(simple().compile(v,{endTokens:[-1]}),RangeError);
  assert.throws(()=>s.fillMask(new Int32Array(3)),TypeError);
  assert.throws(()=>s.commitBytes('a'),TypeError);
  assert.throws(()=>Grammar.fromLark('start: "a"',{profile:'other'}),TypeError);
  await assert.rejects(Grammar.fromEbnf('this is not a grammar').compile(v));
});

test('raw native arguments reject truncation and unsafe typed-array backing',()=>{
  const v=raw.NativeVocab.fromEntries([0,1,31],[b('a'),b('b'),b('ab')],[]);
  const g=raw.NativeGrammar.fromSource('ebnf','start ::= "a" "b"');const s=g.compileSync(v,0,[]).start();
  const initial=s.mask();
  for(const bad of [-1,0.1,NaN,Infinity,4294967296]) {
    assert.throws(()=>s.commitToken(bad),/GLRMASK_INVALID_ARGUMENT/);
    assert.throws(()=>v.tokens([bad]),/GLRMASK_INVALID_ARGUMENT/);
    assert.throws(()=>g.compileSync(v,bad,[]),/GLRMASK_INVALID_ARGUMENT/);
  }
  assert.deepEqual(s.mask(),initial);
  const sharedWords=new Uint32Array(new SharedArrayBuffer(16));
  assert.throws(()=>s.fillMask(sharedWords),/SharedArrayBuffer/);
  const sharedBytes=new Uint8Array(new SharedArrayBuffer(4));
  assert.throws(()=>s.commitBytes(sharedBytes),/SharedArrayBuffer/);
  assert.throws(()=>raw.NativeVocab.fromEntries([0],[sharedBytes],[]),/SharedArrayBuffer/);
  const words=new Uint32Array(4);structuredClone(words.buffer,{transfer:[words.buffer]});
  assert.throws(()=>s.fillMask(words),/detached|typed array/);
  const bytes=new Uint8Array([97]);structuredClone(bytes.buffer,{transfer:[bytes.buffer]});
  assert.throws(()=>s.commitBytes(bytes),/detached|typed array/);
  assert.throws(()=>s.fillMask(new DataView(new ArrayBuffer(16))),/typed array/);
  assert.deepEqual(s.mask(),initial);
  // Direct self-binding is safe even without the public JS facade: two shared
  // references plus separately acquired native mutex guards, never aliased &mut.
  const recursive=raw.NativeGrammar.fromSource('glrm','glrm 1; start root; extern grammar child; nt root = child;');
  recursive.bindGrammar('child',recursive);
  assert.equal(recursive.requirements()[0].name,'child::child');
});

test('garbage collection cannot invalidate a retained state or materialized mask',()=>{
  assert.equal(typeof global.gc,'function','run tests with --expose-gc');
  function stateOnly(){const v=vocab(),g=simple(),c=g.compileSync(v);return c.start();}
  const s=stateOnly(),mask=s.mask();global.gc();global.gc();
  assert.ok(allowed(mask,31));s.commitToken(31);assert.ok(s.isAccepting());
  const kept=s.clone();s.dispose();global.gc();assert.ok(kept.isAccepting());assert.ok(allowed(mask,31));
});

test('ESM and CommonJS share constructor identities',async()=>{
  const esm=await import('../index.mjs');assert.equal(esm.Vocab,Vocab);assert.equal(esm.Grammar,Grammar);assert.equal(esm.version,version);
});

test('named Lark special tokens use explicit mappings, not byte aliases',async()=>{
  const v=new Vocab([[0,b('a')],[1,b('a')]],{exactOnlyIds:[99]});
  for(const optimization of modes) {
    const g=Grammar.fromLark('start: <|begin|> <|body|>',{profile:LarkProfile.Llm,tokenNames:{begin:99,body:[1]}});
    const s=(await g.compile(v,{optimization})).start();assert.ok(allowed(s.mask(),99));s.commitToken(99);
    assert.ok(allowed(s.mask(),1));assert.ok(!allowed(s.mask(),0));s.commitToken(1);assert.ok(s.isAccepting());
  }
  await assert.rejects(Grammar.fromLark('start: <|missing|>',{profile:LarkProfile.Llm}).compile(v));
});
