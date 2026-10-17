'use strict';
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const {spawnSync}=require('node:child_process');
const api=require('../index.cjs');
const cpp=process.argv[2], directory=process.argv[3];
if (!cpp || !directory) throw new Error('Usage: node scripts/parity.cjs CPP_TEST_EXECUTABLE OUTPUT_DIRECTORY');
const executable=path.resolve(cpp), output=path.resolve(directory);
if (!fs.statSync(executable).isFile()) throw new Error('Missing C++ test executable');
fs.mkdirSync(output,{recursive:true});
function run(args) {
  const result=spawnSync(executable,args,{encoding:'utf8',timeout:120000});
  if (result.error) throw result.error;
  if (result.status!==0) throw new Error(`C++ parity helper failed (${result.status}): ${result.stderr}\n${result.stdout}`);
  return result.stdout;
}
const vocab=new api.Vocab([[0,Buffer.from('a')],[1,Buffer.from('b')],[31,Buffer.from('ab')],[32,Buffer.from('a')],[65,Buffer.from('z')]]);
const grammar=api.Grammar.fromEbnf('start ::= "a" "b"');
const modes=Object.values(api.Optimization);
const expected=JSON.parse(run(['--parity']));
const actual=modes.map((optimization,mode)=>{
  const c=grammar.compileSync(vocab,{optimization}),s=c.start(),states=[];
  for (let step=0;step<3;++step) {
    if(step) s.commitToken(step-1);
    states.push({mask:Array.from(s.mask()),accepting:s.isAccepting(),rejected:s.isRejected()});
  }
  s.dispose();c.dispose();return {mode,states};
});
assert.deepEqual(actual,expected,'C++ and Node mask/commit/state results differ');
run(['--save',output]);
for (const [mode,optimization] of modes.entries()) {
  const reference=grammar.compileSync(vocab,{optimization});
  for (const externalVocab of [false,true]) {
    const stem=(externalVocab?'external-':'mode-')+mode+'.glr';
    const cppBytes=fs.readFileSync(path.join(output,'cpp-'+stem));
    const c=api.Constraint.loadSync(cppBytes,externalVocab?vocab:undefined),s=c.start();
    assert.deepEqual(s.mask(),reference.start().mask());s.commitToken(31);assert.ok(s.isAccepting());
    fs.writeFileSync(path.join(output,'node-'+stem),reference.saveSync({externalVocab}));
    s.dispose();c.dispose();
  }
  reference.dispose();
}
run(['--check',output]);
fs.writeFileSync(path.join(output,'parity-report.json'),JSON.stringify({schema:1,version:api.version,modes:actual,artifactChecks:16},null,2)+'\n');
console.log('C++/Node: four-mode masks and state transitions match; 16 two-way embedded/external artifact checks passed.');
