'use strict';
const { Vocab, Grammar } = require('@glrmask/node');
async function main() {
  const vocab=new Vocab([[0,Buffer.from('null')],[1,Buffer.from('true')]]);
  const grammar=Grammar.fromJsonSchema({type:'null'});
  const constraint=await grammar.compile(vocab);
  const state=constraint.start();
  if ((state.mask()[0]&1)===0) throw new Error('null should be allowed');
  state.commitToken(0);
  if (!state.isAccepting()) throw new Error('null should be accepted');
  console.log('JSON Schema: accepted null');
  state.dispose(); constraint.dispose(); grammar.dispose(); vocab.dispose();
}
main().catch(error=>{console.error(error);process.exitCode=1;});
