import { Vocab, Grammar, MaskStrategy } from '@glrmask/node';
const vocab = new Vocab([[0,Buffer.from('hello')],[1,Buffer.from(' ')],[2,Buffer.from('world')]]);
const grammar = Grammar.fromEbnf('start ::= "hello" " " "world"');
const constraint = await grammar.compile(vocab, {maskStrategy: MaskStrategy.ParserDwa});
const state = constraint.start();
const mask = new Uint32Array(constraint.maskLength);
for (const id of [0,1,2]) {
  state.fillMask(mask);
  if ((mask[Math.floor(id/32)] & (1 << (id%32))) === 0) throw new Error(`Token ${id} is not permitted`);
  state.commitToken(id);
}
if (!state.isAccepting()) throw new Error('The completed sequence was not accepted');
console.log('hello world: accepted');
state.dispose(); constraint.dispose(); grammar.dispose(); vocab.dispose();
