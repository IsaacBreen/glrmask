// Compile with TypeScript 5.2+ and NodeNext module resolution.
import { Vocab, Grammar, MaskStrategy, type Constraint } from '@glrmask/node';
const vocab = new Vocab([[0,new TextEncoder().encode('a')]]);
const constraint: Constraint = await Grammar.fromEbnf('start ::= "a"').compile(vocab, {
  maskStrategy: MaskStrategy.TokenTrie,
});
using state = constraint.start();
const mask: Uint32Array = new Uint32Array(constraint.maskLength);
state.fillMask(mask);
state.commitToken(0);
console.log(state.isAccepting());
constraint.dispose(); vocab.dispose();
