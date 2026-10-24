// The same API from TypeScript. Compile with TypeScript 5.2 or newer and NodeNext module resolution.
import { Grammar, MaskStrategy, Vocab, type Constraint, type ConstraintState } from '@glrmask/node';

const vocab = new Vocab([[0, new TextEncoder().encode('yes')], [1, new TextEncoder().encode('no')]]);
const grammar = Grammar.fromEbnf('start ::= "yes" | "no"');
const constraint: Constraint = await grammar.compile(vocab, { maskStrategy: MaskStrategy.TokenTrie });
const state: ConstraintState = constraint.start();
const mask: Uint32Array = state.mask();
console.log(`allowed: ${[0, 1].filter((id) => (mask[0] >>> id) & 1).join(', ')}`);
console.log(`committed "yes": ${state.commitToken(0)}, accepting: ${state.isAccepting()}`);
