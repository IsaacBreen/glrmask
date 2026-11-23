// A generation loop: mask the model's scores, take the best allowed token,
// commit it, and stop when nothing more can be committed.
//
// Random numbers stand in for the model.
import { Expr, Grammar, Vocab } from '@glrmask/node';

// One token per character, and an end-of-sequence token with no bytes.
const chars = '0123456789+*()';
const eos = chars.length;
const vocab = new Vocab(
  [...chars].map((c, id) => [id, Buffer.from(c)]),
);

// Arithmetic, then the end token.
const arithmetic = Grammar.fromGlrm(`
  start sum
  t NUMBER ::= /[0-9]+/
  nt sum ::= product | sum "+" product
  nt product ::= atom | product "*" atom
  nt atom ::= NUMBER | "(" sum ")"
`);
const grammar = Grammar.fromExpr(Expr.sequence(arithmetic, eos));
const constraint = await grammar.compile(vocab);

const state = constraint.start();
const mask = new Uint32Array(constraint.maskLength);
const allowed = (id) => ((mask[id >>> 5] >>> (id & 31)) & 1) === 1;
let text = '';
while (!state.isTerminated()) {
  const scores = Array.from({ length: eos + 1 }, Math.random); // Your model's scores go here.
  // This model grows keener to stop as the text gets longer.
  scores[eos] += 0.1 * text.length;
  state.fillMask(mask);
  let token = -1;
  for (let id = 0; id < scores.length; id++) {
    if (allowed(id) && (token < 0 || scores[id] > scores[token])) token = id;
  }
  if (!state.commitToken(token)) throw new Error(`token ${token} was not allowed`);
  if (token !== eos) text += chars[token];
}
console.log(text);
