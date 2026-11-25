// A generation loop: mask the model's scores, take the best allowed token,
// stop if EOS is chosen, otherwise commit it.
//
// Random numbers stand in for the model.
import { Grammar, Vocab } from '@glrmask/node';

// One token per character, plus a caller-managed EOS ID.
const chars = '0123456789+*()';
const eos = chars.length;
const vocab = new Vocab(
  [...chars].map((c, id) => [id, Buffer.from(c)]),
);

// Arithmetic grammar.
const arithmetic = Grammar.fromGlrm(`
  start sum
  t NUMBER = /[0-9]+/
  nt sum = product | sum "+" product
  nt product = atom | product "*" atom
  nt atom = NUMBER | "(" sum ")"
`);
const constraint = await arithmetic.compile(vocab);

const state = constraint.start();
const mask = new Uint32Array(Math.max(constraint.maskLength, (eos >>> 5) + 1));
const allowed = (id) => ((mask[id >>> 5] >>> (id & 31)) & 1) === 1;
let text = '';
while (true) {
  const scores = Array.from({ length: eos + 1 }, Math.random); // Your model's scores go here.
  // This model grows keener to stop as the text gets longer.
  scores[eos] += 0.1 * text.length;
  state.fillMask(mask);
  if (state.isAccepting()) mask[eos >>> 5] |= 1 << (eos & 31);
  let token = -1;
  for (let id = 0; id < scores.length; id++) {
    if (allowed(id) && (token < 0 || scores[id] > scores[token])) token = id;
  }
  if (token < 0) throw new Error("no allowed token");
  if (token === eos) break;
  if (!state.commitToken(token)) throw new Error(`token ${token} was not allowed`);
  text += chars[token];
}
if (!state.isAccepting()) throw new Error("incomplete output");
console.log(text);
