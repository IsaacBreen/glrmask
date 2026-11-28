```js
import { Vocab, Grammar } from '@glrmask/node';

const grammar = Grammar.fromJsonSchema({
  type: 'object',
  properties: { color: { enum: ['red', 'green', 'blue'] } },
  required: ['color'],
  additionalProperties: false,
});

// One token per byte, plus EOS. With a real model, build this from its tokenizer.
const vocab = new Vocab(Array.from({ length: 256 }, (_, id) => [id, Uint8Array.of(id)]));
const EOS = 256;
const state = (await grammar.compile(vocab)).start();
const sample = Buffer.from('{"color": "blue"}');
const output = [];
while (true) {
  // Stand-in scores for a model that wants to produce sample.
  const logits = new Float32Array(257).fill(-1);
  logits[output.length < sample.length ? sample[output.length] : EOS] = 1;
  const mask = state.mask();
  let token = -1;
  for (let id = 0; id <= EOS; id++) {
    const allowed = id === EOS ? state.isAccepting()
      : ((mask[id >>> 5] >>> (id & 31)) & 1) === 1;
    if (allowed && (token < 0 || logits[id] > logits[token])) token = id;
  }
  if (token < 0) throw new Error('no allowed token and cannot stop');
  if (token === EOS) break;
  if (!state.commitToken(token)) throw new Error('allowed token rejected');
  output.push(token);
}
if (!state.isAccepting()) throw new Error("incomplete output");
console.log(Buffer.from(output).toString());
```
