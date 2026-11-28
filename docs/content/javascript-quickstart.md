```js
// [hidden]
import { Vocab, Grammar } from '@glrmask/node';
// Replace with your tokenizer's token bytes
const tokenBytes = Array.from({ length: 256 }, (_, id) => [id, Uint8Array.of(id)]);
const EOS = 256;
const target = Buffer.from('{"color": "blue"}');
function model(output) {
  const logits = new Float32Array(257).fill(-1);
  logits[output.length < target.length ? target[output.length] : EOS] = 1;
  return logits;
}
function sample(logits, mask, accepting) {
  let token = -1;
  for (let id = 0; id <= EOS; id++) {
    const allowed = id === EOS ? accepting : ((mask[id >>> 5] >>> (id & 31)) & 1) === 1;
    if (allowed && (token < 0 || logits[id] > logits[token])) token = id;
  }
  if (token < 0) throw new Error('no allowed token and cannot stop');
  return token;
}
// [/hidden]
const grammar = Grammar.fromJsonSchema({
  type: 'object',
  properties: {
    color: {
      enum: ['red', 'green', 'blue']
    }
  },
  required: ['color'],
  additionalProperties: false,
});

// Token ID -> exact bytes from tokenizer
const vocab = new Vocab(tokenBytes);
const constraint = await grammar.compile(
  vocab
);
const state = constraint.start();

const output = [];
while (true) {
  const logits = model(output);
  const mask = state.mask();
  const token = sample(
    logits, mask, state.isAccepting()
  );
  if (token === EOS) break;
  if (!state.commitToken(token)) {
    throw new Error('Rejected token');
  }
  output.push(token);
}
// [hidden]
if (!state.isAccepting()) throw new Error('incomplete output');
console.log(Buffer.from(output).toString());
// [/hidden]
```
