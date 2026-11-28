```js
// [hidden]
import { Vocab, Grammar, Module } from '@glrmask/node';
const tokenBytes = Array.from({ length: 256 }, (_, id) => [id, Uint8Array.of(id)]);
// [/hidden]
// Token ID -> exact bytes from tokenizer
const vocab = new Vocab(tokenBytes);

const weather = await Grammar
  .fromJsonSchema({
  type: 'object',
  properties: {
    city: { type: 'string' }
  },
  required: ['city'],
  additionalProperties: false,
}).compile(vocab);

const parent = Grammar.fromGlrm(`
  extern grammar ARGS
  start call
  nt call = "<weather>" ARGS
            "</weather>"
`);
const call = new Module(parent);
call.bind('ARGS', weather);
const constraint = await call.compile(vocab);
// [hidden]
const state = constraint.start();
if (!state.commitBytes(Buffer.from('<weather>{"city": "Perth"}</weather>')) || !state.isAccepting()) throw new Error('composition failed');
// [/hidden]
```
