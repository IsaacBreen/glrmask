```js
import assert from 'node:assert/strict';
import { Vocab, Grammar } from '@glrmask/node';

const vocab = new Vocab([
  [0, Buffer.from('hello')],
  [1, Buffer.from(' ')],
  [2, Buffer.from('world')],
]);
const grammar = Grammar.fromLark('start: "hello" " " "world"');
const constraint = await grammar.compile(vocab);
const state = constraint.start();
const mask = new Uint32Array(constraint.maskLength);

for (const [id, expected] of [[0, 1], [1, 2], [2, 4]]) {
  state.fillMask(mask);
  assert.equal(mask[0] & 7, expected);
  assert.equal(state.commitToken(id), true);
}
assert.equal(state.isAccepting(), true);
```
