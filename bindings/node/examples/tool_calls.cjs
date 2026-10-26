'use strict';
// Tool calls: precompile each tool's arguments once, then bind them into a call
// format. Compiling the combination reuses the precompiled work.
const { Grammar, Module, Vocab } = require('@glrmask/node');

async function main() {
  // One token per byte.
  const vocab = new Vocab(Array.from({ length: 256 }, (_, id) => [id, Uint8Array.of(id)]));

  const weather = new Module(Grammar.fromJsonSchema({
    type: 'object',
    properties: { city: { type: 'string' } },
    required: ['city'],
    additionalProperties: false,
  }));
  const search = new Module(Grammar.fromJsonSchema({
    type: 'object',
    properties: { query: { type: 'string' } },
    required: ['query'],
    additionalProperties: false,
  }));
  await weather.precompile(vocab);
  await search.precompile(vocab);

  const call = new Module(Grammar.fromGlrm(`
    extern grammar WEATHER_ARGS
    extern grammar SEARCH_ARGS
    start call
    nt call ::= "<weather>" WEATHER_ARGS "</weather>" | "<search>" SEARCH_ARGS "</search>"
  `));
  call.bind({ WEATHER_ARGS: weather, SEARCH_ARGS: search });
  const constraint = await call.compile(vocab);

  const accepted = constraint.start();
  if (!accepted.commitBytes(Buffer.from('<weather>{"city": "Perth"}</weather>')) || !accepted.isAccepting()) {
    throw new Error('the weather call should be accepted');
  }
  console.log('accepted a weather call');

  const rejected = constraint.start();
  if (rejected.commitBytes(Buffer.from('<weather>{"query"'))) {
    throw new Error('search arguments should be rejected in a weather call');
  }
  console.log('rejected a weather call with search arguments');
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
