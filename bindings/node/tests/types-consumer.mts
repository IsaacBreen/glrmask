import {Vocab, Grammar, Module, Expr, MaskStrategy, Constraint, type CompileOptions} from '@glrmask/node';
import api = require('@glrmask/node');
const v = new Vocab([[0, new TextEncoder().encode('a')]]);
const sameVocab: api.Vocab = v;
const options: CompileOptions = {maskStrategy: MaskStrategy.ParserDwa};
const c = await Grammar.fromJsonSchema({type: 'string'}).compile(v, options);
const bytes = await c.save(); const loaded = await Constraint.load(bytes);
const words: Uint32Array = loaded.start().mask();

const module = new Module(Grammar.fromEbnf('start ::= "a"'));
const precompiled: Module = await Grammar.fromEbnf('start ::= "a"').precompile(v, options);
void precompiled;
await module.precompile(v, {maskStrategy: MaskStrategy.TokenClassTrie});
const expr = Expr.choice(module, Grammar.fromEbnf('start ::= "a"'));
const restored = await Module.load(await module.save(), v);
const canonical: api.Constraint = await restored.compile(v, {maskStrategy: MaskStrategy.TokenClassTrie});
void [sameVocab, words, expr, canonical];
