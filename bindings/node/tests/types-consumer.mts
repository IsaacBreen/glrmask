import {Vocab, Grammar, Module, Expr, MaskStrategy, ParserBackend, Constraint, type CompileOptions, type TokenTrie} from '@glrmask/node';
import api = require('@glrmask/node');
const v = new Vocab([[0, new TextEncoder().encode('a')]]);
const sameVocab: api.Vocab = v;
const options: CompileOptions = {maskStrategy: MaskStrategy.ParserDwa};
const c = await Grammar.fromJsonSchema({type: 'string'}).compile(v, options);
const bytes = await c.save(); const loaded = await Constraint.load(bytes, v);
const words: Uint32Array = loaded.start().mask();

const module = new Module(Grammar.fromLark('start: "a"'));
const precompiled: Module = await Grammar.fromLark('start: "a"').precompile(v, options);
void precompiled;
const tokenTrie: TokenTrie = {kind: 'token-trie', parserBackend: ParserBackend.TemplateDfa, mergeEquivalentTokens: true};
await module.precompile(v, {maskStrategy: tokenTrie});
const expr = Expr.choice(module, Grammar.fromLark('start: "a"'));
const restored = await Module.load(await module.save());
const canonical: api.Constraint = await restored.compile(v, {maskStrategy: MaskStrategy.TokenTrie});
void [sameVocab, words, expr, canonical];
