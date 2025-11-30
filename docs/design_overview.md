I want to build a parser that tells me which out of 100,000 or so possible next string segments are valid in a grammar.

It'd be expensive to parse each of these possible next strings, so we do some preprocessing at the tokenizer stage. Our goal is to figure out, for each next string, which token sequences could it possibly yield, including the last (possibly incomplete) one.

Each token has its own tokenizer. This means that the grammar is lexically ambiguous.

For example, in Python let's say we're part way through matching an identifier. In fact, suppose we've just parsed `"h"`. We have a tokenizer for an IDENTIFIER (in practice this is just a DFA in a certain state). Now let's say we have the possible next string `"ello, worl"`. What possible next tokens could we have? We don't include the current token. The answer here is `[COMMA, IDENTIFIER]` (whitespace is ignored).

Now suppose the possible next string is `"ello, mat"`. One possible sequence of next tokens would be `[COMMA, IDENTIFIER]`. Another would be `[COMMA, MATCH_KEYWORD]`.

On the other hand, suppose we have just parsed `"1"` using a NUMBER tokenizer. Now let's say we have the same possible next string `"ello, worl"`. What tokens could come next? Well, actually, none. The string `"1ello, worl"` won't tokenize no matter what because `"1ello"` is not a valid token. (An identifier in Python cannot start with a number.) So, the sequence of possible token sequences is empty.

Now suppose the next string is just `"ello"`. This is ok, we haven't mismatched. But we haven't matched any new tokens or started any new tokenizers either. So, there's one possible next token sequence: `[]`.

And anything else necessary.

These requirements aren't strict. You can change them if you want.

The reason we want to set it up this way is that we're building a system to help generate from an LLM constrained by a grammar. A simple way to do this might be to sample a token from the LLM and check if it parses. If it doesn't, just sample another one and check that.

This big issue is that the time complexity in the worst case is proportional to the number of possible tokens. Since inference is often done in parallel for many requests, we can't have one request holding up all others.

Therefore it's prudent to compute a token mask *while* the gpu is computing logits for the next token. That way, the GPU is never waiting for the CPU (assuming we can compute it in time) and we completely avoid this disastrous rejection scenario.

However, it's not practical to try parsing (incrementally) all possible next tokens. The time between LLM steps can be as low as 1 ms, and there can be over one hundred thousand tokens.

Instead, we note that many characters are, in a way, the same from a token matching point of view. For instance, there are many LLM tokens that are just alphabetic characters. When matching an identifier, we can treat these all the same.

(By this point we probably should already have made clear the difference between LLM tokens and grammar tokens. But I'm using the word token loosely in places. Try to correct for that.)

Some LLM tokens don't 'conclude' a token matcher in a given state. Some of them do and then proceed to match many other tokens. Sometimes the tokeniser (the set of all tokens together executed in every possible order) is ambiguous. We need to account for all these things.

What would parsing look like, all things considered?

We have an LR parser with a particular state stack.

We have a set of active marchers, one for each possible token.

We have a map that tells us for each state, for token matcher, for each possible sequence of next grammar tokens, which set of LLM tokens would yield that sequence of next grammar tokens.

We define a map from each LLM token to a flag, which we initially set to false, that tells us whether the LLM token is valid.

We take each next-grammar-token-sequence/LLM-token-set pair and parse the grammar token sequence from the current LR state to make sure it parses. If it does, we go through each LLM token in the set and assign the corresponding flag (defined above (should give it a name) to true.

(We need some better terminology here. 'Sequence of next grammar tokens'? That's a mouthful.)

Note that the notion of ambiguity of the regex/finite automata code attached above is slightly different (but it *is* deterministic in the same sense). That code is greedy in the sense that the finalizer it keeps will be the most recent one - i.e. the longest string. So, in a single execution run for the regex, each group ID (token) can match at most once. But multiple group IDs can match. (And the same group can match differently in a later call regex execution on the same state - but that's not so important here.)

Some of the reasons we want `Tokenizer` to be ambiguous is so we can match 'literals'. For example, in the Python grammar we have *soft keywords* like `match` that can be an identifier or a special 'match' token.

## How to respond

Use CoT to plan and reason. Figure out the best way to achieve the aims laid out here. You aren't bound by my exact requirements.

Write the most mathematically watertight implementation of any code possible. First, just plan. Then, think about what the code should actually do conceptually. Then, think about what the tests should require from the code, precisely. What should the exact output of be in each case? Match this full output precisely in the test assertions. Don't leave any parts of the output to chance.

Think mathematically. Be exhaustive. As comprehensive as possible. Very long response. Cover all possibilities.

Exhaustively reconsider. Improve recursively. Very long response.

Then, finally, write the code (if any code was requested). Ensure the documentation is conceptually complete and all reasoning is transferred there. It should be sufficient to explain the code and all reasoning used to arrive at the choices there.
