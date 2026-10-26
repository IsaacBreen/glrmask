## Design principles

These are the design principles that constrain GLRMask's API.

### Safe by default

Special tokens are privileged. They switch roles, call tools and end turns, and the software running a model acts on them. A grammar, though, may come from an untrusted source, such as an API user, a tool definition or a shared library. If grammar source could allow special tokens, an untrusted grammar could make the model fake a role switch or a tool call, requiring every untrusted grammar to be validated before use.

In GLRMask, grammar source has no way to allow a special token. Allowing one is an explicit step in code, either a `bind` to an extern or an `Expr` that contains the token. A name such as `<|im_end|>` in grammar source is an extern, which allows nothing until it's bound. Wildcards such as `<[*]>` match only ordinary tokens, and a special token's ID in grammar source is an error.

### End tokens are ordinary tokens

When and how generation stops is none of GLRMask's business. GLRMask treats an end token like any other special token. To require one, add it to the end of the grammar.

Without a built-in end token, you may still need to know when the output could be complete, for example to decide whether to let the model stop. `is_accepting` gives you this.

### Working without masks

Suppose you want to know whether any more input can be accepted. A brute-force and obviously impractical way is to commit every possible next byte and special token, each on a copy of the state, and see if any are accepted. You could also check whether the mask is empty, but you shouldn't need to compute masks to use GLRMask. (You might be implementing rejection sampling with the commit functions alone, for example.) This motivates `is_terminated`, which tells you directly.
