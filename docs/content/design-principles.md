## Design principles

GLRMask has a deliberately small API. These principles explain what it leaves out, and why.

### End tokens are yours

GLRMask has no end-token option. What a model's end token means depends on your application: stop generating, call a tool, or start the next turn. GLRMask doesn't decide that for you.

To require an end token after a grammar, put it in the grammar: `Grammar::from_expr(Expr::sequence([Expr::from(grammar), Expr::from(eos_id)]))`. GLRMask recognises this shape and compiles it with no extra work, so it is as fast as the grammar on its own.

### Masks are optional

You never need a mask to drive a state. Commits check every input, `is_accepting` says whether the input so far is complete, and `is_terminated` says whether anything more can be committed. A mask is how you sample only allowed tokens, if you want to.

Without masks, you can use rejection sampling: clone the state, commit the sampled token, and if the commit fails, sample again from the clone.

### Commits say whether they worked

`commit_token` and `commit_bytes` return `true` if the input was allowed and `false` if it wasn't. There is nothing else to report, so there is no error type.

A token ID outside the vocabulary is just not allowed. This matters because a model's logits are often padded past the end of its vocabulary, and a sampler can pick a padded ID.

### A failed commit ends the state

After a commit returns `false`, the state allows nothing more: `is_terminated` is `true`, `is_accepting` is `false`, and the mask is empty. If your code ignores a failed commit, the empty mask catches the mistake at the next step. Otherwise the state would silently drift out of step with the generated text.

To try alternatives, clone the state before committing.

### The API grows only by adding

GLRMask leaves a feature out until it is clearly needed. Adding a function later breaks nobody. Removing one breaks everyone who used it.
