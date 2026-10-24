# Forced tokens

Sometimes the grammar allows only one continuation, such as the rest of a tool call's opening tag once its first
letters are chosen. `forced_tokens()` returns tokens for that text, so you can commit them without running the model for
each one.

```python
for token in state.forced_tokens():
    state.commit_token(token)
```

`forced_tokens()` doesn't change the state; commit the tokens yourself. Your model also has to see them, so add them to
its input as you would tokens it generated.

## Which tokens

When the forced text could be split into tokens in several ways, GLRMask takes the longest allowed token at each step,
and the lowest ID when two tokens have the same bytes. If the grammar requires `ab` and the vocabulary has `a`, `b` and
`ab`, the result is `ab`. This is GLRMask's own rule. It isn't your model's tokenizer, which might have split the text
differently.

The result stops:

- where more than one token could come next;
- where the text so far is a complete match, even if more is forced after it;
- before a token that would cross a point where a special token or token extern could come instead. In `"ab" | "a" T "b"`,
  where `T` is a token extern, `ab` would skip the point after `a` where `T` could go, and `a` alone isn't the longest
  token, so the result is empty;
- after 10,000 tokens or bytes.

An empty result says only that nothing is forced. It doesn't mean the text is complete or that the state has ended.

The result is one way to write the forced text, not the only one: a result of `ab` doesn't mean that `a` followed by `b`
is disallowed.
