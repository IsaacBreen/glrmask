# Failure modes and error semantics

Commit distinguishes API errors from rejection.

- Unknown vocabulary token id is an API error and should not be confused with grammar rejection.
- Grammar rejection during byte commit is represented by an empty resulting state or an explicit `Err` from internal transition.
- A token outside the current mask is allowed to drive the state to rejection; this is part of the runtime model rather than a precondition failure.

The API methods are responsible for token-id lookup errors. The transition methods are responsible for parser/lexer viability errors.
