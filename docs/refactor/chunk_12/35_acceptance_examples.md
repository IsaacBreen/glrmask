# Worked acceptance examples

## Single terminal consuming all bytes

Input bytes scan to one non-ignored terminal of full width. Commit can skip the queue, prune delayed exclusions, advance the parser once, and replace state with the fused result.

## Ignored whitespace

Input bytes scan to an ignored terminal. Commit resets to the tokenizer initial state for that byte segment but does not advance parser stacks.

## Partial lexer state

Input bytes do not complete a terminal but leave a residual tokenizer state. Commit preserves that residual state only if the parser can still accept a terminal that may complete from it.

## Ambiguous tokenization

Input bytes contain multiple normalized terminal matches with different widths. Commit uses the queue to explore each boundary and merges final states at the end.
