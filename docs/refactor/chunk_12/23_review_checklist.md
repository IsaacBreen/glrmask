# Reviewer checklist

- Does `mod.rs` contain only module declarations, shared imports, and high-level module documentation?
- Can a reader find the public commit methods without scrolling through queue logic?
- Can a reader find longest-match pruning without reading fast paths?
- Can a reader distinguish reference Commit from optimized Commit?
- Are diagnostics/profile code separate from ordinary transition code?
- Are all fast paths explicitly fallible back to the general path?
- Does every residual tokenizer state pass through `end_state_can_advance`?
- Does every non-ignored terminal go through parser advance or a proven shortcut?
- Are ignored terminals treated as scanner-only transitions?
- Are the remaining large files documented as follow-up work rather than hidden?
