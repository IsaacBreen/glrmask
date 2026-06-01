# Symbol visibility policy

Most moved helpers are marked `pub(super)` rather than `pub(crate)`. This is intentional: the helpers are shared across Commit submodules but should not become part of the crate-level runtime API.

Visibility levels should mean:

- `pub`: stable public API or public profile structures.
- `pub(crate)`: cross-subsystem implementation API with a named reason.
- `pub(super)`: local Commit helper shared between sibling Commit modules.
- private: helper used only inside one file.

This chunk uses `pub(super)` as a transitional local-sharing boundary. Later compile repair may reduce some helpers back to private if the owning file becomes narrower.
