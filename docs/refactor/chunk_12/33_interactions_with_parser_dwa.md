# Interaction with Parser DWA

The Parser DWA is a compiled read structure used by Mask. Commit does not walk the Parser DWA to update state. Commit uses parser stack-effect semantics through GLR/table/template-DFA machinery.

This distinction is important for paper alignment:

- Mask: active stacks are read through the Parser DWA to combine encountered weights into a vocabulary mask.
- Commit: accepted bytes are scanned into completed terminals, and each completed terminal advances active stacks through stack-effect recognizers.

Both share parser stack-effect concepts, but they use them in different directions.
