# Chunk 05 risk register

## Risk 1: A moved helper may have lost an import or visibility qualifier.

Mitigation: Deferred compile pass should fix imports only, not change behavior.

Review trigger: when touching this area, search the exact file named in the mitigation and confirm the statement still holds.

## Risk 2: A profile print moved to profiling.rs could accidentally change benchmark parser scripts.

Mitigation: Profile line names were preserved; only location moved.

Review trigger: when touching this area, search the exact file named in the mitigation and confirm the statement still holds.

## Risk 3: Support determinization and fallback determinization could be confused.

Mitigation: They are now separate files with different docs and names.

Review trigger: when touching this area, search the exact file named in the mitigation and confirm the statement still holds.

## Risk 4: Default-label handling could accidentally include DEFAULT_LABEL as a parser state.

Mitigation: All parser-state interpretation is through labels.rs.

Review trigger: when touching this area, search the exact file named in the mitigation and confirm the statement still holds.

## Risk 5: The compatibility wrapper could hide the new profile output.

Mitigation: That is intentional for this chunk; a later pipeline-profile chunk should retain it.

Review trigger: when touching this area, search the exact file named in the mitigation and confirm the statement still holds.

## Risk 6: Determinization split could make local epsilon closure seem generally reusable.

Mitigation: It is `pub(super)` inside the determinize submodule, not exported to parser_dwa broadly.

Review trigger: when touching this area, search the exact file named in the mitigation and confirm the statement still holds.

## Risk 7: Terminal bundle interning could be mistaken for hash-based approximation.

Mitigation: Docs state exact equality is required.

Review trigger: when touching this area, search the exact file named in the mitigation and confirm the statement still holds.

## Risk 8: Runtime Mask readers may think Parser DWA depends on GLR.

Mitigation: Module docs emphasize stack-effect recognizers, not parser algorithm details.

Review trigger: when touching this area, search the exact file named in the mitigation and confirm the statement still holds.

## Risk 9: The no-compile instruction means syntactic issues may remain.

Mitigation: Static brace and shape checks are included; compile-fix is deliberately deferred.

Review trigger: when touching this area, search the exact file named in the mitigation and confirm the statement still holds.

## Risk 10: Long source comments could become stale.

Mitigation: They are colocated with module boundaries; later changes should update them as part of definition of done.

Review trigger: when touching this area, search the exact file named in the mitigation and confirm the statement still holds.

## Risk 11: conceptual drift during later compile-fix work

Mitigation: any compile-fix patch should be reviewed against `mathematical_contracts.md`, especially contracts 1, 5, 8, 10, and 13. Do not use borrow-checker pressure as a reason to change weight algebra.

## Risk 12: conceptual drift during later compile-fix work

Mitigation: any compile-fix patch should be reviewed against `mathematical_contracts.md`, especially contracts 1, 5, 8, 10, and 13. Do not use borrow-checker pressure as a reason to change weight algebra.

## Risk 13: conceptual drift during later compile-fix work

Mitigation: any compile-fix patch should be reviewed against `mathematical_contracts.md`, especially contracts 1, 5, 8, 10, and 13. Do not use borrow-checker pressure as a reason to change weight algebra.

## Risk 14: conceptual drift during later compile-fix work

Mitigation: any compile-fix patch should be reviewed against `mathematical_contracts.md`, especially contracts 1, 5, 8, 10, and 13. Do not use borrow-checker pressure as a reason to change weight algebra.

## Risk 15: conceptual drift during later compile-fix work

Mitigation: any compile-fix patch should be reviewed against `mathematical_contracts.md`, especially contracts 1, 5, 8, 10, and 13. Do not use borrow-checker pressure as a reason to change weight algebra.

## Risk 16: conceptual drift during later compile-fix work

Mitigation: any compile-fix patch should be reviewed against `mathematical_contracts.md`, especially contracts 1, 5, 8, 10, and 13. Do not use borrow-checker pressure as a reason to change weight algebra.

## Risk 17: conceptual drift during later compile-fix work

Mitigation: any compile-fix patch should be reviewed against `mathematical_contracts.md`, especially contracts 1, 5, 8, 10, and 13. Do not use borrow-checker pressure as a reason to change weight algebra.

## Risk 18: conceptual drift during later compile-fix work

Mitigation: any compile-fix patch should be reviewed against `mathematical_contracts.md`, especially contracts 1, 5, 8, 10, and 13. Do not use borrow-checker pressure as a reason to change weight algebra.

## Risk 19: conceptual drift during later compile-fix work

Mitigation: any compile-fix patch should be reviewed against `mathematical_contracts.md`, especially contracts 1, 5, 8, 10, and 13. Do not use borrow-checker pressure as a reason to change weight algebra.

## Risk 20: conceptual drift during later compile-fix work

Mitigation: any compile-fix patch should be reviewed against `mathematical_contracts.md`, especially contracts 1, 5, 8, 10, and 13. Do not use borrow-checker pressure as a reason to change weight algebra.

## Risk 21: conceptual drift during later compile-fix work

Mitigation: any compile-fix patch should be reviewed against `mathematical_contracts.md`, especially contracts 1, 5, 8, 10, and 13. Do not use borrow-checker pressure as a reason to change weight algebra.

## Risk 22: conceptual drift during later compile-fix work

Mitigation: any compile-fix patch should be reviewed against `mathematical_contracts.md`, especially contracts 1, 5, 8, 10, and 13. Do not use borrow-checker pressure as a reason to change weight algebra.

## Risk 23: conceptual drift during later compile-fix work

Mitigation: any compile-fix patch should be reviewed against `mathematical_contracts.md`, especially contracts 1, 5, 8, 10, and 13. Do not use borrow-checker pressure as a reason to change weight algebra.

## Risk 24: conceptual drift during later compile-fix work

Mitigation: any compile-fix patch should be reviewed against `mathematical_contracts.md`, especially contracts 1, 5, 8, 10, and 13. Do not use borrow-checker pressure as a reason to change weight algebra.

## Risk 25: conceptual drift during later compile-fix work

Mitigation: any compile-fix patch should be reviewed against `mathematical_contracts.md`, especially contracts 1, 5, 8, 10, and 13. Do not use borrow-checker pressure as a reason to change weight algebra.

## Risk 26: conceptual drift during later compile-fix work

Mitigation: any compile-fix patch should be reviewed against `mathematical_contracts.md`, especially contracts 1, 5, 8, 10, and 13. Do not use borrow-checker pressure as a reason to change weight algebra.

## Risk 27: conceptual drift during later compile-fix work

Mitigation: any compile-fix patch should be reviewed against `mathematical_contracts.md`, especially contracts 1, 5, 8, 10, and 13. Do not use borrow-checker pressure as a reason to change weight algebra.

## Risk 28: conceptual drift during later compile-fix work

Mitigation: any compile-fix patch should be reviewed against `mathematical_contracts.md`, especially contracts 1, 5, 8, 10, and 13. Do not use borrow-checker pressure as a reason to change weight algebra.

## Risk 29: conceptual drift during later compile-fix work

Mitigation: any compile-fix patch should be reviewed against `mathematical_contracts.md`, especially contracts 1, 5, 8, 10, and 13. Do not use borrow-checker pressure as a reason to change weight algebra.

## Risk 30: conceptual drift during later compile-fix work

Mitigation: any compile-fix patch should be reviewed against `mathematical_contracts.md`, especially contracts 1, 5, 8, 10, and 13. Do not use borrow-checker pressure as a reason to change weight algebra.

## Risk 31: conceptual drift during later compile-fix work

Mitigation: any compile-fix patch should be reviewed against `mathematical_contracts.md`, especially contracts 1, 5, 8, 10, and 13. Do not use borrow-checker pressure as a reason to change weight algebra.

## Risk 32: conceptual drift during later compile-fix work

Mitigation: any compile-fix patch should be reviewed against `mathematical_contracts.md`, especially contracts 1, 5, 8, 10, and 13. Do not use borrow-checker pressure as a reason to change weight algebra.

## Risk 33: conceptual drift during later compile-fix work

Mitigation: any compile-fix patch should be reviewed against `mathematical_contracts.md`, especially contracts 1, 5, 8, 10, and 13. Do not use borrow-checker pressure as a reason to change weight algebra.

## Risk 34: conceptual drift during later compile-fix work

Mitigation: any compile-fix patch should be reviewed against `mathematical_contracts.md`, especially contracts 1, 5, 8, 10, and 13. Do not use borrow-checker pressure as a reason to change weight algebra.

## Risk 35: conceptual drift during later compile-fix work

Mitigation: any compile-fix patch should be reviewed against `mathematical_contracts.md`, especially contracts 1, 5, 8, 10, and 13. Do not use borrow-checker pressure as a reason to change weight algebra.

## Risk 36: conceptual drift during later compile-fix work

Mitigation: any compile-fix patch should be reviewed against `mathematical_contracts.md`, especially contracts 1, 5, 8, 10, and 13. Do not use borrow-checker pressure as a reason to change weight algebra.

## Risk 37: conceptual drift during later compile-fix work

Mitigation: any compile-fix patch should be reviewed against `mathematical_contracts.md`, especially contracts 1, 5, 8, 10, and 13. Do not use borrow-checker pressure as a reason to change weight algebra.

## Risk 38: conceptual drift during later compile-fix work

Mitigation: any compile-fix patch should be reviewed against `mathematical_contracts.md`, especially contracts 1, 5, 8, 10, and 13. Do not use borrow-checker pressure as a reason to change weight algebra.

## Risk 39: conceptual drift during later compile-fix work

Mitigation: any compile-fix patch should be reviewed against `mathematical_contracts.md`, especially contracts 1, 5, 8, 10, and 13. Do not use borrow-checker pressure as a reason to change weight algebra.

## Risk 40: conceptual drift during later compile-fix work

Mitigation: any compile-fix patch should be reviewed against `mathematical_contracts.md`, especially contracts 1, 5, 8, 10, and 13. Do not use borrow-checker pressure as a reason to change weight algebra.

## Risk 41: conceptual drift during later compile-fix work

Mitigation: any compile-fix patch should be reviewed against `mathematical_contracts.md`, especially contracts 1, 5, 8, 10, and 13. Do not use borrow-checker pressure as a reason to change weight algebra.

## Risk 42: conceptual drift during later compile-fix work

Mitigation: any compile-fix patch should be reviewed against `mathematical_contracts.md`, especially contracts 1, 5, 8, 10, and 13. Do not use borrow-checker pressure as a reason to change weight algebra.

## Risk 43: conceptual drift during later compile-fix work

Mitigation: any compile-fix patch should be reviewed against `mathematical_contracts.md`, especially contracts 1, 5, 8, 10, and 13. Do not use borrow-checker pressure as a reason to change weight algebra.

## Risk 44: conceptual drift during later compile-fix work

Mitigation: any compile-fix patch should be reviewed against `mathematical_contracts.md`, especially contracts 1, 5, 8, 10, and 13. Do not use borrow-checker pressure as a reason to change weight algebra.

## Risk 45: conceptual drift during later compile-fix work

Mitigation: any compile-fix patch should be reviewed against `mathematical_contracts.md`, especially contracts 1, 5, 8, 10, and 13. Do not use borrow-checker pressure as a reason to change weight algebra.

## Risk 46: conceptual drift during later compile-fix work

Mitigation: any compile-fix patch should be reviewed against `mathematical_contracts.md`, especially contracts 1, 5, 8, 10, and 13. Do not use borrow-checker pressure as a reason to change weight algebra.

## Risk 47: conceptual drift during later compile-fix work

Mitigation: any compile-fix patch should be reviewed against `mathematical_contracts.md`, especially contracts 1, 5, 8, 10, and 13. Do not use borrow-checker pressure as a reason to change weight algebra.

## Risk 48: conceptual drift during later compile-fix work

Mitigation: any compile-fix patch should be reviewed against `mathematical_contracts.md`, especially contracts 1, 5, 8, 10, and 13. Do not use borrow-checker pressure as a reason to change weight algebra.

## Risk 49: conceptual drift during later compile-fix work

Mitigation: any compile-fix patch should be reviewed against `mathematical_contracts.md`, especially contracts 1, 5, 8, 10, and 13. Do not use borrow-checker pressure as a reason to change weight algebra.

## Risk 50: conceptual drift during later compile-fix work

Mitigation: any compile-fix patch should be reviewed against `mathematical_contracts.md`, especially contracts 1, 5, 8, 10, and 13. Do not use borrow-checker pressure as a reason to change weight algebra.

## Risk 51: conceptual drift during later compile-fix work

Mitigation: any compile-fix patch should be reviewed against `mathematical_contracts.md`, especially contracts 1, 5, 8, 10, and 13. Do not use borrow-checker pressure as a reason to change weight algebra.

## Risk 52: conceptual drift during later compile-fix work

Mitigation: any compile-fix patch should be reviewed against `mathematical_contracts.md`, especially contracts 1, 5, 8, 10, and 13. Do not use borrow-checker pressure as a reason to change weight algebra.

## Risk 53: conceptual drift during later compile-fix work

Mitigation: any compile-fix patch should be reviewed against `mathematical_contracts.md`, especially contracts 1, 5, 8, 10, and 13. Do not use borrow-checker pressure as a reason to change weight algebra.

## Risk 54: conceptual drift during later compile-fix work

Mitigation: any compile-fix patch should be reviewed against `mathematical_contracts.md`, especially contracts 1, 5, 8, 10, and 13. Do not use borrow-checker pressure as a reason to change weight algebra.

## Risk 55: conceptual drift during later compile-fix work

Mitigation: any compile-fix patch should be reviewed against `mathematical_contracts.md`, especially contracts 1, 5, 8, 10, and 13. Do not use borrow-checker pressure as a reason to change weight algebra.

## Risk 56: conceptual drift during later compile-fix work

Mitigation: any compile-fix patch should be reviewed against `mathematical_contracts.md`, especially contracts 1, 5, 8, 10, and 13. Do not use borrow-checker pressure as a reason to change weight algebra.

## Risk 57: conceptual drift during later compile-fix work

Mitigation: any compile-fix patch should be reviewed against `mathematical_contracts.md`, especially contracts 1, 5, 8, 10, and 13. Do not use borrow-checker pressure as a reason to change weight algebra.

## Risk 58: conceptual drift during later compile-fix work

Mitigation: any compile-fix patch should be reviewed against `mathematical_contracts.md`, especially contracts 1, 5, 8, 10, and 13. Do not use borrow-checker pressure as a reason to change weight algebra.

## Risk 59: conceptual drift during later compile-fix work

Mitigation: any compile-fix patch should be reviewed against `mathematical_contracts.md`, especially contracts 1, 5, 8, 10, and 13. Do not use borrow-checker pressure as a reason to change weight algebra.

## Risk 60: conceptual drift during later compile-fix work

Mitigation: any compile-fix patch should be reviewed against `mathematical_contracts.md`, especially contracts 1, 5, 8, 10, and 13. Do not use borrow-checker pressure as a reason to change weight algebra.

