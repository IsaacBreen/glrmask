# TODO List

## Tasks (in order of risk/complexity - less risky first)

### 1. [x] Add Makefile target for complexity analysis markdown compilation
- Add target to compile LaTeX to markdown
- DONE: Added 'complexity-md' target to gcg-paper/problems/Makefile

### 2. [x] Rename/restructure complexity analysis to "Formal Treatment"
- Consider renaming the folder
- Prepare for broader scope beyond just complexity
- DONE: Renamed to formal_treatment/, updated Makefile, added legacy aliases

### 3. [x] Create comprehensive "Mathematical Facts" document
- Dense, extensive documentation
- Algorithms, data structures, approach
- Modular structure
- References to code and other documents
- DONE: Created gcg-paper/notes/attachments/mathematical_facts.md

### 4. [x] Review and verify complexity analysis LaTeX
- Fine-tooth comb review of get_mask and commit complexity
- Ensure mathematical accuracy
- DONE: Verified against implementation and solutions:
  - commit = Θ(T_GLR(w)) - CORRECT
  - get_mask can be O(h) per call where h is stack height
  - Total can be O(n²) in worst case - proven with examples
  - Solutions in formal_treatment/solutions/ are mathematically rigorous

### 5. [ ] Expand unified_benchmark_v2.py tests
- More grammars
- More input strings for existing grammars
- LONGER input strings

### 6. [ ] Optimize compile.py performance
- Target: minimize total time for js.ebnf and diff constraint compilation
- Current baseline to measure

### 7. [ ] Add Rust CLI for grammar compilation
- No Python interface needed
- Direct Rust binary

### 8. [ ] Migrate to standard EBNF format
- Update library
- Update all grammars

### 9. [ ] Add stability safeguards for sep1
- Memory limits
- Stress testing
- Graceful error handling for edge cases
- Handle large grammars (50k+ lines)

### 10. [ ] Integrate IELR parser generator crate
- Replace custom table generation

### 11. [ ] Clean up project structure
- Remove junk files
- Reorganize as needed
- Do this LAST

---

## Notes
- Check user.md periodically
- Commit after each task
- Less risky tasks first
