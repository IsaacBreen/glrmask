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

### 5. [x] Expand unified_benchmark_v2.py tests
- More grammars
- More input strings for existing grammars
- LONGER input strings
- DONE: Expanded from ~100 to ~215 total test inputs
  - Added helper functions for generating test data
  - Added new 'imperative_lang' grammar
  - Extended all existing grammars with stress tests
  - Added more large JS files

### 6. [x] Optimize compile.py performance
- Target: minimize total time for js.ebnf and diff constraint compilation
- DONE: 6x serialization speedup (750ms → 110ms) by:
  - Serializing to memory first instead of streaming through gzip
  - Using compression level 3 instead of 6

### 7. [x] Add Rust CLI for grammar compilation
- No Python interface needed
- Direct Rust binary
- DONE: grammar-compiler binary exists and works:
  - `./target/release/grammar-compiler --grammar X --vocab Y --output Z`
  - Compiles js.ebnf in 3.6s with 111ms serialization

### 8. [x] Migrate to standard EBNF format
- Update library
- Update all grammars
- DONE: Added GBNF (llama.cpp) compatibility:
  - Hash `#` comments now supported (in addition to `//`)
  - `root` rule automatically used as start rule if present
  - Dashed identifiers (`add-expr`) now supported

### 8.5 [ ] Add Lark grammar format support
- Add Lark grammar format support

### 9. [x] Add stability safeguards for sep1
- Memory limits
- Stress testing
- Graceful error handling for edge cases
- Handle large grammars (50k+ lines)
- DONE: Stress test in temp/stress_test.py passes:
  - 500 iterations: no memory growth
  - 300 token sequences: work correctly
  - Error handling: graceful exceptions

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
