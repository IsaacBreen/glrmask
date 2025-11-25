# References & Literature Review

> Running collection of papers, articles, and resources for the grammar-constrained decoding paper.

---

## Core Papers (Must Cite)

### Constrained Decoding / Structured Generation

| Paper | Authors | Year | Key Contribution | Notes |
|-------|---------|------|------------------|-------|
| [Grammar-Constrained Decoding for Structured NLP Tasks](../downloads/papers/2023_Geng_GrammarConstrainedDecoding.pdf) | Geng, Josifoski, Peyrard, West | 2023 | GCD framework, input-dependent grammars | EMNLP 2023, foundational paper |
| [Efficient Guided Generation for LLMs](../downloads/papers/2023_Willard_OutlinesEfficient.pdf) | Willard, Louf | 2023 | Outlines library, FSM-based approach | Core baseline |

### Grammar-Based Methods

| Paper | Authors | Year | Key Contribution | Notes |
|-------|---------|------|------------------|-------|
| | | | | |

### Automata Theory for NLP

| Paper | Authors | Year | Key Contribution | Notes |
|-------|---------|------|------------------|-------|
| | | | | |

---

## Related Work

### JSON/Schema Constrained Generation

| Paper | Authors | Year | Notes |
|-------|---------|------|-------|
| | | | |

### Code Generation with Constraints

| Paper | Authors | Year | Notes |
|-------|---------|------|-------|
| | | | |

### Finite State Methods in NLP

| Paper | Authors | Year | Notes |
|-------|---------|------|-------|
| | | | |

---

## Tools & Libraries

| Name | URL | Description | Local Copy |
|------|-----|-------------|------------|
| Outlines | https://github.com/outlines-dev/outlines | Structured generation library | `downloads/repos/outlines-dev_outlines/` |
| Guidance | https://github.com/guidance-ai/guidance | Microsoft's structured generation | |
| LMQL | https://github.com/eth-sri/lmql | Query language for LLMs | |
| XGrammar | https://github.com/mlc-ai/xgrammar | Fast grammar engine | |
| llama.cpp | https://github.com/ggerganov/llama.cpp | Grammar support in inference | |

---

## To Read

- [ ] XGrammar paper (Dong et al., 2024)
- [ ] Grammar-Aligned Decoding (2024)
- [ ] RANLP 2025 paper on constrained decoding performance
- [ ] 

---

## Reading Notes

### Grammar-Constrained Decoding (Geng et al., 2023)
**Citation:** arXiv:2305.13971
**Read Date:** 
**Relevance:** High - foundational paper in the field

**Summary:**
- GCD controls LLM generation to guarantee output follows structure
- Introduces input-dependent grammars
- Evaluated on info extraction, entity disambiguation, constituency parsing

**Key Points:**
- Grammar constraints beat finetuned models in low-resource settings
- Framework applicable to wide range of structured NLP tasks

**Relation to Our Work:**
- We extend this with precomputation for efficiency

---

### Efficient Guided Generation (Willard & Louf, 2023)
**Citation:** arXiv:2307.09702
**Read Date:**
**Relevance:** High - main baseline/comparison

**Summary:**
- Outlines library for structured generation
- FSM-based constraint enforcement
- Regex and CFG support

**Key Points:**
- 

**Relation to Our Work:**
- Direct comparison target for efficiency

---

*Last updated: 2025-11-25*
