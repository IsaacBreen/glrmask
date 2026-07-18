# GLRMask

GLRMask is a grammar-constrained generation library for high-throughput LLM decoding. It is optimized for extremely low next-token mask latency across the distribution, even for complex grammars.

## Performance

Measured with MaskBench on the JSONSchemaBench corpus, using the Llama 3 vocabulary on an Intel Core i7-13620H under Ubuntu 24.04/WSL2.

> **Preliminary:** these timings are not yet accurate and should not be relied on.

### Mask-generation latency

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="../assets/benchmark-mask-tail-2026-07-16-dark.webp">
    <source media="(prefers-color-scheme: light)" srcset="../assets/benchmark-mask-tail-2026-07-16.webp">
    <img src="../assets/benchmark-mask-tail-2026-07-16.webp" alt="Mask-generation latency tail curves for GLRMask and LLGuidance, with high exceedance probabilities on the left and rarer events on the right" width="100%">
  </picture>
</p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="../assets/benchmark-mask-summary-2026-07-16-dark.webp">
    <source media="(prefers-color-scheme: light)" srcset="../assets/benchmark-mask-summary-2026-07-16.webp">
    <img src="../assets/benchmark-mask-summary-2026-07-16.webp" alt="Mask-generation latency percentile summary for GLRMask and LLGuidance" width="88%">
  </picture>
</p>

### Compilation time

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="../assets/benchmark-compilation-summary-2026-07-16-dark.webp">
    <source media="(prefers-color-scheme: light)" srcset="../assets/benchmark-compilation-summary-2026-07-16.webp">
    <img src="../assets/benchmark-compilation-summary-2026-07-16.webp" alt="Compilation-time percentile summary for GLRMask and LLGuidance" width="88%">
  </picture>
</p>

See the [full benchmark report](../benchmark-full-corpus-2026-07-16.md) for methodology.

---

[Compare the three benchmark layouts](README.md)
