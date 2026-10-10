GLRMask is a grammar-constrained decoding library for high-throughput LLM serving. It is optimized for fast mask computation with extremely low tail latency, even for complex grammars.

GLRMask supports general context-free grammars, including ambiguous ones. Unfortunately, there is no universally accepted EBNF dialect. In keeping with this tradition, GLRMask includes its own: GLRM. It also supports JSON Schema and Lark.
