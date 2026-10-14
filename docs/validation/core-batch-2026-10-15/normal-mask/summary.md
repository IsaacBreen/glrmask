# Matched native mask comparison

Exact original-ID words compared: 40,704. Timed masks: {'baseline': 2880000, 'candidate': 2880000}.

| Case | Mode | Mask p50 old/new (ns) | p50 ratio | p99.9 old/new (ns) | p99.9 ratio |
|---|---|---:|---:|---:|---:|
| exact_token_set | Balanced | 4200.0/4250.0 | 1.012 | 9300.0/10350.0 | 1.113 |
| glrm_list | Balanced | 1250.0/1200.0 | 0.960 | 7650.0/7400.0 | 0.967 |
| json_schema | Balanced | 2300.0/2250.0 | 0.978 | 7950.0/7750.0 | 0.975 |
| lark_arithmetic | Balanced | 2500.0/2500.0 | 1.000 | 8350.0/8250.0 | 0.988 |
| scoped_child | Balanced | 83900.0/85200.0 | 1.015 | 189800.0/189000.0 | 0.996 |
| exact_token_set | FastBuild | 2300.0/2250.0 | 0.978 | 7500.0/7550.0 | 1.007 |
| glrm_list | FastBuild | 500.0/500.0 | 1.000 | 2150.0/2300.0 | 1.070 |
| json_schema | FastBuild | 1050.0/1000.0 | 0.952 | 2500.0/2500.0 | 1.000 |
| lark_arithmetic | FastBuild | 1200.0/1100.0 | 0.917 | 2650.0/2500.0 | 0.943 |
| scoped_child | FastBuild | 1100.0/1100.0 | 1.000 | 2550.0/2350.0 | 0.922 |
| exact_token_set | FastRuntime | 4150.0/4300.0 | 1.036 | 10900.0/10050.0 | 0.922 |
| glrm_list | FastRuntime | 1500.0/1500.0 | 1.000 | 8550.0/8150.0 | 0.953 |
| json_schema | FastRuntime | 2550.0/2500.0 | 0.980 | 8950.0/8750.0 | 0.978 |
| lark_arithmetic | FastRuntime | 3700.0/3650.0 | 0.986 | 10650.0/10950.0 | 1.028 |
| scoped_child | FastRuntime | 19800.0/19600.0 | 0.990 | 43800.0/47450.0 | 1.083 |

Repeatability review flags: 0. These require human/code review; their absence does not prove universal zero regression.
