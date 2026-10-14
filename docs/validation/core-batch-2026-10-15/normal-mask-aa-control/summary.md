# Matched native mask comparison

Exact original-ID words compared: 40,704. Timed masks: {'baseline': 1440000, 'candidate': 1440000}.

| Case | Mode | Mask p50 old/new (ns) | p50 ratio | p99.9 old/new (ns) | p99.9 ratio |
|---|---|---:|---:|---:|---:|
| exact_token_set | Balanced | 4200.0/4100.0 | 0.976 | 9350.0/9600.0 | 1.027 |
| glrm_list | Balanced | 1250.0/1200.0 | 0.960 | 7950.0/7750.0 | 0.975 |
| json_schema | Balanced | 2250.0/2300.0 | 1.022 | 7950.0/7950.0 | 1.000 |
| lark_arithmetic | Balanced | 2550.0/2550.0 | 1.000 | 9150.0/8600.0 | 0.940 |
| scoped_child | Balanced | 85550.0/83150.0 | 0.972 | 191550.0/192600.0 | 1.005 |
| exact_token_set | FastBuild | 2300.0/2450.0 | 1.065 | 9050.0/7450.0 | 0.823 |
| glrm_list | FastBuild | 500.0/500.0 | 1.000 | 2050.0/2200.0 | 1.073 |
| json_schema | FastBuild | 1000.0/1050.0 | 1.050 | 2350.0/2750.0 | 1.170 |
| lark_arithmetic | FastBuild | 1200.0/1200.0 | 1.000 | 2500.0/3150.0 | 1.260 |
| scoped_child | FastBuild | 1100.0/1100.0 | 1.000 | 2650.0/2450.0 | 0.925 |
| exact_token_set | FastRuntime | 4350.0/4150.0 | 0.954 | 11500.0/10450.0 | 0.909 |
| glrm_list | FastRuntime | 1500.0/1500.0 | 1.000 | 8950.0/8550.0 | 0.955 |
| json_schema | FastRuntime | 2600.0/2500.0 | 0.962 | 9200.0/8600.0 | 0.935 |
| lark_arithmetic | FastRuntime | 3700.0/3750.0 | 1.014 | 11400.0/11550.0 | 1.013 |
| scoped_child | FastRuntime | 19900.0/20050.0 | 1.008 | 48350.0/49000.0 | 1.013 |

Repeatability review flags: 4. These require human/code review; their absence does not prove universal zero regression.
