# Strict forcing release comparison

3 alternating process pairs, 2000 repeated calls per row. All 42 case/mode/cache outputs match.

| Case | Mode | Cache | IDs | Baseline p50 us | Candidate p50 us | Ratio |
|---|---|---|---:|---:|---:|---:|
| accepting_prefix | BALANCED | cold-state | 0 | 41.600 | 0.200 | 0.005 |
| accepting_prefix | BALANCED | mask-already-computed | 0 | 38.800 | 0.200 | 0.005 |
| accepting_prefix | FAST_BUILD | cold-state | 0 | 43.100 | 0.800 | 0.019 |
| accepting_prefix | FAST_BUILD | mask-already-computed | 0 | 41.600 | 0.800 | 0.019 |
| accepting_prefix | FAST_RUNTIME | cold-state | 0 | 64.100 | 0.200 | 0.003 |
| accepting_prefix | FAST_RUNTIME | mask-already-computed | 0 | 62.500 | 0.200 | 0.003 |
| broad_identifier | BALANCED | cold-state | 0 | 42.700 | 4.500 | 0.105 |
| broad_identifier | BALANCED | mask-already-computed | 0 | 40.400 | 0.200 | 0.005 |
| broad_identifier | FAST_BUILD | cold-state | 0 | 10.400 | 9.700 | 0.933 |
| broad_identifier | FAST_BUILD | mask-already-computed | 0 | 9.800 | 0.400 | 0.041 |
| broad_identifier | FAST_RUNTIME | cold-state | 0 | 41.500 | 0.200 | 0.005 |
| broad_identifier | FAST_RUNTIME | mask-already-computed | 0 | 40.000 | 0.200 | 0.005 |
| deterministic_bytes_aliases | BALANCED | cold-state | 0 | 42.300 | 4.200 | 0.099 |
| deterministic_bytes_aliases | BALANCED | mask-already-computed | 0 | 39.400 | 0.200 | 0.005 |
| deterministic_bytes_aliases | FAST_BUILD | cold-state | 0 | 10.200 | 6.200 | 0.608 |
| deterministic_bytes_aliases | FAST_BUILD | mask-already-computed | 0 | 9.800 | 0.400 | 0.041 |
| deterministic_bytes_aliases | FAST_RUNTIME | cold-state | 0 | 42.200 | 0.200 | 0.005 |
| deterministic_bytes_aliases | FAST_RUNTIME | mask-already-computed | 0 | 40.200 | 0.200 | 0.005 |
| exact_event | BALANCED | cold-state | 1 | 41.500 | 6.600 | 0.159 |
| exact_event | BALANCED | mask-already-computed | 1 | 40.800 | 4.500 | 0.110 |
| exact_event | FAST_BUILD | cold-state | 1 | 11.900 | 8.100 | 0.681 |
| exact_event | FAST_BUILD | mask-already-computed | 1 | 11.500 | 5.500 | 0.478 |
| exact_event | FAST_RUNTIME | cold-state | 1 | 71.400 | 4.600 | 0.064 |
| exact_event | FAST_RUNTIME | mask-already-computed | 1 | 68.700 | 4.500 | 0.066 |
| long_singleton_chain | BALANCED | cold-state | 24 | 70.700 | 52.500 | 0.743 |
| long_singleton_chain | BALANCED | mask-already-computed | 24 | 67.600 | 48.800 | 0.722 |
| long_singleton_chain | FAST_BUILD | cold-state | 24 | 42.200 | 102.900 | 2.438 |
| long_singleton_chain | FAST_BUILD | mask-already-computed | 24 | 41.600 | 97.500 | 2.344 |
| long_singleton_chain | FAST_RUNTIME | cold-state | 24 | 159.300 | 49.600 | 0.311 |
| long_singleton_chain | FAST_RUNTIME | mask-already-computed | 24 | 155.400 | 49.300 | 0.317 |
| nested_inline | BALANCED | cold-state | 3 | 45.200 | 15.800 | 0.350 |
| nested_inline | BALANCED | mask-already-computed | 3 | 44.300 | 11.200 | 0.253 |
| nested_inline | FAST_BUILD | cold-state | 3 | 47.700 | 25.800 | 0.541 |
| nested_inline | FAST_BUILD | mask-already-computed | 3 | 46.700 | 18.400 | 0.394 |
| nested_inline | FAST_RUNTIME | cold-state | 3 | 79.800 | 11.200 | 0.140 |
| nested_inline | FAST_RUNTIME | mask-already-computed | 3 | 77.900 | 11.000 | 0.141 |
| two_singleton_chars | BALANCED | cold-state | 2 | 45.300 | 11.400 | 0.252 |
| two_singleton_chars | BALANCED | mask-already-computed | 2 | 41.600 | 7.500 | 0.180 |
| two_singleton_chars | FAST_BUILD | cold-state | 2 | 13.600 | 17.000 | 1.250 |
| two_singleton_chars | FAST_BUILD | mask-already-computed | 2 | 13.300 | 11.300 | 0.850 |
| two_singleton_chars | FAST_RUNTIME | cold-state | 2 | 45.900 | 7.600 | 0.166 |
| two_singleton_chars | FAST_RUNTIME | mask-already-computed | 2 | 43.800 | 7.600 | 0.174 |
