# Terminal DWA Structure: `Kubernetes---kb_684_Normalized`

This diagram shows the minimized terminal DWA for the slowest example.

```mermaid
flowchart LR
    s3["State 3 [START]<br/>final=none"]
    s2["State 2<br/>final=non-empty"]
    s1["State 1<br/>final=non-empty"]
    s0["State 0<br/>final=non-empty"]

    s3 -- "205" --> s2
    s3 -- "171" --> s1
    s3 -- "4" --> s0
    s2 -- "330" --> s0
    s2 -- "3" --> s1
    s1 -- "249" --> s0

    classDef start fill:#fff3cd,stroke:#b7791f,stroke-width:2px,color:#2d1f00;
    classDef accepting fill:#e6ffed,stroke:#2f855a,stroke-width:2px,color:#123524;
    classDef sink fill:#edf2f7,stroke:#4a5568,stroke-width:2px,color:#1a202c;

    class s3 start;
    class s2,s1 accepting;
    class s0 sink,accepting;
```

## Readout

- `State 3` is the only non-final state and acts as the root dispatch node.
- `State 0` is the terminal sink: it is final and has no outgoing transitions.
- `State 1` always funnels into `State 0`.
- `State 2` mostly funnels into `State 0`, with only `3` transitions going to `State 1`.
- There are no self-loops and no transitions back to the start state.
- The longest path is `State 3 -> State 2 -> State 1 -> State 0`.
