# AlphaFX Architecture

End-to-end flow from data to signal to explanation. The key boundary: the quant
layer owns the signal; the LLM only explains it and never sets it.

```mermaid
flowchart TD
    subgraph DATA["Data layer"]
        Y[yfinance<br/>AUDUSD / DXY / VIX]
        F[FRED / RBA<br/>US2Y / AU2Y / iron ore]
    end
    Y --> DA[DataAgent]
    F --> DA
    DA --> FE[FeatureAgent<br/>11 technical factors]
    FE --> QS[QuantSignalAgent<br/>5 factors scored ±1]

    QS --> CAL{Historical calibration<br/>enough samples?}
    CAL -- yes --> CP[calibrated probability]
    CAL -- no --> FB[fallback score-map]
    CP --> SIG[[signal / score / probability<br/>owned by the quant layer · immutable]]
    FB --> SIG

    SIG --> RISK[RiskAgent<br/>action / leverage / stops]

    %% Numeric-only branch — never calls the LLM
    SIG --> BT[BacktestAgent / WalkForward<br/>fully numeric · no LLM]

    %% LLM explanation layer
    SIG --> EV[build_evidence_pack<br/>numbers only · no raw prices / no future data]
    RISK --> EV
    EV --> TOG{Use LLM?<br/>API key set?}
    TOG -- no --> TPL[template agents<br/>run fully offline]
    TOG -- yes --> LLM[Claude<br/>explain / contrarian / judge]

    LLM --> GUARD{Direction guard<br/>judge: final_signal == quant signal}
    GUARD -- contradiction / error --> TPL
    GUARD -- pass --> OUT[explanation / contrarian / judge]
    TPL --> OUT
    LLM -. every call .-> LOG[(llm_calls<br/>audit log)]

    OUT --> UI[AI Report / Dashboard]
    RISK --> UI
    SIG --> UI

    classDef quant fill:#1f3a5f,stroke:#4da3ff,color:#fff;
    classDef llm fill:#5f3a1f,stroke:#ffae4d,color:#fff;
    class SIG,QS,RISK,BT quant;
    class LLM,EV,GUARD llm;
```

Blue = quant layer (owns the signal). Orange = LLM layer (explains only).

Two boundaries are explicit in the diagram:

1. `BacktestAgent / WalkForward` reads straight from the signal — the LLM is
   never in the backtest or walk-forward loops.
2. The direction guard sends every contradiction or error back to the template
   agents, and the judge's `final_signal` is forced to equal the quant signal.

See [DESIGN.md](../DESIGN.md) and [ROADMAP.md](../ROADMAP.md) for the LLM layer
design and the V2.6 plan.
