# AlphaFX Architecture

End-to-end flow from data to signal to research outputs and explanation.

Two boundaries are load-bearing:

1. **The quant layer owns the signal.** The LLM only explains it (and is never in
   the backtest, walk-forward, or ML loops). The ML model is a research
   comparison only — the rule signal stays primary/live.
2. **No look-ahead.** Macro factors are publication-lagged (point-in-time),
   calibration is expanding-window, and the ML backtest uses out-of-sample
   predictions only.

```mermaid
flowchart TD
    subgraph DATA["Data layer"]
        Y[yfinance<br/>AUDUSD / DXY / VIX]
        F[FRED / RBA<br/>US2Y / AU2Y / iron ore]
    end
    Y --> DA[DataAgent]
    F --> DA
    DA --> FE[FeatureAgent<br/>10 factors · macro publication-lagged · point-in-time]
    FE --> QS[QuantSignalAgent<br/>5 factors ±1 · equal weight by design]

    QS --> CAL{Calibration<br/>expanding-window?}
    CAL -- per-date --> CP[calibrated probability<br/>calibration_frame · no leak]
    CAL -- too few --> FB[fallback score-map]
    CP --> SIG[[signal / score / probability<br/>owned by the quant layer · immutable]]
    FB --> SIG

    SIG --> RISK[RiskAgent<br/>action / leverage / stops]

    %% Numeric research — never calls the LLM
    subgraph NUM["Numeric research · no LLM"]
        BT[BacktestAgent<br/>OOS trades · spread + directional swap]
        WF[WalkForward<br/>degradation guarded]
        DIAG[Diagnostics<br/>overlap-adjusted t-stat · IC/IR]
        CORR[Factor correlation<br/>DXY/VIX overlap monitored]
        RND[Random benchmark<br/>200-seed distribution + percentile]
    end
    SIG --> BT
    SIG --> WF
    SIG --> DIAG
    FE --> CORR
    BT --> RND

    %% ML research comparison — rule signal stays primary
    subgraph MLB["ML research · comparison only"]
        DS[build_dataset<br/>forward-return target · point-in-time X]
        ML2[MLSignalAgent<br/>walk-forward · no leak · small-sample warning]
        MLBT[ML backtest<br/>OOS only · same costs]
    end
    FE --> DS --> ML2 --> MLBT
    BT --> CMP[rule vs ML comparison]
    MLBT --> CMP

    %% LLM explanation layer — explains, never sets the signal
    subgraph LLMX["LLM explanation · explains only"]
        EV[evidence pack<br/>numbers only · no raw prices]
        TOG{Use LLM?<br/>API key?}
        LLM[Claude<br/>explain / contrarian / judge]
        GUARD{direction guard<br/>judge final_signal == quant signal}
        TPL[template agents<br/>offline fallback]
        LOG[(llm_calls<br/>audit log)]
    end
    SIG --> EV
    RISK --> EV
    CMP --> EV
    EV --> TOG
    TOG -- yes --> LLM --> GUARD
    TOG -- no --> TPL
    GUARD -- contradiction/error --> TPL
    GUARD -- pass --> OUT[explanation / contrarian / judge]
    TPL --> OUT
    LLM -. every call .-> LOG

    SIG --> UI[Streamlit · hero + 7 sections]
    RISK --> UI
    NUM --> UI
    CMP --> UI
    OUT --> UI

    classDef quant fill:#1f3a5f,stroke:#4da3ff,color:#fff;
    classDef llm fill:#5f3a1f,stroke:#ffae4d,color:#fff;
    classDef ml fill:#1f5f3a,stroke:#4dffae,color:#fff;
    class SIG,QS,RISK,BT,WF,DIAG,CORR,RND quant;
    class LLM,EV,GUARD llm;
    class DS,ML2,MLBT,CMP ml;
```

Blue = quant layer (owns the signal). Green = ML research (comparison only).
Orange = LLM explanation (explains only).

See [DESIGN.md](../DESIGN.md) and [ROADMAP.md](../ROADMAP.md) for the layer
designs and version history.
