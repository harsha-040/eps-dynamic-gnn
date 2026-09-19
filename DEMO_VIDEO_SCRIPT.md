# Demo Video Script — EPS Dynamic-GNN Fault Diagnosis

Target length: **3–4 minutes**. Screen-record the live demo (local `uvicorn` run or
the deployed URL) while reading this narration. Timestamps are guidance, not
strict cues — pause naturally between sections.

Before recording: run `uvicorn app.backend.main:app --port 8000` (or open the
deployed Space URL) and have the page loaded at the top, zoomed so text is
legible on screen.

---

## 0:00–0:20 — Cold open: state the problem

*(Screen: top of the page, title visible)*

> "Satellites diagnose Electrical Power System faults using Graph Neural
> Networks today — but every existing method, including the base paper we
> compare against, freezes the graph structure once and never updates it.
> This is a live demo of a Dynamic-GNN that recomputes which components
> matter to each other at every timestep instead."

## 0:20–0:50 — Run a scenario

*(Screen: scroll to "① Choose a scenario", select `pcu_regulator_fault` from
the dropdown, click "Run simulation")*

> "I'll pick a PCU regulator fault and run a fresh 96-minute orbit — this
> simulates the satellite from scratch using real governing equations for
> solar power, battery charge, and bus voltage regulation, and injects the
> fault at a random point."

*(Wait for the "done in ... ms" confirmation to appear)*

## 0:50–1:20 — Telemetry

*(Screen: scroll to "② Simulated telemetry", hover over one or two charts)*

> "Here's what each of the five EPS components reported — voltage, current,
> power, temperature, and state of charge. The gray band is eclipse, the red
> band is when the fault is actually active."

## 1:20–2:00 — Model predictions

*(Screen: scroll to "③ Live model predictions")*

> "All four models — CNN, LSTM, a fixed-graph Static-GNN, and our proposed
> Dynamic-GNN — look at the exact same telemetry and independently guess the
> fault. The green bar is the true answer. On the held-out test set, the
> Dynamic-GNN reaches 97.3% accuracy versus 76 to 83% for the other three —
> you can see the full comparison table right here."

*(Point at the metrics table: Acc/Prec/Rec/F1/AUC/Log Loss columns)*

## 2:00–3:00 — The graph itself (the core contribution)

*(Screen: scroll to "⑤ Watch the Dynamic-GNN's graph structure evolve", drag
the time slider slowly from t=0 through the eclipse-to-sunlit transition)*

> "This is the whole point of 'dynamic' in the name. Instead of assuming the
> wiring diagram is always equally important, the model recomputes — at
> every single timestep — which components' readings should influence each
> other most right now. Watch the graph as I scrub through the eclipse-to-
> sunlit transition: the battery is the dominant source early on, and within
> about a minute of sunrise, the solar array visibly takes over. The panel on
> the right shows the same thing as a matrix — A(t) — where each cell answers
> 'how much does this node depend on that one, right now.'"

## 3:00–3:30 — Honest limitation (optional but recommended)

> "We also stress-tested this under noisy and missing sensor data — the
> dynamic graph's advantage holds under moderate corruption but collapses
> under heavy corruption, because the graph is computed from the same
> corrupted signal it's trying to classify. Noise-augmented training closes
> most of that gap. Full numbers are in the paper and README."

## 3:30–3:45 — Close

> "That's the live demo — physics simulator, four trained models, and a
> graph structure you can watch adapt in real time. Code, paper, and
> presentation are all in the repo linked below."

---

## Recording checklist

- [ ] Record at 1280×800 or larger so charts are legible
- [ ] Mute notifications before recording
- [ ] Do one full dry run of clicking through before recording for real
- [ ] Keep the mouse cursor deliberate — pause on what you're describing
- [ ] Export as MP4, upload to YouTube (unlisted is fine) or Drive, and drop the
      link into `README.md` under **Live deployment**
