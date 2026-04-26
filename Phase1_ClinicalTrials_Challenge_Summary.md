# Phase 1 Clinical Trials Challenge Summary

## Context

During the initial project execution, the pipeline started without incorporating the clinical trials dataset.  
At that time, Phase 1 was completed using price, sentiment, and FDA-related sources only.

## What Went Wrong

- The clinical trials dataset was unintentionally excluded during the first Phase 1 build.
- The team moved forward to later phases with this incomplete multimodal setup.
- After progressing past early stages, we identified that clinical trial activity is an important explanatory signal for pharma stock behavior and should be part of the RL state.

## Why It Mattered

- Excluding clinical trials reduced the model's event context and made the state representation less complete.
- It increased the risk of spurious attribution (for example, assigning a move to FDA effects when broader pipeline activity could be involved).
- This created a mismatch between the intended multimodal thesis and the implemented data pipeline.

## Resolution

- We sourced the clinical trials dataset and integrated it into the data workflow.
- Phase 1 was rebuilt to include clinical trial-derived features.
- Because downstream artifacts depended on Phase 1 outputs, we re-ran the full sequence:
  - Phase 1 (data pipeline)
  - Phase 2 (environment alignment with updated features)
  - Phase 3 (price baseline refresh)
  - Phase 4 (price + sentiment refresh)
  - Phase 5 (FDA + clinical trial confound-aware variants)

## Practical Impact on Project Flow

- This introduced additional execution time and rework across the pipeline.
- It improved data completeness and methodological consistency.
- It strengthened confidence that later ablation results were based on the intended multimodal inputs.

## Final Takeaway

The challenge was not a modeling bug but a **pipeline completeness issue** discovered mid-project.  
By integrating clinical trials and re-running the affected phases, the project returned to a consistent and defensible experimental path.
