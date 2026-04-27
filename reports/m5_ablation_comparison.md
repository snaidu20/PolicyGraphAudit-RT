# M5 Unmasked vs. Masked Comparison Table

> **Unmasked** (mask_prob=0.0): trivial tautology — model reconstructs own supervision signal (F1=1.0).
> **Masked** (mask_prob=0.30): defensible evaluation — 30% of label-determining edges hidden.

| Model | Unmasked Macro F1 | Masked Macro F1 | Delta |
|-------|-------------------|-----------------|-------|
| tsne_pca_clustering | 0.2802 | 0.2802 | +0.0000 |
| text_only_logreg | 0.6065 | 0.6065 | +0.0000 |
| policy_only_gnn | 0.6925 | 0.6799 | -0.0125 |
| **full_hetero_gnn** ← headline | 1.0000 | 0.9561 | -0.0439 |

## Notes

- The unmasked full_hetero_gnn achieves F1=1.0000 because it can trivially recover
  the discrepancy label from DECLARES_COLLECTS, DECLARES_SHARES, and COLLECTS_DATATYPE
  edges — the exact edges used to compute the label.
- Under 30% masking, the model must infer labels from policy text (MENTIONS edges) and
  remaining graph context, making this a genuine generalisation test.
- The gap (Δ) for the full model quantifies how much of the prior result was circular.
- Policy-only and text baselines are largely unaffected by masking (they never used
  structural label-determining edges), serving as consistency checks.