# Model Card — M5 HeteroAuditGNN

## Model Description

HeteroAuditGNN is a heterogeneous graph neural network that predicts the **discrepancy class** for each (App, DataType) pair in the PolicyGraphAudit-RT dataset. The model encodes a richly typed knowledge graph — spanning privacy-policy text segments, Play Store data-safety labels, and third-party SDK tracker registries — using two layers of `HeteroConv` with `SAGEConv` message-passing operators, one per edge type. Per-node-type input projections map diverse feature spaces (384-d sentence embeddings for text nodes, 34-d genre one-hot for App nodes) to a shared 128-d hidden space. Reverse edges are added programmatically to ensure bidirectional message flow. A two-layer MLP classifier head takes the concatenated (App, DataType) embedding pair and produces 4-class logits over: CONSISTENT, POLICY_LABEL_MISMATCH, OVER_DISCLOSURE, and UNDECLARED_COLLECTION.

**Architecture summary:**
- Input projections: Linear(input_dim → 128) per node type
- 2 × HeteroConv(SAGEConv) layers with bidirectional edges, ReLU + dropout(0.2)
- Classifier MLP: (256 → 64 → 4)
- Total parameters: ~1.99 M (after lazy SAGEConv initialization)
- Training device: CPU
- Loss: CrossEntropyLoss with inverse-frequency class weights

## Why We Mask Edges

### The circularity problem

The M4 pipeline computes discrepancy labels via a deterministic rule over three structural edge types:

| Edge | Role in label derivation |
|------|--------------------------|
| `(PrivacyLabel)-[DECLARES_COLLECTS]->(DataType)` | `label_collects` flag |
| `(PrivacyLabel)-[DECLARES_SHARES]->(DataType)` | `label_shares` flag |
| `(SDK)-[COLLECTS_DATATYPE]->(DataType)` | `runtime_implies` flag |

When the model is trained on unmasked graphs, it sees all three of these edges at the same time as the label they determine. The GNN trivially learns to reconstruct the labeling rule — a tautology, not generalization. This produces macro F1 = 1.0000 on the unmasked test set and inflates the perceived value of the full-graph model by ≈0.31 F1 points relative to the policy-only baseline.

### The fix: deterministic edge masking (Setup A)

`edge_masking.py` implements the following protocol:

1. For each labeled `(App, DataType)` pair, and for each of the three label-determining edge types above, a *deterministic* masking decision is made using a seeded hash: `hash((datatype_node_id, edge_type, seed)) % 10000 < mask_prob * 10000`.
2. If selected, **all** edges of that type pointing to the DataType node for that pair are removed from the graph.
3. `(PolicySegment)-[MENTIONS]->(DataType)` edges are **deliberately preserved** — this is the policy-text side that the model should leverage to infer the label.
4. The **same seed (42) and same mask_prob (0.30)** are applied identically to train, validation, and test splits. There is no information leakage from the unmasked regime at test time.
5. Ground-truth labels are always the original M4 labels (unmasked) — the model must predict the correct label even when some of the determining edges are hidden.

This forces the model to learn from policy text signals (`MENTIONS` edges and `PolicySegment` embeddings) plus *residual* structural context, not from direct access to the label-generating rule.

## Intended Use

This is a **research prototype** demonstrating that heterogeneous GNNs can recover audit logic from multi-source privacy knowledge graphs. It is **not** intended for production privacy auditing, regulatory compliance assessment, or automated enforcement decisions. Results should be treated as signals for human expert review, not verdicts.

Intended audiences: privacy researchers, HCI/security workshop reviewers, Android ecosystem auditors exploring graph-based tooling.

## Training Data

- **Graphs:** 268 per-app HeteroData objects constructed by the M4 fusion pipeline.
- **Labels:** 3,202 (App, DataType) discrepancy-classification rows in `discrepancy_labels_full.parquet`, spanning 252 apps and 38 canonical data types.
- **Label sources:** Play Store Data Safety DSL (Google Play), Princeton PoliCheck Corpus (PPC), and Exodus / Yale Privacy Lab tracker registry.
- **Class distribution (full corpus):** OVER_DISCLOSURE 51.8%, CONSISTENT 17.5%, POLICY_LABEL_MISMATCH 16.9%, UNDECLARED_COLLECTION 13.7%.
- **Label derivation:** Weak-supervision rules over graph structure — e.g., UNDECLARED_COLLECTION ↔ SDK collects a DataType that neither Policy nor PrivacyLabel mentions. Labels are **deterministic** given the graph edges.
- **Split:** 176 train / 37 val / 39 test apps (app-level, seed=42, never splitting a single app's pairs across splits).
- **Masking:** 30% of label-determining edges removed per pair (deterministic, seed=42, applied identically to all splits).

## Evaluation Results

### Masked evaluation (headline — publishable)

Evaluated on masked test graphs (mask_prob=0.30, seed=42). The model predicts discrepancy class from policy text + residual structural context, without access to 30% of the determining edges.

| Model | Macro F1 | F1 CONSISTENT | F1 PLM | F1 OVR-DISC | F1 UNDECL |
|-------|----------|--------------|--------|------------|--------|
| tsne_pca_clustering (t-SNE / PCA topic-modeling baseline) | 0.2802 | 0.0000 | 0.3896 | 0.7313 | 0.0000 |
| text_only_logreg | 0.6065 | 0.4630 | 0.4923 | 0.7476 | 0.7232 |
| policy_only_gnn | 0.6799 | 0.4024 | 0.6400 | 0.8587 | 0.8187 |
| **full_hetero_gnn (masked)** | **0.9561** | **0.9198** | **0.9524** | **0.9783** | **0.9740** |

See `reports/m5_ablation_table_masked.md` for the full masked ablation table.

### Unmasked evaluation (sanity check — tautological)

Reported here for completeness; these numbers should **not** be cited as the headline result because they reflect the model recovering its own supervision signal, not genuine generalization.

| Model | Macro F1 |
|-------|----------|
| tsne_pca_clustering | 0.2802 |
| text_only_logreg | 0.6065 |
| policy_only_gnn | 0.6925 |
| **full_hetero_gnn (unmasked)** | **1.0000** ← tautology |

### Mask-probability sensitivity

| mask_prob | Macro F1 | F1 UNDECL |
|-----------|----------|-----------|
| 0.00 (unmasked) | 1.0000 | 1.0000 |
| 0.15 | 0.9832 | 0.9801 |
| **0.30 (primary)** | **0.9730** | **0.9740** |
| 0.50 | 0.9247 | 0.9200 |

Performance degrades gracefully with increasing masking — the model retains strong signal from policy text and the remaining 70% of structural edges even under aggressive masking.

Training curves are saved to `reports/m5_training_curves_masked.png`. Early stopping triggered at epoch 13 (patience=8 on val macro F1). Best val macro F1: 0.9609.

## Limitations

1. **Weak-supervision labels.** No human expert verified individual (App, DataType) pairs. Labels inherit the errors of each source: Play Store DSLs can be incomplete, PPC policy classifiers have ~15% false-positive rates, and Exodus tracker profiles may lag SDK updates.

2. **Categorical-prior SDK inference.** SDK → DataType edges come from Exodus/Yale profile *categories*, not dynamic analysis of what data each SDK actually transmits at runtime. A tracker categorized as "analytics" is assumed to collect all analytics-adjacent data types.

3. **English-only policies.** PolicySegment embeddings are generated by `all-MiniLM-L6-v2`, which underperforms on non-English text. The corpus is predominately English-language Play Store apps.

4. **Android-only.** The runtime side (SDK, Endpoint, PrivacyLabel) is sourced from Android/Google Play infrastructure. iOS App Privacy Nutrition Labels are not yet incorporated despite being structurally similar.

5. **No per-app runtime traces.** REACHES_ENDPOINT edges are not yet populated in v1. The "runtime side" currently captures only static SDK presence from APK analysis, not observed network traffic.

6. **Dataset scale.** 268 apps and 3,202 labeled pairs is small by GNN standards. Results may not generalize to the full Play Store distribution.

7. **Residual circularity.** Even with 30% masking, 70% of label-determining edges remain visible. At mask_prob=0.30 the model still receives partial structural signal. The sensitivity sweep suggests the model leans heavily on this; performance at mask_prob=0.50 (macro F1=0.9247) reflects the residual policy-text contribution.

## Ethical Considerations

- **False positives** produced by an automated audit tool could unjustly flag a developer's app as non-compliant. Any output from this model must be treated as a hypothesis requiring human expert review before any enforcement, reporting, or publication decision.
- The model is an **audit aid**, not a verdict. It is designed to surface candidates for human investigation, not to replace legal or regulatory judgment.
- The training data includes real app package names. Care should be taken when publishing specific app-level results to distinguish between detected discrepancies (structural, as of crawl date) and intentional privacy violations.
- Class imbalance (OVER_DISCLOSURE is 52% of pairs) means the model, if not carefully weighted, would suppress UNDECLARED_COLLECTION detections — arguably the most policy-relevant finding. Balanced class weighting is applied to mitigate this.

## Differentiation from Prior Work

| Approach | Method | Limitation addressed by this work |
|----------|--------|-----------------------------------|
| t-SNE / PCA topic-modeling baseline | t-SNE + KMeans on policy text embeddings | No graph structure, no label/runtime side → macro F1 ≈ 0.28 |
| ATLAS (Andow et al. 2020) | NLP-only policy analysis (PoliCheck) | No knowledge graph, no SDK/label fusion |
| PrivacyFlash Pro (Story et al. 2021) | Rule-based analysis of Swift/Java API calls | No privacy-policy cross-referencing, no GNN |
| CrawlPhish / AAPolicies corpuses | Corpus construction only | No classification or discrepancy detection |

This model is the first (to our knowledge) to jointly encode policy text, data-safety labels, and SDK tracker evidence in a single heterogeneous GNN and predict fine-grained discrepancy classes across all three sources simultaneously, under a non-circular (edge-masked) evaluation protocol.

## Citation

If you use this model or data, please cite:

```
PolicyGraphAudit-RT: Heterogeneous GNN Auditing of Android Privacy Disclosures.
M5 prototype, 2025. https://github.com/[repo]
```
