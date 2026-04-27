# M5 Ablation Table (Edge-Masked Evaluation, mask_prob=0.30)

| Model | Macro F1 | F1 Consistent | F1 Pol/Label | F1 Over-Disc | F1 Undecl | Params | Runtime (s) |
|-------|----------|--------------|-------------|------------|--------|--------|-------------|
| tsne_pca_clustering | 0.2802 | 0.0000 | 0.3896 | 0.7313 | 0.0000 | 0 | 0.2 |
| text_only_logreg | 0.6065 | 0.4630 | 0.4923 | 0.7476 | 0.7232 | 3,092 | 1.0 |
| policy_only_gnn | 0.6799 | 0.4024 | 0.6400 | 0.8587 | 0.8187 | 810,180 | 29.1 |
| full_hetero_gnn (masked) | 0.9561 | 0.9198 | 0.9524 | 0.9783 | 0.9740 | 1,994,436 | 47.6 |