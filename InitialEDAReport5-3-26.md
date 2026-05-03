# Feature Selection: Final Variable List for PCVRHyFormer

## Background

The TAAC2026 PCVR dataset contains 121 columns across 5 feature groups. Based on comprehensive quantitative analysis (correlation, mutual information, missing rates, cardinality, redundancy, and CVR variation), this plan proposes which features to **keep**, which to **drop**, and how to update `ns_groups.json`.

The dataset currently uses:
- **46 user int features** (scalar + multi-hot)
- **14 item int features**
- **10 user dense features** (total dim = 623)
- **4 sequence domains** (A: 9 cols, B: 14 cols, C: 12 cols, D: 10 cols)

---

## User Review Required

> [!IMPORTANT]
> The analysis below is based on the 1,000-row demo sample. Correlation and MI estimates on 1K rows have high variance. The directional recommendations (drop vs. keep) are sound, but exact thresholds may shift with the full training data. I recommend running a **single ablation experiment** on the full data after applying these changes.

> [!WARNING]
> Dropping features requires updating both `ns_groups.json` **and** the schema/data pipeline. Features dropped from ns_groups but still present in the parquet data will still be loaded by the dataset (occupying tensor positions), but their embeddings will receive zero-gradient signal. The cleanest approach is to drop them from ns_groups only — the model already handles missing group members gracefully.

---

## Analysis Summary

### Scalar User Int Features (34 scalar features)

| Tier | Criteria | Features (fids) | Count |
|------|----------|-----------------|-------|
| **S — Strong Signal** | \|corr\| ≥ 0.05 | 103, 109, 102, 100, 105, 94, 101, 86, 50 | 9 |
| **A — Moderate Signal** | 0.02 ≤ \|corr\| < 0.05, or MI ≥ 0.01 | 108, 104, 59, 99, 52, 95, 93, 51, 57, 82, 97 | 11 |
| **B — Weak but Usable** | 0.005 ≤ \|corr\| < 0.02, or MI ≥ 0.05 | 96, 4, 49, 53, 56, 48, 3, 106, 55, 54, 98, 1 | 12 |
| **D — Drop Candidates** | \|corr\| < 0.005 AND MI < 0.001, or ≥99% dominant | 92, 58, 107 | 3 |

### Scalar Item Int Features (14 features)

| Tier | Features (fids) | Count |
|------|-----------------|-------|
| **S — Strong Signal** | 13, 9, 16, 7 | 4 |
| **A — Moderate Signal** | 6, 81, 84 | 3 |
| **B — Weak but Usable** | 5, 8, 10, 85, 83, 12 | 6 |
| **D — Drop Candidates** | *(none — all item features carry signal)* | 0 |

### Multi-hot User Int Features (12 features)

> [!IMPORTANT]
> **Paired int/dense features**: Fids 62-66 and 89-91 exist as both `user_int_feats` (categorical IDs) and `user_dense_feats` (continuous statistics). The arrays are **element-aligned**: the int value identifies *which entity/category*, and the dense value provides *a statistic for that entity* (e.g., dwell time, score). Dropping the int part would orphan the dense values, losing the categorical context needed to interpret them. Therefore, **all paired int features must be kept if their dense counterparts are kept**.

| Feature (fid) | Dim | \|corr\| (int) | \|corr\| (dense) | Paired? | Recommendation |
|---------------|-----|----------------|------------------|---------|----------------|
| 65 | 4 | 0.0658 | 0.0118 | Yes | **Keep** — strong int signal |
| 66 | 4 | 0.0694 | 0.0072 | Yes | **Keep** — strong int signal |
| 80 | 2 | 0.0522 | — | No | **Keep** — moderate signal |
| 64 | 3 | 0.0363 | 0.0242 | Yes | **Keep** — moderate signal both sides |
| 62 | 2 | 0.0347 | 0.0359 | Yes | **Keep** — moderate signal both sides |
| 15 | 5 | 0.0155 | — | No | Keep — weak but cheap |
| 89 | 10 | 0.0157 | 0.0211 | Yes | **Keep** — paired; dense has useful signal |
| 90 | 10 | 0.0181 | 0.0174 | Yes | **Keep** — paired; both sides contribute |
| 63 | 4 | 0.0008 | 0.0156 | Yes | **Keep** — int signal weak, but paired dense has 0.0156 |
| 91 | 10 | 0.0012 | 0.0104 | Yes | **Keep** — int signal weak, but paired dense has 0.0104 |
| 60 | 1 | 0.0098 | — | No | **Drop** — single constant value (nuniq=1), no paired dense |
| 11 (item) | 2 | 0.0050 | — | No | Keep — item feature, only 2 dims |

### User Dense Features (10 features, total dim = 623)

| Feature (fid) | Dim | \|corr\| | Zero% | Recommendation |
|---------------|-----|----------|-------|----------------|
| 61 | 256 | 0.0430 | 0% | **Keep** — main embedding, strong signal |
| 62 | 2 | 0.0359 | 0% | Keep |
| 64 | 3 | 0.0242 | 0% | Keep |
| 89 | 10 | 0.0211 | 0% | Keep |
| 90 | 10 | 0.0174 | 0% | Keep |
| 63 | 4 | 0.0156 | 0% | Keep |
| 65 | 4 | 0.0118 | 0% | Keep |
| 91 | 10 | 0.0104 | 0% | Keep |
| 66 | 4 | 0.0072 | 0% | Keep — borderline, but cheap (4 dims) |
| 87 | 320 | 0.0010 | 61% | **Consider Dropping** — 320 dims, 61% zeros, near-zero label correlation |

> [!IMPORTANT]
> **fid 87** is by far the largest dense feature (320 of 623 dims = 51% of total dense dim) but has the weakest label signal (\|corr\|=0.001) and 61% zero values. Dropping it would reduce user_dense_dim from 623 to 303, cutting the dense projection parameters nearly in half. However, sparse signals in 320 dims may be captured by the learned projection that correlation analysis misses. I recommend **trying both** (with and without fid 87) as an ablation.

### Sequence Domains

All 4 sequence domains show non-trivial sequence-length distributions and are structurally important (behavioral history). **Keep all 4 domains and all their features.** The key findings:

| Domain | Features | Mean Len | Empty% | \|len_corr\| | Notes |
|--------|----------|----------|--------|--------------|-------|
| A | 9 | 701 | 0.5% | 0.029 | Dense; fid_38/45 have very high zero% (~95%) — handled by padding |
| B | 14 | 571 | 1.2% | 0.017 | Richest feature count; fid_73/74/76 sparse |
| C | 12 | 449 | 0.2% | 0.042 | Best length-label correlation |
| D | 10 | 1100 | 8.0% | 0.007 | Longest sequences; 8% empty — acceptable |

### Redundancy (High Correlation Between Features)

| Pair | \|corr\| | Recommendation |
|------|----------|----------------|
| u_104 ↔ u_97 | 0.910 | Drop u_97 (weaker signal: 0.0252 vs 0.0412) |
| u_106 ↔ u_98 | 0.771 | Drop u_106 (weaker signal: 0.0147 vs 0.0080). Actually both are weak — drop u_106 |
| u_55 ↔ u_58 | 0.749 | Drop u_58 (near-zero signal: 0.0006) |

> [!NOTE]
> u_58 is already flagged as a drop candidate independently. u_97 and u_106 are borderline-keep features with low correlation to label; dropping one from each pair to reduce redundancy is the cleaner option.

---

## Proposed Changes

### Final Feature Lists

#### User Int Features: **KEEP 42, DROP 4**

**Drop** (fids): `58, 60, 92, 107`

| Dropped fid | Paired Dense? | Reason |
|-------------|---------------|--------|
| 58 | No | Near-zero signal (\|corr\|=0.0006), redundant with 55 (r=0.749) |
| 60 | No | Constant (nuniq=1 across all rows) |
| 92 | No | Zero correlation with label (\|corr\|=0.0000), effectively binary noise |
| 107 | No | Near-zero signal (\|corr\|=0.0012), 90.6% dominant value |

> [!NOTE]
> **fid 63 and 91 retained** despite weak int-only correlation, because they are paired with `user_dense_feats_{63, 91}` which have non-trivial dense correlation (0.0156 and 0.0104). The int part provides categorical identity essential for interpreting the dense values.

**Keep all remaining 42 user int features** (34 scalar + 8 multi-hot).

#### Item Int Features: **KEEP ALL 14**
No item features warrant removal — all carry measurable signal.

#### User Dense Features: **KEEP ALL 10** (initially)
Keep fid 87 initially for a baseline, then try without it as an ablation.

#### Sequence Domains: **KEEP ALL 4 DOMAINS, ALL FEATURES**
No sequence features should be removed.

---

### Updated `ns_groups.json`

#### [MODIFY] [ns_groups.json](file:///c:/Users/murta/Downloads/KDD-tencent-competition/ns_groups.json)

Remove dropped fids from their groups:

```diff
  "user_ns_groups": {
    "U1": [1, 15],
    "U2": [48, 49, 89, 90, 91],
    "U3": [80],
    "U4": [51, 52, 53, 54, 86],
-   "U5": [82, 92, 93],
+   "U5": [82, 93],
-   "U6": [50, 60, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
+   "U6": [50, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 108, 109],
-   "U7": [3, 4, 55, 56, 57, 58, 59, 62, 63, 64, 65, 66]
+   "U7": [3, 4, 55, 56, 57, 59, 62, 63, 64, 65, 66]
  }
```

Changes:
- **U2**: Unchanged — `91` retained (paired with `user_dense_feats_91`)
- **U5**: Remove `92` (zero correlation, no paired dense)
- **U6**: Remove `60` (constant, no paired dense), `107` (near-zero signal, no paired dense)
- **U7**: Remove `58` (redundant with 55, near-zero signal, no paired dense). Keep `63` (paired with `user_dense_feats_63`)

Item NS groups remain **unchanged**.

### Model Constraints Check

With the proposed changes and rankmixer tokenizer (current active config):
- `user_ns_tokens = 5`, `item_ns_tokens = 2`
- `num_ns = 5 + 1 (user_dense) + 2 = 8`
- `T = num_queries * num_sequences + num_ns = 2 * 4 + 8 = 16`
- `d_model = 64`, `64 % 16 = 0` ✅

With group tokenizer (alternative config):
- `num_ns = 7 user groups + 1 user_dense + 4 item groups = 12`
- `T = 1 * 4 + 12 = 16`
- `64 % 16 = 0` ✅

Both configurations remain valid.

---

## Open Questions

> [!IMPORTANT]
> **Dense fid 87 ablation**: Should we commit to dropping fid 87 now (saving ~50% of dense projection parameters) or keep it initially and ablate later? This is the highest-impact single decision in terms of parameter count.

> [!NOTE]
> **Redundant pair u_104/u_97**: Both have low correlation with label (0.041 and 0.025 respectively) but carry different cardinality (3 vs 3). They're 0.91 correlated with each other. My recommendation is to keep both for now since they're cheap (low cardinality = tiny embeddings) and let the model learn to decorrelate. But if you want a leaner model, drop u_97.

---

## Verification Plan

### Automated Tests
1. After updating `ns_groups.json`, run a dry-run model construction to verify:
   - All fids in ns_groups exist in the schema
   - `d_model % T == 0` constraint passes
   - No embedding lookup errors

2. Run 1 training epoch on sample data to confirm no runtime errors.

### Manual Verification
- Compare validation AUC/loss with the pre-pruning baseline after a few training epochs on the full dataset.
- Conduct the fid 87 ablation experiment if desired.
