# V3 runbook

## Main ladder campaign

```bash
python run_small_to_large_ladder_v3.py \
  --small-train-globs "runs/gcp_trm_scaleup_v2/session_1776550919/n20/traces_train/*.pt,runs/gcp_trm_scaleup_v2/session_1776550919/n40/traces_train/*.pt" \
  --small-valid-globs "runs/gcp_trm_scaleup_v2/session_1776550919/n20/traces_valid/*.pt,runs/gcp_trm_scaleup_v2/session_1776550919/n40/traces_valid/*.pt" \
  --curriculum-ckpt runs/gcp_trm_scaleup_v2/session_1776550919/n80/train_run/model-best.pt \
  --sizes "100,200,400" \
  --per-size 10 \
  --budget-ladder "12:24,32:48,64:96" \
  --timeout-sec 60 \
  --profile-every 4
```

## Single-budget ablation

```bash
python run_small_to_large_ablation_v3.py \
  --small-train-globs "runs/gcp_trm_scaleup_v2/session_1776550919/n20/traces_train/*.pt,runs/gcp_trm_scaleup_v2/session_1776550919/n40/traces_train/*.pt" \
  --small-valid-globs "runs/gcp_trm_scaleup_v2/session_1776550919/n20/traces_valid/*.pt,runs/gcp_trm_scaleup_v2/session_1776550919/n40/traces_valid/*.pt" \
  --curriculum-ckpt runs/gcp_trm_scaleup_v2/session_1776550919/n80/train_run/model-best.pt \
  --sizes "100,200" \
  --per-size 2 \
  --simulations 12 \
  --max-depth 24 \
  --timeout-sec 40
```

## Outputs

Each run writes:
- `ladder_results.json`
- `FINAL_LADDER_REPORT.md`
- `merged_traces/train_merged.pt`
- `merged_traces/valid_merged.pt`
- `trained_small/model-best.pt`
- `small_macros.json`
- `untrained_trm.pt`
