# SFT Richness Reports And Probe Tools

This folder keeps the readable reports for the v6.8/v6.9 clean-richness arc and
the RL v6.10 step90 deployment-parameter sweep.

## Reports

| File | Purpose |
|---|---|
| `v68_probe_report.md` / `v68_heldout_probe_report.md` / `v68_rich_opening_probe_report.md` | v6.8 FFN/router LoRA probe results; useful as the "what went wrong" baseline |
| `rl90_temp_sweep_report.md` | Final deployment sampling decision for `/root/rl_v610_step90_merged`: `temperature=0.8, top_p=0.95, presence=0.4, min_p=0` |
| `CODEX_TASK_v68_train_merge_probe.md` | Operational task card for v6.8 merge/probe |

## Reproduction Scripts

The corresponding scripts live under `training/sft_richness/`:

| Path | Purpose |
|---|---|
| `training/sft_richness/gen_clean_probe.py` | Generate the clean-200 probe; supports `--temp`, `--top-p`, `--presence`, `--min-p`, `--seed` |
| `training/sft_richness/gen_attack_probe.py` | Generate the attack-68 probe with the same sampling knobs |
| `training/sft_richness/run_v69b_probe.sh` / `run_v69c_probe.sh` | End-to-end probe pipelines for the v69b/v69c checkpoints |
| `training/sft_richness/probe_analysis/classify_true_breaks.py` | Canonical true-break classifier; starts from `anti_ooc_disclosure` flags and removes echo/code false positives |
| `training/sft_richness/probe_analysis/clean_diversity_v2.py` | Fixed opening-diversity metric: strip leading `*action*`, then average pairwise Jaccard over first tokens |
| `training/sft_richness/probe_analysis/score_attack_rm.py` and related scoring scripts | RM scoring for arbitrary attack JSONL files and sweep outputs |

## Artifact Policy

Large generated JSONL files such as `clean_rl90_t*.jsonl`,
`attack_rl90_t*.jsonl`, and full SFT mix files are not committed. Representative
samples live in `data/samples/`; full local paths and sizes are tracked in
`DATA.md` and `docs/WORKSPACE_ARTIFACT_AUDIT_20260618.md`.
