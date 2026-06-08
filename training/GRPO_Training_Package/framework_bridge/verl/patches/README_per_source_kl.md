# Per-data-source KL patch (ChatJoy v6.3 attack-OOD fix)

Status: **WRITTEN + DRY-RUN VERIFIED on copies, NOT applied to live verl.** Drop-in
for the next restart, to be applied only if the RL plateau persists (decision tree below).

## Diagnosis (2026-05-29, real logs + train.parquet)

Entire RL deficit is on the attack data source:

| metric (old run step-175 val) | clean `chatjoy` (78%) | attack `chatjoy_rm_v6_3_roledrop` (22%) |
|---|---|---|
| is_refusal | 0.28% | **16.4%** |
| RM score | +1.24 | **-1.86** |

Train data is **21.87% attack** (3937/18000). Attack fails two ways: 16% refuse, and
the other 84% don't refuse but score RM -1.86 (break character / follow the OOC
request instead of deflecting in-character). Root cause: the SFT reference is
attack-naive, so per-token KL anchors the policy toward refuse/break-character on the
22% attack samples — KL fights the objective there. A single global `kl_loss_coef`
can't separate "anchor clean (good)" from "don't anchor attack (bad)". This explains
the **plateau** (break_rate ~16% over 175 steps), not a decline. KL is in the loss
(`use_kl_in_reward=False`), so it never enters `critic/rewards/mean`.

## Which code path (important)

This run uses `use_legacy_worker_impl=disable` → the **NEW engine path**. The executed
kl_loss is in `verl/workers/utils/losses.py::ppo_loss` (the `dp_actor.py` update_policy
is the deprecated legacy path and is NOT executed). `data_source` is a NON-tensor field
that does not survive the tensordict worker path, so a single-file actor patch reading
`data_source` would silently no-op. To **guarantee activation** the patch is two files:

- **(A) `verl/trainer/ppo/ray_trainer.py::_update_actor`** — before `to_tensordict()`,
  materialize a `[bsz, T]` float tensor `attack_token_mask` (1 = attack token) from
  `batch.non_tensor_batch["data_source"]`. `to_tensordict()` includes all `batch.batch`
  keys (verified); `left_right_2_no_padding` is a whitelist transform so this
  non-whitelisted tensor passes through unchanged and stays aligned with `response_mask`/`kld`.
- **(B) `verl/workers/utils/losses.py::ppo_loss`** — read `attack_token_mask`, build a
  per-token coef map (attack→`KL_ATTACK_COEF`, clean→`KL_CLEAN_COEF`), fold into `kld`
  before `agg_loss`. Unweighted `kl_loss` kept for the `actor/kl_loss` metric.

Defaults: `KL_ATTACK_COEF=0.0015` (weak residual anchor — keeps attack responses from
drifting into degenerate text, ~3.3× weaker than clean's 0.005). `0.0` = aggressive.

## Safety + verification

- **Env-gated**: `PER_SOURCE_KL` unset/0 → both edits are no-ops (original behavior).
- **Idempotent** (marker check), **per-file `.orig` backup**, `--revert` restores
  byte-identical (verified). Both patched files `py_compile` clean (verified).
- **OOM**: none. Adds only `[bsz, T]` mask + one `kld`-sized temporary (KB–low MB);
  ~6–7 orders smaller than the 26B model state. No change to model/batch/n/seqlen.
- **Guaranteed-ACTIVE check**: first step after relaunch prints BOTH:
  - `[per-source-KL] inject <n>/<batch> attack rows, mask (..,..)`  (trainer A)
  - `[per-source-KL] ACTIVE in ppo_loss: attack@0.0015 clean@0.005`  (actor B)
  If both appear, it is end-to-end active. If neither, the tensor path broke → revert.

## Apply / revert

```bash
cd /root/hanjiaqi/LlamaFactory/chatjoy/GRPO_Training_Package/framework_bridge/verl/patches
python apply_per_source_kl.py --check     # per-file state
python apply_per_source_kl.py             # apply (backs up losses.py.orig + ray_trainer.py.orig)
python apply_per_source_kl.py --revert    # undo both
```

## Relaunch (one command, has guards)

```bash
tmux kill-session -t rl_full_v3            # stop current run first
python apply_per_source_kl.py             # apply patch
bash launch_rl_v4_per_source_kl.sh        # guards: patch present + no rl session + ckpt exists
```

`launch_rl_v4_per_source_kl.sh` mirrors the verified rl_full_v3 launch and only adds
`PER_SOURCE_KL=1 KL_ATTACK_COEF=0.0015 KL_CLEAN_COEF=0.005`. Resumes from latest ckpt,
keeps `load_contents=[model,optimizer]` (LR stays 1.5e-6 constant). Log: `/tmp/rl_v6.3_psKL.log`.

## Decision tree (when to apply)

1. Let `rl_full_v3` run to **step 175 val** then **step 200 val**.
2. If attack break_rate drops <~16% and clean RM holds ≥~1.25 → RL-only is working,
   **do NOT apply**; stay the course.
3. If still plateaued at 200 → apply + relaunch (above). Confirm both ACTIVE log lines.
4. Watch next val: attack RM should rise / break_rate fall, clean RM unchanged (clean KL
   untouched). If clean degrades, raise `KL_ATTACK_COEF` toward 0.003.

## Files

- `apply_per_source_kl.py` — two-file patch installer (apply/--check/--revert), dry-run verified.
- `launch_rl_v4_per_source_kl.sh` — guarded relaunch with per-source KL enabled.
- `README_per_source_kl.md` — this file.
