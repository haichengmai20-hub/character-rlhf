#!/usr/bin/env python3
"""
apply_per_source_kl.py — install/revert the per-data-source KL patch in the
verl NEW worker path (use_legacy_worker_impl=disable), guaranteeing it ACTIVATES.

WHY
---
ChatJoy v6.3 RL: the entire deficit is on the attack data source
(chatjoy_rm_v6_3_roledrop, 21.87% of train). Old step-175 val: clean refuse 0.28%
/ RM +1.24 vs attack refuse 16.4% / RM -1.86. The SFT reference is attack-naive, so
the per-token KL loss anchors the policy toward refusing/breaking character on
exactly the attack samples. A single global kl_loss_coef can't separate "anchor on
clean (good)" from "don't anchor on attack (bad)".

WHICH PATH
----------
This run uses use_legacy_worker_impl=disable -> the NEW engine path. The executed
kl_loss lives in verl/workers/utils/losses.py:ppo_loss (NOT the deprecated
dp_actor.py). data_source is a NON-tensor field that does not survive the
tensordict worker path, so we:
  (A) trainer side  (ray_trainer.py:_update_actor): materialize a [bsz, T] float
      tensor `attack_token_mask` (1=attack token) BEFORE to_tensordict(), where
      data_source is available. It is non-whitelisted in left_right_2_no_padding,
      so it passes through unchanged and stays aligned with response_mask/kld.
  (B) actor side    (losses.py:ppo_loss): read attack_token_mask, build a per-token
      coef map (attack->KL_ATTACK_COEF, clean->KL_CLEAN_COEF), fold into kld before
      aggregation. The unweighted kl_loss is kept for the actor/kl_loss metric.

GUARANTEED ACTIVE: the mask is a tensor (survives), to_tensordict includes all
batch.batch keys (verified), and ppo_loss consumes it. First step prints:
  [per-source-KL] inject N/M attack rows ...        (trainer)
  [per-source-KL] ACTIVE in ppo_loss: attack@.. clean@..   (actor)

SAFETY
------
- Env-gated: with PER_SOURCE_KL unset/0, both edits are no-ops (original behavior).
- Idempotent (marker check), per-file .orig backup, --revert restores byte-identical.

USAGE
-----
  python apply_per_source_kl.py --check
  python apply_per_source_kl.py
  python apply_per_source_kl.py --revert
Then relaunch (restart required) with:
  PER_SOURCE_KL=1 KL_ATTACK_COEF=0.0015 KL_CLEAN_COEF=0.005
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

VERL = Path("/home/ps/hanjiaqi/verl")
LOSSES = VERL / "verl/workers/utils/losses.py"
TRAINER = VERL / "verl/trainer/ppo/ray_trainer.py"
CONSTANTS = VERL / "verl/trainer/constants_ppo.py"
MARKER = "ChatJoy per-data-source KL"

# ---- Edit 1: losses.py add `import os` ----
L_IMPORT_OLD = """import torch
from tensordict import TensorDict
"""
L_IMPORT_NEW = """import os

import torch
from tensordict import TensorDict
"""

# ---- Edit 2: losses.py kl_loss block ----
L_KL_OLD = """    if config.use_kl_loss:
        ref_log_prob = data["ref_log_prob"]
        # compute kl loss
        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=config.kl_loss_type)
        kl_loss = agg_loss(
            loss_mat=kld, loss_mask=response_mask, loss_agg_mode=config.loss_agg_mode, **config.global_batch_info
        )

        policy_loss += kl_loss * config.kl_loss_coef
        metrics["kl_loss"] = Metric(value=kl_loss, aggregation=metric_aggregation)
        metrics["kl_coef"] = config.kl_loss_coef
"""
L_KL_NEW = """    if config.use_kl_loss:
        ref_log_prob = data["ref_log_prob"]
        # compute kl loss
        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=config.kl_loss_type)
        # unweighted kl_loss kept for the actor/kl_loss monitoring metric
        kl_loss = agg_loss(
            loss_mat=kld, loss_mask=response_mask, loss_agg_mode=config.loss_agg_mode, **config.global_batch_info
        )

        # === ChatJoy per-data-source KL (attack-OOD fix) ===
        # attack tokens (attack_token_mask==1) use KL_ATTACK_COEF, clean keep
        # KL_CLEAN_COEF (default = global kl_loss_coef). Per-token coef folded into
        # kld before aggregation. Falls back to global if mask absent / disabled.
        _atm = data.get("attack_token_mask", None)
        if os.environ.get("PER_SOURCE_KL", "0") == "1" and _atm is not None:
            _atk = float(os.environ.get("KL_ATTACK_COEF", "0.0015"))
            _cln = float(os.environ.get("KL_CLEAN_COEF", str(config.kl_loss_coef)))
            _atm = _atm.to(device=kld.device, dtype=kld.dtype)
            if _atm.dim() >= 2 and _atm.shape[1] != kld.shape[1]:
                _atm = _atm[:, :1]  # per-sample flag -> broadcast over the token dim
            _coef_map = _cln + (_atk - _cln) * _atm
            _kl_term = agg_loss(
                loss_mat=kld * _coef_map, loss_mask=response_mask,
                loss_agg_mode=config.loss_agg_mode, **config.global_batch_info
            )
            policy_loss += _kl_term
            metrics["kl_coef"] = _cln
            if not getattr(ppo_loss, "_ps_logged", False):
                print(f"[per-source-KL] ACTIVE in ppo_loss: attack@{_atk} clean@{_cln}", flush=True)
                ppo_loss._ps_logged = True
        else:
            policy_loss += kl_loss * config.kl_loss_coef
            metrics["kl_coef"] = config.kl_loss_coef
        metrics["kl_loss"] = Metric(value=kl_loss, aggregation=metric_aggregation)
"""

# ---- Edit 3: ray_trainer.py _update_actor injection ----
T_OLD = """    def _update_actor(self, batch: DataProto) -> DataProto:
        rollout_config = self.config.actor_rollout_ref.rollout
        batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
        # TODO: Make "temperature" single source of truth from generation.
        batch.meta_info["temperature"] = rollout_config.temperature
        # update actor
"""
T_NEW = """    def _update_actor(self, batch: DataProto) -> DataProto:
        rollout_config = self.config.actor_rollout_ref.rollout
        batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
        # TODO: Make "temperature" single source of truth from generation.
        batch.meta_info["temperature"] = rollout_config.temperature
        # === ChatJoy per-data-source KL: inject per-token attack mask tensor ===
        # data_source is a non-tensor field that does NOT survive the tensordict
        # worker path. Materialize a [bsz, T] float mask (1=attack token) here, where
        # data_source is available; it rides through to_tensordict +
        # left_right_2_no_padding (non-whitelisted -> unchanged) and is consumed in
        # workers/utils/losses.py:ppo_loss. Env-gated; no-op when PER_SOURCE_KL!=1.
        if os.environ.get("PER_SOURCE_KL", "0") == "1" and "response_mask" in batch.batch.keys():
            _ds = batch.non_tensor_batch.get("data_source", None)
            _rm = batch.batch["response_mask"]
            if _ds is not None and len(_ds) == _rm.shape[0]:
                _markers = ("roledrop", "attack")
                _is_atk = torch.tensor(
                    [1.0 if any(m in str(s) for m in _markers) else 0.0 for s in _ds],
                    dtype=torch.float32, device=_rm.device,
                )
                # per-SAMPLE flag [bsz, 1]: broadcasts over the token dim in ppo_loss,
                # so it is robust to kld's per-micro-batch sequence length (avoids a
                # [bsz, T] seq-length mismatch after no_padding_2_padding).
                batch.batch["attack_token_mask"] = _is_atk[:, None].to(torch.float32)
                if not getattr(self, "_ps_kl_logged", False):
                    print(f"[per-source-KL] inject {int(_is_atk.sum().item())}/{len(_ds)} "
                          f"attack rows, mask {tuple(batch.batch['attack_token_mask'].shape)}", flush=True)
                    self._ps_kl_logged = True
        # update actor
"""

# ---- Edit 4: constants_ppo.py — propagate PER_SOURCE_KL envs into ray runtime_env ----
# Ray workers/TaskRunner only get env vars listed in the runtime_env env_vars; the
# launching shell's PER_SOURCE_KL/KL_* are NOT inherited otherwise. Copy them in
# (os.environ values are str, satisfying ray's str-only env_vars requirement).
C_OLD = '''    for key in list(runtime_env["env_vars"].keys()):
        if os.environ.get(key) is not None:
            runtime_env["env_vars"].pop(key, None)
    return runtime_env
'''
C_NEW = '''    for key in list(runtime_env["env_vars"].keys()):
        if os.environ.get(key) is not None:
            runtime_env["env_vars"].pop(key, None)
    # === ChatJoy per-data-source KL: propagate controls to ray TaskRunner + workers ===
    for _k in ("PER_SOURCE_KL", "KL_ATTACK_COEF", "KL_CLEAN_COEF"):
        _v = os.environ.get(_k)
        if _v is not None:
            runtime_env["env_vars"][_k] = _v
    return runtime_env
'''

EDITS = [
    (LOSSES, L_IMPORT_OLD, L_IMPORT_NEW),
    (LOSSES, L_KL_OLD, L_KL_NEW),
    (TRAINER, T_OLD, T_NEW),
    (CONSTANTS, C_OLD, C_NEW),
]


def _files() -> list[Path]:
    seen = []
    for f, _, _ in EDITS:
        if f not in seen:
            seen.append(f)
    return seen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    for f in _files():
        if not f.exists():
            print(f"[err] target not found: {f}")
            return 2

    if args.check:
        for f in _files():
            patched = MARKER in f.read_text()
            backup = f.with_suffix(f.suffix + ".orig")
            print(f"{f}")
            print(f"  state : {'PATCHED' if patched else 'original'}   backup: {'present' if backup.exists() else 'none'}")
        return 0

    if args.revert:
        ok = True
        for f in _files():
            backup = f.with_suffix(f.suffix + ".orig")
            if backup.exists():
                f.write_text(backup.read_text())
                print(f"[ok] reverted {f}")
            else:
                print(f"[warn] no backup for {f}; left as-is")
                ok = False
        return 0 if ok else 1

    # apply
    # 1) verify every OLD anchor is present (and not already patched)
    for f in _files():
        if MARKER in f.read_text():
            print(f"[skip] already patched: {f}. Use --revert first.")
            return 0
    for f, old, _ in EDITS:
        if old not in f.read_text():
            print(f"[err] anchor not found in {f}:\n----\n{old[:120]}...\n----")
            print("      verl may have changed; patch by hand.")
            return 1
    # 2) backup once per file
    for f in _files():
        backup = f.with_suffix(f.suffix + ".orig")
        if not backup.exists():
            backup.write_text(f.read_text())
            print(f"[ok] backup -> {backup}")
    # 3) apply edits (group by file so multiple edits compose)
    for f in _files():
        src = f.read_text()
        for ef, old, new in EDITS:
            if ef == f:
                if old not in src:
                    print(f"[err] anchor vanished mid-apply in {f}; aborting (run --revert)")
                    return 1
                src = src.replace(old, new, 1)
        f.write_text(src)
        print(f"[ok] patched {f}")
    print("\nactivate on relaunch: PER_SOURCE_KL=1 KL_ATTACK_COEF=0.0015 KL_CLEAN_COEF=0.005")
    print("expect both log lines: '[per-source-KL] inject ...' and '[per-source-KL] ACTIVE in ppo_loss ...'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
