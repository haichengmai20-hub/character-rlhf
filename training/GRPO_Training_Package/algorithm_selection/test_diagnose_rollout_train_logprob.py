"""diagnose_rollout_train_logprob 的单测（纯 stdlib，无 GPU 依赖）。"""

import json

import diagnose_rollout_train_logprob as d


def _row(uid, ridx, tidx, ro, tr, slice_="clean", experts_ro=None, experts_tr=None):
    r = {"uid": uid, "response_idx": ridx, "token_idx": tidx,
         "slice": slice_, "logprob_rollout": ro, "logprob_train": tr}
    if experts_ro is not None:
        r["experts_rollout"] = experts_ro
    if experts_tr is not None:
        r["experts_train"] = experts_tr
    return r


def test_join_and_signed_mean():
    train_lps = [-0.99, -0.99, -1.02, -0.99]  # delta = [0.01, 0.01, -0.02, 0.01]
    ro = [_row("u1", 0, t, -1.0, train_lps[t]) for t in range(4)]
    matched, n_un_ro, n_un_tr = d.join_by_token(ro, ro)
    assert (n_un_ro, n_un_tr) == (0, 0) and len(matched) == 4
    deltas = [r["delta"] for r in matched]
    for got, want in zip(deltas, [0.01, 0.01, -0.02, 0.01]):
        assert abs(got - want) < 1e-9
    s = d.summarize_deltas(deltas)
    assert abs(s["signed_mean"] - (0.01 + 0.01 - 0.02 + 0.01) / 4) < 1e-12
    assert s["n"] == 4


def test_percentile_linear_interpolation():
    vals = sorted([float(i) for i in range(1, 101)])  # 1..100
    assert d.percentile(vals, 50) == 50.5
    assert d.percentile(vals, 95) == 95.05
    assert d.percentile(vals, 0) == 1.0
    assert d.percentile(vals, 100) == 100.0
    assert d.percentile([3.5], 95) == 3.5


def test_abs_stats_and_positive_tail():
    # 200 个已知 |delta|，验证 p95/p99 与手算一致
    deltas = [((i % 100) / 1000.0) - 0.05 for i in range(200)]
    s = d.summarize_deltas(deltas)
    assert abs(s["signed_mean"] - (sum(deltas) / 200)) < 1e-12
    assert s["abs"]["p99"] >= s["abs"]["p95"] >= s["abs"]["p50"] >= 0
    assert abs(s["abs"]["mean"] - sum(abs(x) for x in deltas) / 200) < 1e-12


def test_unmatched_counts_excluded():
    ro = [_row("u", 0, t, -1.0, -1.0) for t in range(3)] + [_row("u", 9, 0, -1.0, -1.0)]
    tr = [_row("u", 0, t, -1.0, -1.0) for t in range(3)] + [_row("u", 8, 0, -1.0, -1.0)]
    matched, n_un_ro, n_un_tr = d.join_by_token(ro, tr)
    assert len(matched) == 3 and n_un_ro == 1 and n_un_tr == 1


def test_slice_strata():
    rows = ([_row("a", 0, t, -1.0, -1.01, "clean") for t in range(10)]
            + [_row("b", 0, t, -1.0, -0.98, "attack") for t in range(10)])
    matched, _, _ = d.join_by_token(rows, rows)
    rep = d.build_report(matched, 0, 0)
    assert set(rep["by_slice"]) == {"clean", "attack"}
    assert abs(rep["by_slice"]["clean"]["signed_mean"] - (-0.01)) < 1e-12
    assert abs(rep["by_slice"]["attack"]["signed_mean"] - 0.02) < 1e-12
    assert rep["by_slice"]["clean"]["n"] == 10


def test_length_quartiles_present():
    rows = []
    for ridx, tlen in enumerate([10, 20, 30, 40]):
        rows += [_row("u", ridx, t, -1.0, -1.0) for t in range(tlen)]
    matched, _, _ = d.join_by_token(rows, rows)
    rep = d.build_report(matched, 0, 0)
    assert set(rep["by_length_quartile"]) == {"q1", "q2", "q3", "q4"}
    total = sum(v["n"] for v in rep["by_length_quartile"].values())
    assert total == 100  # 10+20+30+40


def test_routing_agreement_and_absence():
    rows = ([_row("u", 0, 0, -1.0, -1.0, experts_ro=[1, 2], experts_tr=[1, 2]),
             _row("u", 0, 1, -1.0, -1.0, experts_ro=[1], experts_tr=[2]),
             _row("u", 0, 2, -1.0, -1.0, experts_ro=[5], experts_tr=[5]),
             _row("u", 0, 3, -1.0, -1.0, experts_ro=[7], experts_tr=[8])])
    matched, _, _ = d.join_by_token(rows, rows)
    rep = d.build_report(matched, 0, 0)
    assert rep["routing"] == {"matched": 4, "agreement": 0.5}

    no_experts = [_row("u", 0, t, -1.0, -1.0) for t in range(3)]
    m2, _, _ = d.join_by_token(no_experts, no_experts)
    assert d.build_report(m2, 0, 0)["routing"] is None  # 不声称定位 MoE


def test_cli_end_to_end(tmp_path):
    ro_lines = [_row("u", 0, t, -2.0, -2.0 - 0.01) for t in range(5)]
    tr_lines = ro_lines + [_row("u", 1, 0, -2.0, -2.0)]  # 一条 train 侧多余
    p_ro, p_tr, p_out = tmp_path / "ro.jsonl", tmp_path / "tr.jsonl", tmp_path / "rep.json"
    p_ro.write_text("\n".join(json.dumps(r) for r in ro_lines), encoding="utf-8")
    p_tr.write_text("\n".join(json.dumps(r) for r in tr_lines), encoding="utf-8")
    rc = d.main(["--rollout", str(p_ro), "--train", str(p_tr), "--out", str(p_out)])
    assert rc == 0
    rep = json.loads(p_out.read_text(encoding="utf-8"))
    assert rep["n_matched"] == 5 and rep["n_unmatched_train"] == 1
    assert abs(rep["delta"]["signed_mean"] - (-0.01)) < 1e-12
