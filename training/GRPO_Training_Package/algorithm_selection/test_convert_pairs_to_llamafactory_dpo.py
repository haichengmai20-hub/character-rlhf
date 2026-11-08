"""convert_pairs_to_llamafactory_dpo 的单测（纯 stdlib）。"""

import json

import convert_pairs_to_llamafactory_dpo as c


def _pair(pid, chosen_text="好的回答", rejected_text="坏的回答"):
    return {
        "prompt": [{"role": "user", "content": "你好"}],
        "chosen": {"role": "assistant", "content": chosen_text},
        "rejected": {"role": "assistant", "content": rejected_text},
        "prompt_id": pid,
    }


def test_basic_conversion_shape():
    out, skipped = c.convert([_pair("p1")], shuffle=False, seed=0, limit=0)
    assert skipped == {"invalid": 0, "duplicate": 0}
    assert out == [{"messages": [{"role": "user", "content": "你好"}],
                    "chosen": {"role": "assistant", "content": "好的回答"},
                    "rejected": {"role": "assistant", "content": "坏的回答"}}]


def test_invalid_rows_skipped():
    rows = [
        _pair("ok"),
        {"prompt": [], "chosen": {"role": "assistant", "content": "x"},
         "rejected": {"role": "assistant", "content": "y"}},          # 空 prompt
        {"prompt": [{"role": "user", "content": "q"}],
         "chosen": {"role": "assistant", "content": "  "},            # 空 content
         "rejected": {"role": "assistant", "content": "y"}},
        {"prompt": [{"role": "user", "content": "q"}]},               # 缺 chosen/rejected
    ]
    out, skipped = c.convert(rows, shuffle=False, seed=0, limit=0)
    assert len(out) == 1 and skipped["invalid"] == 3


def test_dedup_by_prompt_id_and_hash_fallback():
    rows = [_pair("dup"), _pair("dup")]
    out, skipped = c.convert(rows, shuffle=False, seed=0, limit=0)
    assert len(out) == 1 and skipped["duplicate"] == 1

    a = _pair(None, chosen_text="c1", rejected_text="r1")
    b = _pair(None, chosen_text="c1", rejected_text="r1")  # 无 prompt_id -> 内容哈希去重
    c_ = _pair(None, chosen_text="c2", rejected_text="r1")
    out2, skipped2 = c.convert([a, b, c_], shuffle=False, seed=0, limit=0)
    assert len(out2) == 2 and skipped2["duplicate"] == 1


def test_shuffle_deterministic_and_limit():
    rows = [_pair(f"p{i:03d}", chosen_text=f"c{i}") for i in range(20)]
    a, _ = c.convert(rows, shuffle=True, seed=11, limit=0)
    b, _ = c.convert(rows, shuffle=True, seed=11, limit=0)
    assert [r["chosen"]["content"] for r in a] == [r["chosen"]["content"] for r in b]
    assert a != rows or [r["chosen"]["content"] for r in a] == [f"c{i:03d}" for i in range(20)]
    lim, _ = c.convert(rows, shuffle=False, seed=0, limit=5)
    assert len(lim) == 5
    zero, _ = c.convert(rows, shuffle=False, seed=0, limit=0)  # 0 = 开放完整池
    assert len(zero) == 20


def test_cli_end_to_end(tmp_path):
    src = tmp_path / "pairs.jsonl"
    dst = tmp_path / "dpo.jsonl"
    info = tmp_path / "dataset_info.json"
    src.write_text("\n".join(json.dumps(_pair(f"p{i}")) for i in range(3)), encoding="utf-8")
    rc = c.main(["--input", str(src), "--output", str(dst),
                 "--emit-dataset-info", str(info), "--shuffle", "--seed", "3"])
    assert rc == 0
    lines = [json.loads(x) for x in dst.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(lines) == 3 and all(set(l) == {"messages", "chosen", "rejected"} for l in lines)
    di = json.loads(info.read_text(encoding="utf-8"))
    assert di["character_rlhf_dpo"]["ranking"] is True
