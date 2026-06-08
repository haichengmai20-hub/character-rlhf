#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List


def load_states(input_dir: Path) -> List[Dict]:
    states = []
    for path in sorted(input_dir.glob("pairs.shard*.jsonl.state.json")):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        data["_path"] = str(path)
        states.append(data)
    return states


def pct(part: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(part * 100.0 / total, 2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor pair generation progress from checkpoint state files.")
    parser.add_argument("--input-dir", required=True, help="Directory containing pairs.shard*.jsonl.state.json")
    parser.add_argument("--compact", action="store_true", help="Print a one-line compact progress summary")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    states = load_states(input_dir)
    if not states:
        raise SystemExit(f"No state files found in {input_dir}")

    total_input = max(int(s.get("input_records_total", 0) or 0) for s in states)
    total_assigned = sum(int(s.get("assigned_records", 0) or 0) for s in states)
    total_processed = sum(int(s.get("next_pos", 0) or 0) for s in states)

    total_written = 0
    total_failed = 0
    per_shard = []
    type_counts: Dict[str, int] = {}

    for state in states:
        counts = state.get("counts", {}) or {}
        written = int(counts.get("written", 0) or 0)
        failed = int(counts.get("failed", 0) or 0)
        total_written += written
        total_failed += failed
        for key, value in counts.items():
            if key in {"written", "failed"}:
                continue
            if isinstance(value, int):
                type_counts[key] = type_counts.get(key, 0) + value
        per_shard.append(
            {
                "shard_index": state.get("shard_index"),
                "next_pos": int(state.get("next_pos", 0) or 0),
                "assigned_records": int(state.get("assigned_records", 0) or 0),
                "processed_pct": pct(int(state.get("next_pos", 0) or 0), int(state.get("assigned_records", 0) or 0)),
                "written": written,
                "failed": failed,
                "completed": bool(state.get("completed", False)),
            }
        )

    summary = {
        "input_dir": str(input_dir),
        "total_input_records": total_input,
        "total_assigned_records": total_assigned,
        "processed_records": total_processed,
        "processed_pct_of_input": pct(total_processed, total_input),
        "processed_pct_of_assigned": pct(total_processed, total_assigned),
        "cumulative_output_pairs": total_written,
        "current_run_failed_records": total_failed,
        "all_completed": all(item["completed"] for item in per_shard),
        "note": "processed_* counts track how many original input records have been attempted in the current run, regardless of success or failure. cumulative_output_pairs includes old copied successes when resuming from a retry directory.",
        "per_shard": per_shard,
        "type_counts": dict(sorted(type_counts.items())),
    }

    if args.compact:
        shard_bits = []
        for item in sorted(per_shard, key=lambda x: (x["shard_index"] is None, x["shard_index"])):
            shard_bits.append(f"shard{item['shard_index']} {item['next_pos']}")
        line = (
            f"{total_processed}/{total_input} ({pct(total_processed, total_input)}%)"
            + (" | " + " | ".join(shard_bits) if shard_bits else "")
        )
        print(line)
        return

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
