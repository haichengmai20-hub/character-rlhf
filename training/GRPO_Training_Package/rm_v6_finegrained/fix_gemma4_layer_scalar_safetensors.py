#!/usr/bin/env python3
"""Repair Gemma4 tensors exported by verl legacy_model_merger.

The merger can concatenate FSDP-replicated tensors and save shape
[world_size * D] instead of the model-expected shape [D]. vLLM then fails
with errors like:

    Attempted to load weight (torch.Size([4])) into parameter (torch.Size([1]))
    Attempted to load weight (torch.Size([4608])) into parameter (torch.Size([1152]))

Currently handled rules:

    *.layer_scalar                    shape [4]    -> [1]
    model.vision_tower.std_bias       shape [4608] -> [1152]
    model.vision_tower.std_scale      shape [4608] -> [1152]

All four [1152] segments of std_bias / std_scale have been verified
bytewise identical (FSDP-replicated across TP world size 4), so keeping
the first segment is lossless.

This script rewrites safetensors shards streaming from disk, truncating
matching tensors to the expected shape by keeping the leading slice. All
other tensor bytes are copied unchanged.
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
from pathlib import Path
from typing import Any


DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E5M2": 1,
    "F8_E4M3": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "F64": 8,
    "I64": 8,
    "U64": 8,
}


def tensor_nbytes(info: dict[str, Any]) -> int:
    dtype = info["dtype"]
    size = DTYPE_BYTES[dtype]
    for dim in info["shape"]:
        size *= int(dim)
    return size


EXPECTED_SHAPES: dict[str, list[int]] = {
    "model.vision_tower.std_bias": [1152],
    "model.vision_tower.std_scale": [1152],
}


def target_shape(name: str, old_shape: list[int]) -> list[int] | None:
    """Return the truncated shape for a tensor, or None to keep as-is.

    Tensors are only truncated when the on-disk shape is exactly an integer
    multiple of the expected shape along the leading dim and the leading
    slice corresponds to a single replicated copy from FSDP.
    """
    if name.endswith(".layer_scalar") and old_shape == [4]:
        return [1]
    expected = EXPECTED_SHAPES.get(name)
    if expected is not None and old_shape == [expected[0] * 4]:
        return expected
    return None


def elements_in_shape(shape: list[int]) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n


def read_header(path: Path) -> tuple[int, dict[str, Any]]:
    with path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    return header_len, header


def header_bytes(header: dict[str, Any]) -> bytes:
    raw = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return raw + b" " * ((8 - len(raw) % 8) % 8)


def iter_tensor_items(header: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    items = [(k, v) for k, v in header.items() if k != "__metadata__"]
    return sorted(items, key=lambda item: item[1]["data_offsets"][0])


def rewrite_shard(src: Path, dst: Path) -> tuple[int, int, int, int]:
    old_header_len, old_header = read_header(src)
    old_data_start = 8 + old_header_len
    new_header: dict[str, Any] = {}
    if "__metadata__" in old_header:
        new_header["__metadata__"] = old_header["__metadata__"]

    offset = 0
    repaired = 0
    removed_elements = 0
    removed_bytes = 0
    entries: list[tuple[str, dict[str, Any], int, int, int]] = []

    for name, info in iter_tensor_items(old_header):
        old_begin, old_end = info["data_offsets"]
        old_nbytes = old_end - old_begin
        new_info = dict(info)
        new_nbytes = old_nbytes

        old_shape = list(info.get("shape", []))
        new_shape = target_shape(name, old_shape)
        if new_shape is not None:
            elem_size = DTYPE_BYTES[info["dtype"]]
            new_info["shape"] = new_shape
            new_nbytes = elem_size * elements_in_shape(new_shape)
            repaired += 1
            removed_elements += elements_in_shape(old_shape) - elements_in_shape(new_shape)
            removed_bytes += old_nbytes - new_nbytes

        new_info["data_offsets"] = [offset, offset + new_nbytes]
        new_header[name] = new_info
        entries.append((name, info, old_begin, old_end, new_nbytes))
        offset += new_nbytes

    tmp = dst.with_suffix(dst.suffix + ".tmp")
    encoded_header = header_bytes(new_header)
    with src.open("rb") as fin, tmp.open("wb") as fout:
        fout.write(struct.pack("<Q", len(encoded_header)))
        fout.write(encoded_header)
        for name, info, old_begin, old_end, new_nbytes in entries:
            fin.seek(old_data_start + old_begin)
            remaining = new_nbytes
            while remaining:
                chunk = fin.read(min(1024 * 1024 * 64, remaining))
                if not chunk:
                    raise EOFError(f"Unexpected EOF while copying {name} from {src}")
                fout.write(chunk)
                remaining -= len(chunk)

    tmp.replace(dst)
    return repaired, removed_elements, removed_bytes, offset


def copy_sidecar_files(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for path in src_dir.iterdir():
        if path.name.endswith(".safetensors"):
            continue
        target = dst_dir / path.name
        if path.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(path, target)
        else:
            shutil.copy2(path, target)


def update_index(src_dir: Path, dst_dir: Path, removed_elements: int, removed_bytes: int) -> None:
    index_path = dst_dir / "model.safetensors.index.json"
    if not index_path.exists():
        return
    index = json.loads(index_path.read_text())
    metadata = index.setdefault("metadata", {})
    if "total_parameters" in metadata:
        metadata["total_parameters"] = int(metadata["total_parameters"]) - removed_elements
    if "total_size" in metadata:
        metadata["total_size"] = int(metadata["total_size"]) - removed_bytes
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    src_dir = args.input
    dst_dir = args.output
    if not src_dir.exists():
        raise FileNotFoundError(src_dir)
    if src_dir.resolve() == dst_dir.resolve():
        raise ValueError("input and output must be different directories")

    dst_dir.mkdir(parents=True, exist_ok=True)
    copy_sidecar_files(src_dir, dst_dir)

    total_repaired = 0
    total_removed_elements = 0
    total_removed_bytes = 0
    for shard in sorted(src_dir.glob("*.safetensors")):
        repaired, removed_elements, removed_bytes, new_data_bytes = rewrite_shard(shard, dst_dir / shard.name)
        total_repaired += repaired
        total_removed_elements += removed_elements
        total_removed_bytes += removed_bytes
        print(
            json.dumps(
                {
                    "shard": shard.name,
                    "repaired_tensors": repaired,
                    "removed_elements": removed_elements,
                    "removed_bytes": removed_bytes,
                    "new_data_bytes": new_data_bytes,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    update_index(src_dir, dst_dir, total_removed_elements, total_removed_bytes)
    print(
        json.dumps(
            {
                "output": str(dst_dir),
                "total_repaired_tensors": total_repaired,
                "total_removed_elements": total_removed_elements,
                "total_removed_bytes": total_removed_bytes,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
