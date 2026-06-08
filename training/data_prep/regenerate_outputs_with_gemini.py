import argparse
import json
import os
import time
from typing import Dict, List

import requests

from api import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    build_url,
    extract_text,
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT_PATH = os.path.join(BASE_DIR, "chatjoy_top20_multilingual_27x2200_preapi.json")
DEFAULT_OUTPUT_PATH = os.path.join(BASE_DIR, "chatjoy_top20_multilingual_27x2200_preapi_gemini.json")
DEFAULT_PROGRESS_PATH = os.path.join(BASE_DIR, "chatjoy_top20_multilingual_27x2200_preapi_gemini.progress.jsonl")
DEFAULT_SEXUALLY_EXPLICIT_THRESHOLD = "BLOCK_MEDIUM_AND_ABOVE"
DEFAULT_THINKING_LEVEL = "low"


def load_dataset(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"输入文件不是 JSON 数组: {path}")
    return data


def load_progress(path: str) -> Dict[int, Dict]:
    progress_map: Dict[int, Dict] = {}
    if not os.path.exists(path):
        return progress_map

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            index = int(record["index"])
            progress_map[index] = record["item"]
    return progress_map


def append_progress(path: str, index: int, item: Dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"index": index, "item": item}, ensure_ascii=False) + "\n")
        f.flush()


def call_gemini_for_regeneration(
    api_key: str,
    prompt: str,
    system_instruction: str,
    base_url: str,
    model: str,
    api_version: str,
    temperature: float,
    max_output_tokens: int,
    sexually_explicit_threshold: str,
    thinking_level: str,
    timeout: int,
) -> Dict:
    url = build_url(base_url=base_url, model=model, api_version=api_version)
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }

    payload: Dict = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
        },
        "systemInstruction": {
            "parts": [
                {"text": system_instruction}
            ]
        },
    }

    if thinking_level:
        payload["generationConfig"]["thinkingConfig"] = {
            "thinkingLevel": thinking_level
        }

    # 仅覆盖色情安全阈值，其他安全类别保持默认行为。
    if sexually_explicit_threshold:
        payload["safetySettings"] = [
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": sexually_explicit_threshold,
            }
        ]

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=timeout,
    )

    try:
        data = response.json()
    except Exception:
        response.raise_for_status()
        raise RuntimeError(f"接口返回了非 JSON 内容：{response.text}")

    if response.status_code >= 400:
        raise RuntimeError(
            f"请求失败，HTTP {response.status_code}\n"
            f"URL: {url}\n"
            f"Response: {json.dumps(data, ensure_ascii=False, indent=2)}"
        )

    return data


def regenerate_one_item(
    item: Dict,
    api_key: str,
    base_url: str,
    model: str,
    api_version: str,
    temperature: float,
    max_output_tokens: int,
    sexually_explicit_threshold: str,
    thinking_level: str,
    timeout: int,
) -> Dict:
    prompt = str(item.get("instruction", "")).strip()
    system_prompt = str(item.get("system", "")).strip()

    if not prompt:
        raise ValueError("当前样本 instruction 为空，无法调用 Gemini。")
    if not system_prompt:
        raise ValueError("当前样本 system 为空，无法调用 Gemini。")

    result = call_gemini_for_regeneration(
        api_key=api_key,
        prompt=prompt,
        system_instruction=system_prompt,
        base_url=base_url,
        model=model,
        api_version=api_version,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        sexually_explicit_threshold=sexually_explicit_threshold,
        thinking_level=thinking_level,
        timeout=timeout,
    )
    new_output = extract_text(result).strip()
    if not new_output:
        raise RuntimeError("Gemini 返回了空 output。")

    new_item = dict(item)
    new_item["output"] = new_output
    return new_item


def finalize_output(dataset: List[Dict], progress_map: Dict[int, Dict], output_path: str) -> None:
    missing_indices = [idx for idx in range(len(dataset)) if idx not in progress_map]
    if missing_indices:
        raise RuntimeError(f"仍有未完成样本，无法生成最终文件。缺失数量: {len(missing_indices)}")

    final_data = [progress_map[idx] for idx in range(len(dataset))]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Use Gemini to regenerate outputs for a 10k dataset.")
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help="输入数据集路径")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="最终输出数据集路径")
    parser.add_argument("--progress", default=DEFAULT_PROGRESS_PATH, help="断点续跑进度文件路径")
    parser.add_argument("--api_key", default=os.getenv("GEMINI_API_KEY"), help="Gemini API Key")
    parser.add_argument("--base_url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api_version", default="v1beta")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_output_tokens", type=int, default=2048)
    parser.add_argument(
        "--sexual_threshold",
        type=str,
        default=DEFAULT_SEXUALLY_EXPLICIT_THRESHOLD,
        help=(
            "Gemini 色情内容安全阈值。默认 BLOCK_MEDIUM_AND_ABOVE，表示屏蔽一部分；"
            "传空字符串可不覆盖默认安全设置。"
        ),
    )
    parser.add_argument(
        "--thinking_level",
        type=str,
        default=DEFAULT_THINKING_LEVEL,
        help=(
            "Gemini 3 系列的思考等级。默认 low；"
            "传空字符串可不覆盖模型默认思考设置。"
        ),
    )
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--retry_sleep", type=float, default=3.0)
    parser.add_argument("--limit", type=int, default=0, help="仅处理前 N 条，0 表示处理全部")
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("未提供 API Key，请通过 --api_key 或环境变量 GEMINI_API_KEY 提供。")

    dataset = load_dataset(args.input)
    if args.limit > 0:
        dataset = dataset[:args.limit]

    progress_map = load_progress(args.progress)
    total = len(dataset)

    print(f"1. 输入样本数: {total}", flush=True)
    print(f"2. 已有进度样本数: {sum(1 for idx in progress_map if idx < total)}", flush=True)
    print(f"3. 开始调用模型: {args.model}", flush=True)

    for idx, item in enumerate(dataset):
        if idx in progress_map:
            continue

        last_error = None
        for attempt in range(1, args.max_retries + 1):
            try:
                new_item = regenerate_one_item(
                    item=item,
                    api_key=args.api_key,
                    base_url=args.base_url,
                    model=args.model,
                    api_version=args.api_version,
                    temperature=args.temperature,
                    max_output_tokens=args.max_output_tokens,
                    sexually_explicit_threshold=args.sexual_threshold,
                    thinking_level=args.thinking_level,
                    timeout=args.timeout,
                )
                progress_map[idx] = new_item
                append_progress(args.progress, idx, new_item)

                if (idx + 1) % 50 == 0 or idx == total - 1:
                    print(f"   完成 {idx + 1}/{total}", flush=True)
                break
            except Exception as exc:
                last_error = exc
                sleep_seconds = args.retry_sleep * attempt
                print(
                    f"   样本 {idx} 第 {attempt}/{args.max_retries} 次失败: {exc}. "
                    f"{sleep_seconds:.1f}s 后重试。",
                    flush=True,
                )
                time.sleep(sleep_seconds)
        else:
            raise RuntimeError(f"样本 {idx} 连续失败，停止处理。最后错误: {last_error}") from last_error

    finalize_output(dataset, progress_map, args.output)
    print(f"4. 已生成新数据集: {args.output}", flush=True)
    print(f"5. 进度文件保留在: {args.progress}", flush=True)


if __name__ == "__main__":
    main()
