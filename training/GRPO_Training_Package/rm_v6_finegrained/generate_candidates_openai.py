#!/usr/bin/env python3
"""Generate one ChatJoy RM v6 candidate stream from an OpenAI-compatible vLLM server."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from common import append_jsonl, iter_jsonl


def chat_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


def request_chat(
    url: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    top_p: float,
    max_tokens: int,
    api_key: str,
    timeout: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def load_done_ids(path: Path, source_label: str) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    for row in iter_jsonl(path):
        if row.get("source_model") == source_label and row.get("response"):
            done.add(str(row.get("prompt_id")))
    return done


def build_candidate_row(prompt: dict[str, Any], args: argparse.Namespace, url: str) -> dict[str, Any]:
    prompt_id = str(prompt["prompt_id"])
    started = time.time()
    error = ""
    raw: dict[str, Any] | None = None

    for attempt in range(1, args.max_retries + 1):
        try:
            raw = request_chat(
                url=url,
                model=args.model,
                messages=prompt["messages"],
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                api_key=args.api_key,
                timeout=args.timeout,
            )
            break
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            error = repr(exc)
            if attempt < args.max_retries:
                time.sleep(args.retry_sleep * attempt)

    response_text = ""
    finish_reason = "error"
    usage = {}
    if raw is not None:
        choices = raw.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            response_text = str(message.get("content") or "")
            finish_reason = str(choices[0].get("finish_reason") or "")
        usage = raw.get("usage") or {}

    return {
        "candidate_id": f"{prompt_id}:{args.source_label}",
        "prompt_id": prompt_id,
        "messages": prompt["messages"],
        "language": prompt.get("language"),
        "role_hash": prompt.get("role_hash"),
        "source_index": prompt.get("source_index"),
        "source_model": args.source_label,
        "served_model": args.model,
        "sampling": {"temperature": args.temperature, "top_p": args.top_p, "max_tokens": args.max_tokens},
        "response": response_text,
        "response_chars": len(response_text),
        "prompt_chars": sum(len(m.get("content", "")) for m in prompt["messages"]),
        "usage": usage,
        "finish_reason": finish_reason,
        "latency_s": round(time.time() - started, 3),
        "error": error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8001")
    parser.add_argument("--model", type=str, required=True, help="Served model name.")
    parser.add_argument("--source-label", type=str, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--top-p", type=float, required=True)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--api-key", type=str, default="EMPTY")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    done = load_done_ids(args.output, args.source_label) if args.resume else set()
    url = chat_url(args.base_url)
    pending: list[dict[str, Any]] = []

    for prompt in iter_jsonl(args.prompts):
        prompt_id = str(prompt["prompt_id"])
        if prompt_id in done:
            continue
        if args.limit is not None and len(pending) >= args.limit:
            break
        pending.append(prompt)

    written = 0

    if args.concurrency <= 1:
        for prompt in pending:
            append_jsonl(args.output, build_candidate_row(prompt, args, url))
            written += 1
            if args.progress_every and written % args.progress_every == 0:
                print(json.dumps({"source_model": args.source_label, "written": written, "attempted": len(pending)}, ensure_ascii=False), flush=True)
            if args.sleep:
                time.sleep(args.sleep)
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = [executor.submit(build_candidate_row, prompt, args, url) for prompt in pending]
            for future in as_completed(futures):
                append_jsonl(args.output, future.result())
                written += 1
                if args.progress_every and written % args.progress_every == 0:
                    print(json.dumps({"source_model": args.source_label, "written": written, "attempted": len(pending)}, ensure_ascii=False), flush=True)

    print(json.dumps({"output": str(args.output), "source_model": args.source_label, "attempted": len(pending), "written": written}, indent=2))


if __name__ == "__main__":
    main()
