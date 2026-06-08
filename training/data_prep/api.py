import os
import sys
import json
import argparse
from typing import Any, Dict, Optional

import requests


DEFAULT_BASE_URL = "https://gcpapi.magicutapp.com"
DEFAULT_MODEL = "gemini-3.1-flash-lite-preview"


def build_url(base_url: str, model: str, api_version: str = "v1beta") -> str:
    """
    官方 Gemini generateContent 路径格式：
    /v1beta/models/{model}:generateContent
    这里仅替换域名，不改路径结构。
    """
    base_url = base_url.rstrip("/")
    return f"{base_url}/{api_version}/models/{model}:generateContent"


def extract_text(resp_json: Dict[str, Any]) -> str:
    """
    尽量从 Gemini 风格响应中提取文本。
    官方常见结构：
    candidates[0].content.parts[0].text
    """
    candidates = resp_json.get("candidates", [])
    if not candidates:
        return json.dumps(resp_json, ensure_ascii=False, indent=2)

    parts = candidates[0].get("content", {}).get("parts", [])
    texts = []
    for part in parts:
        if isinstance(part, dict) and "text" in part:
            texts.append(part["text"])

    if texts:
        return "\n".join(texts)

    return json.dumps(resp_json, ensure_ascii=False, indent=2)


def call_gemini(
    api_key: str,
    prompt: str,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    api_version: str = "v1beta",
    temperature: float = 0.7,
    max_output_tokens: int = 2048,
    system_instruction: Optional[str] = None,
    timeout: int = 120,
) -> Dict[str, Any]:
    url = build_url(base_url=base_url, model=model, api_version=api_version)

    # 优先采用官方示例里的 x-goog-api-key
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }

    payload: Dict[str, Any] = {
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
        }
    }

    if system_instruction:
        payload["systemInstruction"] = {
            "parts": [
                {"text": system_instruction}
            ]
        }

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=timeout,
    )

    # 失败时尽量返回详细报错
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


def main():
    parser = argparse.ArgumentParser(description="Call Gemini via custom proxy URL")
    parser.add_argument("--prompt", type=str, help="输入提示词")
    parser.add_argument("--api_key", type=str, default=os.getenv("GEMINI_API_KEY"))
    parser.add_argument("--base_url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--api_version", type=str, default="v1beta")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_output_tokens", type=int, default=2048)
    parser.add_argument(
        "--system",
        type=str,
        default="你是一个严谨、简洁的中文助手。",
        help="系统提示词"
    )
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument(
        "--raw",
        action="store_true",
        help="打印完整 JSON 响应，而不是只打印文本"
    )

    args = parser.parse_args()

    if not args.api_key:
        print("错误：未提供 API Key。请通过 --api_key 或环境变量 GEMINI_API_KEY 提供。", file=sys.stderr)
        sys.exit(1)

    prompt = args.prompt
    if not prompt:
        print("请输入问题，结束后按 Ctrl+D（Linux/macOS）或 Ctrl+Z 回车（Windows）：")
        prompt = sys.stdin.read().strip()

    if not prompt:
        print("错误：prompt 不能为空。", file=sys.stderr)
        sys.exit(1)

    try:
        result = call_gemini(
            api_key=args.api_key,
            prompt=prompt,
            base_url=args.base_url,
            model=args.model,
            api_version=args.api_version,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            system_instruction=args.system,
            timeout=args.timeout,
        )

        if args.raw:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(extract_text(result))

    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
