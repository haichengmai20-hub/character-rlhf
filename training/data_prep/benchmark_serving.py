import argparse
import asyncio
import json
import random
import time
import aiohttp
import numpy as np
import sys

# ---------------- 1. 获取真实模型名 ----------------
async def get_real_model_name(session, api_url):
    try:
        async with session.get(f"{api_url}/v1/models") as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
            return data['data'][0]['id']
    except:
        return None

# ---------------- 2. 发送请求 (核心) ----------------
async def send_request(session, api_url, model, prompt, output_len, api_path, api_key=None):
    url = f"{api_url}{api_path}"
    payload = {
        "model": model,
        "max_tokens": output_len,
        "temperature": 0.7,
        "stream": False # 使用非流式以获得准确的总延迟和吞吐量
    }
    if api_path.endswith("/chat/completions"):
        payload["messages"] = [{"role": "user", "content": prompt}]
    else:
        payload["prompt"] = prompt

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    start_time = time.perf_counter()
    try:
        async with session.post(url, json=payload, headers=headers) as response:
            if response.status != 200:
                return None
            
            result = await response.json()
            end_time = time.perf_counter()
            
            # 提取 Usage 信息
            completion_tokens = result['usage']['completion_tokens']
            prompt_tokens = result['usage']['prompt_tokens']
            
            return {
                "latency": end_time - start_time,
                "output_tokens": completion_tokens,
                "prompt_tokens": prompt_tokens,
                "total_tokens": completion_tokens + prompt_tokens
            }
    except:
        return None

# ---------------- 3. 统计与压测逻辑 ----------------
async def benchmark(api_url, input_model_name, dataset_path, num_prompts, request_rate, api_path, api_key=None):
    async with aiohttp.ClientSession() as session:
        # --- A. 自动获取模型名 ---
        real_model_name = await get_real_model_name(session, api_url)
        target_model = real_model_name if real_model_name else input_model_name

        # --- B. 数据准备 (循环填充) ---
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                dataset = [json.loads(line) for line in f]
        except Exception as e:
            print(f"Error reading dataset: {e}")
            return

        # 数据扩充逻辑
        if len(dataset) < num_prompts:
            while len(dataset) < num_prompts:
                dataset.extend(dataset)
            dataset = dataset[:num_prompts]
        elif len(dataset) > num_prompts:
            dataset = random.sample(dataset, num_prompts)
        
        random.shuffle(dataset)

        # --- C. 执行压测 ---
        tasks = []
        start_benchmark_time = time.perf_counter()
        
        for data in dataset:
            prompt = data.get("prompt", "")
            out_len = data.get("output_len", 256)
            tasks.append(asyncio.create_task(send_request(session, api_url, target_model, prompt, out_len, api_path, api_key)))
            
            if request_rate != float('inf'):
                await asyncio.sleep(1.0 / request_rate)
        
        results = await asyncio.gather(*tasks)
        total_time = time.perf_counter() - start_benchmark_time

    # --- D. 详细统计报告 (仿官方格式) ---
    valid_results = [r for r in results if r is not None]
    success_count = len(valid_results)
    
    if success_count == 0:
        print("RESULT::FAILED")
        return

    # 提取基础数据
    latencies = np.array([r['latency'] for r in valid_results])
    output_tokens = np.array([r['output_tokens'] for r in valid_results])
    total_tokens_sum = sum(output_tokens)
    
    # 计算详细指标
    # 1. 吞吐量
    req_throughput = success_count / total_time
    token_throughput = total_tokens_sum / total_time
    
    # 2. 延迟分布 (Latency Percentiles)
    avg_latency = np.mean(latencies)
    p50_latency = np.percentile(latencies, 50)
    p90_latency = np.percentile(latencies, 90)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    
    # 3. TPOT (Time Per Output Token) - 估算
    # 这里的 TPOT 是平均每个 Token 的生成时间，大致等于 Latency / Output Tokens
    tpot_avg = np.mean(latencies / output_tokens) * 1000 # 转毫秒
    
    # --- E. 格式化输出 (供 auto_benchmark.py 抓取) ---
    print("\n" + "="*50)
    print(f"📊 Detailed Benchmark Result (QPS Setting: {request_rate})")
    print("="*50)
    print(f"Requests:")
    print(f"  Total:        {num_prompts}")
    print(f"  Successful:   {success_count}")
    print(f"  Duration:     {total_time:.2f} s")
    print("-" * 50)
    print(f"Throughput:")
    print(f"  Requests/s:   {req_throughput:.2f} req/s")
    print(f"  Tokens/s:     {token_throughput:.2f} tokens/s")
    print("-" * 50)
    print(f"Latency (End-to-End):")
    print(f"  Avg:          {avg_latency:.4f} s")
    print(f"  P50 (Median): {p50_latency:.4f} s")
    print(f"  P90:          {p90_latency:.4f} s")
    print(f"  P99:          {p99_latency:.4f} s")
    print("-" * 50)
    print(f"Est. Speed:")
    print(f"  TPOT (Avg):   {tpot_avg:.2f} ms/token")
    print("="*50 + "\n")

    # 打印一行简报供机器快速读取 (可选)
    print(f"RESULT_SUMMARY_JSON::{json.dumps({'qps': request_rate, 'rps': req_throughput, 'tps': token_throughput, 'p99': p99_latency})}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", type=str, default="unknown")
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--request-rate", type=float, default=5.0)
    parser.add_argument(
        "--api-path",
        type=str,
        default="/v1/completions",
        choices=["/v1/completions", "/v1/chat/completions"],
    )
    parser.add_argument("--api-key", type=str, default=None)
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/") if args.base_url else f"http://localhost:{args.port}"
    asyncio.run(benchmark(
        api_url=base_url,
        input_model_name=args.model,
        dataset_path=args.dataset_path,
        num_prompts=args.num_prompts,
        request_rate=args.request_rate,
        api_path=args.api_path,
        api_key=args.api_key,
    ))
