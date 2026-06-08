#!/bin/bash
# test_rm_vllm_classify.sh
# 验证 chatjoy_rm_v3_merged_vllm 模型可以被 vLLM 加载并通过 /classify 端点返回 reward score
#
# 用法:
#   bash test_rm_vllm_classify.sh
#
# 前提:
#   1. 已运行 python3 GRPO_Training_Package/merge_rm_lora_for_vllm.py (生成 merged 模型)
#   2. vLLM 已 patch (Qwen3MoeForSequenceClassification 注册)
#   3. GPU 4-7 空闲

set -euo pipefail

RM_MODEL="/home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/qwen3-30b-moe/chatjoy_rm_v3_merged_vllm"
PORT=8001
TP=4
GPU_UTIL=0.62
CUDA_DEVICES="4,5,6,7"

echo "=== 启动 vLLM RM server (GPU ${CUDA_DEVICES}, TP=${TP}) ==="
CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" conda run -n gemma4-rl \
  python3 -m vllm.entrypoints.openai.api_server \
    --model "${RM_MODEL}" \
    --task classify \
    --tensor-parallel-size "${TP}" \
    --gpu-memory-utilization "${GPU_UTIL}" \
    --port "${PORT}" \
    --host 127.0.0.1 \
    --trust-remote-code &

SERVER_PID=$!
echo "vLLM server PID: ${SERVER_PID}"

# 等待 server 就绪
echo "等待 server 启动..."
for i in $(seq 1 120); do
  if curl -s "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then
    echo "Server ready after ${i}s"
    break
  fi
  sleep 5
done

echo ""
echo "=== 测试 /classify 端点 ==="
PROMPT="[INST] 你是一个角色扮演助手。请扮演龙傲天，一个霸道总裁。\n用户: 你好！[/INST]\n当然，我就是龙傲天！"

RESPONSE=$(curl -s -X POST "http://127.0.0.1:${PORT}/classify" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${RM_MODEL}\",
    \"input\": \"${PROMPT}\",
    \"use_activation\": false
  }")

echo "Response: ${RESPONSE}"
echo ""

# 提取 reward score
SCORE=$(echo "${RESPONSE}" | python3 -c "
import sys, json
data = json.load(sys.stdin)
if 'data' in data and len(data['data']) > 0:
    probs = data['data'][-1].get('probs', [])
    print(f'reward score (probs[-1]): {probs[-1] if probs else \"N/A\"}')
    print(f'all probs: {probs}')
else:
    print(f'ERROR: {data}')
")
echo "${SCORE}"

echo ""
echo "=== 清理 ==="
kill "${SERVER_PID}" 2>/dev/null || true
echo "Done."
