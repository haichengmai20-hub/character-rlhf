# 1. 设置显卡可见性 (你的配置是 2,3,6,7)
export CUDA_VISIBLE_DEVICES=2,3,6,7

# 2. 启动 vLLM API 服务器
# 注意：这将占用终端，请不要关闭。如果需要后台运行可以使用 nohup
python -m vllm.entrypoints.openai.api_server \
    --model /home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/qwen3-30b-moe/chatjoy_v1_merged \
    --served-model-name qwen-ft \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \  # 32K
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --port 8000