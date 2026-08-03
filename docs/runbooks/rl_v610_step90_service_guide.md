# RL v6.10 Step90 模型服务配置指导

本文用于指导其他人复现 `/root/rl_v610_step90_merged` 的 Gradio/vLLM 服务启动方式。

## 1. 模型与代码路径

- 模型目录：`/root/rl_v610_step90_merged`
- Gradio 应用入口：`/root/hanjiaqi/LlamaFactory/chatjoy/demo_character.py`
- 已验证启动包装脚本：`/root/run_rl_v610_step90_gradio.py`
- 服务日志：`/tmp/gradio_rl_v610_step90.log`
- 公网隧道日志：`/tmp/gradio_rl_v610_step90_tunnel.log`

模型目录应至少包含：

```text
config.json
generation_config.json
model.safetensors.index.json
model-00001-of-00002.safetensors
model-00002-of-00002.safetensors
tokenizer.json
tokenizer_config.json
processor_config.json
chat_template.jinja
```

当前模型大小约 `49G`。

## 2. 环境要求

推荐使用现有环境：

```bash
source /home/ps/miniconda3/etc/profile.d/conda.sh
conda activate gemma4-character
export PYTHONNOUSERSITE=1
```

已验证框架/运行栈：

- Python：`gemma4-character` conda 环境
- 推理框架：vLLM `v0.19.0`
- 模型架构：`Gemma4ForConditionalGeneration`
- dtype：`bfloat16`
- 分布式后端：NCCL
- Gradio：由 `demo_character.py` 内部启动

注意：不要临时升级/降级 transformers、vLLM、torch，除非重新做完整加载验证。

## 3. 硬件条件

已验证硬件：

- GPU：4 张 NVIDIA RTX 5090
- 单卡显存：约 32GB
- Tensor Parallel：`TP=4`
- 启动后单卡显存占用：约 `27.9GiB`
- 可用 KV cache：约 `13.96GiB`
- `max_model_len=8192` 下最大并发容量日志显示约 `79x`

最低建议：

- 至少 4 张 32GB 级别 GPU
- 启动前每张目标 GPU 空闲显存最好大于 `28GiB`
- 如果同机还跑其他 vLLM 服务，务必分配不同 GPU 和端口

检查 GPU：

```bash
nvidia-smi
```

## 4. 推荐服务参数

当前产品试用配置：

```text
temperature=0.8
top_p=0.95
presence_penalty=0.4
min_p=0.0
max_model_len=8192
max_response_tokens=1536
dtype=bfloat16
quantization=None
gpu_memory_utilization=0.84
enforce_eager=True
tensor_parallel_size=4
max_num_seqs=8
max_num_batched_tokens=1024
prefix_caching=True
chunked_prefill=True
text_only=True
```

`text_only=True` 是通过限制多模态输入实现的：

```python
limit_mm_per_prompt={"image": 0, "video": 0}
skip_mm_profiling=True
```

这样可避免 Gemma4 多模态 profile 带来的额外显存和启动不稳定。

## 5. 启动脚本

建议保存为 `/root/run_rl_v610_step90_gradio.py`：

```python
import importlib.util
import os

os.environ.setdefault("DEMO_SELECTED_GPUS", "0,1,2,3")
os.environ.setdefault("DEMO_BASE_MODEL_PATH", "/root/rl_v610_step90_merged")
os.environ.setdefault("DEMO_MODEL_BACKEND", "local")
os.environ.setdefault("DEMO_VLLM_QUANTIZATION", "")
os.environ.setdefault("DEMO_VLLM_DTYPE", "bfloat16")
os.environ.setdefault("DEMO_VLLM_GPU_MEMORY_UTILIZATION", "0.84")
os.environ.setdefault("DEMO_VLLM_MAX_MODEL_LEN", "8192")
os.environ.setdefault("DEMO_VLLM_ENFORCE_EAGER", "true")
os.environ.setdefault("DEMO_RESPONSE_MAX_TOKENS", "1536")
os.environ.setdefault("DEMO_GRADIO_SHARE", "false")
os.environ.setdefault("GRADIO_SERVER_PORT", "7860")
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,0.0.0.0")
os.environ.setdefault("no_proxy", os.environ["NO_PROXY"])

SCRIPT_PATH = "/root/hanjiaqi/LlamaFactory/chatjoy/demo_character.py"
spec = importlib.util.spec_from_file_location("demo_character_runtime", SCRIPT_PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

_sampling_params = mod.SamplingParams
_async_engine_args = mod.AsyncEngineArgs


def patched_sampling_params(*args, **kwargs):
    kwargs["temperature"] = 0.8
    kwargs["top_p"] = 0.95
    kwargs["presence_penalty"] = 0.4
    kwargs["min_p"] = 0.0
    kwargs["max_tokens"] = int(os.environ.get("DEMO_RESPONSE_MAX_TOKENS", "1536"))
    kwargs.pop("repetition_penalty", None)
    return _sampling_params(*args, **kwargs)


def patched_async_engine_args(*args, **kwargs):
    kwargs.setdefault("skip_mm_profiling", True)
    kwargs.setdefault("limit_mm_per_prompt", {"image": 0, "video": 0})
    return _async_engine_args(*args, **kwargs)


mod.SamplingParams = patched_sampling_params
mod.AsyncEngineArgs = patched_async_engine_args
mod.main()
```

如需改用 GPU 4-7，只改：

```python
os.environ.setdefault("DEMO_SELECTED_GPUS", "4,5,6,7")
os.environ.setdefault("GRADIO_SERVER_PORT", "7861")
```

## 6. tmux 启动命令

GPU 0-3、端口 7860：

```bash
tmux new -d -s gradio_rl_v610_step90 \
  'source /home/ps/miniconda3/etc/profile.d/conda.sh && \
   conda activate gemma4-character && \
   export PYTHONNOUSERSITE=1 && \
   cd /root && \
   python /root/run_rl_v610_step90_gradio.py > /tmp/gradio_rl_v610_step90.log 2>&1'
```

查看日志：

```bash
tail -f /tmp/gradio_rl_v610_step90.log
```

成功标志：

```text
Resolved architecture: Gemma4ForConditionalGeneration
Using max model len 8192
All limits of multimodal modalities supported by the model are set to 0, running in text-only mode.
启动服务... 本地访问 http://127.0.0.1:7860
```

## 7. 公网链接

优先使用 Gradio 自带公网 share；如果 share 失败，可用本机已有 frpc：

```bash
TOKEN=$(python - <<'PY'
import secrets
print(secrets.token_urlsafe(24))
PY
)

tmux new -d -s gradio_rl_v610_step90_tunnel \
  "bash -lc '/home/ps/.cache/huggingface/gradio/frpc/frpc_linux_amd64_v0.3 \
  http -n ${TOKEN} -l 7860 -i 0.0.0.0 --uc --sd random --ue \
  --server_addr gradio-live.com:7000 --disable_log_color \
  --tls_enable --tls_trusted_ca_file /root/.gradio/certificate.pem \
  > /tmp/gradio_rl_v610_step90_tunnel.log 2>&1'"
```

获取链接：

```bash
sed -n 's/.*start proxy success: //p' /tmp/gradio_rl_v610_step90_tunnel.log | tail -n 1
```

公网暴露模型前需要确认授权；不要默认把内部模型服务暴露到公网。

## 8. 停止服务

只停止本服务：

```bash
tmux kill-session -t gradio_rl_v610_step90
tmux kill-session -t gradio_rl_v610_step90_tunnel
```

停止后确认显存释放：

```bash
nvidia-smi
```

不要用模糊 `pkill vllm` 或 `killall python`，同机可能有其他训练/服务。

## 9. 常见问题

### tokenizer 兼容补丁

日志里可能出现：

```text
检测到 Gemma tokenizer 配置兼容问题，使用补丁目录: /tmp/gemma4_tokenizer_patch_...
```

这是正常现象，`demo_character.py` 会自动生成兼容补丁目录。

### 显存不足

如果报：

```text
Free memory on device ... is less than desired GPU memory utilization
```

先检查目标 GPU 是否有旧 vLLM worker 残留。若目标 GPU 确实空闲但仍 OOM，可按优先级下调：

1. `DEMO_VLLM_GPU_MEMORY_UTILIZATION=0.80`
2. `DEMO_VLLM_MAX_MODEL_LEN=6144`
3. `DEMO_RESPONSE_MAX_TOKENS=1024`

不要优先改 dtype；当前 `bfloat16` 已验证。

### 公网 share 失败

如果日志出现：

```text
Gradio Share 创建失败，已回退到公网直连地址
```

可用第 7 节的 frpc 方式创建 gradio.live 链接。

## 10. 当前已验证快照

最后一次验证时：

- GPU 0-3：`/root/rl_v610_step90_merged`
- 每卡显存：约 `27918MiB`
- 本地端口：`7860`
- 模型加载耗时：约 5 秒读取权重，约 2.6 秒初始化 KV cache/warmup
- 角色卡加载：23 个文件

该配置可以作为对外人工评测/产品试用的标准启动模板。
