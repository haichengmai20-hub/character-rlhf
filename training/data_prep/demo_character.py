import os
import sys
import uuid
import asyncio
import json
import glob
import time
import re
import socket
import shutil
import hashlib
import ssl
import urllib.request
import urllib.error
from datetime import datetime

# ==================== 【环境配置】 ====================
# 旧版 vLLM 曾支持 VLLM_USE_V1；当前环境里该变量已失效，这里不再设置，
# 避免启动时出现 "Unknown vLLM environment variable" 警告。
# ====================================================

# ==================== 【用户配置区域】 ====================
# 1. 显卡选择
SELECTED_GPUS = os.environ.get("DEMO_SELECTED_GPUS", "0")

# 2. 模型选择 ("base" 或 "ft")
CURRENT_MODEL_TYPE = "base"

# 2.1 推理后端选择（默认本地 vLLM，可切换到 API）
# - local: 使用本地模型路径 + vLLM
# - api: 使用 OpenAI 兼容 Chat Completions API
MODEL_BACKEND = os.environ.get("DEMO_MODEL_BACKEND", "local").strip().lower()
API_BASE_URL = os.environ.get("DEMO_API_BASE_URL", "").strip().rstrip("/")
API_CHAT_COMPLETIONS_PATH = os.environ.get("DEMO_API_CHAT_COMPLETIONS_PATH", "/v1/chat/completions").strip()
API_MODEL_NAME = os.environ.get("DEMO_API_MODEL_NAME", "").strip()
API_KEY = os.environ.get("DEMO_API_KEY", "").strip()
API_TIMEOUT_SECONDS = float(os.environ.get("DEMO_API_TIMEOUT_SECONDS", "120"))
API_VERIFY_SSL = (os.environ.get("DEMO_API_VERIFY_SSL", "true").strip().lower() in {"1", "true", "yes", "on"})

# 3. 路径配置
BASE_MODEL_PATH = os.environ.get(
    "DEMO_BASE_MODEL_PATH",
    "/home/ps/.cache/huggingface/hub/models--cyankiwi--gemma-4-26B-A4B-it-AWQ-4bit/snapshots/c5b961cae592047a529e32a8fd740998fbac16be",
)
FT_MODEL_PATH = "/home/ps/hanjiaqi/LlamaFactory/chatjoy/saves/qwen3-30b-moe/clean32_for_sft1_fsdp_qlora_merged"

# 4. 角色卡路径配置
CHARACTERS_DIR = "/home/ps/hanjiaqi/LlamaFactory/chatjoy/TOP20CHARACTERS_systemprompt"

# 5. A/B Test: 权重加载与量化配置（默认不改变当前行为）
# - VLLM_LOAD_FORMAT: auto / safetensors / pt / npcache / bitsandbytes / tensorizer / sharded_state
# - VLLM_DTYPE: auto / half / float16 / bfloat16 / float32
# - VLLM_QUANTIZATION: None / awq / gptq / fp8 / bitsandbytes
VLLM_LOAD_FORMAT = os.environ.get("DEMO_VLLM_LOAD_FORMAT", "auto")
VLLM_DTYPE = os.environ.get("DEMO_VLLM_DTYPE", "auto")
VLLM_QUANTIZATION = os.environ.get("DEMO_VLLM_QUANTIZATION", "awq_marlin") or None
VLLM_GPU_MEMORY_UTILIZATION = float(os.environ.get("DEMO_VLLM_GPU_MEMORY_UTILIZATION", "0.84"))
VLLM_MAX_MODEL_LEN = int(os.environ.get("DEMO_VLLM_MAX_MODEL_LEN", "8192"))
VLLM_MAX_NUM_SEQS = int(os.environ.get("DEMO_VLLM_MAX_NUM_SEQS", "8"))
VLLM_MAX_NUM_BATCHED_TOKENS = int(os.environ.get("DEMO_VLLM_MAX_NUM_BATCHED_TOKENS", "1024"))
RESPONSE_MAX_TOKENS = int(os.environ.get("DEMO_RESPONSE_MAX_TOKENS", "512"))
TRANSLATION_MAX_TOKENS = int(os.environ.get("DEMO_TRANSLATION_MAX_TOKENS", "512"))
CHAT_LOG_PATH = "/home/ps/hanjiaqi/LlamaFactory/chatjoy/logs/chat_demo_logs.jsonl"
LOG_FULL_PROMPT = False

# =========================================================

# 6. Gradio 公网分享配置（默认开启，可通过环境变量关闭）
GRADIO_SHARE = (os.environ.get("DEMO_GRADIO_SHARE", "true").strip().lower() in {"1", "true", "yes", "on"})
GRADIO_SERVER_NAME = "0.0.0.0"
GRADIO_SERVER_PORT = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
GRADIO_PORT_SEARCH_LIMIT = int(os.environ.get("GRADIO_PORT_SEARCH_LIMIT", "30"))
GRADIO_SHARE_SERVER_ADDRESS = os.environ.get("GRADIO_SHARE_SERVER_ADDRESS") or None
GRADIO_SHARE_SERVER_PROTOCOL = os.environ.get("GRADIO_SHARE_SERVER_PROTOCOL") or None
GRADIO_PUBLIC_HOST = (os.environ.get("GRADIO_PUBLIC_HOST") or "").strip()


def append_no_proxy(*hosts: str) -> str:
    existing = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
    items = [x.strip() for x in existing.split(",") if x.strip()]
    for host in hosts:
        if host not in items:
            items.append(host)
    return ",".join(items)


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def is_api_backend() -> bool:
    return MODEL_BACKEND == "api"


def get_api_endpoint() -> str:
    path = API_CHAT_COMPLETIONS_PATH or "/v1/chat/completions"
    if not path.startswith("/"):
        path = "/" + path
    if not API_BASE_URL:
        return path
    return API_BASE_URL + path


def parse_api_message_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" and item.get("text"):
                    text_parts.append(str(item.get("text")))
                elif item.get("content"):
                    text_parts.append(str(item.get("content")))
            elif isinstance(item, str):
                text_parts.append(item)
        return "\n".join(text_parts)
    if isinstance(content, dict):
        if "text" in content:
            return str(content.get("text") or "")
        if "content" in content:
            return str(content.get("content") or "")
    return str(content or "")


def request_chat_completion_api(messages, temperature, top_p, max_tokens):
    endpoint = get_api_endpoint()
    if not API_BASE_URL:
        raise RuntimeError("未配置 DEMO_API_BASE_URL。")
    if not API_MODEL_NAME:
        raise RuntimeError("未配置 DEMO_API_MODEL_NAME。")

    payload = {
        "model": API_MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stream": False,
    }

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
    }
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    ssl_context = None
    if endpoint.startswith("https") and not API_VERIFY_SSL:
        ssl_context = ssl._create_unverified_context()

    req = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=API_TIMEOUT_SECONDS, context=ssl_context) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
        raise RuntimeError(f"API 请求失败: HTTP {e.code} - {detail}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"API 请求失败: {e}")

    try:
        body = json.loads(raw)
    except json.JSONDecodeError:
        raise RuntimeError(f"API 返回非 JSON: {raw[:500]}")

    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"API 响应缺少 choices: {body}")

    message = choices[0].get("message", {})
    content = parse_api_message_content(message.get("content", ""))
    return content


async def request_chat_completion_api_async(messages, temperature, top_p, max_tokens):
    return await asyncio.to_thread(
        request_chat_completion_api,
        messages,
        temperature,
        top_p,
        max_tokens,
    )


# --- 设置显卡 ---
os.environ["CUDA_VISIBLE_DEVICES"] = SELECTED_GPUS
os.environ["NO_PROXY"] = append_no_proxy("localhost", "127.0.0.1", "0.0.0.0")
os.environ["no_proxy"] = os.environ["NO_PROXY"]
TP_SIZE = len(SELECTED_GPUS.split(','))
VLLM_ENFORCE_EAGER = env_bool("DEMO_VLLM_ENFORCE_EAGER", False)
VLLM_ENABLE_PREFIX_CACHING = env_bool("DEMO_VLLM_ENABLE_PREFIX_CACHING", True)
FAST_RESPONSE_MODE = env_bool("DEMO_FAST_RESPONSE_MODE", True)

print(f"当前配置显卡: {SELECTED_GPUS} (并行度: {TP_SIZE})")

try:
    import gradio as gr
except ImportError:
    print("错误：请安装 gradio")
    sys.exit(1)

if is_api_backend():
    SamplingParams = None
    AsyncEngineArgs = None
    AsyncLLMEngine = None
    AutoTokenizer = None
else:
    try:
        from vllm import SamplingParams
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        from transformers import AutoTokenizer
    except ImportError:
        print("错误：local 后端需要安装 vllm 和 transformers")
        sys.exit(1)

try:
    from deep_translator import GoogleTranslator, MyMemoryTranslator
    TRANSLATOR_READY = True
except ImportError:
    GoogleTranslator = None
    MyMemoryTranslator = None
    TRANSLATOR_READY = False
    print("⚠️ 未安装 deep-translator，左侧翻译链路不可用。安装命令: pip install deep-translator")

# 全局变量
llm_engine = None
tokenizer = None
CHARACTER_DICT = {} 
CHARACTER_FILE_MAP = {}

TRANSLATION_LANGUAGES = {
    "英语 (English)": "en",
    "日语 (Japanese)": "ja",
    "韩语 (Korean)": "ko",
    "法语 (French)": "fr",
    "德语 (German)": "de",
    "乌克兰语 (Ukrainian)": "uk",
    "瑞典语 (Swedish)": "sv",
    "波兰语 (Polish)": "pl",
    "葡萄牙语 (Portuguese)": "pt",
    "意大利语 (Italian)": "it",
    "波斯语 (Persian)": "fa",
    "罗马尼亚语 (Romanian)": "ro",
    "斯洛伐克语 (Slovak)": "sk",
    "印地语 (Hindi)": "hi",
    "克罗地亚语 (Croatian)": "hr",
    "丹麦语 (Danish)": "da",
    "土耳其语 (Turkish)": "tr",
    "西班牙语 (Spanish)": "es",
    "俄语 (Russian)": "ru",
    "阿拉伯语 (Arabic)": "ar",
    "泰语 (Thai)": "th",
    "越南语 (Vietnamese)": "vi",
}
DEFAULT_TRANSLATION_LANGUAGE = "英语 (English)"
TRANSLATION_FALLBACK_TO_MODEL = True
# 默认优先本地模型，避免外网受限导致超时
TRANSLATION_PROVIDER_PRIORITY = ["model", "mymemory", "google"]
# 强制输入翻译与回译优先使用本地模型；如需外部翻译可改为 False。
FORCE_LOCAL_MODEL_TRANSLATION = True


def append_chat_log(record):
    """Append one chat record as JSONL."""
    try:
        log_dir = os.path.dirname(CHAT_LOG_PATH)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(CHAT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"❌ 写日志失败: {e}")

def load_characters():
    global CHARACTER_DICT, CHARACTER_FILE_MAP
    CHARACTER_DICT = {}
    CHARACTER_FILE_MAP = {}
    
    if not os.path.exists(CHARACTERS_DIR):
        print(f"⚠️ 警告: 角色目录不存在: {CHARACTERS_DIR}")
        return []

    json_files = glob.glob(os.path.join(CHARACTERS_DIR, "*.json"))
    print(f"📂 正在加载角色卡，找到 {len(json_files)} 个文件...")

    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                file_stem = os.path.splitext(os.path.basename(file_path))[0]
                char_name = data.get("name") or data.get("character_name") or file_stem
                char_desc = data.get("description", "")
                
                if char_name and char_desc:
                    CHARACTER_DICT[char_name] = char_desc
                    CHARACTER_FILE_MAP[char_name] = file_path
        except Exception as e:
            print(f"❌ 读取文件 {file_path} 失败: {e}")
    
    names = sorted(CHARACTER_DICT.keys(), key=lambda x: str(x).lower())
    return names

def refresh_character_choices(current_selected_name):
    char_names = load_characters()
    selected_name = current_selected_name if current_selected_name in CHARACTER_DICT else None
    if selected_name is None and char_names:
        selected_name = char_names[0]
    current_prompt = CHARACTER_DICT.get(selected_name, "") if selected_name else ""
    return gr.Dropdown(choices=char_names, value=selected_name), current_prompt

def save_character_prompt(selected_char_name, prompt_text):
    if not selected_char_name:
        return "请先选择角色后再保存。"

    file_path = CHARACTER_FILE_MAP.get(selected_char_name)
    if not file_path:
        return f"未找到角色文件：{selected_char_name}"

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        data["name"] = selected_char_name
        data["description"] = (prompt_text or "").strip()

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        CHARACTER_DICT[selected_char_name] = data["description"]
        return f"已保存到角色文件：{os.path.basename(file_path)}"
    except Exception as e:
        return f"保存失败：{e}"


def make_safe_character_filename(name):
    cleaned = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", (name or "").strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("._")
    return cleaned or f"character_{int(time.time())}"


def create_character_prompt(new_char_name, prompt_text):
    new_char_name = (new_char_name or "").strip()
    prompt_text = (prompt_text or "").strip()

    if not new_char_name:
        dropdown, current_prompt = refresh_character_choices(None)
        return dropdown, current_prompt, "请先输入新角色名称。", ""
    if not prompt_text:
        dropdown, current_prompt = refresh_character_choices(None)
        return dropdown, current_prompt, "请先输入新角色的 System Prompt。", new_char_name
    if new_char_name in CHARACTER_DICT:
        dropdown, current_prompt = refresh_character_choices(new_char_name)
        return dropdown, current_prompt, f"角色已存在：{new_char_name}", new_char_name

    file_name = make_safe_character_filename(new_char_name) + ".json"
    file_path = os.path.join(CHARACTERS_DIR, file_name)

    if os.path.exists(file_path):
        dropdown, current_prompt = refresh_character_choices(None)
        return dropdown, current_prompt, f"文件已存在：{file_name}", new_char_name

    try:
        os.makedirs(CHARACTERS_DIR, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "name": new_char_name,
                    "description": prompt_text,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        CHARACTER_DICT[new_char_name] = prompt_text
        CHARACTER_FILE_MAP[new_char_name] = file_path
        dropdown, current_prompt = refresh_character_choices(new_char_name)
        return dropdown, current_prompt, f"已创建新角色：{new_char_name}", ""
    except Exception as e:
        dropdown, current_prompt = refresh_character_choices(None)
        return dropdown, current_prompt, f"创建失败：{e}", new_char_name


def find_available_port(start_port, search_limit):
    for port in range(start_port, start_port + max(search_limit, 1)):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    raise OSError(
        f"Cannot find empty port in range: {start_port}-{start_port + max(search_limit, 1) - 1}"
    )


def detect_public_host():
    if GRADIO_PUBLIC_HOST:
        return GRADIO_PUBLIC_HOST

    services = [
        "https://api.ipify.org",
        "https://ifconfig.me/ip",
        "https://icanhazip.com",
    ]
    for url in services:
        try:
            from urllib.request import urlopen

            with urlopen(url, timeout=5) as response:
                value = response.read().decode("utf-8").strip()
                if value:
                    return value
        except Exception:
            continue
    return None


def build_public_url(host, port):
    if not host:
        return None
    return f"http://{host}:{port}"


def prepare_compatible_tokenizer_dir(model_path: str) -> str:
    model_dir = os.path.abspath(model_path)
    tokenizer_config_path = os.path.join(model_dir, "tokenizer_config.json")
    if not os.path.exists(tokenizer_config_path):
        return model_dir

    try:
        with open(tokenizer_config_path, "r", encoding="utf-8") as f:
            tokenizer_config = json.load(f)
    except Exception:
        return model_dir

    extra_special_tokens = tokenizer_config.get("extra_special_tokens")
    if not isinstance(extra_special_tokens, list):
        return model_dir

    patch_key = hashlib.md5(model_dir.encode("utf-8")).hexdigest()[:12]
    patched_dir = os.path.join("/tmp", f"gemma4_tokenizer_patch_{patch_key}")
    os.makedirs(patched_dir, exist_ok=True)

    for name in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "config.json", "generation_config.json"]:
        src = os.path.join(model_dir, name)
        dst = os.path.join(patched_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, dst)

    tokenizer_config.pop("extra_special_tokens", None)
    with open(os.path.join(patched_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)

    return patched_dir


def detect_model_quantization_method(model_path: str):
    config_path = os.path.join(os.path.abspath(model_path), "config.json")
    if not os.path.exists(config_path):
        return None
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception:
        return None

    quant_config = config.get("quantization_config")
    if not isinstance(quant_config, dict):
        return None

    quant_method = quant_config.get("quant_method")
    return quant_method if isinstance(quant_method, str) and quant_method.strip() else None


def load_model_engine():
    global llm_engine, tokenizer

    if is_api_backend():
        endpoint = get_api_endpoint()
        if not API_BASE_URL:
            raise RuntimeError("当前为 API 后端，但未设置 DEMO_API_BASE_URL")
        if not API_MODEL_NAME:
            raise RuntimeError("当前为 API 后端，但未设置 DEMO_API_MODEL_NAME")

        llm_engine = None
        tokenizer = None
        print(f"正在使用 API 后端: model={API_MODEL_NAME} endpoint={endpoint}")
        return f"API模型 ({API_MODEL_NAME})"
    
    if CURRENT_MODEL_TYPE == "base":
        path = BASE_MODEL_PATH
        name = "原模型 (Base)"
    else:
        path = FT_MODEL_PATH
        name = "微调模型 (Fine-tuned)"

    print(f"正在加载: {name} | 路径: {path}")
    tokenizer_path = prepare_compatible_tokenizer_dir(path)
    model_quantization_method = detect_model_quantization_method(path)
    if tokenizer_path != path:
        print(f"⚠️ 检测到 Gemma tokenizer 配置兼容问题，使用补丁目录: {tokenizer_path}")
    
    try:
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        except AttributeError as exc:
            if "keys" not in str(exc):
                raise
            print("⚠️ Fast tokenizer 与当前 Gemma 配置不兼容，回退到 slow tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, use_fast=False)

        if tokenizer.chat_template is None:
            print("⚠️ 未检测到 Chat Template，应用默认 ChatML 模板...")
            tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

        # 通过可配置参数做权重/量化 A/B test；默认配置等价于当前行为
        engine_kwargs = {
            "model": path,
            "tokenizer": tokenizer_path,
            "trust_remote_code": True,
            "gpu_memory_utilization": VLLM_GPU_MEMORY_UTILIZATION,
            "tensor_parallel_size": TP_SIZE,
            "max_model_len": VLLM_MAX_MODEL_LEN,
            "max_num_seqs": VLLM_MAX_NUM_SEQS,
            "max_num_batched_tokens": VLLM_MAX_NUM_BATCHED_TOKENS,
            "enforce_eager": VLLM_ENFORCE_EAGER,
            "enable_prefix_caching": VLLM_ENABLE_PREFIX_CACHING,
            "load_format": VLLM_LOAD_FORMAT,
            "dtype": VLLM_DTYPE,
        }
        explicit_quantization = VLLM_QUANTIZATION
        if explicit_quantization and model_quantization_method and explicit_quantization != model_quantization_method:
            print(
                "⚠️ 检测到模型自带量化配置与手动 quantization 参数冲突："
                f"model={model_quantization_method}, arg={explicit_quantization}。"
                "将优先使用模型配置，忽略手动 quantization 参数。"
            )
            explicit_quantization = None
        if explicit_quantization:
            engine_kwargs["quantization"] = explicit_quantization
        elif model_quantization_method:
            print(f"⚠️ 检测到模型自带量化配置: {model_quantization_method}，将由 vLLM 自动读取。")

        print(
            f"vLLM配置: prefix_caching={VLLM_ENABLE_PREFIX_CACHING}, "
            f"enforce_eager={VLLM_ENFORCE_EAGER}, "
            f"max_num_seqs={VLLM_MAX_NUM_SEQS}, "
            f"max_num_batched_tokens={VLLM_MAX_NUM_BATCHED_TOKENS}, "
            f"load_format={VLLM_LOAD_FORMAT}, dtype={VLLM_DTYPE}, "
            f"quantization={explicit_quantization or model_quantization_method}"
        )
        engine_args = AsyncEngineArgs(**engine_kwargs)
        
        llm_engine = AsyncLLMEngine.from_engine_args(engine_args)
        return name
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        raise e

def parse_gradio_content(content):
    if content is None: return ""
    if isinstance(content, str): return content
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get('type') == 'text':
                text_parts.append(str(item.get('text', '')))
            elif isinstance(item, str):
                text_parts.append(item)
        return "\n".join(text_parts)
    if isinstance(content, dict) and content.get('type') == 'text':
        return str(content.get('text', ''))
    return str(content)


def strip_thinking_content(text):
    cleaned = text or ""
    if not cleaned:
        return ""

    # Remove completed <think>...</think> blocks first.
    previous = None
    while cleaned != previous:
        previous = cleaned
        cleaned = re.sub(r"(?is)<think>\s*.*?\s*</think>\s*", "", cleaned)

    # If the model is still streaming an open <think> block, hide it from UI/back-translation.
    cleaned = re.sub(r"(?is)<think>\s*.*$", "", cleaned)

    # Remove any leftover bare tags.
    cleaned = re.sub(r"(?is)</?think>\s*", "", cleaned)
    return cleaned.strip()


def apply_fast_response_hint(system_prompt: str) -> str:
    prompt = (system_prompt or "").strip()
    if not FAST_RESPONSE_MODE:
        return prompt

    speed_hint = (
        "\n\n[Response Style Override]\n"
        "Respond directly and naturally. Do not output chain-of-thought, hidden reasoning, or <think> tags. "
        "Avoid unnecessary preambles and repetition. Keep replies concise by default, unless the user explicitly asks for detail."
    )
    return prompt + speed_hint if prompt else speed_hint.strip()

def translate_text(text, source_lang, target_lang):
    clean_text = (text or "").strip()
    if not clean_text:
        return ""
    if source_lang == target_lang:
        return clean_text
    if not TRANSLATOR_READY:
        raise RuntimeError("未安装 deep-translator，请先执行: pip install deep-translator")
    return GoogleTranslator(source=source_lang, target=target_lang).translate(clean_text)

def translate_text_mymemory(text, source_lang, target_lang):
    clean_text = (text or "").strip()
    if not clean_text:
        return ""
    if source_lang == target_lang:
        return clean_text
    if not TRANSLATOR_READY:
        raise RuntimeError("未安装 deep-translator，请先执行: pip install deep-translator")
    return MyMemoryTranslator(source=source_lang, target=target_lang).translate(clean_text)

async def translate_text_with_model(text, source_lang_label, target_lang_label):
    clean_text = (text or "").strip()
    if not clean_text:
        return ""
    if is_api_backend():
        if not API_BASE_URL or not API_MODEL_NAME:
            raise RuntimeError("API 后端未配置完整，无法使用模型翻译。")
    elif llm_engine is None or tokenizer is None:
        raise RuntimeError("模型未加载，无法使用模型翻译兜底。")

    # 从类似“德语 (German)”中提取英文名，避免模型对混合标签理解不稳定。
    def normalize_language_label(label):
        if not label:
            return ""
        if "(" in label and ")" in label:
            inner = label.split("(", 1)[1].split(")", 1)[0].strip()
            if inner:
                return inner
        return label

    source_name = normalize_language_label(source_lang_label)
    target_name = normalize_language_label(target_lang_label)

    language_meta = {
        "Arabic": {"code": "ar", "native": "العربية"},
        "Chinese": {"code": "zh-CN", "native": "中文"},
        "Croatian": {"code": "hr", "native": "Hrvatski"},
        "Danish": {"code": "da", "native": "Dansk"},
        "English": {"code": "en", "native": "English"},
        "French": {"code": "fr", "native": "Français"},
        "German": {"code": "de", "native": "Deutsch"},
        "Hindi": {"code": "hi", "native": "हिन्दी"},
        "Italian": {"code": "it", "native": "Italiano"},
        "Japanese": {"code": "ja", "native": "日本語"},
        "Korean": {"code": "ko", "native": "한국어"},
        "Persian": {"code": "fa", "native": "فارسی"},
        "Polish": {"code": "pl", "native": "Polski"},
        "Portuguese": {"code": "pt", "native": "Português"},
        "Romanian": {"code": "ro", "native": "Română"},
        "Russian": {"code": "ru", "native": "Русский"},
        "Slovak": {"code": "sk", "native": "Slovenčina"},
        "Spanish": {"code": "es", "native": "Español"},
        "Swedish": {"code": "sv", "native": "Svenska"},
        "Thai": {"code": "th", "native": "ไทย"},
        "Turkish": {"code": "tr", "native": "Türkçe"},
        "Ukrainian": {"code": "uk", "native": "Українська"},
        "Vietnamese": {"code": "vi", "native": "Tiếng Việt"},
    }

    def get_language_code(name):
        return language_meta.get(name, {}).get("code", "")

    def get_language_native(name):
        return language_meta.get(name, {}).get("native", name)

    def clean_translation_output(raw_text):
        cleaned = (raw_text or "").strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\\n?", "", cleaned)
            cleaned = re.sub(r"\\n?```$", "", cleaned)
        cleaned = strip_thinking_content(cleaned)
        cleaned = re.sub(r"^(Translation|译文|翻译)\s*[:：]\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.strip()
        if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in ['\"', "'"]:
            cleaned = cleaned[1:-1].strip()
        return cleaned

    def cjk_ratio(s):
        text_local = s or ""
        if not text_local:
            return 0.0
        cjk_chars = re.findall(r"[\\u4e00-\\u9fff]", text_local)
        return len(cjk_chars) / max(len(text_local), 1)

    def should_repair_direction(translated_text):
        # 目标是中文时，若几乎没有中文字符，则认为方向可能错了。
        if target_name.lower() in ["chinese", "zh", "zh-cn", "zh_cn"]:
            return cjk_ratio(translated_text) < 0.08
        # 源是中文、目标非中文时，若输出仍大量中文，认为未真正翻译。
        if source_name.lower() in ["chinese", "zh", "zh-cn", "zh_cn"]:
            return cjk_ratio(translated_text) > 0.35

        # 德语专项：防止误译成英语。
        if target_name.lower() == "german":
            t = (translated_text or "").lower()
            if len(t) >= 24:
                german_markers = [
                    " der ", " die ", " das ", " und ", " ist ", " nicht ",
                    " ich ", " du ", " wir ", " mit ", " fur ", " fuer ", " ein ", " eine ",
                    " auf ", " zu ", " im ", " den ", " dem ", " des ", " es ", " sie "
                ]
                english_markers = [
                    " the ", " and ", " is ", " are ", " i ", " you ", " we ", " they ",
                    " to ", " of ", " in ", " on ", " with ", " for ", " this ", " that "
                ]
                has_german_chars = any(ch in t for ch in ["ä", "ö", "ü", "ß"])
                german_score = sum(tok in f" {t} " for tok in german_markers) + (1 if has_german_chars else 0)
                english_score = sum(tok in f" {t} " for tok in english_markers)
                if german_score == 0 and english_score >= 2:
                    return True
        return False

    source_code = get_language_code(source_name)
    target_code = get_language_code(target_name)
    source_native = get_language_native(source_name)
    target_native = get_language_native(target_name)
    german_extra = ""
    if target_name.lower() == "german":
        german_extra = " Important: Output must be in German (Deutsch), not English."

    translation_messages = [
        {
            "role": "system",
            "content": (
                "You are a professional translation engine. "
                "Translate faithfully from source language to target language. "
                "Keep semantics, tone, and formatting (including newlines, punctuation, *action markers*). "
                "Return ONLY the translation text. Do not explain. Do not add any prefix. "
                "Do not output <think> tags or reasoning."
                + german_extra
            ),
        },
        {
            "role": "user",
            "content": (
                f"Source language: {source_name}\\n"
                f"Source language code: {source_code}\\n"
                f"Source language native name: {source_native}\\n"
                f"Target language: {target_name}\\n"
                f"Target language code: {target_code}\\n"
                f"Target language native name: {target_native}\\n"
                "Translate the following text exactly:\n"
                f"{clean_text}"
            ),
        },
    ]
    if is_api_backend():
        latest_text = await request_chat_completion_api_async(
            translation_messages,
            temperature=0.0,
            top_p=1.0,
            max_tokens=TRANSLATION_MAX_TOKENS,
        )
    else:
        prompt = tokenizer.apply_chat_template(
            translation_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.0,
            max_tokens=TRANSLATION_MAX_TOKENS
        )
        request_id = str(uuid.uuid4())
        results_generator = llm_engine.generate(prompt, sampling_params, request_id)

        latest_text = ""
        async for request_output in results_generator:
            latest_text = request_output.outputs[0].text

    translated = clean_translation_output(latest_text)
    if translated and should_repair_direction(translated):
        repair_messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict translation fixer. "
                    "Output must be in the TARGET language only. "
                    "Return only corrected translation text. "
                    "Do not output <think> tags or reasoning."
                    + german_extra
                ),
            },
            {
                "role": "user",
                "content": (
                    f"TARGET language is {target_name}.\\n"
                    f"TARGET language code is {target_code}.\\n"
                    f"TARGET language native name is {target_native}.\\n"
                    f"SOURCE language is {source_name}.\\n"
                    f"SOURCE language code is {source_code}.\\n"
                    "Original source text:\n"
                    f"{clean_text}\\n\\n"
                    "Previous translation (wrong direction or mixed language):\n"
                    f"{translated}"
                ),
            },
        ]
        if is_api_backend():
            repair_text = await request_chat_completion_api_async(
                repair_messages,
                temperature=0.0,
                top_p=1.0,
                max_tokens=TRANSLATION_MAX_TOKENS,
            )
        else:
            repair_prompt = tokenizer.apply_chat_template(
                repair_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            repair_request_id = str(uuid.uuid4())
            repair_generator = llm_engine.generate(repair_prompt, sampling_params, repair_request_id)
            repair_text = ""
            async for request_output in repair_generator:
                repair_text = request_output.outputs[0].text
        repaired = clean_translation_output(repair_text)
        if repaired:
            translated = repaired

    return translated

async def translate_text_auto(text, source_lang, target_lang, source_lang_label, target_lang_label, providers=None):
    clean_text = (text or "").strip()
    if not clean_text:
        return ""
    if source_lang == target_lang:
        return clean_text

    provider_errors = []
    provider_list = providers if providers else TRANSLATION_PROVIDER_PRIORITY
    for provider in provider_list:
        try:
            if provider == "mymemory":
                return translate_text_mymemory(clean_text, source_lang, target_lang)
            if provider == "google":
                return translate_text(clean_text, source_lang, target_lang)
            if provider == "model":
                return await translate_text_with_model(clean_text, source_lang_label, target_lang_label)
        except Exception as e:
            provider_errors.append(f"{provider}: {e}")

    raise RuntimeError("所有翻译后端均失败: " + " | ".join(provider_errors))

def get_latest_user_message(history):
    for turn in reversed(history):
        if isinstance(turn, dict) and turn.get("role") == "user":
            return parse_gradio_content(turn.get("content")).strip()
    return ""

# ==================== 【修复部分 1：用户发送处理函数】 ====================
# 这一步只负责：把用户说的话上屏，清空输入框
def add_user_message(message, history):
    if message is None or message.strip() == "":
        return "", history
    # 将用户消息加入历史
    new_history = history + [{"role": "user", "content": message}]
    # 返回：清空的输入框, 更新后的历史
    return "", new_history

def add_cn_user_message(message, cn_history):
    if message is None or message.strip() == "":
        return "", cn_history
    new_cn_history = cn_history + [{"role": "user", "content": message.strip()}]
    return "", new_cn_history

async def inject_translated_user(cn_history, model_history, selected_target_language):
    cn_user_input = get_latest_user_message(cn_history)
    if not cn_user_input:
        return cn_history, model_history

    target_lang = TRANSLATION_LANGUAGES.get(selected_target_language, "en")
    try:
        if FORCE_LOCAL_MODEL_TRANSLATION:
            providers = ["model"]
        else:
            providers = TRANSLATION_PROVIDER_PRIORITY if TRANSLATION_FALLBACK_TO_MODEL else [p for p in TRANSLATION_PROVIDER_PRIORITY if p != "model"]
        translated_input = await translate_text_auto(
            cn_user_input,
            "zh-CN",
            target_lang,
            "Chinese",
            selected_target_language,
            providers
        )
    except Exception as e:
        err_msg = {
            "role": "assistant",
            "content": f"翻译错误（中文 -> {selected_target_language}）: {e}",
        }
        return cn_history + [err_msg], model_history

    updated_model_history = model_history + [{"role": "user", "content": translated_input}]
    return cn_history, updated_model_history

# ==================== 【修复部分 2：模型生成函数】 ====================
# 这一步负责：读取历史，生成回复，实时更新历史
async def chat_response(history, selected_char_name, manual_system_prompt):
    if (not is_api_backend()) and llm_engine is None:
        # 如果模型没加载，yield 当前历史（不做改变）
        yield history
        return

    req_start_ts = time.perf_counter()
    ttft_ms = None
    final_output_text = ""

    # 1. 获取 System Prompt
    system_prompt = (manual_system_prompt or "").strip()
    if not system_prompt:
        system_prompt = "You are a helpful assistant."
        if selected_char_name in CHARACTER_DICT:
            system_prompt = CHARACTER_DICT[selected_char_name]
    system_prompt = apply_fast_response_hint(system_prompt)
    
    messages = []
    
    # 添加 System Prompt
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # 2. 构建 Prompt
    # 注意：history 中已经是包含了刚才用户发送的那句话了
    for turn in history:
        if isinstance(turn, dict):
            role = turn.get("role")
            content = parse_gradio_content(turn.get("content"))
            if role and content: 
                messages.append({"role": role, "content": content})

    prompt = None
    if not is_api_backend():
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except Exception as e:
            # 如果报错，追加一条报错信息给用户看
            yield history + [{"role": "assistant", "content": f"Template Error: {e}"}]
            return

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.8,
        repetition_penalty=1.05,
        max_tokens=RESPONSE_MAX_TOKENS
    ) if not is_api_backend() else None
    request_id = str(uuid.uuid4())
    user_input = ""
    for turn in reversed(history):
        if isinstance(turn, dict) and turn.get("role") == "user":
            user_input = parse_gradio_content(turn.get("content"))
            break
    
    try:
        if is_api_backend():
            latest_text = await request_chat_completion_api_async(
                messages,
                temperature=0.7,
                top_p=0.8,
                max_tokens=RESPONSE_MAX_TOKENS,
            )
            final_output_text = strip_thinking_content(latest_text)
            if final_output_text:
                if ttft_ms is None:
                    ttft_ms = round((time.perf_counter() - req_start_ts) * 1000, 2)
                yield history + [{"role": "assistant", "content": final_output_text}]
            total_ms = round((time.perf_counter() - req_start_ts) * 1000, 2)
            log_record = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "request_id": request_id,
                "character": selected_char_name,
                "input": user_input,
                "output": final_output_text,
                "ttft_ms": ttft_ms,
                "total_ms": total_ms,
                "input_chars": len(user_input),
                "output_chars": len(final_output_text),
                "backend": "api",
                "api_model": API_MODEL_NAME,
            }
            if LOG_FULL_PROMPT:
                log_record["system_prompt"] = system_prompt
                log_record["messages"] = messages
            append_chat_log(log_record)
            return

        results_generator = llm_engine.generate(prompt, sampling_params, request_id)

        # --- 流式输出核心逻辑 ---
        # 节流更新：减少前端重绘频率，提升长文本时的体感速度
        loop = asyncio.get_running_loop()
        emit_interval_s = 0.08
        min_chars_delta = 12
        last_emit_time = 0.0
        last_emitted_text = ""
        latest_text = ""

        async for request_output in results_generator:
            generated_text = request_output.outputs[0].text
            latest_text = generated_text
            display_text = strip_thinking_content(generated_text)
            now = loop.time()
            finished = bool(getattr(request_output, "finished", False))
            delta_chars = len(generated_text) - len(last_emitted_text)

            # 满足任一条件才刷新：
            # 1) 距上次刷新超过阈值；2) 新增字符足够多；3) 生成结束
            if (
                finished
                or delta_chars >= min_chars_delta
                or (now - last_emit_time) >= emit_interval_s
            ):
                if ttft_ms is None and display_text:
                    ttft_ms = round((time.perf_counter() - req_start_ts) * 1000, 2)
                if display_text or finished:
                    current_bot_message = {"role": "assistant", "content": display_text}
                    yield history + [current_bot_message]
                    last_emitted_text = generated_text
                    last_emit_time = now

        # 兜底：确保最终完整文本一定展示出来
        final_display_text = strip_thinking_content(latest_text if latest_text else last_emitted_text)
        if final_display_text and latest_text and latest_text != last_emitted_text:
            yield history + [{"role": "assistant", "content": final_display_text}]

        final_output_text = strip_thinking_content(latest_text if latest_text else last_emitted_text)
        total_ms = round((time.perf_counter() - req_start_ts) * 1000, 2)
        log_record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "request_id": request_id,
            "character": selected_char_name,
            "input": user_input,
            "output": final_output_text,
            "ttft_ms": ttft_ms,
            "total_ms": total_ms,
            "input_chars": len(user_input),
            "output_chars": len(final_output_text),
            "backend": "local",
        }
        if LOG_FULL_PROMPT:
            log_record["system_prompt"] = system_prompt
            log_record["messages"] = messages
            log_record["prompt"] = prompt
            log_record["prompt_chars"] = len(prompt)
        append_chat_log(log_record)
            
    except Exception as e:
        total_ms = round((time.perf_counter() - req_start_ts) * 1000, 2)
        log_record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "request_id": request_id,
            "character": selected_char_name,
            "input": user_input,
            "output": "",
            "ttft_ms": ttft_ms,
            "total_ms": total_ms,
            "error": str(e),
        }
        if LOG_FULL_PROMPT:
            log_record["system_prompt"] = system_prompt
            log_record["messages"] = messages
            log_record["prompt"] = prompt
            log_record["prompt_chars"] = len(prompt)
        append_chat_log(log_record)
        yield history + [{"role": "assistant", "content": f"Generation Error: {e}"}]

async def chat_response_with_back_translation(model_history, cn_history, selected_char_name, selected_target_language, manual_system_prompt):
    if (not is_api_backend()) and llm_engine is None:
        yield model_history, cn_history
        return

    if not model_history or model_history[-1].get("role") != "user":
        yield model_history, cn_history
        return

    req_start_ts = time.perf_counter()
    ttft_ms = None
    final_output_text = ""
    target_lang = TRANSLATION_LANGUAGES.get(selected_target_language, "en")

    system_prompt = (manual_system_prompt or "").strip()
    if not system_prompt:
        system_prompt = "You are a helpful assistant."
        if selected_char_name in CHARACTER_DICT:
            system_prompt = CHARACTER_DICT[selected_char_name]
    system_prompt = apply_fast_response_hint(system_prompt)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    for turn in model_history:
        if isinstance(turn, dict):
            role = turn.get("role")
            content = parse_gradio_content(turn.get("content"))
            if role and content:
                messages.append({"role": role, "content": content})

    prompt = None
    if not is_api_backend():
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except Exception as e:
            yield model_history + [{"role": "assistant", "content": f"Template Error: {e}"}], cn_history + [{"role": "assistant", "content": f"模板错误: {e}"}]
            return

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.8,
        repetition_penalty=1.05,
        max_tokens=RESPONSE_MAX_TOKENS
    ) if not is_api_backend() else None
    request_id = str(uuid.uuid4())
    user_input = get_latest_user_message(model_history)

    try:
        if is_api_backend():
            latest_text = await request_chat_completion_api_async(
                messages,
                temperature=0.7,
                top_p=0.8,
                max_tokens=RESPONSE_MAX_TOKENS,
            )
            final_output_text = strip_thinking_content(latest_text)
            final_model_history = model_history
            if final_output_text:
                if ttft_ms is None:
                    ttft_ms = round((time.perf_counter() - req_start_ts) * 1000, 2)
                final_model_history = model_history + [{"role": "assistant", "content": final_output_text}]
                yield final_model_history, cn_history

            back_translated_text = ""
            if final_output_text:
                try:
                    if FORCE_LOCAL_MODEL_TRANSLATION:
                        providers = ["model"]
                    else:
                        output_priority = ["mymemory", "google", "model"]
                        if TRANSLATION_FALLBACK_TO_MODEL:
                            providers = [p for p in output_priority if p in TRANSLATION_PROVIDER_PRIORITY]
                        else:
                            providers = [p for p in output_priority if p in TRANSLATION_PROVIDER_PRIORITY and p != "model"]
                    back_translated_text = await translate_text_auto(
                        final_output_text,
                        target_lang,
                        "zh-CN",
                        selected_target_language,
                        "Chinese",
                        providers
                    )
                    if not (back_translated_text or "").strip():
                        back_translated_text = f"回译为空，原文：\n\n{final_output_text}"
                except Exception as e:
                    back_translated_text = f"回译错误（{selected_target_language} -> 中文）: {e}\n\n原文：{final_output_text}"
                cn_history = cn_history + [{"role": "assistant", "content": back_translated_text}]

            total_ms = round((time.perf_counter() - req_start_ts) * 1000, 2)
            log_record = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "request_id": request_id,
                "character": selected_char_name,
                "input": user_input,
                "output": final_output_text,
                "ttft_ms": ttft_ms,
                "total_ms": total_ms,
                "input_chars": len(user_input),
                "output_chars": len(final_output_text),
                "translation_target_language": selected_target_language,
                "back_translation": back_translated_text,
                "backend": "api",
                "api_model": API_MODEL_NAME,
            }
            if LOG_FULL_PROMPT:
                log_record["system_prompt"] = system_prompt
                log_record["messages"] = messages
            append_chat_log(log_record)
            yield final_model_history, cn_history
            return

        results_generator = llm_engine.generate(prompt, sampling_params, request_id)

        loop = asyncio.get_running_loop()
        emit_interval_s = 0.08
        min_chars_delta = 12
        last_emit_time = 0.0
        last_emitted_text = ""
        latest_text = ""

        async for request_output in results_generator:
            generated_text = request_output.outputs[0].text
            latest_text = generated_text
            display_text = strip_thinking_content(generated_text)
            now = loop.time()
            finished = bool(getattr(request_output, "finished", False))
            delta_chars = len(generated_text) - len(last_emitted_text)

            if (
                finished
                or delta_chars >= min_chars_delta
                or (now - last_emit_time) >= emit_interval_s
            ):
                if ttft_ms is None and display_text:
                    ttft_ms = round((time.perf_counter() - req_start_ts) * 1000, 2)
                if display_text or finished:
                    current_bot_message = {"role": "assistant", "content": display_text}
                    yield model_history + [current_bot_message], cn_history
                    last_emitted_text = generated_text
                    last_emit_time = now

        final_display_text = strip_thinking_content(latest_text if latest_text else last_emitted_text)
        if final_display_text and latest_text and latest_text != last_emitted_text:
            yield model_history + [{"role": "assistant", "content": final_display_text}], cn_history

        final_output_text = strip_thinking_content(latest_text if latest_text else last_emitted_text)
        final_model_history = model_history
        if final_output_text:
            final_model_history = model_history + [{"role": "assistant", "content": final_output_text}]

        back_translated_text = ""
        if final_output_text:
            try:
                if FORCE_LOCAL_MODEL_TRANSLATION:
                    providers = ["model"]
                else:
                    # 非本地强制模式时，回译优先走专用翻译器，模型做兜底。
                    output_priority = ["mymemory", "google", "model"]
                    if TRANSLATION_FALLBACK_TO_MODEL:
                        providers = [p for p in output_priority if p in TRANSLATION_PROVIDER_PRIORITY]
                    else:
                        providers = [p for p in output_priority if p in TRANSLATION_PROVIDER_PRIORITY and p != "model"]
                back_translated_text = await translate_text_auto(
                    final_output_text,
                    target_lang,
                    "zh-CN",
                    selected_target_language,
                    "Chinese",
                    providers
                )
                if not (back_translated_text or "").strip():
                    back_translated_text = f"回译为空，原文：\n\n{final_output_text}"
            except Exception as e:
                back_translated_text = f"回译错误（{selected_target_language} -> 中文）: {e}\n\n原文：{final_output_text}"
            cn_history = cn_history + [{"role": "assistant", "content": back_translated_text}]

        total_ms = round((time.perf_counter() - req_start_ts) * 1000, 2)
        log_record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "request_id": request_id,
            "character": selected_char_name,
            "input": user_input,
            "output": final_output_text,
            "ttft_ms": ttft_ms,
            "total_ms": total_ms,
            "input_chars": len(user_input),
            "output_chars": len(final_output_text),
            "translation_target_language": selected_target_language,
            "back_translation": back_translated_text,
            "backend": "local",
        }
        if LOG_FULL_PROMPT:
            log_record["system_prompt"] = system_prompt
            log_record["messages"] = messages
            log_record["prompt"] = prompt
            log_record["prompt_chars"] = len(prompt)
        append_chat_log(log_record)
        yield final_model_history, cn_history

    except Exception as e:
        total_ms = round((time.perf_counter() - req_start_ts) * 1000, 2)
        log_record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "request_id": request_id,
            "character": selected_char_name,
            "input": user_input,
            "output": "",
            "ttft_ms": ttft_ms,
            "total_ms": total_ms,
            "error": str(e),
            "translation_target_language": selected_target_language,
        }
        if LOG_FULL_PROMPT:
            log_record["system_prompt"] = system_prompt
            log_record["messages"] = messages
            log_record["prompt"] = prompt
            log_record["prompt_chars"] = len(prompt)
        append_chat_log(log_record)
        yield model_history + [{"role": "assistant", "content": f"Generation Error: {e}"}], cn_history + [{"role": "assistant", "content": f"模型生成错误: {e}"}]

def main():
    model_name = load_model_engine()
    char_names = load_characters()
    default_char = char_names[0] if char_names else None

    with gr.Blocks(title=f"ChatJoy: {model_name}", fill_height=True) as demo:
        gr.Markdown(f"## 🎭 ChatJoy 角色扮演 - {model_name}")

        with gr.Row():
            with gr.Column(scale=2):
                char_dropdown = gr.Dropdown(
                    choices=char_names,
                    value=default_char,
                    label="选择角色 (Character)",
                    interactive=True
                )
            with gr.Column(scale=8):
                current_prompt_display = gr.Textbox(
                    label="当前 System Prompt（可手动修改）",
                    value=CHARACTER_DICT.get(default_char, "") if default_char else "",
                    lines=12,
                    interactive=True
                )
            with gr.Column(scale=1, min_width=180):
                refresh_char_btn = gr.Button("🔄 刷新角色卡", variant="secondary")
                save_prompt_btn = gr.Button("💾 保存当前Prompt", variant="secondary")
                clear_btn = gr.Button("🗑️ 清除左右对话", variant="secondary")

        with gr.Row():
            with gr.Column(scale=3):
                new_char_name_input = gr.Textbox(
                    label="新增角色名称",
                    placeholder="例如：温柔学姐"
                )
            with gr.Column(scale=7):
                create_char_btn = gr.Button("➕ 新建角色卡并切换到该角色", variant="primary")

        save_status = gr.Textbox(
            label="Prompt保存状态",
            value="",
            interactive=False,
            lines=1
        )

        def update_prompt_preview(name):
            return CHARACTER_DICT.get(name, "")
        char_dropdown.change(fn=update_prompt_preview, inputs=char_dropdown, outputs=current_prompt_display)
        refresh_char_btn.click(
            fn=refresh_character_choices,
            inputs=[char_dropdown],
            outputs=[char_dropdown, current_prompt_display]
        )

        save_prompt_btn.click(
            fn=save_character_prompt,
            inputs=[char_dropdown, current_prompt_display],
            outputs=[save_status]
        )

        create_char_btn.click(
            fn=create_character_prompt,
            inputs=[new_char_name_input, current_prompt_display],
            outputs=[char_dropdown, current_prompt_display, save_status, new_char_name_input]
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🌐 翻译联动窗口（中文）")
                target_lang_dropdown = gr.Dropdown(
                    choices=list(TRANSLATION_LANGUAGES.keys()),
                    value=DEFAULT_TRANSLATION_LANGUAGE,
                    label="翻译目标语言（固定输入中文）",
                    interactive=True
                )
                cn_chatbot = gr.Chatbot(height=650, value=[], label="中文对话（回译结果）")
                with gr.Row():
                    cn_msg_input = gr.Textbox(
                        show_label=False,
                        placeholder="在此输入中文，自动翻译后发送到右侧模型...",
                        container=False,
                        scale=8
                    )
                    cn_submit_btn = gr.Button("翻译并发送", variant="primary", scale=1)

            with gr.Column(scale=1):
                gr.Markdown("### 🤖 模型对话窗口（目标语言）")
                chatbot = gr.Chatbot(height=650, value=[], label="模型原始对话")
                with gr.Row():
                    msg_input = gr.Textbox(
                        show_label=False,
                        placeholder="在此直接输入任意语言与模型对话...",
                        container=False,
                        scale=8
                    )
                    submit_btn = gr.Button("发送", variant="primary", scale=1)

        # ==================== 【修复部分 3：事件绑定逻辑】 ====================
        
        # 逻辑链：
        # 1. 执行 add_user_message -> 更新 Chatbot(显示用户话语), 清空 Input
        # 2. 执行 chat_response    -> 读取 Chatbot 历史, 调用模型, 流式更新 Chatbot(显示机器人回复)

        # 回车提交
        msg_input.submit(
            fn=add_user_message,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot]
        ).then(
            fn=chat_response,
            inputs=[chatbot, char_dropdown, current_prompt_display],
            outputs=[chatbot]
        )

        # 按钮提交
        submit_btn.click(
            fn=add_user_message,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot]
        ).then(
            fn=chat_response,
            inputs=[chatbot, char_dropdown, current_prompt_display],
            outputs=[chatbot]
        )

        # 左侧中文输入 -> 翻译到目标语言 -> 发送右侧模型 -> 回译到中文显示在左侧
        cn_msg_input.submit(
            fn=add_cn_user_message,
            inputs=[cn_msg_input, cn_chatbot],
            outputs=[cn_msg_input, cn_chatbot]
        ).then(
            fn=inject_translated_user,
            inputs=[cn_chatbot, chatbot, target_lang_dropdown],
            outputs=[cn_chatbot, chatbot]
        ).then(
            fn=chat_response_with_back_translation,
            inputs=[chatbot, cn_chatbot, char_dropdown, target_lang_dropdown, current_prompt_display],
            outputs=[chatbot, cn_chatbot]
        )

        cn_submit_btn.click(
            fn=add_cn_user_message,
            inputs=[cn_msg_input, cn_chatbot],
            outputs=[cn_msg_input, cn_chatbot]
        ).then(
            fn=inject_translated_user,
            inputs=[cn_chatbot, chatbot, target_lang_dropdown],
            outputs=[cn_chatbot, chatbot]
        ).then(
            fn=chat_response_with_back_translation,
            inputs=[chatbot, cn_chatbot, char_dropdown, target_lang_dropdown, current_prompt_display],
            outputs=[chatbot, cn_chatbot]
        )

        # 清除按钮
        clear_btn.click(lambda: ([], []), None, [cn_chatbot, chatbot])

    server_port = find_available_port(GRADIO_SERVER_PORT, GRADIO_PORT_SEARCH_LIMIT)
    public_host = detect_public_host()
    public_url = build_public_url(public_host, server_port)
    print(f"启动服务... 本地访问 http://127.0.0.1:{server_port}")
    if public_url:
        print(f"公网直连地址（需放通 {server_port} 端口）: {public_url}")

    _, local_url, share_url = demo.queue().launch(
        server_name=GRADIO_SERVER_NAME,
        server_port=server_port,
        share=GRADIO_SHARE,
        share_server_address=GRADIO_SHARE_SERVER_ADDRESS,
        share_server_protocol=GRADIO_SHARE_SERVER_PROTOCOL,
        prevent_thread_lock=True,
    )

    if share_url:
        print(f"Gradio Share 地址: {share_url}")
    elif public_url:
        print(f"Gradio Share 创建失败，已回退到公网直连地址: {public_url}")
    else:
        print(f"Gradio Share 创建失败，请直接使用本地地址: {local_url}")

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("Keyboard interruption in main thread... closing server.")

if __name__ == "__main__":
    main()
