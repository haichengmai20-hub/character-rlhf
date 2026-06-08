import glob
import json
import os
import re
from collections import defaultdict

# ================= ⚙️ 核心配置区域 ⚙️ =================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. 路径配置
INPUT_LOGS_GLOB = os.path.join(BASE_DIR, "chatjoy_dataset", "chatjoy_part_*.jsonl")
CHAR_CONFIG_DIR = os.path.join(BASE_DIR, "TOP20CHARACTERS_systemprompt")
OUTPUT_DATASET_PATH = os.path.join(BASE_DIR, "chatjoy_top20_hq_full.json")

# 2. 采样配置
MAX_HISTORY_TURNS = 0

# 3. 质量筛选配置
MIN_INSTRUCT_LEN = 5
MAX_INSTRUCT_LEN = 300
MIN_OUTPUT_LEN = 10
MAX_OUTPUT_LEN = 512

# 文件名格式: "xx-xxx人物描述"
CHAR_FILENAME_PATTERN = re.compile(r"^\d+-(\d+)")

# =======================================================


def get_role_from_message_id(message_id):
    if "_human_" in message_id:
        return "user"
    if "_ai_" in message_id:
        return "assistant"
    return "unknown"


def normalize_character_id(char_id):
    return str(char_id).strip()


def extract_char_id_from_filename(filename):
    stem, _ = os.path.splitext(filename)
    match = CHAR_FILENAME_PATTERN.match(stem)
    if not match:
        return None
    return match.group(1)


def extract_system_prompt(file_content):
    """
    按新需求优先把整文件内容当成 system prompt。
    同时兼容当前目录里仍然是 {"description": "..."} 的历史格式。
    """
    raw_text = file_content.strip()
    if not raw_text:
        return None

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        return raw_text

    if isinstance(parsed, dict):
        for key in ("system", "system_prompt", "description", "description2"):
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    return raw_text


def load_top20_character_prompts():
    char_prompt_map = {}

    pattern = os.path.join(CHAR_CONFIG_DIR, "*")
    for file_path in sorted(glob.glob(pattern)):
        if not os.path.isfile(file_path):
            continue

        filename = os.path.basename(file_path)
        char_id = extract_char_id_from_filename(filename)
        if char_id is None:
            print(f"   跳过无法解析角色ID的文件: {filename}")
            continue

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                system_prompt = extract_system_prompt(f.read())
        except Exception as exc:
            print(f"   读取角色文件失败: {filename} | 错误: {exc}")
            continue

        if not system_prompt:
            print(f"   跳过空 system prompt 文件: {filename}")
            continue

        char_prompt_map[char_id] = system_prompt

    return char_prompt_map


def is_high_quality(instruction, output):
    if len(instruction) < MIN_INSTRUCT_LEN or len(instruction) > MAX_INSTRUCT_LEN:
        return False
    if len(output) < MIN_OUTPUT_LEN or len(output) > MAX_OUTPUT_LEN:
        return False
    if "*" not in output:
        return False
    return True


def process_dataset():
    print("0. 正在加载 TOP20 角色 system prompt ...")
    char_prompt_map = load_top20_character_prompts()
    allowed_char_ids = set(char_prompt_map.keys())

    if not allowed_char_ids:
        print(f"错误：没有从目录中加载到任何角色配置: {CHAR_CONFIG_DIR}")
        return

    print(f"   成功加载 {len(allowed_char_ids)} 个角色配置。")

    sessions = defaultdict(list)
    print("1. 正在读取原始日志分片...")

    input_paths = sorted(glob.glob(INPUT_LOGS_GLOB))
    if not input_paths:
        print(f"错误：没有找到输入文件，匹配规则: {INPUT_LOGS_GLOB}")
        return

    print(f"   检测到 {len(input_paths)} 个分片文件。")
    total_lines = 0
    bad_json_lines = 0
    skipped_non_top20_lines = 0

    for path in input_paths:
        print(f"   读取: {path}")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                total_lines += 1
                line = line.strip()
                if not line:
                    continue

                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    bad_json_lines += 1
                    continue

                cid = normalize_character_id(msg.get("character_id", ""))
                if cid not in allowed_char_ids:
                    skipped_non_top20_lines += 1
                    continue

                sessions[(msg["user_id"], cid)].append(msg)

    final_dataset = []

    print(f"2. 开始处理 {len(sessions)} 个 TOP20 角色会话流...")

    for (uid, cid), msgs in sessions.items():
        _ = uid
        system_prompt = char_prompt_map[cid]

        msgs.sort(key=lambda x: int(x["timestamp"]))

        history = []
        current_user_buffer = []

        for msg in msgs:
            role = get_role_from_message_id(msg["message_id"])
            text = msg["text"]

            if role == "user":
                current_user_buffer.append(text)
                continue

            if role != "assistant" or not current_user_buffer:
                continue

            instruction = "\n".join(current_user_buffer)
            output = text

            if is_high_quality(instruction, output):
                final_dataset.append(
                    {
                        "instruction": instruction,
                        "input": "",
                        "output": output,
                        "system": system_prompt,
                    }
                )

            history.append([instruction, output])
            current_user_buffer = []

    print(f"3. 筛选完成。TOP20有效角色数: {len(allowed_char_ids)}")
    print(f"   原始总行数: {total_lines}，JSON 解析失败: {bad_json_lines}")
    print(f"   非 TOP20 角色数据行数: {skipped_non_top20_lines}")
    print(f"   最终高质量训练样本数: {len(final_dataset)}")

    print(f"4. 开始保存数据集，大小: {len(final_dataset)} 条。")
    with open(OUTPUT_DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(final_dataset, f, ensure_ascii=False, indent=2)

    print(f"   文件已保存至: {OUTPUT_DATASET_PATH}")
    print("   Done.")


if __name__ == "__main__":
    process_dataset()
