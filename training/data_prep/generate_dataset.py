import glob
import json
import os
from collections import defaultdict

# ================= ⚙️ 核心配置区域 ⚙️ =================

# 1. 路径配置
INPUT_LOGS_GLOB = "/home/ps/hanjiaqi/LlamaFactory/chatjoy/chatjoy_dataset/chatjoy_part_*.jsonl"
CHAR_CONFIG_DIR = "/home/ps/hanjiaqi/LlamaFactory/chatjoy/characters"
OUTPUT_DATASET_PATH = "chatjoy_hq_full.json"

# 2. 采样配置
MAX_HISTORY_TURNS = 3        # 【需求点1】History 只保留最近 3 轮

# 3. 质量筛选配置 (字数限制)
# Instruction (用户输入)
MIN_INSTRUCT_LEN = 5         # 太短的不要 (比如 "嗨")
MAX_INSTRUCT_LEN = 300       # 太长的可能是脏数据
# Output (AI回复)
MIN_OUTPUT_LEN = 10          # 【需求点3】太短的不要 (比如 "嗯嗯")
MAX_OUTPUT_LEN = 512         # 保持 30B 模型的回复精简，防止显存溢出

# =======================================================

def get_role_from_message_id(message_id):
    if "_human_" in message_id: return "user"
    elif "_ai_" in message_id: return "assistant"
    return "unknown"

def load_character_config(char_id):
    """
    【需求点2】严格加载角色配置，如果文件不存在返回 None
    """
    file_path = os.path.join(CHAR_CONFIG_DIR, f"{char_id}.json")
    if not os.path.exists(file_path):
        return None # 文件不存在，返回 None 标记废弃
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            char_data = json.load(f)
            # 优先用 description，没有则用 description2
            system = char_data.get("description", "")
            if not system:
                system = char_data.get("description2", "")
            return system
    except Exception:
        return None # 读取错误也算不存在

def is_high_quality(instruction, output):
    """
    【需求点3】字数质量筛选
    """
    if len(instruction) < MIN_INSTRUCT_LEN or len(instruction) > MAX_INSTRUCT_LEN:
        return False
    if len(output) < MIN_OUTPUT_LEN or len(output) > MAX_OUTPUT_LEN:
        return False
    # 额外规则：AI回复必须包含动作描写 (带星号) -> 恋爱/RP数据质量的关键
    if "*" not in output:
        return False
    return True

def process_dataset():
    # --- 第一步：读取原始数据并按 Session 分组 ---
    sessions = defaultdict(list)
    print("1. 正在读取原始日志分片...")

    input_paths = sorted(glob.glob(INPUT_LOGS_GLOB))
    if not input_paths:
        print(f"错误：没有找到输入文件，匹配规则: {INPUT_LOGS_GLOB}")
        return

    print(f"   检测到 {len(input_paths)} 个分片文件。")
    total_lines = 0
    bad_json_lines = 0
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
                    # Session Key: 用户ID + 角色ID
                    sessions[(msg["user_id"], msg["character_id"])].append(msg)
                except json.JSONDecodeError:
                    bad_json_lines += 1
                    continue

    # --- 第二步：重建对话并进行质量筛选 ---
    final_dataset = []
    
    # 缓存 System Prompt，避免重复 IO
    char_system_cache = {}
    valid_char_ids = set()
    missing_char_ids = set()

    print(f"2. 开始处理 {len(sessions)} 个会话流...")

    for (uid, cid), msgs in sessions.items():
        # 2.1 检查角色是否存在 (缓存优化)
        if cid in missing_char_ids: continue # 之前确认过不存在，直接跳过
        
        if cid not in char_system_cache:
            sys_prompt = load_character_config(cid)
            if sys_prompt is None:
                missing_char_ids.add(cid)
                continue # 【需求点2】角色文件不存在，整条线丢弃
            char_system_cache[cid] = sys_prompt
            valid_char_ids.add(cid)
        
        system_prompt = char_system_cache[cid]
        
        # 2.2 按时间排序
        msgs.sort(key=lambda x: int(x['timestamp']))

        # 2.3 重建上下文
        history = [] 
        current_user_buffer = []

        for msg in msgs:
            role = get_role_from_message_id(msg['message_id'])
            text = msg['text']

            if role == "user":
                current_user_buffer.append(text)
            
            elif role == "assistant":
                if not current_user_buffer: continue # 只有AI说话，跳过
                
                instruction = "\n".join(current_user_buffer)
                output = text

                # 2.4 【需求点3】字数质量筛选
                # 即使当前这条不符合质量被筛掉，它依然要加入 history 供后续对话使用
                # 但我们只把高质量的加入最终训练集
                if is_high_quality(instruction, output):
                    
                    # 【需求点1】History 截断：只取最近 MAX_HISTORY_TURNS 轮
                    truncated_history = history[-MAX_HISTORY_TURNS:]
                    
                    data_item = {
                        "instruction": instruction,
                        "input": "",
                        "output": output,
                        "system": system_prompt,
                        "history": truncated_history 
                    }
                    final_dataset.append(data_item)

                # 更新历史 (History 记录完整的上下文流)
                history.append([instruction, output])
                current_user_buffer = []

    # --- 第三步：统计并保存 ---
    print(f"3. 筛选完成。有效角色数: {len(valid_char_ids)}。被丢弃的角色ID数: {len(missing_char_ids)}")
    print(f"   原始总行数: {total_lines}，JSON 解析失败: {bad_json_lines}")
    print(f"   最终高质量训练样本数: {len(final_dataset)}")

    # --- 第四步：保存 ---
    print(f"4. 开始保存数据集，大小: {len(final_dataset)} 条。")
    
    with open(OUTPUT_DATASET_PATH, 'w', encoding='utf-8') as f:
        json.dump(final_dataset, f, ensure_ascii=False, indent=2)
    
    print(f"   文件已保存至: {OUTPUT_DATASET_PATH}")
    print("   Done.")

if __name__ == "__main__":
    process_dataset()
