import json
import random
import re

# ================= 配置区域 =================
INPUT_FILE = "/home/ps/hanjiaqi/LlamaFactory/chatjoy/chatjoy_alpaca_data.json"      # 你的原始数据文件名
OUTPUT_FILE = "chatjoy_filtered.json" # 输出的文件名
TARGET_COUNT = 5000                   # 你最终想要的数据量 (建议 3000-5000)
MIN_OUTPUT_LEN = 10                   # 回复的最短长度（过滤掉"嗯"、"哦"）
MAX_OUTPUT_LEN = 1024                 # 回复的最长长度（防止显存爆炸）
# ===========================================

def is_high_quality(item):
    """
    判断单条数据是否高质量
    """
    instruction = item.get("instruction", "")
    output = item.get("output", "")
    
    # 1. 基础非空检查
    if not instruction or not output:
        return False
        
    # 2. 剔除 AI 拒答或味儿太重的回复
    # 如果回复里包含这些词，直接不要
    ai_keywords = [
        "作为一名AI", "人工智能", "语言模型", "我无法", "我不能", 
        "As an AI", "language model", "I cannot"
    ]
    for kw in ai_keywords:
        if kw in output:
            return False

    # 3. 格式强制检查 (针对你的数据集特性)
    # 必须包含 #emotion:xxx# 标签
    if not re.search(r"#emotion:.*?#", output):
        return False
    
    # 必须包含动作描写 (即必须有星号 *)
    # 角色扮演的精髓在于动作描写，没有动作的对话在此时价值较低
    if "*" not in output:
        return False

    # 4. 长度过滤
    # 剔除太短的回复（例如只有表情或几个字），保留信息量大的
    if len(output) < MIN_OUTPUT_LEN:
        return False
    
    # 剔除过长的回复（显存杀手）
    if len(output) > MAX_OUTPUT_LEN:
        return False

    return True

def main():
    print(f"正在读取文件: {INPUT_FILE} ...")
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误：找不到文件 {INPUT_FILE}，请确认文件名正确。")
        return

    print(f"原始数据总量: {len(data)} 条")

    # 第一步：清洗
    clean_data = []
    seen_conversations = set() # 用于去重

    for item in data:
        # 去重逻辑：如果输入和输出完全一样，就跳过
        unique_key = (item.get("instruction"), item.get("output"))
        if unique_key in seen_conversations:
            continue
        
        if is_high_quality(item):
            clean_data.append(item)
            seen_conversations.add(unique_key)

    print(f"清洗后合格数据: {len(clean_data)} 条 (剔除了格式错误、重复、AI味、过短数据)")

    # 第二步：采样 (如果数据还是太多)
    if len(clean_data) > TARGET_COUNT:
        print(f"数据仍多于 {TARGET_COUNT} 条，正在进行随机采样...")
        final_data = random.sample(clean_data, TARGET_COUNT)
    else:
        final_data = clean_data

    # 第三步：保存
    print(f"正在保存最终数据到: {OUTPUT_FILE} ...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)

    print(f"完成！最终数据集包含 {len(final_data)} 条数据。")
    print("建议：请使用新生成的 json 文件进行微调。")

if __name__ == "__main__":
    main()