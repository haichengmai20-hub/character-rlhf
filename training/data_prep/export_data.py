import json
import time
import os
from pymilvus import connections, db, Collection

# --- 1. 配置区域 ---
HOST = '<prod-milvus-host>'  # 正式服 IP
PORT = '19530'
DB_NAME = '<prod-db>'
COLLECTION_NAME = 'vectors'

# --- 2. 导出设置 ---
MAX_LIMIT = 10000000         # 总计划导出数量 (1000万)
SPLIT_SIZE = 1000000         # 单个文件包含的数据量 (100万)
OUTPUT_DIR = 'chatjoy_dataset' # 数据保存的根目录文件夹名称
BASE_FILENAME = 'chatjoy_part' # 文件名前缀，生成如 chatjoy_part_1.jsonl
BATCH_SIZE = 2000            # 单次查询批大小

TARGET_FIELDS = ["id", "text", "user_id", "character_id", "timestamp", "message_id"]

def export_data():
    try:
        # --- 初始化文件夹 ---
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
            print(f"已创建数据目录: {OUTPUT_DIR}")
        else:
            print(f"数据目录已存在: {OUTPUT_DIR}")

        # --- 连接数据库 ---
        print(f"正在连接正式服务器: {HOST}...")
        connections.connect(alias="default", host=HOST, port=PORT)
        db.using_database(DB_NAME)
        
        collection = Collection(COLLECTION_NAME)
        collection.load()
        
        total_in_db = collection.num_entities
        print(f"连接成功！正式服共有数据: {total_in_db} 条")
        print(f"计划导出: {MAX_LIMIT} 条 | 分片大小: {SPLIT_SIZE} 条/文件")
        
        # 只导出最后 MAX_LIMIT 条（当库里不足 MAX_LIMIT 时，导出全部）
        target_total = min(MAX_LIMIT, total_in_db)
        skip_remaining = max(total_in_db - target_total, 0)
        print(f"本次实际导出目标: {target_total} 条 | 将跳过前置旧数据: {skip_remaining} 条")

        # --- 变量初始化 ---
        total_fetched = 0       # 总累计条数
        current_file_count = 0  # 当前文件内的条数
        file_index = 1          # 当前文件序号
        initialized_files = set()  # 记录已初始化写入的分片文件，避免重复运行时直接追加旧内容

        print("开始分片流式导出... 模式: 最近数据(尾部窗口分页)")
        start_time = time.time()

        iterator = collection.query_iterator(
            batch_size=BATCH_SIZE,
            limit=-1,
            expr="",
            output_fields=TARGET_FIELDS
        )

        while total_fetched < target_total:
            results = iterator.next()
            if not results:
                print("数据库已无更多数据。")
                break

            # 先跳过旧数据，只保留尾部 target_total 条
            if skip_remaining >= len(results):
                skip_remaining -= len(results)
                continue
            elif skip_remaining > 0:
                results = results[skip_remaining:]
                skip_remaining = 0

            # 最后一批可能超出目标，裁剪到精确条数
            remain_needed = target_total - total_fetched
            if len(results) > remain_needed:
                results = results[:remain_needed]
            
            # --- 构造当前写入的文件路径 ---
            # 格式: chatjoy_dataset/chatjoy_part_1.jsonl
            current_filename = f"{BASE_FILENAME}_{file_index}.jsonl"
            file_path = os.path.join(OUTPUT_DIR, current_filename)

            # --- 写入数据 ---
            file_mode = 'a' if file_path in initialized_files else 'w'
            with open(file_path, file_mode, encoding='utf-8') as f:
                for item in results:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            initialized_files.add(file_path)
            
            # --- 更新计数器 ---
            count = len(results)
            total_fetched += count
            current_file_count += count
            
            print(f"总进度: {total_fetched}/{target_total} | 当前文件({current_filename}): {current_file_count}/{SPLIT_SIZE}")

            # --- 检查是否需要切换到下一个文件 ---
            if current_file_count >= SPLIT_SIZE:
                print(f"--- 文件 {current_filename} 已存满 {current_file_count} 条，切换下一个文件 ---")
                file_index += 1
                current_file_count = 0 # 重置当前文件计数

        iterator.close()

        duration = time.time() - start_time
        print(f"\n================ 导出完成 ================")
        print(f"总耗时: {duration:.2f}秒")
        print(f"总获取数据: {total_fetched} 条")
        generated_files = file_index if current_file_count > 0 else max(file_index - 1, 0)
        print(f"生成文件数: {generated_files} 个")
        print(f"文件位置: ./{OUTPUT_DIR}/")

    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    export_data()
