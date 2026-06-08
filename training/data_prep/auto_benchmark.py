import subprocess
import time
import os
import sys

# ================= 配置区域 =================
BENCHMARK_SCRIPT = "benchmark_serving.py"
RESULT_FILE = "benchmark_report_chatml_uid.txt" # 结果文件名
DATASET_PATH = "/home/ps/hanjiaqi/LlamaFactory/chatjoy/test_dataset.jsonl"
MODEL_NAME = "qwen-ft"
BASE_URL = None  # 例如: "http://<example-host>:8000"
PORT = "8000"
API_PATH = "/v1/completions"
API_KEY = None

# 测试梯度：从 5 到 30，步长 5
# range(5, 31, 5) -> 5, 10, 15, 20, 25, 30
RATES = list(range(5, 31, 5))

# 每个梯度测试时长（秒）
TARGET_DURATION_SEC = 60
# ===========================================

def run_test():
    # 1. 检查文件
    if not os.path.exists(BENCHMARK_SCRIPT):
        print(f"❌ 错误: 找不到 {BENCHMARK_SCRIPT}")
        return

    # 2. 初始化结果文件
    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        f.write(f"🚀 vLLM 详细压力测试报告\n")
        f.write(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"模型: {MODEL_NAME}\n")
        f.write(f"注意: 数据不足时会自动循环填充 (Data Recycling)\n")
        f.write("="*60 + "\n\n")

    print(f"📂 测试开始！结果将实时写入: {RESULT_FILE}")
    print(f"📋 计划测试 QPS 档位: {RATES}")

    # 3. 循环执行测试
    for rate in RATES:
        # 动态计算 prompt 数量
        num_prompts = int(rate * TARGET_DURATION_SEC)
        if num_prompts < 50: num_prompts = 50 # 最小保底

        header = f"\n>>> 正在进行测试: Target QPS={rate} | Total Prompts={num_prompts} (约 {TARGET_DURATION_SEC}s) <<<"
        print(header)
        
        # 写入文件分隔符
        with open(RESULT_FILE, "a", encoding="utf-8") as f:
            f.write("\n\n" + "#"*60 + "\n")
            f.write(header + "\n")
            f.write("#"*60 + "\n")

        # 构造命令
        cmd = [
            "python", BENCHMARK_SCRIPT,
            "--model", MODEL_NAME,
            "--dataset-path", DATASET_PATH,
            "--num-prompts", str(num_prompts),
            "--request-rate", str(rate),
            "--port", PORT,
            "--api-path", API_PATH,
        ]
        if BASE_URL:
            cmd.extend(["--base-url", BASE_URL])
        if API_KEY:
            cmd.extend(["--api-key", API_KEY])

        # 执行并实时记录
        try:
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            # 实时读取输出，同时打印到屏幕和写入文件
            with open(RESULT_FILE, "a", encoding="utf-8") as f:
                for line in process.stdout:
                    # 可以在这里做过滤，如果觉得屏幕太乱，就不print line
                    print(line, end="") 
                    f.write(line)
            
            process.wait()
            
            if process.returncode != 0:
                print(f"❌ QPS {rate} 测试似乎遇到错误，请检查日志。")
        
        except KeyboardInterrupt:
            print("\n⚠️ 用户手动中断测试！")
            sys.exit(0)
        except Exception as e:
            print(f"执行出错: {e}")

        # 冷却时间
        print("❄️  冷却 5 秒等待显存回收...")
        time.sleep(5)

    print(f"\n✅ 所有测试完成！请打开文件查看完整报告: {RESULT_FILE}")

if __name__ == "__main__":
    run_test()
