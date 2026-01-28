import subprocess
import sys
import time
import argparse
import pynvml
from threading import Thread

def monitor_gpu_memory(pid, interval=0.1):
    """监控指定 PID 进程运行期间的 GPU 显存峰值（MB）"""
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 默认监控 GPU 0
    max_mem = 0

    try:
        while True:
            try:
                # 检查进程是否还在运行
                proc = subprocess.run(['ps', '-p', str(pid)], 
                                    stdout=subprocess.DEVNULL, 
                                    stderr=subprocess.DEVNULL)
                if proc.returncode != 0:
                    break  # 进程已退出
            except Exception:
                break

            # 获取当前 GPU 显存使用量
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            current_mem_mb = mem_info.used // (1024 * 1024)
            if current_mem_mb > max_mem:
                max_mem = current_mem_mb

            time.sleep(interval)

    except KeyboardInterrupt:
        pass
    finally:
        pynvml.nvmlShutdown()

    return max_mem

def main():
    parser = argparse.ArgumentParser(description="Monitor peak GPU memory of a Python script")
    parser.add_argument("script", help="Path to the Python script to run")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments to pass to the script")
    args = parser.parse_args()

    # 启动目标脚本作为子进程
    cmd = [sys.executable, args.script] + args.args
    print(f"[Monitor] Running: {' '.join(cmd)}")
    
    process = subprocess.Popen(cmd)
    pid = process.pid

    # 在后台线程中监控显存
    max_mem = [0]
    def run_monitor():
        max_mem[0] = monitor_gpu_memory(pid)

    monitor_thread = Thread(target=run_monitor)
    monitor_thread.start()

    # 等待子进程结束
    process.wait()
    monitor_thread.join()

    print("\n" + "="*50)
    print(f"✅ Script finished with exit code: {process.returncode}")
    print(f"📊 Peak GPU Memory Usage: {max_mem[0]} MB ({max_mem[0]/1024:.2f} GB)")
    print("="*50)

if __name__ == "__main__":
    main()