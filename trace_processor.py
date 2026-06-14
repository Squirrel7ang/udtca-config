#!/usr/bin/env python3
"""
trace_processor - 处理 tfevent 文件
步骤1: 提取每个 exp 的 tb_scalars 中字典序最大的文件
步骤2: 分析 tfevent 文件，计算平均迭代时间（排除第一轮）
"""

import os
import subprocess
import sys

# 配置
SOURCE_DIR = "/data1/tangruijing/trace"
OUTPUT_DIR = "/data1/tangruijing/trace_process"


def find_exp_dirs():
    """查找所有 exp 目录"""
    exp_dirs = []
    for item in os.listdir(SOURCE_DIR):
        item_path = os.path.join(SOURCE_DIR, item)
        if os.path.isdir(item_path) and item.startswith("exp"):
            exp_dirs.append(item)
    return sorted(exp_dirs)


def get_largest_file(tb_scalars_dir):
    """获取 tb_scalars 目录中字典序最大的文件"""
    if not os.path.exists(tb_scalars_dir):
        return None
    
    files = []
    for f in os.listdir(tb_scalars_dir):
        if f.startswith("events.out.tfevents"):
            files.append(f)
    
    if not files:
        return None
    
    return max(files)


def save_processed():
    """提取每个 exp 的字典序最大 tfevent 文件并保存"""
    print(f"=== 步骤1: 提取文件 ===")
    print(f"源目录: {SOURCE_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 清空已有内容
    for item in os.listdir(OUTPUT_DIR):
        item_path = os.path.join(OUTPUT_DIR, item)
        if os.path.isdir(item_path):
            subprocess.run(["rm", "-rf", item_path])
        else:
            os.remove(item_path)
    
    # 查找所有 exp 目录
    exp_dirs = find_exp_dirs()
    print(f"找到 {len(exp_dirs)} 个 exp 目录")
    
    # 处理每个 exp
    for exp in exp_dirs:
        exp_path = os.path.join(SOURCE_DIR, exp)
        
        # 查找 tb_scalars 目录
        tb_scalars_path = None
        for root, dirs, files in os.walk(exp_path):
            if "tb_scalars" in dirs:
                tb_scalars_path = os.path.join(root, "tb_scalars")
                break
        
        if tb_scalars_path is None:
            print(f"警告: {exp} 中未找到 tb_scalars 目录")
            continue
        
        # 获取字典序最大的文件
        largest_file = get_largest_file(tb_scalars_path)
        if largest_file is None:
            print(f"警告: {tb_scalars_path} 中未找到 tfevent 文件")
            continue
        
        # 创建输出子目录
        output_subdir = os.path.join(OUTPUT_DIR, exp)
        os.makedirs(output_subdir, exist_ok=True)
        
        # 复制文件
        src_file = os.path.join(tb_scalars_path, largest_file)
        dst_file = os.path.join(output_subdir, largest_file)
        subprocess.run(["cp", src_file, dst_file])
        
        print(f"{exp} -> {largest_file}")
    
    print(f"\n保存完成！结果保存在: {OUTPUT_DIR}")


def analyze_tfevents():
    """分析 tfevent 文件，计算平均迭代时间（排除第一轮）"""
    print("\n=== 步骤2: 分析时间 ===")
    
    if not os.path.exists(OUTPUT_DIR):
        print(f"错误: 输出目录不存在，请先运行 save 命令")
        return
    
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("错误: 未安装 tensorboard，请先安装: pip install tensorboard")
        return
    
    exp_dirs = []
    for item in os.listdir(OUTPUT_DIR):
        item_path = os.path.join(OUTPUT_DIR, item)
        if os.path.isdir(item_path) and item.startswith("exp"):
            exp_dirs.append(item)
    
    if not exp_dirs:
        print("未找到处理过的 exp 目录")
        return
    
    results = []
    for exp in sorted(exp_dirs):
        exp_path = os.path.join(OUTPUT_DIR, exp)
        
        tfevent_file = None
        for f in os.listdir(exp_path):
            if f.startswith("events.out.tfevents"):
                tfevent_file = os.path.join(exp_path, f)
                break
        
        if tfevent_file is None:
            print(f"{exp}: 未找到 tfevent 文件")
            continue
        
        try:
            event_acc = EventAccumulator(tfevent_file)
            event_acc.Reload()
            
            tags = event_acc.Tags().get('scalars', [])
            if not tags:
                print(f"{exp}: 未找到标量数据")
                continue
            
            # 使用字典去重，每个 step 只保留一个时间戳
            step_times = {}
            for tag in tags:
                events = event_acc.Scalars(tag)
                for event in events:
                    if event.step not in step_times:
                        step_times[event.step] = event.wall_time
            
            # 转换为排序后的列表
            times = sorted(step_times.items(), key=lambda x: x[0])
            print(f"{exp}: 共 {len(times)} 个 step")
            
            if len(times) < 2:
                print(f"{exp}: 数据点不足")
                continue
            
            # 计算每轮时间
            durations = []
            for i in range(1, len(times)):
                duration = times[i][1] - times[i-1][1]
                durations.append(duration)
            
            if len(durations) < 1:
                print(f"{exp}: 无法计算时间")
                continue
            
            # 排除第一轮后计算平均吞吐量
            if len(durations) > 1:
                # 计算每一轮的吞吐量，然后取算数平均
                throughputs = [16384 / d for d in durations[1:]]
                avg_throughput = sum(throughputs) / len(throughputs)
                avg_time = sum(durations[1:]) / len(durations[1:])
                count = len(durations[1:])
                total_rounds = len(durations)
            else:
                avg_throughput = 16384 / durations[0]
                avg_time = durations[0]
                count = 1
                total_rounds = 1
            
            results.append((exp, avg_time, avg_throughput, count, total_rounds))
            print(f"{exp}: 总轮次={total_rounds}, 平均时间={avg_time:.3f}s, 平均吞吐量={avg_throughput:.1f} tokens/s (排除第一轮后，共 {count} 轮)")
        
        except Exception as e:
            print(f"{exp}: 分析失败 - {e}")
    
    if results:
        print("\n=== 汇总 ===")
        for exp, avg_time, avg_throughput, count, total_rounds in sorted(results):
            print(f"{exp}: 总轮次={total_rounds}, 平均时间={avg_time:.3f}s, 平均吞吐量={avg_throughput:.1f} tokens/s")


def show_usage():
    """显示用法"""
    print("""
用法:
    python trace_processor.py [命令]

命令:
    save    - 提取每个 exp 的字典序最大 tfevent 文件
    analyze - 分析时间，计算平均迭代时间（排除第一轮）
    all     - 执行 save + analyze

示例:
    python trace_processor.py all
""")


def main():
    if len(sys.argv) < 2:
        show_usage()
        return
    
    command = sys.argv[1]
    
    if command == "save":
        save_processed()
    elif command == "analyze":
        analyze_tfevents()
    elif command == "all":
        save_processed()
        analyze_tfevents()
    else:
        print(f"未知命令: {command}")
        show_usage()


if __name__ == "__main__":
    main()
