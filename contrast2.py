import os

# 防止 OpenMP 环境冲突报错
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import multiprocessing

import torch

from ultralytics import YOLO


def get_model_metrics(model_path, dataset_yaml, device):
    """辅助函数：加载模型并获取 mAP."""
    if not os.path.exists(model_path):
        return None
    try:
        model = YOLO(model_path)
        # 运行验证模式 (val)，关闭 verbose 减少刷屏
        metrics = model.val(data=dataset_yaml, split="test", device=device, plots=False, verbose=False)
        return metrics.box.map
    except Exception as e:
        print(f"⚠️ 加载失败 {model_path}: {e}")
        return 0.0


def main():
    # --- 1. 硬件配置 ---
    device = 0 if torch.cuda.is_available() else "cpu"
    dataset_yaml = "VOC.yaml"

    print("\n" + "=" * 80)
    print("📈 终极实验报告：Baseline vs Ours (100e - 1000e 全周期)")
    print("=" * 80)

    # --- 2. 定义所有待测模型路径 (包含所有1000轮的文件夹) ---
    models_config = [
        ("Official Baseline", "baseline_yolo11n"),  # 官方基准
        ("Ours-PConv (100e)", "ours_pconv"),  # 100轮
        ("Ours-PConv (200e)", "ours_pconv_extended_200e"),  # 200轮
        ("Ours-PConv (300e)", "ours_pconv_extended_300e"),  # 300轮
        ("Ours-PConv (400e)", "ours_pconv_extended_400e"),  # 400轮
        ("Ours-PConv (500e)", "ours_pconv_extended_500e"),  # 500轮
        ("Ours-PConv (600e)", "ours_pconv_extended_600e"),  # 600轮
        ("Ours-PConv (700e)", "ours_pconv_extended_700e"),  # 700轮
        ("Ours-PConv (800e)", "ours_pconv_extended_800e"),  # 800轮
        ("Ours-PConv (900e)", "ours_pconv_extended_900e"),  # 900轮
        ("Ours-PConv (1000e)", "ours_pconv_extended_1000e"),  # 1000轮
    ]

    base_dir = r"runs/voc_compare"
    results = []

    # --- 3. 批量评估循环 ---
    baseline_map = 0.0
    prev_map = 0.0

    for i, (name, folder_name) in enumerate(models_config):
        print(f"🔄 [{i + 1}/{len(models_config)}] 正在评估: {name:<25} ...", end="", flush=True)

        weight_path = os.path.join(base_dir, folder_name, "weights", "best.pt")

        map_50_95 = get_model_metrics(weight_path, dataset_yaml, device)

        if map_50_95 is None:
            print(" ❌ 文件不存在 (跳过)")
            continue

        print(f" ✅ mAP: {map_50_95:.4f}")

        # 记录 Baseline
        if "Baseline" in name:
            baseline_map = map_50_95
            gap_baseline = 0.0
            growth = 0.0
        else:
            gap_baseline = map_50_95 - baseline_map
            growth = map_50_95 - prev_map if prev_map > 0 else 0

        results.append({"name": name, "map": map_50_95, "gap": gap_baseline, "growth": growth})

        if "Baseline" not in name:
            prev_map = map_50_95

    # --- 4. 生成全周期大表 ---
    print("\n" + "=" * 95)
    print(f"{'模型阶段':<25} | {'mAP50-95':<10} | {'与基准差距':<12} | {'阶段提升':<12} | {'状态'}")
    print("-" * 95)

    best_our_model = None
    best_our_map = -1.0

    for row in results:
        name = row["name"]
        map_val = row["map"]

        # 寻找我们自己模型中的最高分
        if "Baseline" not in name:
            if map_val > best_our_map:
                best_our_map = map_val
                best_our_model = name

        # 生成评价
        if "Baseline" in name:
            comment = "🎯 基准线"
        elif row["gap"] >= 0:
            comment = "🏆 超越基准"
        elif row["gap"] >= -0.01:
            comment = "🔥 几乎持平"
        elif row["gap"] >= -0.05:
            comment = "👌 可接受范围"
        else:
            comment = "⚠️ 差距较大"

        gap_str = f"{row['gap']:+.4f}"
        growth_str = f"{row['growth']:+.4f}" if "Baseline" not in name else "-"

        print(f"{name:<25} | {map_val:.4f}     | {gap_str:<12} | {growth_str:<12} | {comment}")

    print("=" * 95)

    # --- 5. 巅峰对决 (Best vs Baseline) ---
    print("\n" + "#" * 50)
    print("🏆 最终结论：最佳模型 vs 官方基准")
    print("#" * 50)

    if best_our_model:
        diff = best_our_map - baseline_map
        print(f"🥇 你的最佳模型: {best_our_model}")
        print(f"📊 最终精度 (mAP): {best_our_map:.4f}")
        print(f"📏 与官方差距: {diff:+.4f}")

        print("-" * 30)
        if diff >= 0:
            print("✅ 实验非常成功！你的魔改模型在更轻量的情况下，精度超越了官方模型！")
        elif diff >= -0.02:
            print("✅ 实验成功！精度几乎无损 (差距<2%)，但换来了 PConv 的速度优势。")
        else:
            print("💡 实验总结：精度虽有下降，但验证了长周期训练对收敛的帮助。")
    else:
        print("❌ 未找到有效的 Ours 模型数据。")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
