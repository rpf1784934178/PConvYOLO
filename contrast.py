import torch
from ultralytics import YOLO
import multiprocessing
import os

def main():
    # --- 1. 硬件配置 ---
    if torch.cuda.is_available():
        device = 0
        print(f"🔥 显卡就绪: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'

    dataset_yaml = "VOC.yaml"

    # ⚠️ 不需要 Batch Size 和 Workers 了，因为我们只做验证(Validation)，不训练

    print("\n" + "=" * 60)
    print("📊 最终实验对比报告生成器 (Official-100e vs Ours-400e)")
    print("=" * 60)

    # ==============================================
    # 1️⃣ 获取 Baseline 的成绩
    # ==============================================
    baseline_path = r"runs/voc_compare/baseline_yolo11n/weights/best.pt"

    if os.path.exists(baseline_path):
        print(f"\n✅ [1/2] 正在加载 Baseline 模型 (100 Epochs): {baseline_path}")
        try:
            model_base = YOLO(baseline_path)
            # 运行验证模式 (val) 获取指标
            metrics_base = model_base.val(data=dataset_yaml, split='test', device=device, plots=False)
            map_base = metrics_base.box.map
            print(f"   ---> Baseline mAP50-95: {map_base:.4f}")
        except Exception as e:
            print(f"❌ 读取 Baseline 失败: {e}")
            map_base = 0
    else:
        print(f"❌ 找不到 Baseline 模型文件: {baseline_path}")
        map_base = 0

    # ==============================================
    # 2️⃣ 获取你刚刚跑完的 400轮 PConv 成绩
    # ==============================================
    # ⚠️ 关键点：这里指向你刚刚生成的 extended 400e 文件夹
    pconv_path = r"runs/voc_compare/ours_pconv_extended_500e/weights/best.pt"

    if os.path.exists(pconv_path):
        # 【修正】文字标签改为 400 Epochs
        print(f"\n✅ [2/2] 正在加载 Ours (500 Epochs) 模型: {pconv_path}")
        try:
            model_our = YOLO(pconv_path)
            # 运行验证模式 (val)
            metrics_our = model_our.val(data=dataset_yaml, split='test', device=device, plots=False)
            map_our = metrics_our.box.map
            print(f"   ---> Ours mAP50-95: {map_our:.4f}")
        except Exception as e:
            print(f"❌ 读取 Ours 模型失败: {e}")
            map_our = 0
    else:
        print(f"❌ 找不到 PConv (400e) 模型文件，请检查训练是否已完成！")
        print(f"   路径: {pconv_path}")
        map_our = 0

    # ==============================================
    # 📊 最终实验报告
    # ==============================================
    print("\n" + "=" * 60)
    # 【修正】报告标题改为 400e
    print("📑 最终对比报告 (Baseline-100e vs Ours-400e)")
    print("=" * 60)
    print(f"{'模型':<20} | {'mAP50-95':<10} | {'结论'}")
    print("-" * 60)

    # 防止 map 为 0 导致报错
    diff = map_our - map_base

    print(f"{'Official (100e)':<20} | {map_base:.4f}     | 基准线")
    # 【修正】表格行名改为 400e
    print(f"{'PConv-Ours (400e)':<20} | {map_our:.4f}     | {diff:+.4f}")

    print("-" * 60)
    if diff >= -0.01:
        print("\n🎉 恭喜！通过增加训练轮数，轻量化模型成功追平或超越了官方基准！")
    elif diff >= -0.03:
        print("\n👌 结果不错！差距在 0.03 以内，考虑到计算量下降，这是一个可用的轻量化模型。")
    else:
        print("\n💡 分析：精度仍有差距。可能需要检查 PConv 模块是否替换得太多，影响了特征提取能力。")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()