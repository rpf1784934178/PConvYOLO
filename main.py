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
    BATCH_SIZE = 64

    # 【尝试修改】改为 2
    # 如果再次报错 "error code: <1455>"，请必须改回 0
    WORKERS = 2

    EPOCHS = 100

    print("\n" + "=" * 50)
    print("⚔️ 实验开始：Baseline (官方) vs Ours (PConv)")
    print("=" * 50)

    # ==============================================
    # 1️⃣ 第一轮：训练官方基准模型 (Baseline)
    # ==============================================
    baseline_path = "runs/voc_compare/baseline_yolo11n/weights/best.pt"

    if os.path.exists(baseline_path):
        print(f"\n✅ [1/2] 检测到 Baseline 模型已存在: {baseline_path}")
        print("⏩ 跳过训练，直接加载模型...")
        try:
            model_base = YOLO(baseline_path)
            metrics_base = model_base.val(data=dataset_yaml, split='test', device=device, plots=False)
            map_base = metrics_base.box.map
        except Exception as e:
            print(f"⚠️ 读取旧模型失败: {e}")
            map_base = 0
    else:
        print("\n📦 [1/2] 正在训练官方 YOLO11n (Baseline)...")
        try:
            model_base = YOLO("yolo11n.yaml")
            model_base.load("yolo11n.pt")
            results_base = model_base.train(
                data=dataset_yaml, epochs=EPOCHS, imgsz=640, batch=BATCH_SIZE,
                device=device, workers=WORKERS, project="runs/voc_compare",
                name="baseline_yolo11n", exist_ok=True, amp=True, cache=False
            )
            map_base = results_base.box.map
        except Exception as e:
            print(f"❌ Baseline 训练失败: {e}")
            return

    # ==============================================
    # 2️⃣ 第二轮：训练你的魔改模型 (Ours)
    # ==============================================
    print("\n🚀 [2/2] 正在训练 PConv 魔改模型 (Ours)...")
    try:
        model_our = YOLO("yolo11-pconv.yaml")
        model_our.load("yolo11n.pt")

        results_our = model_our.train(
            data=dataset_yaml,
            epochs=EPOCHS,
            imgsz=640,
            batch=BATCH_SIZE,
            device=device,
            workers=WORKERS,  # 这里尝试用 2
            project="runs/voc_compare",
            name="ours_pconv",
            exist_ok=True,
            amp=True,
            cache=False
        )
        map_our = results_our.box.map
    except Exception as e:
        print(f"❌ PConv 训练失败: {e}")
        # 如果这里报错，你就知道必须把 WORKERS 改回 0 了
        return

    # ==============================================
    # 📊 最终实验报告
    # ==============================================
    print("\n" + "=" * 50)
    print("📑 最终对比报告")
    print("=" * 50)
    print(f"{'模型':<15} | {'mAP50-95':<10} | {'结论'}")
    print("-" * 50)
    print(f"{'Official':<15} | {map_base:.4f}     | 基准线")
    print(f"{'PConv-Ours':<15} | {map_our:.4f}     | {(map_our - map_base):.4f}")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()