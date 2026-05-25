import multiprocessing
import os

import torch

from ultralytics import YOLO


def main():
    # =========================================================
    # ⚙️ 核心配置区域
    # =========================================================

    # 1. 【关键修改】这里必须指向 "extended_200e" 文件夹里的 last.pt
    # 这样才是基于 200 轮的智商继续往下学
    last_weight_path = (
        r"D:\PythonSoftware\code\YoloSsd\ultralytics\runs\voc_compare\ours_pconv_extended_500e\weights\last.pt"
    )

    # 2. 【关键修改】改个新名字，代表冲击 300 轮
    new_project_name = "ours_pconv_extended_600e"

    # 3. 继续训练的轮数 (200 -> 300)
    ADDITIONAL_EPOCHS = 100

    # 4. 显卡与内存设置
    device = 0 if torch.cuda.is_available() else "cpu"
    # 既然你上次用 2 跑通了，就保持 2。如果报错再改回 0。
    WORKERS = 2
    BATCH_SIZE = 64
    dataset_yaml = "VOC.yaml"
    # =========================================================

    print("\n" + "=" * 60)
    print("🔄 正在加载第 300 轮的最终模型 (last.pt)...")
    print(f"📂 读取路径: {last_weight_path}")
    print("=" * 60)

    # 检查文件是否存在
    if not os.path.exists(last_weight_path):
        print("❌ 错误：找不到文件！\n请去文件夹确认 'ours_pconv_extended_300e' 是否存在，或者文件名是否正确。")
        return

    try:
        # 1️⃣ 加载 200轮 的权重
        model = YOLO(last_weight_path)
        print("✅ 权重加载成功！准备开始第三阶段训练 (400 -> 500 Epochs)...")

        # 2️⃣ 开始训练
        model.train(
            data=dataset_yaml,
            epochs=ADDITIONAL_EPOCHS,
            imgsz=640,
            batch=BATCH_SIZE,
            device=device,
            workers=WORKERS,
            project="runs/voc_compare",
            name=new_project_name,
            exist_ok=True,
            amp=True,
            cache=False,
        )

        print("\n" + "=" * 60)
        print("🎉 500轮训练完成！")
        print(f"📂 结果保存在: runs/voc_compare/{new_project_name}")
        print("=" * 60)

    except Exception as e:
        print(f"❌ 训练启动失败: {e}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
