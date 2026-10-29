import os

# 【✅ 修复核心】必须放在 import torch 之前！
# 这行代码告诉系统：允许 OpenMP 库共存，忽略 Error #15
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import multiprocessing

import torch

from ultralytics import YOLO


def main():
    # =========================================================
    # ⚙️ 核心配置区域
    # =========================================================

    # 1. 【起点】从 500轮 的 last.pt 开始
    # ⚠️ 请确认这个路径下的文件存在
    current_weight_path = (
        r"D:\PythonSoftware\code\YoloSsd\ultralytics\runs\voc_compare\ours_pconv_extended_500e\weights\last.pt"
    )

    # 2. 训练计划：我们要跑 5 次，每次 100 轮 (总共冲击 1000 轮)
    START_EPOCH = 500  # 当前进度
    TOTAL_TARGET = 1000  # 最终目标
    STEP_EPOCHS = 100  # 每多少轮保存一次

    # 3. 显卡与内存设置
    device = 0 if torch.cuda.is_available() else "cpu"

    # 【⚠️ 保持为 0】
    # 既然已经出现了环境库冲突，多进程更不稳定，务必保持 0
    WORKERS = 0
    BATCH_SIZE = 64
    dataset_yaml = "VOC.yaml"
    # =========================================================

    print("\n" + "=" * 60)
    print(f"🔄 启动接力训练计划：从 {START_EPOCH} 轮 -> {TOTAL_TARGET} 轮")
    print(f"📂 起始权重: {current_weight_path}")
    print("   (已添加 KMP_DUPLICATE_LIB_OK=TRUE 修复环境冲突)")
    print("=" * 60)

    if not os.path.exists(current_weight_path):
        print(f"❌ 错误：找不到文件！\n请确认路径是否正确: {current_weight_path}")
        return

    # --- 循环逻辑 ---
    for target_stage in range(START_EPOCH + STEP_EPOCHS, TOTAL_TARGET + STEP_EPOCHS, STEP_EPOCHS):
        new_project_name = f"ours_pconv_extended_{target_stage}e"

        print(f"\n🚀 [阶段启动] 目标: {target_stage} Epochs (当前读取: {os.path.basename(current_weight_path)})")

        try:
            model = YOLO(current_weight_path)

            model.train(
                data=dataset_yaml,
                epochs=STEP_EPOCHS,
                imgsz=640,
                batch=BATCH_SIZE,
                device=device,
                workers=WORKERS,
                project="runs/voc_compare",
                name=new_project_name,
                exist_ok=True,
                amp=True,
                cache=False,
                # 【再次确认】关闭画图
                # 刚才的日志显示 Plotting labels 时也触发了 OpenMP 错误
                plots=False,
            )

            current_weight_path = os.path.join("runs", "voc_compare", new_project_name, "weights", "last.pt")

            print(f"✅ {target_stage}轮 存档完成！保存在: runs/voc_compare/{new_project_name}")

        except Exception as e:
            print(f"❌ 训练在冲击 {target_stage}轮 时启动失败: {e}")
            break

    print("\n" + "=" * 60)
    print("🎉 全部训练结束！")
    print("=" * 60)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
