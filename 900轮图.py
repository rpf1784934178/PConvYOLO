import os

from ultralytics import YOLO


def main():
    # ==========================================
    # 1. 路径配置
    # ==========================================
    # 你的 900轮 模型权重路径 (请确认路径正确)
    weights_path = (
        r"D:\PythonSoftware\code\YoloSsd\ultralytics\runs\voc_compare\ours_pconv_extended_900e\weights\best.pt"
    )

    # 你的数据集配置文件
    data_yaml = "VOC.yaml"

    # 输出结果保存的文件夹名称
    project_dir = "runs/paper_plots"
    name_dir = "PConv_900e_Detailed_Metrics"

    # ==========================================
    # 2. 执行详细验证
    # ==========================================
    if os.path.exists(weights_path):
        print(f"🚀 正在加载模型: {weights_path}")
        model = YOLO(weights_path)

        print("📊 开始生成全套评估图表 (Confusion Matrix, PR Curve, etc.)...")

        # 关键参数解释:
        # split='test': 使用测试集 (更严谨)
        # plots=True:  强制生成所有图表
        # save_json=True: 保存原始数据方便后续自己画图
        # workers=0:   防止内存溢出报错
        model.val(
            data=data_yaml,
            split="test",  # 或者 'val'
            imgsz=640,
            batch=4,  # 以此降低显存压力
            workers=0,  # 内存保护
            conf=0.001,  # 置信度阈值设低一点，为了画出完整的 PR 曲线
            iou=0.6,  # NMS IoU 阈值
            plots=True,  # ✅ 核心：必须为 True 才会画图
            save_json=True,
            project=project_dir,
            name=name_dir,
            exist_ok=True,  # 覆盖旧结果
        )

        print("\n✅ 图表已全部生成！")
        print(f"📂 请打开此文件夹查看图片: {os.path.join(project_dir, name_dir)}")

    else:
        print(f"❌ 找不到权重文件: {weights_path}")


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()
