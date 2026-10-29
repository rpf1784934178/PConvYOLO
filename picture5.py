import os

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# ==========================================
# 1. 字体设置 (防乱码)
# ==========================================
def get_chinese_font():
    font_candidates = [r"C:\Windows\Fonts\msyh.ttf", r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\simsun.ttc"]
    for f in font_candidates:
        if os.path.exists(f):
            return fm.FontProperties(fname=f)
    return fm.FontProperties(family="sans-serif")


zh_font = get_chinese_font()

# 英文用 Times New Roman
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]


def main():
    root_dir = r"D:\PythonSoftware\code\YoloSsd\ultralytics\runs\voc_compare"

    # -------------------------------------------------------
    # 2. 数据收集
    # -------------------------------------------------------
    data = []

    # A. 读取 Baseline
    try:
        df_base = pd.read_csv(os.path.join(root_dir, "baseline_yolo11n", "results.csv"))
        df_base.columns = [c.strip() for c in df_base.columns]
        best_row = df_base.iloc[df_base["metrics/mAP50-95(B)"].idxmax()]
        data.append(
            {
                "模型名称": "Baseline (YOLO11n)",
                "mAP 50-95": best_row["metrics/mAP50-95(B)"],
                "mAP 50": best_row["metrics/mAP50(B)"],
                "Precision": best_row["metrics/precision(B)"],
                "Recall": best_row["metrics/recall(B)"],
            }
        )
    except:
        print("未找到 Baseline 数据")

    # B. 读取 Ours (100e - 1000e)
    ours_folders = [
        "ours_pconv",
        "ours_pconv_extended_200e",
        "ours_pconv_extended_300e",
        "ours_pconv_extended_400e",
        "ours_pconv_extended_500e",
        "ours_pconv_extended_600e",
        "ours_pconv_extended_700e",
        "ours_pconv_extended_800e",
        "ours_pconv_extended_900e",
        "ours_pconv_extended_1000e",
    ]

    current_epoch = 100
    for folder in ours_folders:
        try:
            csv_path = os.path.join(root_dir, folder, "results.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                df.columns = [c.strip() for c in df.columns]
                # 取该阶段的最优值
                best_row = df.iloc[df["metrics/mAP50-95(B)"].idxmax()]

                # 标记第900轮为特殊名字
                name = f"PConv-YOLO ({current_epoch}e)"
                if current_epoch == 900:
                    name = "★ PConv-YOLO (900e 最优)"

                data.append(
                    {
                        "模型名称": name,
                        "mAP 50-95": best_row["metrics/mAP50-95(B)"],
                        "mAP 50": best_row["metrics/mAP50(B)"],
                        "Precision": best_row["metrics/precision(B)"],
                        "Recall": best_row["metrics/recall(B)"],
                    }
                )
        except:
            pass
        current_epoch += 100

    # -------------------------------------------------------
    # 3. 绘制热力图
    # -------------------------------------------------------
    df_plot = pd.DataFrame(data).set_index("模型名称")

    plt.figure(figsize=(10, 8))

    # 颜色选择：YlGnBu (黄绿蓝) 或 RdBu (红蓝)
    # 这种配色方案在论文里看起来很干净
    ax = sns.heatmap(
        df_plot,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        linewidths=1,
        linecolor="white",
        cbar_kws={"label": "Metric Value"},
        annot_kws={"size": 11, "fontfamily": "Times New Roman"},
    )

    # 调整布局
    ax.xaxis.tick_top()  # 列名放上面
    ax.xaxis.set_label_position("top")

    # 标题
    plt.title(
        "所有模型性能对比热力图 (Baseline vs. 10阶段)", fontproperties=zh_font, fontsize=16, pad=25, fontweight="bold"
    )
    plt.ylabel("")

    # 字体调整
    plt.xticks(fontproperties=zh_font, fontsize=12)  # 如果列名是中文
    # plt.xticks(fontname='Times New Roman', fontsize=12) # 如果列名是英文

    plt.yticks(fontproperties=zh_font, fontsize=12, rotation=0)

    plt.tight_layout()
    plt.savefig("11模型对比热力图.png", dpi=300)
    print("✅ [方案A] 11模型对比热力图已生成！")
    plt.show()


if __name__ == "__main__":
    main()
