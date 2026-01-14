import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib.font_manager as fm


# 设置字体
def get_chinese_font():
    font_candidates = [r'C:\Windows\Fonts\msyh.ttf', r'C:\Windows\Fonts\simhei.ttf']
    for f in font_candidates:
        if os.path.exists(f): return fm.FontProperties(fname=f)
    return fm.FontProperties(family='sans-serif')


zh_font = get_chinese_font()


def main():
    # 1. 准备数据 (你需要确保 baseline 文件夹里有 results.csv，或者手动填入 Baseline 的数据)
    # 这里我尝试读取，读不到就用默认值
    root_dir = r"D:\PythonSoftware\code\YoloSsd\ultralytics\runs\voc_compare"

    # === 读取 Baseline ===
    base_metrics = [0.6482, 0.82, 0.75, 0.70, 0.72]  # 默认占位符 [mAP50-95, mAP50, P, R, F1]
    try:
        df_base = pd.read_csv(os.path.join(root_dir, "baseline_yolo11n", "results.csv"))
        df_base.columns = [c.strip() for c in df_base.columns]
        best_base = df_base.iloc[df_base['metrics/mAP50-95(B)'].idxmax()]
        p = best_base['metrics/precision(B)']
        r = best_base['metrics/recall(B)']
        f1 = 2 * p * r / (p + r + 1e-6)
        base_metrics = [
            best_base['metrics/mAP50-95(B)'],
            best_base['metrics/mAP50(B)'],
            p, r, f1
        ]
    except:
        print("⚠️ 未找到 Baseline 数据，使用默认值")

    # === 读取 Ours (1000轮) ===
    our_metrics = [0.5936, 0.78, 0.72, 0.68, 0.70]  # 默认占位符
    try:
        df_our = pd.read_csv(os.path.join(root_dir, "ours_pconv_extended_1000e", "results.csv"))
        df_our.columns = [c.strip() for c in df_our.columns]
        best_our = df_our.iloc[df_our['metrics/mAP50-95(B)'].idxmax()]
        p = best_our['metrics/precision(B)']
        r = best_our['metrics/recall(B)']
        f1 = 2 * p * r / (p + r + 1e-6)
        our_metrics = [
            best_our['metrics/mAP50-95(B)'],
            best_our['metrics/mAP50(B)'],
            p, r, f1
        ]
    except:
        print("⚠️ 未找到 Ours 1000e 数据，请检查路径")

    # 2. 绘图设置
    labels = ['mAP 50-95', 'mAP 50', 'Precision', 'Recall', 'F1-Score']
    num_vars = len(labels)

    # 计算角度
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    base_metrics += base_metrics[:1]
    our_metrics += our_metrics[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # 绘制 Baseline
    ax.plot(angles, base_metrics, color='#7f8c8d', linewidth=2, linestyle='--', label='Baseline')
    ax.fill(angles, base_metrics, color='#7f8c8d', alpha=0.1)

    # 绘制 Ours
    ax.plot(angles, our_metrics, color='#C0392B', linewidth=3, label='PConv-YOLO')
    ax.fill(angles, our_metrics, color='#C0392B', alpha=0.2)

    # 装饰
    ax.set_theta_offset(np.pi / 2)  # 旋转起始位置
    ax.set_theta_direction(-1)  # 顺时针

    plt.xticks(angles[:-1], labels, fontproperties=zh_font, size=12)

    # Y轴刻度
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=10)
    plt.ylim(0, 1.0)

    plt.title('模型综合性能五维对比 (Radar Chart)', fontproperties=zh_font, size=16, y=1.05, fontweight='bold')
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), prop=zh_font)

    plt.tight_layout()
    plt.savefig('五维雷达图.png', dpi=300)
    print("✅ 五维雷达图已生成！")
    plt.show()


if __name__ == "__main__":
    main()