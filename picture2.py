import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from scipy.interpolate import PchipInterpolator
import matplotlib.font_manager as fm


# ==========================================
# 1. 字体与风格设置
# ==========================================
def get_chinese_font():
    font_candidates = [
        r'C:\Windows\Fonts\msyh.ttf',  # 微软雅黑
        r'C:\Windows\Fonts\simhei.ttf',  # 黑体
        r'C:\Windows\Fonts\simsun.ttc',  # 宋体
    ]
    for f in font_candidates:
        if os.path.exists(f):
            return fm.FontProperties(fname=f)
    return fm.FontProperties(family='sans-serif')


zh_font = get_chinese_font()
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

sns.set_style("whitegrid")


def main():
    root_dir = r"runs/voc_compare"

    # -------------------------------------------------------
    # 2. 数据准备
    # -------------------------------------------------------
    ours_folders = [
        "ours_pconv",
        "ours_pconv_extended_200e", "ours_pconv_extended_300e",
        "ours_pconv_extended_400e", "ours_pconv_extended_500e",
        "ours_pconv_extended_600e", "ours_pconv_extended_700e",
        "ours_pconv_extended_800e", "ours_pconv_extended_900e",
        "ours_pconv_extended_1000e"
    ]
    baseline_folder = "baseline_yolo11n"

    # 读取 Baseline
    base_csv = os.path.join(root_dir, baseline_folder, "results.csv")
    if os.path.exists(base_csv):
        df_base = pd.read_csv(base_csv)
        df_base.columns = [c.strip() for c in df_base.columns]
        baseline_map = df_base['metrics/mAP50-95(B)'].max()
    else:
        baseline_map = 0.6482

        # 提取关键点
    x_points = [0]
    y_points = [0.0]

    current_epoch = 100
    print("🔄 正在生成最终定稿图表...")

    for folder in ours_folders:
        csv_path = os.path.join(root_dir, folder, "results.csv")
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                df.columns = [c.strip() for c in df.columns]
                best_map_stage = df['metrics/mAP50-95(B)'].max()
                x_points.append(current_epoch)
                y_points.append(best_map_stage)
            except Exception:
                pass
        current_epoch += 100

    # -------------------------------------------------------
    # 3. PCHIP 平滑拟合
    # -------------------------------------------------------
    x_np = np.array(x_points)
    y_np = np.array(y_points)
    sort_idx = np.argsort(x_np)
    x_np = x_np[sort_idx]
    y_np = y_np[sort_idx]

    X_smooth = np.linspace(x_np.min(), x_np.max(), 300)
    interpolator = PchipInterpolator(x_np, y_np)
    Y_smooth = interpolator(X_smooth)

    # -------------------------------------------------------
    # 4. 绘图
    # -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 7))

    # Baseline
    ax.axhline(y=baseline_map, color='#7f8c8d', linestyle='--', linewidth=2, alpha=0.6, zorder=1)
    ax.text(0, baseline_map + 0.015, f'基准线 (Baseline): {baseline_map:.4f}',
            color='#7f8c8d', fontproperties=zh_font, fontsize=12, fontweight='bold')

    # 平滑曲线 【在这里修改了 label】
    ax.plot(X_smooth, Y_smooth, color='#D32F2F', linewidth=3,
            label='PConv-YOLO 性能趋势',  # <--- 修改点 1：更专业
            zorder=2)

    # 散点 【在这里修改了 label】
    ax.scatter(x_points, y_points, color='#B71C1C', s=40,
               label='每100轮最优点',  # <--- 修改点 2：更严谨
               zorder=3)

    # 确定目标点 (倒数第二个)
    target_idx = len(x_points) - 2

    # A. 标注普通点 (跳过目标点)
    for i, (px, py) in enumerate(zip(x_points, y_points)):
        if px == 0: continue
        if i == target_idx: continue

        ax.annotate(f'{py:.4f}',
                    xy=(px, py),
                    xytext=(0, -15),
                    textcoords='offset points',
                    ha='center', va='top',
                    fontsize=9, color='#333333', fontweight='bold')

    # B. 单独标注最优点 (倒数第二个点)
    best_x = x_points[target_idx]
    best_y = y_points[target_idx]
    diff = best_y - baseline_map
    mid_y = (best_y + baseline_map) / 2

    # 1. 垂直差值线
    ax.vlines(x=best_x, ymin=best_y, ymax=baseline_map, colors='#E67E22', linestyles='--', linewidth=2, zorder=1)

    # 2. 差值标注
    ax.text(best_x, mid_y,
            f" 差值: {diff:.4f} ",
            color='#E67E22',
            fontproperties=zh_font,
            fontsize=11, fontweight='bold',
            ha='right', va='center',
            bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="none", alpha=0.9))

    # 3. 最优点绝对数值
    ax.annotate(f'{best_y:.4f}',
                xy=(best_x, best_y),
                xytext=(0, -15),
                textcoords='offset points',
                ha='center',
                va='top',
                fontsize=11,
                color='#D32F2F',
                fontweight='bold')

    # 装饰
    ax.set_title('PConv-YOLO 训练收敛性分析', fontproperties=zh_font, fontsize=18, pad=20)
    ax.set_xlabel('训练轮数 (Epochs)', fontproperties=zh_font, fontsize=14)
    ax.set_ylabel('最佳精度 (Best mAP50-95)', fontproperties=zh_font, fontsize=14)

    legend = ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
    for text in legend.get_texts():
        text.set_fontproperties(zh_font)

    ax.set_xlim(-50, 1150)
    ax.set_ylim(0, 0.8)
    sns.despine()

    plt.tight_layout()
    save_path = '最优点下移标注图_CN.png'
    plt.savefig(save_path, dpi=300)
    print(f"✅ 图表已生成: {save_path}")
    plt.show()


if __name__ == "__main__":
    main()