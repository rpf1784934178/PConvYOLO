import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.font_manager as fm
from scipy.interpolate import PchipInterpolator


# ==========================================
# 1. 基础设置
# ==========================================
def get_chinese_font():
    font_candidates = [r'C:\Windows\Fonts\msyh.ttf', r'C:\Windows\Fonts\simhei.ttf', r'C:\Windows\Fonts\simsun.ttc']
    for f in font_candidates:
        if os.path.exists(f): return fm.FontProperties(fname=f)
    return fm.FontProperties(family='sans-serif')


zh_font = get_chinese_font()
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'


def main():
    root_dir = r"D:\PythonSoftware\code\YoloSsd\ultralytics\runs\voc_compare"
    ours_folders = [
        "ours_pconv", "ours_pconv_extended_200e", "ours_pconv_extended_300e",
        "ours_pconv_extended_400e", "ours_pconv_extended_500e",
        "ours_pconv_extended_600e", "ours_pconv_extended_700e",
        "ours_pconv_extended_800e", "ours_pconv_extended_900e",
        "ours_pconv_extended_1000e"
    ]

    # 数据容器
    x_points = [0]
    y_map5095 = [0.0]
    y_map50 = [0.0]  # 新增指标：mAP@50

    current_epoch = 100
    print("🔄 正在读取多维指标数据...")

    for folder in ours_folders:
        csv_path = os.path.join(root_dir, folder, "results.csv")
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                df.columns = [c.strip() for c in df.columns]  # 去除空格

                # 提取阶段最大值
                best_idx = df['metrics/mAP50-95(B)'].idxmax()

                # 获取 mAP50-95 和 mAP50
                val_5095 = df.iloc[best_idx]['metrics/mAP50-95(B)']
                val_50 = df.iloc[best_idx]['metrics/mAP50(B)']  # 读取 mAP50

                x_points.append(current_epoch)
                y_map5095.append(val_5095)
                y_map50.append(val_50)
            except Exception as e:
                print(f"Skipping {folder}: {e}")
        current_epoch += 100

    # 平滑处理
    def get_smooth_curve(x, y):
        x = np.array(x);
        y = np.array(y)
        idx = np.argsort(x);
        x = x[idx];
        y = y[idx]
        X_smooth = np.linspace(x.min(), x.max(), 300)
        interpolator = PchipInterpolator(x, y)
        return X_smooth, interpolator(X_smooth)

    X_new, Y_5095_smooth = get_smooth_curve(x_points, y_map5095)
    _, Y_50_smooth = get_smooth_curve(x_points, y_map50)

    # ================= 绘图 =================
    fig, ax = plt.subplots(figsize=(12, 8))

    # 1. 绘制 mAP@50 (高位曲线，用虚线或浅色)
    ax.plot(X_new, Y_50_smooth, color='#2980B9', linewidth=2.5, linestyle='--', alpha=0.8, label='mAP@50 (宽松标准)')
    ax.scatter(x_points, y_map50, color='#2980B9', s=30, alpha=0.6)

    # 2. 绘制 mAP@50-95 (核心曲线，用实线深色)
    ax.plot(X_new, Y_5095_smooth, color='#C0392B', linewidth=3.5, label='mAP@50-95 (严苛标准)')
    ax.scatter(x_points, y_map5095, color='#C0392B', s=50, zorder=5)

    # 3. 填充中间区域 (增加丰满度)
    ax.fill_between(X_new, Y_5095_smooth, Y_50_smooth, color='#D4E6F1', alpha=0.3, label='性能鲁棒区间')

    # 装饰
    ax.set_title('PConv-YOLO 多维精度指标演变分析 (900轮)', fontproperties=zh_font, fontsize=18, pad=20,
                 fontweight='bold')
    ax.set_xlabel('训练轮数 (Epochs)', fontproperties=zh_font, fontsize=14, fontweight='bold')
    ax.set_ylabel('评估指标 (Metrics)', fontproperties=zh_font, fontsize=14, fontweight='bold')

    # 标注最终值
    ax.text(x_points[-1] + 10, y_map50[-1], f"{y_map50[-1]:.3f}", color='#2980B9', fontweight='bold', va='center')
    ax.text(x_points[-1] + 10, y_map5095[-1], f"{y_map5095[-1]:.3f}", color='#C0392B', fontweight='bold', va='center')

    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_xlim(-50, 1150)
    ax.set_ylim(0, 1.0)  # 0-1 的范围能容纳 mAP50
    ax.legend(loc='lower right', prop=zh_font, fontsize=12)

    plt.tight_layout()
    plt.savefig('双指标趋势图.png', dpi=300)
    print("✅ 双指标趋势图已生成！")
    plt.show()


if __name__ == "__main__":
    main()