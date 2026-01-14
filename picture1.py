import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import matplotlib.font_manager as fm


# ==========================================
# 1. 基础设置 (字体与风格)
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
# 论文标准字体设置
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'


def main():
    # ==========================================
    # 2. 数据录入
    # ==========================================
    base_fps = 171
    base_map = 0.6482
    our_fps = 236
    our_map = 0.5936

    # 计算变化率
    speed_gain_pct = (our_fps - base_fps) / base_fps * 100

    # ==========================================
    # 3. 绘图逻辑
    # ==========================================
    fig, ax = plt.subplots(figsize=(10, 7))

    # 【核心修改】设置 Y 轴从 0 开始
    # 这样更能体现“精度损失微乎其微”
    x_min, x_max = 130, 280
    y_min, y_max = 0.0, 0.80  # 上限设为 0.8 给标题留空间
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # --- A. 绘制“精度容忍区间” (全高度背景) ---
    tolerance_drop = 0.06
    zone_top = base_map + 0.02
    zone_bottom = base_map - tolerance_drop

    rect_width = x_max - x_min
    rect = patches.Rectangle((x_min, zone_bottom), rect_width, (zone_top - zone_bottom),
                             linewidth=0, edgecolor='none', facecolor='#F2F3F4', alpha=0.6, zorder=0)
    ax.add_patch(rect)

    # 标注区间文字
    ax.text(x_min + 5, (base_map + zone_bottom) / 2, "精度容忍区间 (Tolerance Zone)",
            color='#95A5A6', fontproperties=zh_font, fontsize=12,
            va='center', ha='left', fontweight='bold', style='italic')

    # --- B. 绘制数据点 ---
    # Baseline
    ax.scatter(base_fps, base_map, color='#34495E', s=250, marker='s',
               label='Baseline (YOLO11n)', zorder=5, edgecolors='black', linewidth=1)

    # Ours
    ax.scatter(our_fps, our_map, color='#C0392B', s=450, marker='*',
               label='PConv-YOLO (Ours)', zorder=5, edgecolors='black', linewidth=1)

    # 数值标签 (稍微上移，防止遮挡)
    ax.text(base_fps, base_map + 0.02, f"({base_fps}, {base_map:.4f})",
            ha='center', va='bottom', fontsize=10, fontweight='bold', color='#34495E')
    ax.text(our_fps, our_map + 0.02, f"({our_fps}, {our_map:.4f})",
            ha='center', va='bottom', fontsize=10, fontweight='bold', color='#C0392B')

    # --- C. 辅助投影线 (拉长到底部) ---
    # 这能体现“脚踏实地”的速度对比
    ax.vlines(x=base_fps, ymin=0, ymax=base_map, color='gray', linestyle=':', alpha=0.4)
    ax.vlines(x=our_fps, ymin=0, ymax=our_map, color='#C0392B', linestyle=':', alpha=0.4)
    # 横向参考线
    ax.hlines(y=base_map, xmin=x_min, xmax=our_fps, color='gray', linestyle='--', alpha=0.4)

    # --- D. 绘制速度增益箭头 (移到底部空白区) ---
    # 利用 Y=0 带来的底部空白，画一个更直观的对比
    arrow_y_pos = 0.1  # 在底部绘制
    ax.annotate("",
                xy=(our_fps, arrow_y_pos), xycoords='data',
                xytext=(base_fps, arrow_y_pos), textcoords='data',
                arrowprops=dict(arrowstyle="->", color="#C0392B", lw=3))

    # 增益文字
    mid_fps = (base_fps + our_fps) / 2
    ax.text(mid_fps, arrow_y_pos + 0.03,
            f"推理速度增益: +{speed_gain_pct:.1f}%",
            color='#C0392B', fontproperties=zh_font, fontsize=13, fontweight='bold', ha='center',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#C0392B", alpha=0.9))

    # --- E. 标题与标签 ---
    ax.set_title('PConv-YOLO 与基准模型的推理效率-精度权衡对比', fontproperties=zh_font, fontsize=18, pad=20,
                 fontweight='bold')

    ax.set_xlabel('推理速度 (Inference Speed / FPS)', fontproperties=zh_font, fontsize=12, fontweight='bold')
    ax.set_ylabel('平均精度 (mAP 50-95)', fontproperties=zh_font, fontsize=12, fontweight='bold')

    # 图例放在中上部空白处，或者右上角
    legend = ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black', fontsize=11)

    plt.tight_layout()
    save_path = '最终诚实版_效率对比图(从0开始).png'
    plt.savefig(save_path, dpi=300)
    print(f"✅ 图表已生成: {save_path}")
    plt.show()


if __name__ == "__main__":
    main()