import math
import matplotlib.pyplot as plt


# ---------------------- 原无人机坐标分配函数 ----------------------
def assign_drone_square_formation(n, start_x=0, start_y=0, spacing=10):
    positions = []
    if n <= 0:
        return positions
    if n == 1:
        positions.append((start_x, start_y))
        return positions

    sqrt_n = math.sqrt(n)
    min_size = math.ceil(sqrt_n)
    rows, cols = min_size, min_size

    # 调整行列数以容纳所有无人机
    while rows * cols < n:
        if rows <= cols:
            rows += 1
        else:
            cols += 1

    # 计算偏移量（确保方阵以(start_x, start_y)为中心）
    total_width = (cols - 1) * spacing
    total_height = (rows - 1) * spacing
    start_offset_x = start_x - total_width / 2
    start_offset_y = start_y - total_height / 2

    # 生成所有可能位置（含冗余）
    # 关键修改：使用reversed(range(rows))使行从下到上遍历
    all_positions = []
    for i in reversed(range(rows)):  # 从下到上遍历行
        for j in range(cols):  # 从左到右遍历列
            x = start_offset_x + j * spacing
            y = start_offset_y + i * spacing
            all_positions.append((x, y))

    # 计算需跳过的冗余位置（对称跳过）
    skip_count = rows * cols - n
    skip_indices = []
    for k in range(skip_count):
        if k % 2 == 0:
            skip_indices.append(k // 2)  # 从前面跳过
        else:
            skip_indices.append(len(all_positions) - 1 - (k // 2))  # 从后面跳过

    # 筛选最终位置（按顺序保留非跳过位置）
    count = 0
    for idx, (x, y) in enumerate(all_positions):
        if idx not in skip_indices:
            positions.append((round(x, 1), round(y, 1)))
            count += 1
            if count >= n:
                break
    return positions


# ---------------------- 可视化函数 ----------------------
def visualize_drone_formation(n_list, start_x=10, start_y=10, spacing=2):
    # 设置画布大小和子图布局（2行3列，适配6个案例）
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()  # 展平轴数组，方便循环

    for idx, n in enumerate(n_list):
        ax = axes[idx]
        # 1. 生成当前n的无人机坐标
        positions = assign_drone_square_formation(n, start_x, start_y, spacing)
        # 2. 提取x、y坐标列表
        x_coords = [p[0] for p in positions]
        y_coords = [p[1] for p in positions]

        # 3. 绘制无人机位置（蓝色散点）
        ax.scatter(x_coords, y_coords, s=200, c='skyblue', edgecolors='blue', zorder=3)

        # 4. 标注顺序编号（1~n，对应代码返回列表的顺序）
        for i, (x, y) in enumerate(positions):
            ax.annotate(
                str(i + 1),  # 编号从1开始（更直观）
                xy=(x, y),
                xytext=(0, 0),
                textcoords='offset points',
                ha='center',
                va='center',
                fontsize=12,
                fontweight='bold',
                color='darkblue'
            )

        # 5. 绘制方阵中心（红色×）
        ax.scatter(start_x, start_y, s=300, c='red', marker='*', zorder=2, label='方阵中心')

        # 6. 计算方阵的行列数（复用原代码逻辑，用于标题说明）
        if n == 1:
            rows, cols = 1, 1
        else:
            sqrt_n = math.sqrt(n)
            min_size = math.ceil(sqrt_n)
            rows, cols = min_size, min_size
            while rows * cols < n:
                if rows <= cols:
                    rows += 1
                else:
                    cols += 1
        skip_count = rows * cols - n

        # 7. 设置子图样式
        ax.set_title(
            f'无人机数量 n={n}\n方阵规格 {rows}×{cols}（跳过{skip_count}个冗余位置）',
            fontsize=14,
            fontweight='bold',
            pad=20
        )
        ax.set_xlabel('X坐标', fontsize=11)
        ax.set_ylabel('Y坐标', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')  # 网格线（辅助观察间距）
        ax.legend(loc='upper right', fontsize=10)

        # 8. 调整坐标轴范围（确保所有点和中心都在视野内）
        all_x = x_coords + [start_x]
        all_y = y_coords + [start_y]
        x_margin = spacing * 1.5
        y_margin = spacing * 1.5
        ax.set_xlim(min(all_x) - x_margin, max(all_x) + x_margin)
        ax.set_ylim(min(all_y) - y_margin, max(all_y) + y_margin)

    # 9. 整体标题
    fig.suptitle(
        '无人机方阵位置分配可视化（按顺序编号）',
        fontsize=18,
        fontweight='bold',
        y=0.98
    )

    # 10. 调整子图间距
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    # 显示图像
    plt.show()


# ---------------------- 运行可视化 ----------------------
if __name__ == "__main__":
    # 选择典型的无人机数量案例
    typical_n = [1, 4, 5, 7, 9, 100]
    # 调用可视化函数（默认中心(0,0)，间距10）
    visualize_drone_formation(typical_n)