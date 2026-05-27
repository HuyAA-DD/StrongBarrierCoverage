import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Patch
from matplotlib.lines import Line2D

# =========================
# Input
# =========================
fileData = "./dataset/100_1.txt"
x_corr = np.loadtxt(fileData, dtype=int)

k = 2
alpha = 0.7
n_layers = 100

# =========================
# Style xanh dương
# =========================
primary_color = "#1E88E5"      # xanh dương chính
light_blue = "#BBDEFB"         # xanh dương nhạt
dark_blue = "#0D47A1"          # xanh dương đậm
uncertain_edge = "#1565C0"     # viền vùng không chắc chắn

# =========================
# Radii
# =========================
radii = np.array([
    0, 0, 26, 0, 0, 0, 0, 0, 0, 3,
    20, 0, 0, 0, 0, 19, 0, 0, 0, 0,
    11, 0, 0, 0, 0, 0, 13, 0, 0, 0,
    0, 0, 26, 0, 0, 0, 29, 0, 0, 0,
    5, 0, 0, 11, 0, 0, 31, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 27, 0,
    16, 0, 0, 0, 0, 0, 0, 39, 0, 0,
    0, 0, 0, 0, 0, 36, 0, 0, 0, 0,
    28, 25, 17, 0, 0, 0, 0, 0, 0, 26,
    0, 0, 0, 30, 0, 0, 48, 0, 0, 0
])

N = min(len(x_corr), len(radii))

# =========================
# Plot
# =========================
fig, ax = plt.subplots(figsize=(14, 6))

# Vẽ đường barrier
ax.hlines(
    y=0,
    xmin=0,
    xmax=1000,
    color="black",
    linewidth=2.2,
    zorder=1
)

for i in range(N):
    if radii[i] > 0:
        r_u = radii[i]
        R_s = k * r_u
        R_certain = R_s - r_u

        # =========================
        # Uncertain region nhiều lớp xanh mờ
        # =========================
        radius_layers = np.linspace(R_certain, R_s, n_layers)

        alphas = np.exp(
            -np.linspace(-np.log(alpha), -np.log(0.01), n_layers)
        )

        for j in range(n_layers):
            circle = Circle(
                (x_corr[i], 0),
                radius_layers[j],
                color=primary_color,
                fill=False,
                alpha=alphas[j] * 0.35,
                linewidth=0.7,
                zorder=2
            )
            ax.add_artist(circle)

        # =========================
        # Certain region
        # =========================
        inner_circle = Circle(
            (x_corr[i], 0),
            R_certain,
            facecolor=primary_color,
            edgecolor=dark_blue,
            alpha=0.42,
            linewidth=1.0,
            zorder=3
        )
        ax.add_artist(inner_circle)

        # =========================
        # Outer sensing boundary
        # =========================
        outer_circle = Circle(
            (x_corr[i], 0),
            R_s,
            edgecolor=uncertain_edge,
            linestyle="--",
            linewidth=1.4,
            fill=False,
            alpha=0.65,
            zorder=4
        )
        ax.add_patch(outer_circle)

        # Active sensor
        ax.scatter(
            x_corr[i],
            0,
            marker="o",
            color="red",
            edgecolor="black",
            linewidth=0.5,
            s=35,
            zorder=6
        )

    else:
        # Inactive sensor
        ax.scatter(
            x_corr[i],
            0,
            marker="x",
            color="gray",
            s=22,
            alpha=0.75,
            zorder=5
        )

# =========================
# Layout
# =========================
max_Rs = np.max(radii[:N]) * k

ax.set_xlim(0, 1000)
ax.set_ylim(-max_Rs * 1.08, max_Rs * 1.08)

ax.set_xlabel("Position on barrier", fontsize=14)
ax.set_ylabel("Sensing range", fontsize=14)

ax.set_title(
    "Illustration of Probabilistic Strong Barrier Coverage",
    fontsize=15,
    fontweight="bold"
)

ax.grid(
    True,
    linestyle="--",
    alpha=0.35
)

ax.tick_params(axis="both", labelsize=12)

# Giữ tỉ lệ hình tròn đúng
ax.set_aspect("equal", adjustable="box")

# =========================
# Legend
# =========================
legend_elements = [
    Line2D(
        [0], [0],
        marker="o",
        color="w",
        label="Active sensor",
        markerfacecolor="red",
        markeredgecolor="black",
        markersize=8
    ),
    Line2D(
        [0], [0],
        marker="x",
        color="gray",
        label="Inactive sensor",
        linestyle="None",
        markersize=8
    ),
    Patch(
        facecolor=primary_color,
        edgecolor=dark_blue,
        alpha=0.42,
        label="Certain region"
    ),
    Line2D(
        [0], [0],
        color=uncertain_edge,
        linestyle="--",
        linewidth=1.5,
        label="Uncertain boundary"
    )
]

ax.legend(
    handles=legend_elements,
    fontsize=12,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.18),
    ncol=4,
    frameon=True
)

plt.tight_layout()

# Nếu muốn lưu ảnh thì mở 2 dòng này
# plt.savefig("sensor_probabilistic_coverage_blue.png", dpi=300, bbox_inches="tight")
# plt.savefig("sensor_probabilistic_coverage_blue.pdf", bbox_inches="tight")

plt.show()