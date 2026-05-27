import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

# Parameters
R_U = 1.0
kappa = 2.0
R_S = kappa * R_U
R_certain = R_S - R_U

n_layers = 80

# Figure
fig, ax = plt.subplots(figsize=(6, 6))

# Colors
certain_color = "#4C72B0"
attenuation_color = "#D9DFEB"

# Attenuation region (gradient)
radii = np.linspace(R_S, R_certain, n_layers)
alphas = np.linspace(0.04, 0.25, n_layers)

for r, a in zip(radii, alphas):
    ax.add_patch(
        Circle(
            (0, 0),
            r,
            facecolor=attenuation_color,
            edgecolor=None,
            alpha=a,
            zorder=1
        )
    )

# Clear inside
ax.add_patch(
    Circle(
        (0, 0),
        R_certain,
        facecolor="white",
        edgecolor=None,
        zorder=2
    )
)

# Certain region
ax.add_patch(
    Circle(
        (0, 0),
        R_certain,
        facecolor=certain_color,
        edgecolor="black",
        linewidth=1.3,
        alpha=0.55,
        zorder=3
    )
)

# Boundary of outer radius
ax.add_patch(
    Circle(
        (0, 0),
        R_S,
        facecolor="none",
        edgecolor="black",
        linestyle="--",
        linewidth=1.8,
        zorder=4
    )
)

# Sensor point
ax.plot(0, 0, "ko", markersize=7, zorder=5)
ax.text(-0.12, -0.18, r"$S_i$", fontsize=16)

# Label RS = kRU
ax.annotate(
    "",
    xy=(R_S, 0),
    xytext=(0, 0),
    arrowprops=dict(arrowstyle="<->", linewidth=1.5),
    zorder=6
)
ax.text(
    R_S / 2 - 0.35,
    -0.22,
    r"$R_S= kR_U$",
    fontsize=14
)

# plot RU
ax.annotate(
    "",
    xy=(0, R_certain),
    xytext=(0, R_S),
    arrowprops=dict(arrowstyle="<->", linewidth=1.5),
)
ax.text(0.1, (R_certain + R_S) / 2, "$R_U$", fontsize=14)

# Optional region text (can remove if you want even cleaner)
ax.text(0, 0.25, "Certain region", fontsize=10, ha="center")
ax.text(0.95, 1.15, "Attenuation region", fontsize=10, ha="center")

# Layout
margin = 0.5
ax.set_aspect("equal")
ax.set_xlim(-R_S - margin, R_S + margin)
ax.set_ylim(-R_S - margin, R_S + margin)
ax.axis("off")

plt.tight_layout()
plt.savefig("sensing_model_clean.png", dpi=300, bbox_inches="tight")
plt.savefig("sensing_model_clean.pdf", bbox_inches="tight")
plt.show()