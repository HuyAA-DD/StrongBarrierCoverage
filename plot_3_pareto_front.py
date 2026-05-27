import pandas as pd
import matplotlib.pyplot as plt

dataset = 100
subset = 8

algos = ["moead", "nsga", "nspso"]

filenames = [
    f"./result/pareto/approx/{algo}_{dataset}_{subset}.csv"
    for algo in algos
]

styles = {
    "moead": {
        "color": "#E41A1C",
        "marker": "o",
        "linestyle": "-",
        "label": "MOEA/D",
        "alpha": 0.35,
        "linewidth": 1.8,
        "size": 55,
        "zorder": 2
    },
    "nsga": {
        "color": "#377EB8",
        "marker": "D",
        "linestyle": "-",
        "label": "NSGA-II",
        "alpha": 1.0,
        "linewidth": 3.8,
        "size": 115,
        "zorder": 5
    },
    "nspso": {
        "color": "#4DAF4A",
        "marker": "X",
        "linestyle": "-.",
        "label": "NSPSO",
        "alpha": 0.5,
        "linewidth": 2.0,
        "size": 70,
        "zorder": 3
    }
}

plt.figure(figsize=(10, 8))

dataframes = {}

for algo, filename in zip(algos, filenames):
    df = pd.read_csv(filename, header=None, names=["f1", "f2"])
    df = df.drop_duplicates()
    df = df.sort_values(by=["f1", "f2"])
    dataframes[algo] = df

    style = styles[algo]

    plt.plot(
        df["f1"],
        df["f2"],
        linestyle=style["linestyle"],
        color=style["color"],
        linewidth=style["linewidth"],
        alpha=style["alpha"],
        label=style["label"],
        zorder=style["zorder"]
    )

    plt.scatter(
        df["f1"],
        df["f2"],
        color=style["color"],
        marker=style["marker"],
        s=style["size"],
        edgecolor="black",
        linewidth=0.8,
        alpha=style["alpha"],
        zorder=style["zorder"] + 1
    )

#plt.xlim(20, 45)
#plt.ylim(0, 500)

# Chú thích điểm tốt nhất theo năng lượng của NSGA-II
nsga_df = dataframes["nsga"]
best_nsga = nsga_df.loc[nsga_df["f2"].idxmin()]

plt.annotate(
    "NSGA-II obtains lower energy",
    xy=(best_nsga["f1"], best_nsga["f2"]),
    xytext=(best_nsga["f1"] + 2, best_nsga["f2"] * 1.05),
    arrowprops=dict(
        arrowstyle="->",
        linewidth=1.5,
        color="#377EB8"
    ),
    fontsize=12,
    color="#377EB8",
    fontweight="bold"
)

plt.xlabel("Number of active sensors", fontsize=14)
plt.ylabel("Total energy consumption", fontsize=14)

plt.title(
    f"Comparison of Pareto Fronts on Dataset {dataset}, Subset {subset}",
    fontsize=15,
    fontweight="bold"
)

plt.legend(fontsize=13)
plt.grid(True, linestyle="--", alpha=0.45)

plt.tight_layout()
plt.show()