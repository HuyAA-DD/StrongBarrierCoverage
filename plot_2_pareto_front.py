import pandas as pd
import matplotlib.pyplot as plt

dataset = 100
algos = ["moead", "nsga", "nspso"]
subset = 7

filenames = [
    f"./result/pareto/approx/moead_{dataset}_{subset}.csv",
    f"./result/pareto/approx/nsga_{dataset}_{subset}.csv",
    f"./result/pareto/approx/{dataset}_{subset}.csv",
]

colors = ["r", "b", "g"]
markers = ["o", "D", "^"]
lines = ["-", "--", ":"]
plt.figure(figsize=(10, 9))

for i, filename in enumerate(filenames):
    df = pd.read_csv(filename, header=None, names=["f1", "f2"])
    plt.scatter(
        df["f1"],
        df["f2"],
        label=algos[i],
        color=colors[i],
        marker=markers[i],
        s=70,
    )
    plt.plot(
        df["f1"],
        df["f2"],
        lines[i],
        color=colors[i],
        linewidth=2.5,
    )

plt.xlabel("Number of active sensors", fontsize=14)
plt.ylabel("Total energy consumption", fontsize=14)
plt.legend(fontsize=19)
plt.grid(True)
plt.tight_layout()
plt.show()
