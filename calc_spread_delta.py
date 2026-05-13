import numpy as np
import matplotlib.pyplot as plt

run_start = 0
run_end = 10
base_link_moead = "./result/pareto/moead/moead_"
base_link_nsga = "./result/pareto/nsga/nsga_"
base_link_nspso = "./result/pareto/nspso/nspso_"
datasets = ["100_1", "150_1", "200_1", "250_1"]


def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


def spread_metric(pareto_front):
    pareto_front = sorted(pareto_front, key=lambda x: x[0])
    pareto_front = np.unique(np.array(pareto_front), axis=0)

    f1_max = pareto_front[-1][0]
    f2_max = pareto_front[-1][1]
    f1_min = pareto_front[0][0]
    f2_min = pareto_front[0][1]

    for point in pareto_front:
        point[0] = (point[0] - f1_min) / (f1_max - f1_min)
        point[1] = (point[1] - f2_min) / (f2_max - f2_min)

    distances = [
        euclidean_distance(pareto_front[i], pareto_front[i + 1])
        for i in range(1, len(pareto_front) - 2)
    ]

    d_f = euclidean_distance(pareto_front[0], pareto_front[1])
    d_l = euclidean_distance(pareto_front[-1], pareto_front[-2])

    d_avg = np.mean(distances)

    spread = (d_f + d_l + sum(abs(d - d_avg) for d in distances)) / (
        d_f + d_l + (len(pareto_front) - 1) * d_avg
    )

    return spread


spread_values_per_dataset_moead = []
spread_values_per_dataset_nsga = []
spread_values_per_dataset_nspso = []

for dataset in datasets:
    spread_values_moead = []
    spread_values_nsga = []
    spread_values_nspso = []
    for run in range(run_start, run_end):
        pareto_front_moead = np.loadtxt(f"{base_link_moead}{dataset}_{run}.csv", dtype=float, delimiter=",")
        pareto_front_nsga = np.loadtxt(f"{base_link_nsga}{dataset}_{run}.csv", dtype=float, delimiter=",")
        pareto_front_nspso = np.loadtxt(f"{base_link_nspso}{dataset}_{run}.csv", dtype=float, delimiter=",")

        spread_values_moead.append(spread_metric(pareto_front_moead))
        spread_values_nsga.append(spread_metric(pareto_front_nsga))
        spread_values_nspso.append(spread_metric(pareto_front_nspso))

    spread_values_per_dataset_moead.append(spread_values_moead)
    spread_values_per_dataset_nsga.append(spread_values_nsga)
    spread_values_per_dataset_nspso.append(spread_values_nspso)

positions = np.arange(1, len(datasets) + 1)
offset = 0.25

plt.boxplot(
    spread_values_per_dataset_moead,
    positions=positions - offset,
    widths=0.2,
    patch_artist=True,
    boxprops=dict(facecolor="red", alpha=0.5),
    medianprops=dict(color="black"),
)
plt.boxplot(
    spread_values_per_dataset_nsga,
    positions=positions,
    widths=0.2,
    patch_artist=True,
    boxprops=dict(facecolor="blue", alpha=0.5),
    medianprops=dict(color="black"),
)
plt.boxplot(
    spread_values_per_dataset_nspso,
    positions=positions + offset,
    widths=0.2,
    patch_artist=True,
    boxprops=dict(facecolor="green", alpha=0.5),
    medianprops=dict(color="black"),
)

plt.ylabel("Spread Delta Value")
plt.xlabel("Number of Sensors")
plt.xticks(positions, ["100", "150", "200", "250"])
plt.legend(["MOEA/D", "NSGA-II", "NSPSO"], loc="best")
plt.show()
