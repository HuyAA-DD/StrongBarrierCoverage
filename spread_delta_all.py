import numpy as np
import os

run_start = 0
run_end = 10
num_subdatasets = 10

datasets = ["100", "150", "200", "250", "300"]
base_link = "./result/pareto"

algorithms = {
    "MOEA/D": "moead",
    "NSGA-II": "nsga",
    "NSPSO": "nspso"
}


def euclidean_distance(point1, point2):
    return np.linalg.norm(point1 - point2)


def normalize(data, min_vals, max_vals):
    denom = max_vals - min_vals
    denom[denom == 0] = 1.0
    return (data - min_vals) / denom


def load_front(file_path):
    try:
        data = np.loadtxt(file_path, dtype=float, delimiter=",")

        if data.size == 0:
            return np.empty((0, 2))

        if data.ndim == 1:
            data = data.reshape(1, -1)

        data = np.unique(data, axis=0)
        return data

    except (OSError, ValueError):
        return np.empty((0, 2))


def spread_delta(pareto_front, extreme_point_1, extreme_point_2):
    if len(pareto_front) < 3:
        return np.nan

    pareto_front = np.unique(pareto_front, axis=0)

    if len(pareto_front) < 3:
        return np.nan

    # Sort theo f1, dùng cho bài toán 2 mục tiêu
    pareto_front = pareto_front[np.argsort(pareto_front[:, 0])]

    distances = np.array([
        euclidean_distance(pareto_front[i], pareto_front[i + 1])
        for i in range(len(pareto_front) - 1)
    ])

    d_avg = np.mean(distances)

    d_f = euclidean_distance(extreme_point_1, pareto_front[0])
    d_l = euclidean_distance(extreme_point_2, pareto_front[-1])

    sum_diff = np.sum(np.abs(distances - d_avg))

    denominator = d_f + d_l + (len(pareto_front) - 1) * d_avg

    if denominator == 0:
        return np.nan

    return (d_f + d_l + sum_diff) / denominator


all_spread_results = {
    "MOEA/D": [],
    "NSGA-II": [],
    "NSPSO": []
}


for dataset in datasets:
    for sub_id in range(num_subdatasets):

        # Reference front chung cho subdataset này
        reference_file = f"{base_link}/approx/{dataset}_{sub_id}.csv"
        reference_front = load_front(reference_file)

        if len(reference_front) < 3:
            print(f"Skip reference front: {reference_file}")
            continue

        # Normalize theo reference front
        min_vals = np.min(reference_front, axis=0)
        max_vals = np.max(reference_front, axis=0)
        denom = max_vals - min_vals
        denom[denom == 0] = 1.0

        reference_norm = normalize(reference_front, min_vals, max_vals)

        # Lấy extreme points từ reference front đã chuẩn hóa
        reference_sorted = reference_norm[np.argsort(reference_norm[:, 0])]

        extreme_point_1 = reference_sorted[0]
        extreme_point_2 = reference_sorted[-1]

        for run in range(run_start, run_end):
            for alg_name, folder_name in algorithms.items():

                file_path = (
                    f"{base_link}/{folder_name}/"
                    f"{folder_name}_{dataset}_{sub_id}_{run}.csv"
                )

                approx_front = load_front(file_path)

                if len(approx_front) < 3:
                    continue

                approx_norm = normalize(approx_front, min_vals, max_vals)

                spread = spread_delta(
                    approx_norm,
                    extreme_point_1,
                    extreme_point_2
                )

                if not np.isnan(spread):
                    all_spread_results[alg_name].append(spread)


print("===== Tổng hợp Spread Delta =====")

for alg_name, values in all_spread_results.items():
    values = np.array(values)

    print(f"\n{alg_name}")
    print("Số giá trị hợp lệ:", len(values))
    print("Mean Spread:", np.mean(values))
    print("Std Spread:", np.std(values))
    print("Min Spread:", np.min(values))
    print("Max Spread:", np.max(values))

print("===== Tổng hợp Spread Delta =====")

for alg_name, values in all_spread_results.items():
    values = np.array(values)

    print(f"\n{alg_name}")
    print("Số giá trị hợp lệ:", len(values))

    if len(values) == 0:
        print("Không có giá trị hợp lệ.")
        continue

    print("Mean Spread:", np.mean(values))
    print("Std Spread:", np.std(values))
    print("Min Spread:", np.min(values))
    print("Max Spread:", np.max(values))


# ===== VẼ BIỂU ĐỒ =====

import matplotlib.pyplot as plt

alg_names = list(all_spread_results.keys())

mean_values = []
std_values = []

for alg_name in alg_names:
    values = np.array(all_spread_results[alg_name])

    if len(values) == 0:
        mean_values.append(np.nan)
        std_values.append(np.nan)
    else:
        mean_values.append(np.mean(values))
        std_values.append(np.std(values))


# Bar chart Mean ± Std
plt.figure(figsize=(8, 6))

x = np.arange(len(alg_names))
colors = ["red", "blue", "green"]

plt.bar(
    x,
    mean_values,
    yerr=std_values,
    capsize=6,
    width=0.5,
    color=colors
)

plt.xticks(x, alg_names, fontsize=12)
plt.ylabel("Spread Delta", fontsize=13)
plt.xlabel("Algorithm", fontsize=13)
plt.title("Overall Spread Delta Comparison\nLower is Better", fontsize=14)
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()


# Boxplot
box_data = []
box_labels = []

for alg_name in alg_names:
    values = np.array(all_spread_results[alg_name])

    if len(values) > 0:
        box_data.append(values)
        box_labels.append(alg_name)

plt.figure(figsize=(8, 6))

bp = plt.boxplot(
    box_data,
    labels=box_labels,
    patch_artist=True
)

for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)

plt.ylabel("Spread Delta", fontsize=13)
plt.xlabel("Algorithm", fontsize=13)
plt.title("Distribution of Spread Delta over All Datasets\nLower is Better", fontsize=14)
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()