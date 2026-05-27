import numpy as np
import pandas as pd
from pygmo import hypervolume
import matplotlib.pyplot as plt

pop_size = 32
max_generation = 10000

dts_list = [100, 150, 200, 250, 300]
num_subdatasets = 10
num_runs = 10


def load_data(file_name):
    data = pd.read_csv(file_name, header=None, sep=r"\s+")
    z_nad = data.iloc[0].to_numpy().astype(float)
    values = data.iloc[1:].values.astype(float)
    return z_nad, values


def calculate_hv(values, z_nad):
    hv_values = []

    for generation in range(max_generation):
        start_idx = generation * pop_size
        end_idx = start_idx + pop_size

        front = values[start_idx:end_idx]

        if len(front) == 0:
            hv_values.append(np.nan)
            continue

        normalized_front = []

        for point in front:
            f1 = point[0] / z_nad[0]
            f2 = point[1] / z_nad[1]

            # Chỉ giữ điểm nằm trong reference point [1, 1]
            if f1 <= 1 and f2 <= 1:
                normalized_front.append([f1, f2])

        if len(normalized_front) == 0:
            hv_values.append(0.0)
        else:
            hv = hypervolume(normalized_front)
            hv_values.append(hv.compute([1, 1]))

    return hv_values


def calculate_statistics(filenames):
    all_hv_values = []

    for file_name in filenames:
        try:
            z_nad, values = load_data(file_name)
            hv_values = calculate_hv(values, z_nad)
            all_hv_values.append(hv_values)
        except Exception as e:
            print(f"Error reading file: {file_name}")
            print(e)

    all_hv_values = np.array(all_hv_values, dtype=float)

    mean_hv = np.nanmean(all_hv_values, axis=0)
    std_hv = np.nanstd(all_hv_values, axis=0)

    return mean_hv, std_hv


def build_files(algorithm):
    files = []

    for dts in dts_list:
        datasets = [f"{dts}_{i}" for i in range(num_subdatasets)]

        for dataset in datasets:
            for run in range(num_runs):
                files.append(
                    f"./result/f/{algorithm}/{algorithm}_{dataset}_{run}.csv"
                )

    return files


moead_files = build_files("moead")
nsga_files = build_files("nsga")
nspso_files = build_files("nspso")

mean_hv_moead, std_hv_moead = calculate_statistics(moead_files)
mean_hv_nsga, std_hv_nsga = calculate_statistics(nsga_files)
mean_hv_nspso, std_hv_nspso = calculate_statistics(nspso_files)


plt.figure(figsize=(12, 6))

x = range(max_generation)

plt.plot(
    mean_hv_moead,
    label="MOEA/D Mean",
    color="red",
    linestyle="--",
    linewidth=2
)
plt.fill_between(
    x,
    mean_hv_moead - std_hv_moead,
    mean_hv_moead + std_hv_moead,
    color="lightcoral",
    alpha=0.3
)

plt.plot(
    mean_hv_nsga,
    label="NSGA-II Mean",
    color="blue",
    linestyle="-",
    linewidth=2
)
plt.fill_between(
    x,
    mean_hv_nsga - std_hv_nsga,
    mean_hv_nsga + std_hv_nsga,
    color="lightblue",
    alpha=0.3
)

plt.plot(
    mean_hv_nspso,
    label="NSPSO Mean",
    color="green",
    linestyle="-",
    linewidth=2
)
plt.fill_between(
    x,
    mean_hv_nspso - std_hv_nspso,
    mean_hv_nspso + std_hv_nspso,
    color="lightgreen",
    alpha=0.3
)

plt.xlabel("Generation", fontsize=14)
plt.ylabel("Hypervolume", fontsize=14)
plt.title("Overall Hypervolume Convergence over All Datasets", fontsize=16)
plt.legend(fontsize=14, loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()