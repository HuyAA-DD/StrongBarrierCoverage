import numpy as np
import pandas as pd
from pygmo import hypervolume

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


def calculate_final_hv(values, z_nad):
    """
    Tính Hypervolume ở generation cuối cùng.
    Mỗi generation gồm pop_size cá thể.
    """

    start_idx = (max_generation - 1) * pop_size
    end_idx = max_generation * pop_size

    final_front = values[start_idx:end_idx]

    # Nếu file không đủ 10000 generation thì lấy pop_size cá thể cuối cùng
    if len(final_front) == 0:
        final_front = values[-pop_size:]

    normalized_front = []

    for point in final_front:
        f1 = point[0] / z_nad[0]
        f2 = point[1] / z_nad[1]

        # Chỉ giữ điểm hợp lệ với reference point [1, 1]
        if f1 <= 1 and f2 <= 1:
            normalized_front.append([f1, f2])

    if len(normalized_front) == 0:
        return 0.0

    hv = hypervolume(normalized_front)
    return hv.compute([1, 1])


def calculate_hv_statistics(file_list):
    hv_values = []

    for file_name in file_list:
        try:
            z_nad, values = load_data(file_name)
            hv = calculate_final_hv(values, z_nad)
            hv_values.append(hv)

        except Exception as e:
            print(f"Lỗi file: {file_name}")
            print(e)

    hv_values = np.array(hv_values)

    mean_hv = np.mean(hv_values)
    std_hv = np.std(hv_values)
    best_hv = np.max(hv_values)
    worst_hv = np.min(hv_values)

    return mean_hv, std_hv, best_hv, worst_hv


def build_files(algorithm, dts):
    datasets = [f"{dts}_{i}" for i in range(num_subdatasets)]

    files = [
        f"./result/f/{algorithm}/{algorithm}_{dataset}_{run}.csv"
        for dataset in datasets
        for run in range(num_runs)
    ]

    return files


results = []

for dts in dts_list:
    moead_files = build_files("moead", dts)
    nsga_files = build_files("nsga", dts)
    nspso_files = build_files("nspso", dts)

    moead_mean, moead_std, moead_best, moead_worst = calculate_hv_statistics(moead_files)
    nsga_mean, nsga_std, nsga_best, nsga_worst = calculate_hv_statistics(nsga_files)
    nspso_mean, nspso_std, nspso_best, nspso_worst = calculate_hv_statistics(nspso_files)

    results.append({
        "Dataset": dts,

        "MOEA/D Mean": moead_mean,
        "MOEA/D Std": moead_std,
        "MOEA/D Best": moead_best,
        "MOEA/D Worst": moead_worst,

        "NSGA-II Mean": nsga_mean,
        "NSGA-II Std": nsga_std,
        "NSGA-II Best": nsga_best,
        "NSGA-II Worst": nsga_worst,

        "NSPSO Mean": nspso_mean,
        "NSPSO Std": nspso_std,
        "NSPSO Best": nspso_best,
        "NSPSO Worst": nspso_worst,
    })


df = pd.DataFrame(results)

print(df)

df.to_csv("hv_summary_by_dataset.csv", index=False)