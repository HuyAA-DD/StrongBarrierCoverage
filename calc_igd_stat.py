import numpy as np
import pandas as pd

run_start = 0
run_end = 10
num_subdatasets = 10

base_link = "./result/pareto"
datasets = ["100", "150", "200", "250", "300"]


def calculate_igd(approx_front, reference_front):
    """
    IGD: với mỗi điểm trong reference front,
    tìm khoảng cách gần nhất đến approximate front.
    IGD càng nhỏ càng tốt.
    """
    distances = []

    for point in reference_front:
        min_distance = np.min(
            np.linalg.norm(approx_front - point, axis=1)
        )
        distances.append(min_distance)

    return np.mean(distances)


def normalize(data, min_vals, max_vals):
    denom = max_vals - min_vals
    denom[denom == 0] = 1.0
    return (data - min_vals) / denom


def load_data(file):
    data = pd.read_csv(file, header=None, delimiter=",")
    data.drop_duplicates(inplace=True)
    return data.values.astype(float)


def calculate_statistics(values):
    values = np.array(values)

    mean_value = np.mean(values)
    std_value = np.std(values)

    # Với IGD: càng nhỏ càng tốt
    best_value = np.min(values)
    worst_value = np.max(values)

    return mean_value, std_value, best_value, worst_value


results = []

for dataset in datasets:
    print(f"Processing dataset {dataset}...")

    moead_igd_values = []
    nsga_igd_values = []
    nspso_igd_values = []

    for i in range(num_subdatasets):
        true_pareto_front = load_data(
            f"{base_link}/approx/{dataset}_{i}.csv"
        )

        min_vals = np.min(true_pareto_front, axis=0)
        max_vals = np.max(true_pareto_front, axis=0)

        norm_true_pareto_front = normalize(
            true_pareto_front,
            min_vals,
            max_vals
        )

        for run in range(run_start, run_end):
            try:
                moead_pareto_front = load_data(
                    f"{base_link}/moead/moead_{dataset}_{i}_{run}.csv"
                )

                nsga_pareto_front = load_data(
                    f"{base_link}/nsga/nsga_{dataset}_{i}_{run}.csv"
                )

                nspso_pareto_front = load_data(
                    f"{base_link}/nspso/nspso_{dataset}_{i}_{run}.csv"
                )

                norm_moead_pareto_front = normalize(
                    moead_pareto_front,
                    min_vals,
                    max_vals
                )

                norm_nsga_pareto_front = normalize(
                    nsga_pareto_front,
                    min_vals,
                    max_vals
                )

                norm_nspso_pareto_front = normalize(
                    nspso_pareto_front,
                    min_vals,
                    max_vals
                )

                moead_igd = calculate_igd(
                    norm_moead_pareto_front,
                    norm_true_pareto_front
                )

                nsga_igd = calculate_igd(
                    norm_nsga_pareto_front,
                    norm_true_pareto_front
                )

                nspso_igd = calculate_igd(
                    norm_nspso_pareto_front,
                    norm_true_pareto_front
                )

                moead_igd_values.append(moead_igd)
                nsga_igd_values.append(nsga_igd)
                nspso_igd_values.append(nspso_igd)

            except Exception as e:
                print(f"Lỗi ở dataset={dataset}, subdataset={i}, run={run}")
                print(e)

    moead_mean, moead_std, moead_best, moead_worst = calculate_statistics(
        moead_igd_values
    )

    nsga_mean, nsga_std, nsga_best, nsga_worst = calculate_statistics(
        nsga_igd_values
    )

    nspso_mean, nspso_std, nspso_best, nspso_worst = calculate_statistics(
        nspso_igd_values
    )

    results.append({
        "Dataset": dataset,

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

print("\n===== IGD summary by dataset =====")
print(df)

df.to_csv("igd_summary_by_dataset.csv", index=False)