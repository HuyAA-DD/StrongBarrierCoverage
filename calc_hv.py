import numpy as np
import matplotlib.pyplot as plt

pop_size = 32
max_generation = 10000
dts = 200 #gốc là 250
datasets = [f"{dts}_{i}" for i in range(10)] #gốc là range 10
num_runs = 10


def load_data(file_name):
    data = np.loadtxt(file_name, dtype=float)
    z_nad = data[0]
    aaa = data[1:].tolist()
    return z_nad, aaa


def calculate_hv(aaa, z_nad):
    def hypervolume_2d(points, reference_point=(1.0, 1.0)):
        if not points:
            return 0.0

        ref_x, ref_y = reference_point
        filtered_points = [
            (x, y)
            for x, y in points
            if x <= ref_x and y <= ref_y
        ]
        if not filtered_points:
            return 0.0

        filtered_points.sort(key=lambda point: (point[0], point[1]))

        frontier = []
        best_y = float("inf")
        for x, y in filtered_points:
            if y < best_y:
                frontier.append((x, y))
                best_y = y

        hv = 0.0
        for index, (x, y) in enumerate(frontier):
            next_x = frontier[index + 1][0] if index + 1 < len(frontier) else ref_x
            hv += max(0.0, next_x - x) * max(0.0, ref_y - y)
        return hv

    hv_value = []
    generation_count = min(max_generation, len(aaa) // pop_size)
    for generation in range(generation_count):
        bbb = []
        for j in range(pop_size):
            idx = generation * pop_size + j
            if idx < len(aaa):
                bbb.append([aaa[idx][0] / z_nad[0], aaa[idx][1] / z_nad[1]])
        hv_value.append(hypervolume_2d(bbb, (1.0, 1.0)))
    return hv_value


def calculate_statistics(filenames):
    all_hv_values = []
    for file_name in filenames:
        z_nad, aaa = load_data(file_name)
        hv_values = calculate_hv(aaa, z_nad)
        all_hv_values.append(hv_values)

    all_hv_values = np.array(all_hv_values)
    mean_hv = np.mean(all_hv_values, axis=0)
    std_hv = np.std(all_hv_values, axis=0)
    return mean_hv, std_hv


moead_files = [
    f"./result/f/moead/moead_{dataset}_{run}.csv"
    for dataset in datasets
    for run in range(num_runs)
]


nsga_files = [
    f"./result/f/nsga/nsga_{dataset}_{run}.csv"
    for dataset in datasets
    for run in range(num_runs)
]


nspso_files = [
    f"./result/f/nspso/nspso_{dataset}_{run}.csv"
    for dataset in datasets
    for run in range(num_runs)
]

mean_hv_moead, std_hv_moead = calculate_statistics(moead_files)
mean_hv_nsga, std_hv_nsga = calculate_statistics(nsga_files)
mean_hv_nspso, std_hv_nspso = calculate_statistics(nspso_files)


# Plot the results for all three algorithms
plt.figure(figsize=(12, 6))
generation_count = min(len(mean_hv_moead), len(mean_hv_nsga), len(mean_hv_nspso))
generation_axis = range(generation_count)

plt.plot(mean_hv_moead[:generation_count], label="MOEA/D Mean", color="red", linestyle="--", linewidth=2)
plt.fill_between(
    generation_axis,
    mean_hv_moead[:generation_count] - std_hv_moead[:generation_count],
    mean_hv_moead[:generation_count] + std_hv_moead[:generation_count],
    color="lightcoral",
    alpha=0.3,
)

plt.plot(mean_hv_nsga[:generation_count], label="NSGA-II Mean", color="blue", linestyle="-", linewidth=2)
plt.fill_between(
    generation_axis,
    mean_hv_nsga[:generation_count] - std_hv_nsga[:generation_count],
    mean_hv_nsga[:generation_count] + std_hv_nsga[:generation_count],
    color="lightblue",
    alpha=0.3,
)

plt.plot(mean_hv_nspso[:generation_count], label="NSPSO Mean", color="green", linestyle=":", linewidth=2)
plt.fill_between(
    generation_axis,
    mean_hv_nspso[:generation_count] - std_hv_nspso[:generation_count],
    mean_hv_nspso[:generation_count] + std_hv_nspso[:generation_count],
    color="lightgreen",
    alpha=0.3,
)


plt.xlabel("Generation", fontsize=14)
plt.ylabel("Hyper Volume", fontsize=14)
plt.legend(fontsize=20)
plt.legend(loc="lower right")
output_path = "./result/compare_metrics/hv.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Saved hypervolume plot to {output_path}")
