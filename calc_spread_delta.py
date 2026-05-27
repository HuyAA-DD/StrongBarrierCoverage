import numpy as np
import matplotlib.pyplot as plt

run_start = 0
run_end = 10
algo = "moead"
base_link = f"./result/pareto/{algo}/{algo}_"
datasets = ["100_4", "150_4", "200_4", "250_4", "300_4"]

def euclidean_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

def spread_metric(pareto_front, extreme_point_1, extreme_point_2):
    """
    Tính Spread (Delta) metric chuẩn cho bài toán 2 mục tiêu.
    Đầu vào đã phải được chuẩn hóa (normalized) từ trước.
    """
    if len(pareto_front) < 3:
        return np.nan

    # Sort theo f1 (Chỉ đúng với 2 mục tiêu)
    pareto_front = pareto_front[np.argsort(pareto_front[:, 0])]

    # Tính các khoảng cách liên tiếp d_i
    distances = np.array([
        euclidean_distance(pareto_front[i], pareto_front[i + 1])
        for i in range(len(pareto_front) - 1)
    ])

    d_avg = np.mean(distances)

    # Tính khoảng cách tới các extreme points của True/Reference Front (d_f, d_l)
    # Vì đã sort theo f1: 
    # pareto_front[0] là điểm có f1 nhỏ nhất
    # pareto_front[-1] là điểm có f1 lớn nhất
    d_f = euclidean_distance(extreme_point_1, pareto_front[0])
    d_l = euclidean_distance(extreme_point_2, pareto_front[-1])

    # Công thức Spread (Delta) chuẩn của Deb
    sum_diff = np.sum(np.abs(distances - d_avg))
    spread = (d_f + d_l + sum_diff) / (d_f + d_l + (len(pareto_front) - 1) * d_avg)

    return spread

# --- MAIN EXECUTION ---
spread_values_per_dataset = []

for dataset in datasets:
    # BƯỚC 1: Đọc toàn bộ dữ liệu của dataset này để tìm Global Min/Max và Extreme Points
    all_solutions = []
    fronts_raw = []
    
    for run in range(run_start, run_end):
        input_file = f"{base_link}{dataset}_{run}.csv"
        try:
            front = np.loadtxt(input_file, dtype=float, delimiter=",")
            if front.ndim == 1:
                front = front.reshape(1, -1)
            front = np.unique(front, axis=0)
            fronts_raw.append(front)
            all_solutions.append(front)
        except OSError:
            print(f"Warning: File {input_file} not found. Skipping.")
            fronts_raw.append(np.array([]))
            
    if not all_solutions:
        spread_values_per_dataset.append([])
        continue

    # Gom tất cả các điểm lại để tìm biên giới (Reference Front)
    all_solutions_matrix = np.vstack(all_solutions)
    
    global_min = all_solutions_matrix.min(axis=0)
    global_max = all_solutions_matrix.max(axis=0)
    denom = global_max - global_min
    denom[denom == 0] = 1.0 # Tránh chia cho 0

    # Lấy Extreme points (đã chuẩn hóa) dựa trên toàn bộ các điểm thu được
    # Ở bài toán 2 mục tiêu, extreme point 1: f1 min, f2 max. Extreme point 2: f1 max, f2 min.
    # Ta sort toàn bộ dữ liệu theo f1 để lấy 2 đầu mút
    all_solutions_sorted = all_solutions_matrix[np.argsort(all_solutions_matrix[:, 0])]
    
    ext_1_raw = all_solutions_sorted[0]  # f1 nhỏ nhất
    ext_2_raw = all_solutions_sorted[-1] # f1 lớn nhất
    
    # Chuẩn hóa extreme points
    ext_1_norm = (ext_1_raw - global_min) / denom
    ext_2_norm = (ext_2_raw - global_min) / denom

    # BƯỚC 2 & 3: Chuẩn hóa từng front và tính Spread Delta
    spread_values = []
    
    for front in fronts_raw:
        if len(front) == 0:
            continue
            
        # Chuẩn hóa front theo GLOBAL min/max
        front_norm = (front - global_min) / denom
        
        # Tính Spread
        spread_value = spread_metric(front_norm, ext_1_norm, ext_2_norm)
        
        if not np.isnan(spread_value):
            spread_values.append(spread_value)

    spread_values_per_dataset.append(spread_values)

# --- VẼ BIỂU ĐỒ ---
plt.boxplot(
    spread_values_per_dataset,
    labels=["100", "150", "200", "250", "300"],
    patch_artist=True
)

plt.ylabel("Spread ($\Delta$) Value")
plt.xlabel("Number of Sensors")
plt.title("Spread Metric over 10 Runs (Lower is Better)")
plt.show()