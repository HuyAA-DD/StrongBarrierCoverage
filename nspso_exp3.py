import argparse
import math
import time
import random
import os
import numpy as np

try:
    from gpu_utils import xp, has_gpu
except Exception:
    xp = np
    has_gpu = False


# =========================================================
# Argument
# =========================================================
parser = argparse.ArgumentParser()
parser.add_argument("dataset", nargs="?", default="300_0")
parser.add_argument("--gpu", action="store_true")
args = parser.parse_args()

dataset = args.dataset
USE_GPU = args.gpu and has_gpu

input_file = f"./dataset/{dataset}.txt"
print("-nspso-real-position", dataset, "backend=GPU" if USE_GPU else "backend=CPU")

x_corr = np.loadtxt(input_file, dtype=int)
num_sensor = len(x_corr)


# =========================================================
# Parameters
# =========================================================
INF = 10**18

pop_size = 32
max_gen = 10000

run_start = 0
run_end = 1

k = 2
k_minus_1 = k - 1
beta = 1
gamma = 0.5
barrier_length = 1000

# NSPSO parameters
c1 = 2.05
c2 = 2.05
phi = c1 + c2

if phi <= 4:
    raise ValueError("Constriction factor requires c1 + c2 > 4.")

chi = 2.0 / abs(2.0 - phi - math.sqrt(phi**2 - 4.0 * phi))

# Vì position nằm trong [0,1], v_max không nên quá lớn
v_max = 0.3

# mutation trên position
mutation_prob = 0.02
mutation_sigma = 0.1

eval_cache = {}


# =========================================================
# Strong Barrier Coverage Model
# =========================================================
def exp_approx_beta(term):
    beta_term = beta * term
    return 1 - beta_term + (beta_term**2) / 2 - (beta_term**3) / 6


def radius_formalize_outermost_sensor(x1, isFirst=True):
    if not isFirst:
        x1 = barrier_length - x1

    for r_u1_val in range(1, x1 + 1):
        term = x1 - k_minus_1 * r_u1_val

        if exp_approx_beta(term) >= gamma:
            return r_u1_val

    return x1


def radius_formalize_sensor(r_u1, x1, x2):
    if x1 + r_u1 * k_minus_1 >= x2:
        return 0

    k_minus_1_r_u1 = k_minus_1 * r_u1
    x1_certain = None

    for x in range(x1 + r_u1, x2 + 1):
        if exp_approx_beta(x - x1 - k_minus_1_r_u1) >= gamma:
            x1_certain = x
            break

    if x1_certain is None or x1_certain >= x2:
        return 0

    x1_certain = math.floor(x1_certain)

    r_u2_max = (x2 - x1_certain) / (k - 1)
    r_u2_min = (x2 - x1_certain) / k

    for r_u2_val in range(math.ceil(r_u2_min), math.ceil(r_u2_max) + 1):
        k_minus_1_r_u2 = k_minus_1 * r_u2_val

        x_min = (x1 + x2 - (k - 1) * (r_u1 - r_u2_val)) / 2

        expr_val = (
            exp_approx_beta(x_min - x1 - k_minus_1_r_u1)
            + exp_approx_beta(x2 - x_min - k_minus_1_r_u2)
            - gamma
        )

        if expr_val >= 0:
            return r_u2_val

    return 0


def radius_formalize(gene):
    active_idx = np.flatnonzero(gene).tolist()

    if not active_idx:
        return [0] * num_sensor

    r_u = [radius_formalize_outermost_sensor(x_corr[active_idx[0]])]
    r_0_count = 0

    for i in range(1, len(active_idx)):
        prev_idx = active_idx[i - 1 - r_0_count]
        curr_idx = active_idx[i]

        r_temp = radius_formalize_sensor(
            r_u[i - 1 - r_0_count],
            x_corr[prev_idx],
            x_corr[curr_idx],
        )

        if r_temp == 0:
            r_0_count += 1
        else:
            r_0_count = 0

        r_u.append(r_temp)

    r_last = radius_formalize_outermost_sensor(
        x_corr[active_idx[-1]],
        isFirst=False,
    )

    if r_last > r_u[-1]:
        r_u[-1] = r_last

    all_r_u = [0] * num_sensor

    for idx, rv in zip(active_idx, r_u):
        all_r_u[idx] = rv

    return all_r_u


def calc_energy_consumption(r_u):
    term1 = 0.5 * (k_minus_1 * r_u) ** 2
    exp_beta = math.exp(-beta * r_u)
    term2 = (1 - exp_beta * (1 + beta * r_u)) / (beta**2)
    term3 = (k_minus_1 * r_u * (1 - exp_beta)) / beta

    return term1 + term2 + term3


def evaluate(gene):
    key = tuple(int(x) for x in gene)
    cached = eval_cache.get(key)

    if cached is not None:
        return cached

    all_r_u = radius_formalize(gene)

    total_active_sensor = int(np.sum(gene))

    if USE_GPU:
        arr = xp.asarray(all_r_u, dtype=xp.float64)

        term1 = 0.5 * (k_minus_1 * arr) ** 2
        exp_beta = xp.exp(-beta * arr)
        term2 = (1 - exp_beta * (1 + beta * arr)) / (beta**2)
        term3 = (k_minus_1 * arr * (1 - exp_beta)) / beta

        energy_arr = term1 + term2 + term3
        total_energy_consumption = float(xp.sum(energy_arr[arr > 0]))

    else:
        total_energy_consumption = 0.0

        for r in all_r_u:
            if r > 0:
                total_energy_consumption += calc_energy_consumption(r)

    result = (total_active_sensor, total_energy_consumption)
    eval_cache[key] = result

    return result


# =========================================================
# Particle
# =========================================================
class Particle:
    __slots__ = (
        "position",
        "vel",
        "gene",
        "f1",
        "f2",
        "rank",
        "crowding_distance",
        "pbest_position",
        "pbest_gene",
        "pbest_f1",
        "pbest_f2",
        "pbest_rank",
    )

    def __init__(self):
        self.position = None
        self.vel = None
        self.gene = None

        self.f1 = None
        self.f2 = None

        self.rank = None
        self.crowding_distance = 0.0

        self.pbest_position = None
        self.pbest_gene = None
        self.pbest_f1 = None
        self.pbest_f2 = None
        self.pbest_rank = None


class FitnessItem:
    __slots__ = ("f1", "f2", "rank", "crowding_distance")

    def __init__(self, f1, f2):
        self.f1 = f1
        self.f2 = f2
        self.rank = None
        self.crowding_distance = 0.0


def copy_particle(p):
    q = Particle()

    q.position = p.position.copy()
    q.vel = p.vel.copy()
    q.gene = p.gene.copy()

    q.f1 = p.f1
    q.f2 = p.f2

    q.rank = p.rank
    q.crowding_distance = p.crowding_distance

    q.pbest_position = p.pbest_position.copy()
    q.pbest_gene = p.pbest_gene.copy()
    q.pbest_f1 = p.pbest_f1
    q.pbest_f2 = p.pbest_f2
    q.pbest_rank = p.pbest_rank

    return q


# =========================================================
# Decode real position -> binary gene
# =========================================================
def decode_position(position):
    gene = (position >= 0.5).astype(np.int8)

    if np.sum(gene) == 0:
        idx = np.argmax(position)
        gene[idx] = 1

    return gene


# =========================================================
# Multi-objective utilities
# =========================================================
def dominates_obj(f1a, f2a, f1b, f2b):
    return (
        (f1a <= f1b and f2a < f2b)
        or
        (f1a < f1b and f2a <= f2b)
    )


def non_dominated_rank(population):
    n = len(population)
    ranks = [0] * n

    dominating_list = [[] for _ in range(n)]
    dominated_count = [0 for _ in range(n)]

    for i in range(n):
        pi = population[i]

        for j in range(i + 1, n):
            pj = population[j]

            if dominates_obj(pi.f1, pi.f2, pj.f1, pj.f2):
                dominating_list[i].append(j)
                dominated_count[j] += 1

            elif dominates_obj(pj.f1, pj.f2, pi.f1, pi.f2):
                dominating_list[j].append(i)
                dominated_count[i] += 1

    current_rank = 0
    current_front = [i for i in range(n) if dominated_count[i] == 0]

    while current_front:
        next_front = []

        for i in current_front:
            ranks[i] = current_rank

            for j in dominating_list[i]:
                dominated_count[j] -= 1

                if dominated_count[j] == 0:
                    next_front.append(j)

        current_front = next_front
        current_rank += 1

    return ranks


def get_fronts(population):
    ranks = non_dominated_rank(population)
    fronts_dict = {}

    for ind, rank in zip(population, ranks):
        ind.rank = rank
        fronts_dict.setdefault(rank, []).append(ind)

    return [fronts_dict[r] for r in sorted(fronts_dict.keys())]


def calc_crowding_distance(front):
    n = len(front)

    if n == 0:
        return

    if n <= 2:
        for ind in front:
            ind.crowding_distance = INF
        return

    for ind in front:
        ind.crowding_distance = 0.0

    sorted_f1 = sorted(front, key=lambda x: x.f1)
    sorted_f1[0].crowding_distance = INF
    sorted_f1[-1].crowding_distance = INF

    f1_min = sorted_f1[0].f1
    f1_max = sorted_f1[-1].f1

    if f1_max != f1_min:
        denom = f1_max - f1_min

        for i in range(1, n - 1):
            sorted_f1[i].crowding_distance += (
                sorted_f1[i + 1].f1 - sorted_f1[i - 1].f1
            ) / denom

    sorted_f2 = sorted(front, key=lambda x: x.f2)
    sorted_f2[0].crowding_distance = INF
    sorted_f2[-1].crowding_distance = INF

    f2_min = sorted_f2[0].f2
    f2_max = sorted_f2[-1].f2

    if f2_max != f2_min:
        denom = f2_max - f2_min

        for i in range(1, n - 1):
            sorted_f2[i].crowding_distance += (
                sorted_f2[i + 1].f2 - sorted_f2[i - 1].f2
            ) / denom


def assign_rank_and_crowding(population):
    fronts = get_fronts(population)

    for front in fronts:
        calc_crowding_distance(front)

    return fronts


def better_by_rank_and_crowding(a, b):
    if a.rank < b.rank:
        return a

    if b.rank < a.rank:
        return b

    if a.crowding_distance > b.crowding_distance:
        return a

    if b.crowding_distance > a.crowding_distance:
        return b

    return a if random.random() < 0.5 else b


def rank_based_selection(population):
    a, b = random.sample(population, 2)
    return better_by_rank_and_crowding(a, b)


def environmental_selection(combined_population, pop_size):
    fronts = assign_rank_and_crowding(combined_population)

    new_population = []

    for front in fronts:
        if len(new_population) + len(front) <= pop_size:
            new_population.extend(front)

        else:
            front.sort(key=lambda x: x.crowding_distance, reverse=True)

            remain = pop_size - len(new_population)
            new_population.extend(front[:remain])
            break

    assign_rank_and_crowding(new_population)

    return new_population


# =========================================================
# NSPSO operators
# =========================================================
def mutate_position(position):
    position = position.copy()

    mask = np.random.rand(num_sensor) < mutation_prob

    if np.any(mask):
        noise = np.random.normal(0.0, mutation_sigma, size=num_sensor)
        position[mask] += noise[mask]
        position = np.clip(position, 0.0, 1.0)

    return position


def init_swarm(pop_size):
    swarm = []

    for _ in range(pop_size):
        p = Particle()

        p.position = np.random.rand(num_sensor)
        p.vel = np.random.uniform(-v_max, v_max, size=num_sensor)

        p.gene = decode_position(p.position)
        p.f1, p.f2 = evaluate(p.gene)

        p.pbest_position = p.position.copy()
        p.pbest_gene = p.gene.copy()
        p.pbest_f1 = p.f1
        p.pbest_f2 = p.f2
        p.pbest_rank = 0

        swarm.append(p)

    assign_rank_and_crowding(swarm)

    return swarm


def choose_gbest(front0):
    return random.choice(front0)


def create_offspring(parent, gbest):
    child = copy_particle(parent)

    r1 = np.random.rand(num_sensor)
    r2 = np.random.rand(num_sensor)

    x = child.position
    pbest = child.pbest_position
    leader = gbest.position

    child.vel = chi * (
        child.vel
        + c1 * r1 * (pbest - x)
        + c2 * r2 * (leader - x)
    )

    child.vel = np.clip(child.vel, -v_max, v_max)

    child.position = child.position + child.vel

    low_mask = child.position < 0.0
    high_mask = child.position > 1.0

    if np.any(low_mask):
        child.position[low_mask] = 0.0
        child.vel[low_mask] = -random.random() * v_max

    if np.any(high_mask):
        child.position[high_mask] = 1.0
        child.vel[high_mask] = random.random() * (-v_max)

    child.position = mutate_position(child.position)

    child.gene = decode_position(child.position)

    child.f1, child.f2 = evaluate(child.gene)

    return child


def update_pbest_by_rank(swarm):
    temp_items = []
    pair_refs = []

    for p in swarm:
        current_item = FitnessItem(p.f1, p.f2)
        pbest_item = FitnessItem(p.pbest_f1, p.pbest_f2)

        temp_items.append(current_item)
        temp_items.append(pbest_item)

        pair_refs.append((p, current_item, pbest_item))

    assign_rank_and_crowding(temp_items)

    for p, current_item, pbest_item in pair_refs:
        should_update = False

        if current_item.rank < pbest_item.rank:
            should_update = True

        elif current_item.rank == pbest_item.rank:
            if current_item.crowding_distance > pbest_item.crowding_distance:
                should_update = True

            elif current_item.crowding_distance == pbest_item.crowding_distance:
                should_update = random.random() < 0.5

        if should_update:
            p.pbest_position = p.position.copy()
            p.pbest_gene = p.gene.copy()
            p.pbest_f1 = p.f1
            p.pbest_f2 = p.f2
            p.pbest_rank = current_item.rank


# =========================================================
# Output
# =========================================================
def ensure_dirs():
    os.makedirs("./result/r/nspso", exist_ok=True)
    os.makedirs("./result/f/nspso", exist_ok=True)
    os.makedirs("./result/pareto/nspso", exist_ok=True)


def write_results(run, population, history_f, elapsed_time):
    final_front = assign_rank_and_crowding(population)[0]

    with open(f"./result/r/nspso/nspso_{dataset}_{run}.txt", "w") as file:
        for p in final_front:
            r = radius_formalize(p.gene)
            file.write(f"{r}\n")

    with open(f"./result/f/nspso/nspso_{dataset}_{run}.csv", "w") as file:
        if history_f:
            z1 = max(pair[0] for pair in history_f)
            z2 = max(pair[1] for pair in history_f)

            file.write(f"{z1} {z2}\n")

            for f1, f2 in history_f:
                file.write(f"{f1} {f2}\n")
        else:
            file.write("0 0\n")

    with open("./result/nspso_time.csv", "a") as file:
        file.write(f"{pop_size}, {max_gen}, {dataset}, {run}, {elapsed_time}\n")

    with open(f"./result/pareto/nspso/nspso_{dataset}_{run}.csv", "w") as file:
        for p in final_front:
            file.write(f"{p.f1}, {p.f2}\n")


# =========================================================
# Main
# =========================================================
def main():
    ensure_dirs()

    for run in range(run_start, run_end):
        output_path = f"./result/r/nspso/nspso_{dataset}_{run}.txt"

        # if os.path.exists(output_path):
        #     print("Skip", dataset, run)
        #     continue

        print("-nspso-real-position run", run)

        random.seed(run)
        np.random.seed(run)
        eval_cache.clear()

        time_start = time.time()

        history_f = []

        population = init_swarm(pop_size)

        for generation in range(max_gen):
            if (generation + 1) % 100 == 0:
                print("-nspso-real-position", generation + 1)

            fronts = assign_rank_and_crowding(population)
            front0 = fronts[0]

            offspring = []

            for _ in range(pop_size):
                parent = rank_based_selection(population)
                gbest = choose_gbest(front0)

                child = create_offspring(parent, gbest)
                offspring.append(child)

            combined_population = population + offspring

            population = environmental_selection(combined_population, pop_size)

            update_pbest_by_rank(population)

            history_f.extend((p.f1, p.f2) for p in population)

        elapsed_time = time.time() - time_start

        write_results(run, population, history_f, elapsed_time)


if __name__ == "__main__":
    main()