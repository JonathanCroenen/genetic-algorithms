import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import Reporter


@njit(nogil=True)
def initialize(weights, lam):
    size = weights.shape[0]
    population = np.empty((lam, size), dtype=np.int32)
    sample_size = weights.shape[0] // 2

    for i in range(lam):
        valid = np.arange(size, dtype=np.int32)
        population[i][0] = np.random.randint(size)

        for j in range(1, size):
            previous = population[i][j - 1]
            valid = np.delete(valid, np.where(valid == previous)[0])

            candidates = np.random.choice(
                valid, min(sample_size, valid.size), replace=False
            )
            best = np.argmin(weights[population[i][j - 1]][candidates])
            next = candidates[best]

            population[i][j] = next

    return population


@njit(nogil=True)
def fitness(weights, individual):
    result = 0.0
    for (start, end) in zip(individual, np.roll(individual, -1)):
        result += weights[start][end]

    return result


@njit(nogil=True)
def population_fitness(weights, population):
    result = np.empty(population.shape[0], dtype=np.float32)
    for i, individual in enumerate(population):
        result[i] = fitness(weights, individual)

    return result


@njit(nogil=True)
def k_tournament(available, objective, k):
    idx = np.random.choice(available, k, replace=False)
    best = np.argmin(objective[idx])
    return idx[best]  # type: ignore


@njit(nogil=True)
def combine(parent1, parent2, path_length, result):
    start, end = np.random.choice(path_length, 2, replace=False)

    idx_to_fill = None
    if start <= end:
        result[start:end] = parent1[start:end]
        idx_to_fill = np.arange(start + path_length - end)
        idx_to_fill[-path_length + end :] = np.arange(end, path_length)
    else:
        result[start:path_length] = parent1[start:path_length]
        result[0:end] = parent1[0:end]
        idx_to_fill = np.arange(end, start)

    last_idx = 0
    for j in idx_to_fill:
        for n in range(last_idx, path_length):
            if parent2[n] not in result:
                result[j] = parent2[n]
                last_idx = n + 1
                break




@njit(nogil=True)
def crossover(population, fitnesses, mu, k):
    path_length = population[0].size
    new_population = np.full((population.shape[0] + mu, path_length), -1, dtype=np.int32)
    new_population[:population.shape[0]] = population

    available = np.arange(population.shape[0], dtype=np.int32)
    for i in range(mu):
        p1_idx = k_tournament(available.shape[0], fitnesses[available], k)
        p2_idx = k_tournament(available.shape[0], fitnesses[available], k)

        parent1 = population[available[p1_idx]]
        parent2 = population[available[p2_idx]]

        result = new_population[i + population.shape[0]]
        combine(parent1, parent2, path_length, result)

    return new_population



@njit(nogil=True)
def mutation(population, rate):
    length = population.shape[1]

    for individual in population:
        while np.random.rand() < rate:
            start = np.random.randint(length - 1)
            size = np.random.randint(length - start - 1)
            end = start + size

            individual[start:end] = individual[start:end][::-1]


@njit(nogil=True)
def path_distance(perm1, perm2):
    length = perm1.size

    distance = 0
    for i in range(length):
        for j in range(length):
            if perm1[i] == perm2[j]:
                distance += perm1[(i + 1) % length] != perm2[(j + 1) % length]

    return distance


@njit(nogil=True)
def calculate_required_distances(population, candidates, chosen, out_distance_matrix):
    for i in candidates:
        out_distance_matrix[i][chosen] = path_distance(population[i], population[chosen])


@njit(nogil=True)
def fitness_sharing(candidates_idx, selected_idx, fitnesses, distances, alpha, sigma, out_fitnesses):
    for i in candidates_idx:
        one_plus_beta = 1.0
        for j in selected_idx:
            dist = distances[i][j]
            if dist >= sigma:
                continue

            one_plus_beta += 1.0 - (dist / sigma) ** alpha

        out_fitnesses[i] = fitnesses[i] * one_plus_beta



@njit(nogil=True)
def elimination(population, fitnesses, lam, k):
    result = np.empty((lam, population.shape[1]), dtype=np.int32)

    available = np.arange(population.shape[0])
    for i in range(lam):
        chosen_idx = k_tournament(available.shape[0], fitnesses, k)
        result[i] = population[available[chosen_idx]]
        available = np.delete(available, chosen_idx)

    return result


@njit(nogil=True)
def fitness_sharing_elimination(population, fitnesses, lam, k, alpha, sigma):
    result = np.empty((lam, population.shape[1]), dtype=np.int32)
    out_fitnesses = np.empty_like(fitnesses)

    distance_matrix = np.empty((population.shape[0], population.shape[0]), dtype=np.float32)

    candidates = np.arange(population.shape[0], dtype=np.int32)
    chosen = np.empty(lam, dtype=np.int32)
    for i in range(lam):
        fitness_sharing(candidates, chosen[:i], fitnesses, distance_matrix , alpha, sigma, out_fitnesses)

        chosen_idx = k_tournament(candidates.shape[0],  out_fitnesses, k)
        chosen[i] = chosen_idx
        result[i] = population[candidates[chosen_idx]]
        candidates = np.delete(candidates, chosen_idx)

        calculate_required_distances(population, candidates, chosen_idx, distance_matrix)

    return result


@njit(nogil=True)
def local_search(weights, population, attempts):
    for individual in population:
        best_fitness = fitness(weights, individual)
        best_swap = (0, 0)

        for _ in range(attempts):
            start = np.random.randint(individual.size - 1)
            size = np.random.randint(individual.size - start - 1)
            end = start + size

            individual[start:end] = individual[start:end][::-1]
            new_fitness = fitness(weights, individual)
           
            if new_fitness < best_fitness:
                best_fitness = new_fitness
                best_swap = start, end
           
            individual[start:end] = individual[start:end][::-1]

        individual[best_swap[0]:best_swap[1]] = individual[best_swap[0]:best_swap[1]][::-1]



@njit(nogil=True)
def ga_iteration(weights, population, mu, lam, selection_k, elimination_k, mutation_rate):
    fitnesses = population_fitness(weights, population)

    population = crossover(population, fitnesses, mu, selection_k)
    mutation(population, mutation_rate)

    local_search(weights, population, int(0.7 * weights.shape[0]))
    fitnesses = population_fitness(weights, population)

    # population = elimination(population, fitnesses, lam, elimination_k)
    population = fitness_sharing_elimination(
        population,
        fitnesses,
        lam, 
        elimination_k,
        0.5, 
        int(0.2 * weights.shape[0])
    )
    
    return population



def get_stall_criterion(n):
    repeats = n
    prev_fitness = -1

    def func(fitness):
        nonlocal repeats, prev_fitness, n
        if fitness == prev_fitness:
            repeats -= 1
        else:
            repeats = n

        prev_fitness = fitness

        return repeats > 0

    return func



def optimize(filename):
    lam = 150
    mu = 300
    selection_k = lam // 20
    elimination_k = (lam + mu) // 20
    mutation_rate = 0.1

    file = open(filename)
    weights = np.loadtxt(file, delimiter=",", dtype=np.float32)
    file.close()

    reporter = Reporter.Reporter("output.txt")

    population = initialize(weights, lam)

    mean_objective = np.inf
    best_objective = np.inf
    best_solution = None
    iteration = 0
    time = 5 * 60
    plot_data = []

    stall_criterion = get_stall_criterion(20)
    while (
        np.nan_to_num(mean_objective / best_objective, nan=np.inf) >= 1.001
        and stall_criterion(best_objective)
        or iteration < 100
    ):
        population = ga_iteration(weights, population, mu, lam, selection_k, elimination_k, mutation_rate)
        fitnesses = population_fitness(weights, population)

        mean_objective = np.mean(fitnesses)
        best_objective = np.min(fitnesses)
        best_solution = population[np.argmin(fitnesses)]


        time_left = reporter.report(
            mean_objective, best_objective, best_solution
        )
        if time_left < 0:
            break

        print("It: ", iteration, "Mean: ", mean_objective, "Best: ", best_objective, "Time: ", time - time_left)
        time = time_left

        plot_data.append((mean_objective, best_objective))

        iteration += 1

    print(
        "Mean: ", mean_objective, "Best: ", best_objective, "Solution: ", best_solution
    )

    plt.plot(plot_data)
    plt.legend(["mean objective", "best objective"])
    plt.show()


if __name__ == "__main__":
    optimize("./tours/tour500.csv")
