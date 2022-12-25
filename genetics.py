import os

os.environ["NUMBA_PARALLEL_DIAGNOSTICS"] = "4"


import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import Reporter


@njit(nogil=True)
def initialize(weights, lam):
    size = weights.shape[0]
    population = np.empty((lam, size), dtype=np.int32)
    sample_size = lam // 20

    for i in range(lam):
        valid = np.arange(size, dtype=np.int32)
        population[i][0] = np.random.randint(size)

        valid = np.delete(
            valid, np.where(valid == population[i][0])[0].astype(np.int32)
        )

        for j in range(1, size):
            candidates = np.random.choice(
                valid, min(sample_size, valid.size), replace=False
            )
            best = np.argmin(weights[population[i][j - 1]][candidates])
            next = candidates[best]

            valid = np.delete(valid, np.where(valid == next)[0].astype(np.int32))
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
def k_tournament(range, objective, k):
    idx = np.random.choice(range, k, replace=False)
    best = np.argmin(objective[idx])
    return idx[best]  # type: ignore


@njit(nogil=True)
def crossover(population, fitnesses, mu, k):
    path_length = population[0].size
    results = np.full((mu, path_length), -1, dtype=np.int32)

    for i in range(mu):
        p1_idx = k_tournament(population.shape[0], fitnesses, k)
        p2_idx = k_tournament(population.shape[0], fitnesses, k)

        parent1 = population[p1_idx]
        parent2 = population[p2_idx]

        result = results[i]
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

    return results



@njit(nogil=True)
def mutation(population, rate):
    length = population.shape[1]

    for individual in population:
        while np.random.rand() < rate:
            size = np.random.randint(0.2 * length)
            start = np.random.randint(length - size - 1) + 1
            end = start + size

            if start == 0:
                individual[start:end] = individual[end-1::-1]
            else:
                individual[start:end] = individual[end-1:start-1:-1]



@njit(nogil=True)
def path_distance(perm1, perm2):
    length = perm1.size

    distance = 0
    for i in range(length + 1):
        for j in range(length):
            if perm1[i] != perm2[j]:
                continue

            if perm1[(i + 1) % length] != perm2[(j + 1) % length]:
                distance += 1

    return distance



@njit(nogil=True)
def fitness_sharing(population, fitnesses, alpha, sigma):
    lam = population.shape[0]
    new_fitnesses = np.empty_like(fitnesses)

    for i in range(lam):
        one_plus_beta = 1
        for j in np.random.choice(lam, lam // 5, replace=False): #type: ignore
            if i == j or fitnesses[i] > fitnesses[j]:
                continue

            dist = path_distance(population[i], population[j])
            if dist >= sigma:
                continue

            one_plus_beta += 1 - (dist / sigma) ** alpha

        new_fitnesses[i] = fitnesses[i] * one_plus_beta

    return new_fitnesses


# @njit()
# def fitness_sharing(candidates, selected, fitnesses, alpha, sigma):
#     new_fitnesses = np.empty_like(fitnesses)
#
#     for i in range(candidates.shape[0]):
#         one_plus_beta = 1.0
#         for j in np.random.choice(selected.shape[0], selected.shape[0] // 5, replace=False): #type: ignore
#             dist = path_distance(candidates[i], selected[j])
#             if dist >= sigma:
#                 continue
#
#             one_plus_beta += 1.0 - (dist / sigma) ** alpha
#
#         new_fitnesses[i] = fitnesses[i] * one_plus_beta
#
#     return new_fitnesses


@njit(nogil=True)
def elimination(population, fitnesses, lam, k):
    result = np.empty((lam, population.shape[1]), dtype=np.int32)

    available = np.arange(population.shape[0])
    for i in range(lam):
        chosen_idx = k_tournament(available.shape[0], fitnesses, k)
        result[i] = population[available[chosen_idx]]
        available = np.delete(available, chosen_idx)

    return result


# @njit(nogil=True)
# def fitness_sharing_elimination(population, fitnesses, lam, k, alpha, sigma):
#     result = np.empty((lam, population.shape[1]), dtype=np.int32)
#
#     available = np.arange(population.shape[0], dtype=np.int32)
#     for i in range(lam):
#         new_fitnesses = fitness_sharing(population[available], result[:i], fitnesses[available], alpha, sigma)
#
#         chosen_idx = k_tournament(available.shape[0], new_fitnesses , k)
#         result[i] = population[available[chosen_idx]]
#         available = np.delete(available, chosen_idx)
#
#     return result


@njit(nogil=True)
def local_search(weights, population, rate, attempts):
    for individual in population:
        if np.random.rand() < rate:
            best_fitness = fitness(weights, individual)
            best_swap = (0, 0)

            for _ in range(attempts):
                idx1, idx2 = np.random.choice(individual.size, 2, replace=False)
                individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
                new_fitness = fitness(weights, individual)
                
                if new_fitness < best_fitness:
                    best_fitness = new_fitness
                    best_swap = (idx1, idx2)

                individual[idx1], individual[idx2] = individual[idx2], individual[idx1]

            individual[best_swap[0]], individual[best_swap[1]] = individual[best_swap[1]], individual[best_swap[0]]



@njit(nogil=True)
def ga_iteration(weights, population, mu, lam, k1, k2, mutation_rate):
    initial_fitnesses = population_fitness(weights, population)

    # initial_fitnesses = fitness_sharing(population, initial_fitnesses, 2, lam / 20)
    offspring = crossover(population, initial_fitnesses, mu, k1)

    total = np.concatenate((population, offspring))
    local_search(weights, total, 0.3, 20)
    mutation(total, mutation_rate)
    final_fitnesses = population_fitness(weights, total)

    new_population = elimination(total, final_fitnesses, lam, k2)

    return new_population



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
    lam = 600
    mu = 1000
    selection_k = 16
    elimination_k = 18
    mutation_rate = 0.4

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
    optimize("./tours/tour50.csv")
