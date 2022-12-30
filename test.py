import numpy as np
from numpy.typing import NDArray

import random
import time

from numba import int32, float32
from numba.experimental import jitclass

import matplotlib.pyplot as plt


Genome = NDArray[np.int32]
Weights = NDArray[np.float32]
Population = list['Individual']



@jitclass([
              ("genome", int32[:]),
              ("size", int32),
              ("fitness", float32),
              ("cached_fitness", float32),
              ("fitness_penalty", float32)
          ])
class Individual:
    def __init__(self, weights: Weights, genome: Genome | None=None, initialization_k: int | None=None) -> None:
        if genome is not None:
            self.genome = genome
        elif initialization_k is not None:
            self.genome = self.random_genome(weights, initialization_k)

        self.size = self.genome.size

        self.fitness = self.calc_fitness(weights)

        self.cached_fitness = self.fitness
        self.fitness_penalty = 1.0


    def random_genome(self, weights: Weights, k: int) -> Genome:
        size = weights.shape[0]
        genome = np.empty(size, dtype=np.int32)
        sample_size = k

        valid = np.arange(size, dtype=np.int32)
        genome[0] = np.random.randint(size)

        for j in range(1, size):
            previous = genome[j - 1]
            
            valid = np.delete(valid, np.where(valid == previous)[0].astype(np.int32))
            
            candidates = np.random.choice(
                valid, min(sample_size, valid.size), replace=False
            )

            best = np.argmin(weights[previous][candidates])
            next = candidates[best]

            genome[j] = next

        return genome


    def calc_fitness(self, weights: Weights) -> float:
        result = 0.0
        for (start, end) in zip(self.genome, np.roll(self.genome, -1)):
            result += weights[start][end]

        return result


    def mutate(self, weights: Weights, rate: float):
        if random.random() < rate:
            size = int(random.normalvariate(self.size // 10, self.size // 20))
            start = random.randint(0, self.size - size - 1)
            end = start + size
    
            np.random.shuffle(self.genome[start:end])
    
            self.fitness = self.calc_fitness(weights)


    @staticmethod
    def combine(parent1: 'Individual', parent2: 'Individual', weights: Weights) -> 'Individual':
        child_genome = np.full_like(parent1.genome, -1)
        start, end = np.random.choice(parent1.size, 2, replace=False)

        idx_to_fill = None
        if start <= end:
            child_genome[start:end] = parent1.genome[start:end]
            idx_to_fill = np.arange(start + parent1.size - end)
            idx_to_fill[-parent1.size + end :] = np.arange(end, parent1.size)
        else:
            child_genome[start:parent1.size] = parent1.genome[start:parent1.size]
            child_genome[0:end] = parent1.genome[0:end]
            idx_to_fill = np.arange(end, start)

        last_idx = 0
        for j in idx_to_fill:
            for n in range(last_idx, parent1.size):
                if parent2.genome[n] not in child_genome:
                    child_genome[j] = parent2.genome[n]
                    last_idx = n + 1
                    break

        return Individual(weights, child_genome, None)


    # def local_search(self, weights: Weights) -> None:
    #     best_fitness = self.fitness
    #     best_swap = self.genome.copy()
    #    
    #     N = 8
    #     for start in range(self.size - N):
    #         backup = self.genome[start:start+N].copy()
    #        
    #         for _ in range(2):
    #             np.random.shuffle(self.genome[start:start+N])
    #             new_fitness = self.calc_fitness(weights)
    #             if new_fitness < best_fitness:
    #                 best_swap = self.genome.copy()
    #                 best_fitness = new_fitness
    #
    #         self.genome[start:start+N] = backup
    #
    #     self.genome = best_swap
    #     self.fitness = best_fitness

    def local_search(self, weights: Weights):
        best_fitness = self.fitness
        best_swap = 0, 0

        for i in range(self.size):
            for j in range(self.size):
                if i == j:
                    continue

                self.genome[i], self.genome[j] = self.genome[j], self.genome[i]
                new_fitness = self.calc_fitness(weights)
                if new_fitness < best_fitness:
                    best_fitness = new_fitness
                    best_swap = i, j

                self.genome[i], self.genome[j] = self.genome[j], self.genome[i]

        self.genome[best_swap[0]], self.genome[best_swap[1]] = self.genome[best_swap[1]], self.genome[best_swap[0]]
        self.fitness = best_fitness


    def cache_fitness(self):
        self.cached_fitness = self.fitness


    def restore_fitness(self):
        self.fitness = self.cached_fitness
        self.fitness_penalty = 1.0


    def update_fitness_penalty(self, distance: int, sigma: float, alpha: float):
        if distance < sigma:
            self.fitness_penalty += 1.0 - (distance / sigma) ** alpha

        self.fitness = self.cached_fitness * self.fitness_penalty


    def distance(self, other: 'Individual') -> int:
        distance = 0
        for i in range(self.size):
            for j in range(other.size):
                if self.genome[i] == other.genome[j]:
                    distance += self.genome[(i + 1) % self.size] != other.genome[(j + 1) % other.size]
                    break

        return distance


    def standard_repr(self):
        i = np.where(self.genome == 0)[0].astype(np.int32)[0]
        return np.roll(self.genome, -i)


class Algorithm:
    def __init__(
            self,
            weights: Weights,
            lam: int, 
            mu: int,
            crossover_k: int,
            elimination_k: int,
            initialization_k: int,
            mutation_rate: float,
            fitness_sharing_sigma: float,
            fitness_sharing_alpha: float
        ) -> None:
        
        self.weights = weights

        self.lam = lam
        self.mu = mu
        self.crossover_k = crossover_k
        self.elimination_k = elimination_k
        self.initialization_k = initialization_k

        self.mutation_rate = mutation_rate

        self.fitness_sharing_sigma = fitness_sharing_sigma
        self.fitness_sharing_alpha = fitness_sharing_alpha


    def initialize(self) -> Population:
        population: Population = []
        for _ in range(self.lam):
            population.append(Individual(self.weights, initialization_k=self.initialization_k))

        return population


    @staticmethod
    def k_tournament(population: Population, k: int, exclude: Individual | None=None) -> Individual:
        candidates = random.sample(population, k)
        if exclude is not None and exclude in candidates:
            candidates.remove(exclude)

        return min(candidates, key=lambda x: x.fitness)


    def crossover(self, population: Population) -> Population:
        new_population = population.copy()

        for _ in range(self.mu):
            parent1 = Algorithm.k_tournament(population, self.crossover_k)
            parent2 = Algorithm.k_tournament(population, self.crossover_k, exclude=parent1)
            
            child = Individual.combine(parent1, parent2, self.weights)
            new_population.append(child)

        return new_population


    def mutate_inplace(self, population: Population) -> None:
        for individual in population:
            individual.mutate(self.weights, self.mutation_rate)


    def local_search_inplace(self, population: Population) -> None:
        for individual in population:
            individual.local_search(self.weights)


    def elimination(self, population: Population) -> Population:
        def update_penalties(winner: Individual):
            nonlocal population
            for individual in random.sample(population, len(population) // 15):
                individual.update_fitness_penalty(
                    individual.distance(winner),
                    self.fitness_sharing_sigma,
                    self.fitness_sharing_alpha
                )

        survivors: Population = []

        for individual in population:
            individual.cache_fitness()

        elite = self.best_individual(population)
        survivors.append(elite)
        population.remove(elite)

        update_penalties(elite)
        
        for _ in range(self.lam):
            winner = self.k_tournament(population, self.elimination_k)
        
            survivors.append(winner)
            population.remove(winner)

            update_penalties(winner)

        for individual in survivors:
            individual.restore_fitness()

        return survivors


    def run(self, population: Population) -> Population:
        population = self.crossover(population)
        self.mutate_inplace(population)

        self.local_search_inplace(population)
        population = self.elimination(population)

        return population


    def best_individual(self, population):
        return min(population, key=lambda x: x.fitness)


    def average_fitness(self, population: Population):
        fitnesses = [x.fitness for x in population]
        return sum(fitnesses) / len(fitnesses)



def load_tour(file_path: str) -> Weights:
    with open(file_path, "r") as file:
        return np.loadtxt(file, delimiter=",", dtype=np.float32)


def plot(runtime_data):

    plt.plot([(x[1:]) for x in runtime_data])
    plt.legend(["best", "average"])
    plt.show()



def convergence_criterion(max: int):
    prev_best = 0
    num_same = 0

    def func(best_fitness: float):
        nonlocal num_same, prev_best
        num_same = num_same + 1 if prev_best == best_fitness else 0
        prev_best = best_fitness
        return num_same < max

    return func


def optimize(tour: str):
    weights = load_tour(tour)
    size = weights.shape[0]

    lam = 200
    mu = 400

    algorithm = Algorithm(
        weights,
        lam, mu,
        lam // 10, (lam + mu) // 4,
        int(size * 0.8),
        0.05,
        int(size * 0.05), 0.5
    )

    start_time = time.time()
    population = algorithm.initialize()
    total_time = time.time() - start_time

    runtime_data = []
    i = 0

    best = algorithm.best_individual(population)
    best_fit = best.fitness
    average_fit = algorithm.average_fitness(population)
    runtime_data.append((0.0, best_fit, average_fit, best.standard_repr().tolist()))

    criterion = convergence_criterion(20)

    while total_time < 5 * 60 and criterion(best_fit):

        population = algorithm.run(population)

        best = algorithm.best_individual(population)
        best_fit = best.fitness
        average_fit = algorithm.average_fitness(population)
        runtime_data.append((total_time, best_fit, average_fit, best.standard_repr().tolist()))

        total_time = time.time() - start_time
        print(f"It: {i} Average: {average_fit:.4f} Best: {best_fit:.4f} Time: {total_time: .2f}")
        i += 1

    return runtime_data


def save_to_csv(file_name: str, runtime_data):
    with open(file_name, "w+") as file:
        file.write('"iteration","time","best-fitness","average-fitness","best-genome"\n')
        for i, (time, best_fit, average_fit, best_genome) in enumerate(runtime_data):
            file.write(f'"{i}","{time}","{best_fit}","{average_fit}","{best_genome}"\n')


def save_runs():
    for tour in [50, 100, 250, 500, 750, 1000]:
        for i in range(30):
            print(f"Tour {tour}, iteration: {i}")
            runtime_data = optimize(f"./tours/tour{tour}.csv")
            save_to_csv(f"runtime-data/tour{tour}/run{i}.csv", runtime_data)


optimize("./tours/tour50.csv")
