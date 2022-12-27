import numpy as np
from numpy.typing import NDArray

import random
import time

from numba import int32, float32
from numba.experimental import jitclass

import matplotlib.pyplot as plt


Genome = NDArray[np.int32]
Weights = NDArray[np.float32]
Distance = np.float32
Distances = NDArray[Distance]
Population = list['Individual']



@jitclass([
              ("genome", int32[:]),
              ("size", int32),
              ("fitness", float32),
          ])
class Individual:
    def __init__(self, weights: Weights, genome: Genome | None=None, initialization_k: int | None=None) -> None:
        if genome is not None:
            self.genome = genome
        elif initialization_k is not None:
            self.genome = self.random_genome(weights, initialization_k)
        self.size = self.genome.size
        self.fitness = self.calc_fitness(weights)


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


    def mutate(self, weights: Weights, rate: float) -> None:
        while np.random.rand() < rate:
            size = np.random.randint(self.size // 5)
            start = np.random.randint(self.size - size - 1) + 1
            end = start + size

            self.genome[start:end] = self.genome[start:end][::-1]

        self.fitness = self.calc_fitness(weights)


    @staticmethod
    def combine(parent1: 'Individual', parent2: 'Individual', weights: Weights) -> 'Individual':
        parent1.genome = np.roll(parent1.genome, np.random.randint(parent1.size))
        parent2.genome = np.roll(parent2.genome, np.random.randint(parent2.size))

        child_genome = np.full_like(parent1.genome, -1)

        start = random.randint(0, parent1.size - 2)
        size = random.randint(start, parent1.size - 1)
        end = start + size

        child_genome[start:end] = parent1.genome[start:end]

        last_idx = 0
        for i in [i for i in range(child_genome.size) if i < start or i >= end]:
            for j in range(last_idx, parent2.size):
                if parent2.genome[j] in child_genome:
                    continue

                child_genome[i] = parent2.genome[j]
                last_idx = j
                break

        return Individual(weights, child_genome, None)


    def local_search(self, weights: Weights, rate: float, attempts: int) -> None:
        if np.random.rand() < rate:
            best_fitness = self.fitness
            best_swap = (0, 0)
    
            for _ in range(attempts):
                size = np.random.randint(self.size // 5)
                start = np.random.randint(self.size - size - 1) + 1
                end = start + size
    
                self.genome[start:end] = self.genome[start:end][::-1]
                new_fitness = self.calc_fitness(weights)
    
                if new_fitness < best_fitness:
                    best_fitness = new_fitness
                    best_swap = (start, end)
    
                self.genome[start:end] = self.genome[start:end][::-1]
    
            self.genome[best_swap[0]:best_swap[1]] = self.genome[best_swap[0]:best_swap[1]][::-1]
            self.fitness = best_fitness


    def share_fitness(self, distances: Distances, sigma: float, alpha: float):
        one_plus_beta = 1.0

        for dist in distances:
            if dist >= sigma:
                continue

            one_plus_beta += 1.0 - (dist / sigma) ** alpha

        self.fitness *= one_plus_beta


    def distance(self, other: 'Individual'):
        distance = 0
        for i in range(self.size):
            for j in range(other.size):
                if self.genome[i] == other.genome[j]:
                    distance += self.genome[(i + 1) & self.size] != other.genome[(j + 1) % other.size]
                    # TODO: test break speedup
                    # break



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
            local_search_rate: float,
            local_search_attempts: int
        ) -> None:
        
        self.weights = weights

        self.lam = lam
        self.mu = mu
        self.crossover_k = crossover_k
        self.elimination_k = elimination_k
        self.initialization_k = initialization_k

        self.mutation_rate = mutation_rate
        
        self.local_search_rate = local_search_rate
        self.local_search_attempts = local_search_attempts


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
            parent1 = self.k_tournament(population, self.crossover_k)
            parent2 = self.k_tournament(population, self.crossover_k, exclude=parent1)
            
            if random.random() < 0.1:
                population.remove(parent1)

            if random.random() < 0.1:
                population.remove(parent2)

            child = Individual.combine(parent1, parent2, self.weights)
            new_population.append(child)

        return new_population


    def mutate_inplace(self, population: Population) -> None:
        for individual in population:
            individual.mutate(self.weights, self.mutation_rate)


    def local_search_inplace(self, population: Population) -> None:
        for individual in population:
            individual.local_search(self.weights, self.local_search_rate, self.local_search_attempts)


    def elimination(self, population: Population) -> Population:
        survivors: Population = []

        for _ in range(self.lam):
            winner = self.k_tournament(population, self.elimination_k)
        
            survivors.append(winner)
            population.remove(winner)

        return survivors


    def __call__(self, population: Population) -> Population:
        population = self.crossover(population)
        self.mutate_inplace(population)

        self.local_search_inplace(population)
        population = self.elimination(population)

        return population




def load_tour(file_path: str) -> Weights:
    with open(file_path, "r") as file:
        return np.loadtxt(file, delimiter=",", dtype=np.float32)


def optimize():
    weights = load_tour("./tours/tour750.csv")
    size = weights.shape[0]

    lam = 600
    mu = 1000

    algorithm = Algorithm(
        weights,
        lam, mu,
        lam // 20, (lam + mu) // 20,
        int(size * 0.5),
        0.1,
        5, 80
    )

    population = algorithm.initialize()

    plot_data = []
    for i in range(250):
        t1 = time.time_ns()

        population = algorithm(population)

        best = min(population, key=lambda x: x.fitness)
        fitnesses = [x.fitness for x in population]
        average_fit = sum(fitnesses) / len(fitnesses)
        plot_data.append((best.fitness, average_fit))

        t2 = time.time_ns()
        print(f"It: {i} Average: {average_fit:.4f} Best: {best.fitness:.4f} Time: {(t2 - t1) * 1e-9:.2f}")

    plt.plot(plot_data)
    plt.legend(["best", "average"])
    plt.show()



optimize()
