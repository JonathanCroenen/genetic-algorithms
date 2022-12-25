import numpy as np
from numpy.typing import NDArray

import random
import time

from numba import int32, float32
from numba.experimental import jitclass



Genome = NDArray[np.int32]
Weights = NDArray[np.float32]
Population = list['Individual']



@jitclass([
              ("genome", int32[:]),
              ("size", int32),
              ("fitness", float32),
              ("mutation_rate", float32),
              ("local_search_rate", float32),
              ("local_search_attempts", float32)
          ])
class Individual:
    def __init__(self, weights: Weights, initial_genome: Genome | None=None) -> None:
        if initial_genome is not None:
            self.genome = initial_genome
        else:
            self.genome = self.initialize(weights)

        self.size = self.genome.size

        self.fitness = self.calc_fitness(weights)

        self.mutation_rate = 0.1
        
        self.local_search_rate = 0.3
        self.local_search_attempts = 10


    def initialize(self, weights: Weights):
        size = weights.shape[0]
        genome = np.empty(size, dtype=np.int32)
        sample_size = size // 20

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


    def calc_fitness(self, weights: Weights):
        result = 0.0
        for (start, end) in zip(self.genome, np.roll(self.genome, -1)):
            result += weights[start][end]

        return result


    def mutate(self, weights: Weights):
        while np.random.rand() < self.mutation_rate:
            size = np.random.randint(self.size // 5)
            start = np.random.randint(self.size - size - 1) + 1
            end = start + size

            self.genome[start:end] = self.genome[start:end][::-1]

        self.fitness = self.calc_fitness(weights)


    @staticmethod
    def combine(parent1: 'Individual', parent2: 'Individual', weights: Weights):
        child_genome = np.full_like(parent1.genome, -1)

        start = random.randint(0, parent1.size - 2)
        size = random.randint(start, parent1.size - 1)
        end = start + size

        child_genome[start:end] = parent1.genome[start:end]

        last_idx = 0
        for i in (i for i in range(child_genome.size) if i < start or i >= end):
            for j in range(last_idx, parent2.size):
                if parent2.genome[j] in child_genome:
                    continue

                child_genome[i] = parent2.genome[j]
                last_idx = j
                break

        return Individual(weights, initial_genome=child_genome)


    def local_search(self, weights: Weights):
        if np.random.rand() < self.local_search_rate:
            best_fitness = self.fitness
            best_swap = (0, 0)

            for _ in range(self.local_search_attempts):
                idx1, idx2 = np.random.choice(self.size, 2, replace=False)
                self.genome[idx1], self.genome[idx2] = self.genome[idx2], self.genome[idx1]
                new_fitness = self.calc_fitness(weights)
                
                if new_fitness < best_fitness:
                    best_fitness = new_fitness
                    best_swap = (idx1, idx2)

                self.genome[idx1], self.genome[idx2] = self.genome[idx2], self.genome[idx1]

            self.genome[best_swap[0]], self.genome[best_swap[1]] = self.genome[best_swap[1]], self.genome[best_swap[0]]

            self.fitness = self.calc_fitness(weights)


        
def k_tournament(population: Population, k: int, exclude: Individual | None=None):
    if exclude is not None:
        population = list(filter(lambda x: x is not exclude, population))

    candidates = random.sample(population, k)
    return min(candidates, key=lambda x: x.fitness)


def crossover(population: Population, weights: Weights, mu: int, k: int):
    new_population = population.copy()

    for _ in range(mu):
        parent1 = k_tournament(population, k)
        parent2 = k_tournament(population, k, exclude=parent1)

        child = Individual.combine(parent1, parent2, weights)
        new_population.append(child)

    return new_population


def mutate_inplace(population: Population, weights: Weights):
    for individual in population:
        individual.mutate(weights)


def local_search_inplace(population: Population, weights: Weights):
    for individual in population:
        individual.local_search(weights)


def elimination(population: Population, lam: int, k: int):
    survivors: Population = []

    for _ in range(lam):
        winner = k_tournament(population, k)
    
        survivors.append(winner)
        population.remove(winner)

    return survivors


def algorithm():
    with open("./tours/tour50.csv", "r") as file:
        weights = np.loadtxt(file, delimiter=",", dtype=np.float32)

    lam = 500
    mu = 800
    crossover_k = lam // 20
    elimination_k = (lam + mu) // 20


    population: Population = []
    for _ in range(lam):
        population.append(Individual(weights))

    for i in range(1000):
        t1 = time.time_ns()

        population = crossover(population, weights, mu, crossover_k)
        mutate_inplace(population, weights)

        population = elimination(population, lam, elimination_k)
        local_search_inplace(population, weights)

        best = min(population, key=lambda x: x.fitness)

        t2 = time.time_ns()
        print("It:", i, "Best:", best.fitness, "Time:", (t2 - t1) * 12-6)


