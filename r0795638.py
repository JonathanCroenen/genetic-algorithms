import numpy as np
from numpy.typing import NDArray

import random

from numba import int32, float32
from numba.experimental import jitclass

import Reporter


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
        genome[0] = random.randint(0, size - 1)

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


    def __swap_fitness(self, weights: Weights, i: int, j: int):
        temp1 = weights[self.genome[i - 1]][self.genome[i]]
        temp2 = weights[self.genome[i]][self.genome[i + 1]]

        temp3 = weights[self.genome[j - 1]][self.genome[j]]
        temp4 = weights[self.genome[j]][self.genome[j + 1]]

        temp5 = weights[self.genome[i - 1]][self.genome[j]]
        temp6 = weights[self.genome[j]][self.genome[i + 1]]

        temp7 = weights[self.genome[j - 1]][self.genome[i]]
        temp8 = weights[self.genome[i]][self.genome[j + 1]]


        return self.fitness - temp1 - temp2 - temp3 - temp4 + temp5 + temp6 + temp7 + temp8


    def __swap_adjacent_fitness(self, weights: Weights, i: int):
        j = i + 1
    
        temp1 = weights[self.genome[i - 1]][self.genome[i]]
        temp2 = weights[self.genome[i]][self.genome[j]] 
        temp3 = weights[self.genome[j]][self.genome[j + 1]]

        temp4 = weights[self.genome[i - 1]][self.genome[j]]
        temp5 = weights[self.genome[j]][self.genome[i]]

        temp6 = weights[self.genome[i]][self.genome[j + 1]]

        return self.fitness - temp1 - temp2 - temp3 + temp4 + temp5 + temp6


    def local_search(self, weights: Weights):
        if self.fitness == np.inf:
            return

        best_fitness = self.fitness
        best_swap = 0, 0

        for i in range(1, self.size - 1):
            for j in range(i + 1, self.size - 1):
                if abs(i - j) == 1:
                    new_fitness = self.__swap_adjacent_fitness(weights, min(i, j))
                else:
                    new_fitness = self.__swap_fitness(weights, i, j)

                if new_fitness < best_fitness:
                    best_fitness = new_fitness
                    best_swap = i, j

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
        edges = set()
        for edge in zip(self.genome, np.roll(self.genome, 1)):
            edges.add(edge)

        distance = 0
        for edge in zip(other.genome, np.roll(other.genome, 1)):
            distance += 0 if edge in edges else 1

        return distance


class Algorithm:
    def __init__(
            self,
            weights: Weights,
            lam: int, 
            mu: int,
            initialization_k: int,
            crossover_k: int,
            elimination_k: int,
            mutation_rate: float,
            fitness_sharing_sigma: float,
            fitness_sharing_alpha: float
        ) -> None:
        
        self.weights = weights

        self.lam = lam
        self.mu = mu

        self.initialization_k = initialization_k
        self.crossover_k = crossover_k
        self.elimination_k = elimination_k

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
            for individual in random.sample(population, len(population) // 5):
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


    def best_individual(self, population) -> Individual:
        return min(population, key=lambda x: x.fitness)


    def average_fitness(self, population: Population) -> float:
        fitnesses = [x.fitness for x in population]
        return sum(fitnesses) / len(fitnesses)




class r0795638:
    def __init__(self) -> None:
        self.reporter = Reporter.Reporter(self.__class__.__name__)

        self.prev_best = 0.0
        self.num_not_improved = 0


    @staticmethod
    def load_tour(file_path: str) -> Weights:
        with open(file_path, "r") as file:
            return np.loadtxt(file, delimiter=",", dtype=np.float32)


    def convergence_criterion(self, max: int, best_fitness: float) -> bool:
        self.num_not_improved  = self.num_not_improved + 1 if self.prev_best / best_fitness - 1 < 0.001 else 0
        self.prev_best = best_fitness
        return self.num_not_improved < max


    @staticmethod
    def standard_repr(individual: Individual) -> Genome:
        i = np.where(individual.genome == 0)[0].astype(np.int32)[0]
        return np.roll(individual.genome, -i)


    def configure_algorithm(self, tour: str, log: bool) -> Algorithm:
        weights = self.load_tour(tour)
        size = weights.shape[0]

        lam = int(-0.333 * (size - 1000) + 200)
        mu = 2 * lam

        config = {
            "lam": lam, 
            "mu": mu,
            "initialization_k": int(size * 0.8),
            "crossover_k": lam // 20,
            "elimination_k": (lam + mu) // 4,
            "mutation_rate": 0.05,
            "fitness_sharing_sigma": int(size * 0.15),
            "fitness_sharing_alpha": 4
        }

        if log:
            print(f"Current config: {config}")

        return Algorithm(weights, **config)


    def optimize(self, tour: str, log=False) -> Genome:
        algorithm = self.configure_algorithm(tour, log)
        population = algorithm.initialize()

        best = algorithm.best_individual(population)
        best_fitness = best.fitness
        average_fitness = algorithm.average_fitness(population)
        time_left = self.reporter.report(average_fitness, best_fitness, self.standard_repr(best))

        while self.convergence_criterion(15, best_fitness) and time_left > 0:
            population = algorithm.run(population)

            best = algorithm.best_individual(population)
            best_fitness = best.fitness
            average_fitness = algorithm.average_fitness(population)
            time_left = self.reporter.report(average_fitness, best_fitness, self.standard_repr(best))
            
            if log:
                print(f"Average: {average_fitness:.4f} Best: {best_fitness:.4f} Time left: {time_left: .2f}")

        return self.standard_repr(best)



if __name__ == "__main__":
    best_genome = r0795638().optimize("./tours/tour1000.csv", log=True)
    print(best_genome)


