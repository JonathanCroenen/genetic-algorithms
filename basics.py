import Reporter
import numpy as np
import csv

# Modify the class name to match your student number.
class Basics:
    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)
        self.lam = 200
        self.mu = 400
        self.k = 4
        self.mutation_rate = 0.5

    def initialize(self, weights):
        size = weights.shape[0]
        population = np.empty((self.lam, size), dtype=np.int64)
        sample_size = 50

        for i in range(self.lam):
            valid = np.arange(size, dtype=np.int64)
            population[i][0] = np.random.randint(size)
            valid = np.delete(valid, np.where(valid == population[i][0]))

            for j in range(1, size):
                candidates = np.random.choice(
                    valid, min(sample_size, valid.size), replace=False
                )
                best = np.argmin(weights[population[i][j - 1]][candidates])
                next = candidates[best]

                valid = np.delete(valid, np.where(valid == next))
                population[i][j] = next

        for i in range(10):
            self.mutation(population)

        return population

    def fitness(self, weights, population):
        result = np.zeros(population.shape[0], dtype=np.int64)
        for i, individual in enumerate(population):
            for (start, end) in zip(individual, np.roll(individual, -1)):
                result[i] += weights[start][end]

        return result

    def _kTournament(self, population, fitnesses):
        idx = np.random.choice(population.shape[0], self.k, replace=False)
        candidates = population[idx]
        best = np.argmin(fitnesses[idx])
        return candidates[best]

    def selection(self, population, fitnesses):
        parents = np.empty((self.mu, 2, population.shape[1]), dtype="i4")
        for i in range(self.mu):
            parents[i][0] = self._kTournament(population, fitnesses)
            parents[i][1] = self._kTournament(population, fitnesses)

        return parents

    def crossover(self, parents):
        path_length = parents[0][0].size
        results = np.full((self.mu, path_length), -1, dtype="i4")

        for i, (parent1, parent2) in enumerate(parents):
            result = results[i]
            start, end = np.random.choice(path_length, 2, replace=False)

            idx_to_fill = None
            if start <= end:
                result[start:end] = parent1[start:end]
                idx_to_fill = np.r_[0:start, end:path_length]
            else:
                result[start:path_length] = parent1[start:path_length]
                result[0:end] = parent1[0:end]
                idx_to_fill = np.arange(end, start)

            last_idx = 0
            for j in idx_to_fill:
                for k in range(last_idx, path_length):
                    if parent2[k] not in result:
                        result[j] = parent2[k]
                        last_idx = k + 1
                        break

        return results

    def mutation(self, population):
        for individual in population:
            while np.random.rand() < self.mutation_rate:
                idx1, idx2 = np.random.choice(population.shape[1], 2, replace=False)
                individual[idx1], individual[idx2] = individual[idx2], individual[idx1]

    def elimination(self, population, fitnesses):
        idx = np.argpartition(fitnesses, self.lam)[0 : self.lam]

        return population[idx], fitnesses[idx]

    # The evolutionary algorithm's main loop
    def optimize(self, filename):
        # Read distance matrix from file.
        file = open(filename)
        distanceMatrix = np.loadtxt(file, delimiter=",", dtype="f4")
        file.close()

        population = self.initialize(distanceMatrix)

        with open("./plots/result.csv", "w") as file:
            writer = csv.writer(file)
            writer.writerow(["Iteration", "Mean Value"])

            meanObjective = np.inf
            bestObjective = np.inf
            bestSolution = None
            iteration = 0
            while (
                np.nan_to_num(meanObjective / bestObjective, nan=np.inf)
                >= 1.001
                # or iteration < 135
            ):
                fitnesses = self.fitness(distanceMatrix, population)
                parents = self.selection(population, fitnesses)
                offspring = self.crossover(parents)

                total = np.concatenate([population, offspring])
                self.mutation(total)
                fitnesses = self.fitness(distanceMatrix, total)
                population, fitnesses = self.elimination(total, fitnesses)

                meanObjective = np.mean(fitnesses)
                bestObjective = np.min(fitnesses)
                bestSolution = population[np.argmin(fitnesses)]
                # Call the reporter with:
                #  - the mean objective function value of the population
                #  - the best objective function value of the population
                #  - a 1D numpy array in the cycle notation containing the best solution
                #    with city numbering starting from 0

                print("Mean: ", meanObjective, "Best: ", bestObjective)

                writer.writerow([iteration, meanObjective])

                timeLeft = self.reporter.report(
                    meanObjective, bestObjective, bestSolution
                )
                if timeLeft < 0:
                    break

                iteration += 1

        print(
            "Mean: ", meanObjective, "Best: ", bestObjective, "Solution: ", bestSolution
        )
        return 0


if __name__ == "__main__":
    test = Basics()
    test.optimize("./tours/tour50.csv")
