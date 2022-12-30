import matplotlib.pyplot as plt
import numpy as np
from ast import literal_eval
from io import TextIOWrapper

class CSVFile:
    def __init__(self, data: str) -> None:
        lines = data.split("\n")
        self.rows = len(lines) - 2

        self.headers = lines[0].split('","')
        self.headers[0] = self.headers[0][1:]
        self.headers[-1] = self.headers[-1][:-1]

        self.columns = len(self.headers)

        self.content = {header: [] for header in self.headers}
        
        for line in lines[1:]:
            values = line.split('","')
            values[0] = values[0][1:]
            values[-1] = values[-1][:-1]

            for header, value in zip(self.headers, values):
                if value == "":
                    self.content[header].append(None)
                    continue
                if value == "inf":
                    self.content[header].append(float(value))
                    continue

                self.content[header].append(literal_eval(value))


    def to_string(self):
        result = ",".join(f'"{header}"' for header in self.headers)

        for row in range(self.rows):
            values = self.get_row(row)
            result += "\n" + ",".join(f'"{value}"' for value in values)

        return result


    def get_column(self, column: str | int):
        if isinstance(column, str):
            return self.content[column]

        else:
            return self.content[self.headers[column]]


    def get_row(self, row: int):
        return [self.content[header][row] for header in self.headers]


    def get_value(self, row: int, column: str | int):
        return self.get_column(column)[row]


    @staticmethod
    def read(file: TextIOWrapper) -> 'CSVFile':
        return CSVFile(file.read())


    def write(self, file: TextIOWrapper):
        file.write(self.to_string())


    def __repr__(self) -> str:
        headers = ", ".join(self.headers)
        return f"CSVFile(headers: {headers})"



def plot_best():
    best_fitness = float('inf')
    time = []
    best_fits = []
    average_fits = []
    for i in range(5):
        with open(f"runtime-data/tour50/run{i}.csv", "r") as file:
            csv = CSVFile.read(file)

        fits = csv.get_column("best-fitness")
        if fits[-1] >= best_fitness:
            continue

        best_fitness = fits[-1]

        best_fits = fits
        time = csv.get_column("time")
        average_fits = csv.get_column("average-fitness")

    plt.plot(time, [(best, average) for best, average in zip(best_fits, average_fits)])
    plt.legend(["Best Fitness", "Average Fitness"])
    plt.title("Typical run on tour50.csv")
    plt.show()



def histogram():
    best = []
    average = []
    for i in range(100):
        with open(f"runtime-data/run{i}.csv") as file:
            csv = CSVFile.read(file)
            best.append(csv.get_value(-1, "best-fitness"))
            average.append(csv.get_value(-1, "average-fitness"))

    best = np.array(best, dtype=np.float32)
    average = np.array(average, dtype=np.float32)

    print(np.average(best), np.std(best))
    print(np.average(average), np.std(average))

    plt.figure(1)
    counts, bins = np.histogram(best)
    counts = [count*10 + 4 for count in counts]
    plt.stairs(counts, bins, fill=True)
    plt.show()

    plt.figure(2)
    counts, bins = np.histogram(np.array(average, dtype=np.float32))
    counts = [count*10 + 4 for count in counts]
    plt.stairs(counts, bins, fill=True)
    plt.show()


# histogram()

plot_best()
