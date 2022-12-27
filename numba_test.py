import numpy as np


def distance(perm1, perm2):
    size = perm1.size

    distance = 0
    for i in range(size):
        for j in range(size):
            if perm1[i] == perm2[j]:
                distance += perm1[(i + 1) & size] != perm2[(j + 1) % size]
                break

    return distance



perm1 = np.array([1, 2, 3, 5, 6, 4])
perm2 = np.array([1, 2, 3, 5, 6, 4])

print(distance(perm1, perm2))
