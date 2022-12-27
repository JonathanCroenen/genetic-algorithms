
import numpy as np
from numba import int32, float32
from numba.typed import List
from numba.experimental import jitclass


@jitclass([
              ("size", int32),
              ("distances", float32[:])
          ])
class A():
    def __init__(self) -> None:
        self.size = 5
        self.distances = np.empty(5, dtype=np.float32)


    def update(self, a):
        for i in a:
            self.distances[0] = i.size

a = List()

b = A()
b.update(a)
