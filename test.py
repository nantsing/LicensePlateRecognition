import numpy as np

a = np.array([1, 1, 1, 1, 2, 3])

print(a[::-1])

print(len(a) - np.argmin(a[::-1]) - 1)