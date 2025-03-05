import numpy as np

x = np.array([1,2, 3, 2, 4, 2, 3, 1, 2])
vals, counts = np.unique(x, return_counts=True)
max_count = vals[np.argmax(counts)]
print(max_count)