import numpy as np

def sample_weighted(ps):
    c = 0.0
    r = np.random.rand()
    for i in range(len(ps)):
        c += ps[i]
        if c >= r:
            return i

