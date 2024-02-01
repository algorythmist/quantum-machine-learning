import numpy as np


def inverse_stereographic_projection(X):
    s2 = np.sum(np.square(X))
    return np.hstack([(s2-1)/(s2+1), 2*X/(s2+1)])


def inverse_gnomonic_projection(X):
    embedded = [1] + X
    norm = np.linalg.norm(embedded)
    return embedded / norm
