import numpy as np


def inverse_stereographic_projection(X):
    s2 = np.sum(np.square(X))
    return np.hstack([(s2 - 1) / (s2 + 1), 2 * X / (s2 + 1)])


def arbitrary_inverse_projection(a, x_dash_in_Rn):
    # a should be the x_0 coordinate of the centre of projection. x should be a list of length n, corresponding to (x_1', ... , x_n') as in the diagram
    s = np.linalg.norm(x_dash_in_Rn)

    # New part - if a <-1, then this extends the domain of the projection to the entirety of Rn
    if (a < -1) and (s > np.sqrt((a - 1) / (a + 1))):
        x_dash_in_Rn = [np.sqrt((a - 1) / (a + 1)) * x_i / s for x_i in x_dash_in_Rn]
        s = np.sqrt((a - 1) / (a + 1))

    # quadratic formula
    A = (s ** 2 + (1 - a) ** 2)
    B = -2 * a * s ** 2
    C = s ** 2 * a ** 2 - (1 - a) ** 2

    x_0 = (-B + np.sqrt(B ** 2 - 4 * A * C)) / (2 * A)

    return [x_0] + [((x_0 - a) / (1 - a)) * x_i for x_i in x_dash_in_Rn]
