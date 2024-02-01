from utils.embeddings import *
from pytest import approx


def test_stereographic():
    X = np.array([0.3, 0.7], dtype=float)
    P = inverse_stereographic_projection(X)
    assert np.sum(P**2) == approx(1.00)


def test_gnomonic():
    X = np.array([0.3, 0.7], dtype=float)
    P = inverse_gnomonic_projection(X)
    assert np.sum(P ** 2) == approx(1.00)
