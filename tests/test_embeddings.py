from utils.embeddings import *
from pytest import approx


def test_stereographic():
    X = np.array([1.2, 0.3, -0.7], dtype=float)
    P = inverse_stereographic_projection(X)
    assert len(P) == 4
    assert np.sum(P**2) == approx(1.00)


def test_gnomonic():
    X = np.array([1.2, 0.3, -0.7], dtype=float)
    P = arbitrary_inverse_projection(0, X)
    p = np.array(P)
    assert np.sum(p ** 2) == approx(1.00)
    assert len(p) == 4


def test_twilight():
    X = np.array([1.2, 0.3, -0.7], dtype=float)
    P = arbitrary_inverse_projection(-1-np.sqrt(2)/2, X)
    p = np.array(P)
    assert np.sum(p ** 2) == approx(1.00)
    assert len(p) == 4
