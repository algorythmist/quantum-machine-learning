import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from penny.classifier import *
from penny.models import StronglyEntangledBinaryModel, amplitude_embedding_fn

def get_binary_scaled_iris():
    X, y = load_iris(return_X_y=True)
    features = X[y != 2]
    labels = y[y != 2]
    labels = np.where(labels == 0, -1, 1)
    features_scaled = StandardScaler().fit_transform(features)
    return labels, features_scaled

def test_binary_iris():
    random_seed=623
    pnp.random.seed(random_seed)

    labels, features_scaled = get_binary_scaled_iris()
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, shuffle=True, test_size=0.2,
                                                        random_state=random_seed)

    qubits = 4
    classifier = BinaryClassifier(
        model=StronglyEntangledBinaryModel(qubits),
        weights_shape=(3, qubits, 3),
        iterations=100
    )
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    assert accuracy == 1.0


def test_amplitude_embedding_iris():
    random_seed = 12939
    pnp.random.seed(random_seed)

    labels, features_scaled = get_binary_scaled_iris()
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, shuffle=True, test_size=0.2,
                                                        random_state=random_seed)
    qubits = 2
    classifier = BinaryClassifier(
        model=StronglyEntangledBinaryModel(qubits,
                                           embedding_fn=amplitude_embedding_fn),
        weights_shape=(3, qubits, 3),
        iterations=100
    )

    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(accuracy)