import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from penny.models import MultiClassQMLModel
from penny.torch_classifier import IterationTorchClassifier
from utils.preprocess import tensorize


def test_torch_iris():
    np.random.seed(24)

    X, y = load_iris(return_X_y=True)
    X_scaled = StandardScaler().fit_transform(X)
    y_hot = OneHotEncoder(sparse_output=False).fit_transform(y.reshape(-1, 1))
    X_scaled, y, y_hot = tensorize((X_scaled, y, y_hot))
    X_train, X_test, y_train, y_test, y_train_hot, y_test_hot = train_test_split(X_scaled, y, y_hot,
                                                                                 shuffle=True, test_size=0.2)

    qubits = 4
    num_classes = 3
    classifier = IterationTorchClassifier(
        model=MultiClassQMLModel(qubits, num_classes=num_classes),
        num_classes=num_classes,
        weights_shape=(3, qubits, 3),
        iterations=80
    )
    classifier.fit(X_train, y_train_hot)
    predictions = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    assert accuracy > 0.9
