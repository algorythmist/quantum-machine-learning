from abc import ABC, abstractmethod

import pennylane as qml


class QMLModel(ABC):

    @property
    @abstractmethod
    def num_qubits(self):
        pass

    @abstractmethod
    def evaluate(self, weights, features):
        pass


def angle_embedding_fn(features, num_qubits):
    return qml.AngleEmbedding(features, wires=range(num_qubits))


def amplitude_embedding_fn(features, num_qubits):
    return qml.AmplitudeEmbedding(features, wires=range(num_qubits),
                                  pad_with=0.0, normalize=True)


class StronglyEntangledBinaryModel(QMLModel):

    def __init__(self, num_qubits: int,
                 embedding_fn=angle_embedding_fn):
        self.num_qubits = num_qubits
        self.embedding = embedding_fn

    def num_qubits(self):
        return self.num_qubits

    def evaluate(self, weights, features):
        self.embedding(features, self.num_qubits)
        qml.StronglyEntanglingLayers(weights, wires=range(self.num_qubits))
        return qml.expval(qml.PauliZ(0))


class MultiClassQMLModel(QMLModel):

    def __init__(self, num_qubits, num_classes,
                 embedding_fn=angle_embedding_fn):
        if num_qubits < num_classes:
            raise ValueError("num_qubits must be greater than or equal to num_classes")
        self.num_qubits = num_qubits
        self.num_classes = num_classes
        self.embedding = embedding_fn

    def layer(self, W):
        for i in range(self.num_qubits):
            qml.Rot(W[i, 0], W[i, 1], W[i, 2], wires=i)

        for i in range(self.num_qubits):
            qml.CNOT(wires=[i, (i + 1) % self.num_qubits])

    def evaluate(self, weights, features):
        self.embedding(features, self.num_qubits)
        for W in weights:
            self.layer(W)
        return [qml.expval(qml.PauliZ(i)) for i in range(self.num_classes)]

    def num_qubits(self):
        return self.num_qubits
