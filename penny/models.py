from abc import ABC, abstractmethod
from typing import Callable, Any

import pennylane as qml


class QMLModel(ABC):

    @property
    @abstractmethod
    def num_qubits(self):
        pass

    @abstractmethod
    def evaluate(self, weights, features):
        pass


class StronglyEntangledBinaryModel(QMLModel):

    def __init__(self, num_qubits: int,
                 embedding_fn=qml.AngleEmbedding):
        self.num_qubits = num_qubits
        self.embedding = embedding_fn

    def num_qubits(self):
        return self.num_qubits

    def evaluate(self, weights, features):
        self.embedding(features, wires=range(self.num_qubits))
        qml.StronglyEntanglingLayers(weights, wires=range(self.num_qubits))
        return qml.expval(qml.PauliZ(0))
