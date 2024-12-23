from abc import ABC, abstractmethod
from typing import Tuple

import pennylane as qml
from pennylane import GradientDescentOptimizer, AdamOptimizer
from pennylane import numpy as pnp
from sklearn.metrics import accuracy_score

from penny.models import QMLModel
from utils.metrics import square_loss


class ClassifierContext:

    def __init__(self, classifier, current_iteration,
                 current_parameters,
                 current_batch_X, current_batch_y,
                 current_cost=None):
        self.classifier = classifier
        self.current_iteration = current_iteration
        self.current_parameters = current_parameters
        self.current_batch_X = current_batch_X
        self.current_batch_y = current_batch_y
        self.current_cost = current_cost


class Classifier(ABC):

    @abstractmethod
    def fit(self, features, labels):
        pass

    @abstractmethod
    def predict(self, features):
        pass

    @abstractmethod
    def set_parameters(self, parameters):
        pass

    @abstractmethod
    def get_parameters(self):
        pass


def report_binary_progress(classifier_context: ClassifierContext):
    iteration = classifier_context.current_iteration
    classifier = classifier_context.classifier
    features = classifier_context.current_batch_X
    labels = classifier_context.current_batch_y
    if (iteration + 1) % 10 == 0:
        predictions = classifier.predict(features)
        accuracy = accuracy_score(labels, predictions)
        penalty = classifier.cost(classifier.weights, classifier.bias, features, labels)
        print(f"Iter: {iteration + 1:5d} | Cost: {penalty:0.7f} | Accuracy: {accuracy:0.7f} ")
        return penalty, accuracy


class BinaryClassifier(Classifier):

    def __init__(self,
                 model: QMLModel,
                 weights_shape: Tuple[int, int, int],
                 optimizer: GradientDescentOptimizer = AdamOptimizer(0.02),
                 loss_fn=square_loss,
                 device="default.qubit",
                 iterations=100,
                 batch_size=5,
                 report_fn=report_binary_progress
                 ):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.iterations = iterations
        self.circuit = qml.QNode(model.evaluate,
                                 qml.device(device, wires=model.num_qubits),
                                 interface="autograd")
        self.weights = 0.01 * pnp.random.randn(*weights_shape, requires_grad=True)
        self.bias = pnp.array(0.01, requires_grad=True)
        self.loss_fn = loss_fn
        self.report_fn = report_fn

    def _output(self, weights, bias, features):
        return self.circuit(weights, features) + bias

    def predict(self, features):
        return pnp.sign(self._output(self.weights, self.bias, features))

    def cost(self, weights, bias, features, labels):
        predictions = [self._output(weights, bias, f) for f in features]
        return self.loss_fn(labels, predictions)

    def fit(self, features, labels):
        self.optimizer.reset()
        for iteration in range(self.iterations):
            # Update the weights by one optimizer step
            batch_index = pnp.random.randint(0, len(labels), (self.batch_size,))
            features_batch = features[batch_index]
            labels_batch = labels[batch_index]
            self.weights, self.bias, _, _ = self.optimizer.step(self.cost, self.weights, self.bias,
                                                                features_batch, labels_batch)
            if self.report_fn:
                self.report_fn(ClassifierContext(self, iteration, (self.weights, self.bias),
                                                 features_batch, labels_batch))

    def get_parameters(self):
        return self.weights, self.bias

    def set_parameters(self, parameters):
        weights, bias = parameters
        self.weights = pnp.array(weights, requires_grad=True)
        self.bias = pnp.array(bias, requires_grad=True)
