from abc import ABC
from typing import Tuple

import numpy as np
import pennylane as qml
import torch
from sklearn.metrics import accuracy_score
from torch import optim
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss

from penny.classifier import Classifier, ClassifierContext
from penny.models import QMLModel


class BaseTorchClassifier(Classifier, ABC):

    def __init__(self):
        self.params = None
        self.circuit = None

    def classify_probabilities(self, features):
        weights, bias = self.params
        values = []
        for f in features:
            value = torch.stack(self.circuit(weights, f)) + bias
            values.append(value)
        values = torch.stack(values)
        return values

    def predict(self, features):
        return torch.argmax(self.classify_probabilities(features), axis=1)

    def evaluate(self, X, y):
        predictions = self.predict(X)
        return accuracy_score(predictions, y)

    def set_parameters(self, params):
        weights, bias = params
        weights = Variable(torch.from_numpy(weights), requires_grad=True)
        bias = Variable(torch.from_numpy(bias), requires_grad=True)
        self.params = (weights, bias)

    def get_parameters(self):
        weights, bias = self.params
        return weights.detach().numpy(), bias.detach().numpy()

    @staticmethod
    def one_hot_to_label(one_hot):
        return torch.argmax(one_hot, axis=1)


class IterationTorchClassifier(BaseTorchClassifier):

    def __init__(self, model: QMLModel,
                 num_classes: int,
                 weights_shape: Tuple[int, int, int],
                 optimizer_builder=lambda params: optim.Adam(params, lr=0.1),
                 loss_builder=lambda: CrossEntropyLoss(),
                 batch_size=5,
                 iterations=100,
                 device="default.qubit",
                 report_fn=None):
        self.model = model
        self.optimizer_builder = optimizer_builder
        self.loss_builder = loss_builder
        self.circuit = qml.QNode(model.evaluate,
                                 qml.device(device, wires=model.num_qubits),
                                 interface="torch")
        weights = Variable(0.1 * torch.randn(*weights_shape, ), requires_grad=True)
        bias = Variable(0.1 * torch.ones(num_classes), requires_grad=True)
        self.params = (weights, bias)
        self.shape = weights.shape
        self.batch_size = batch_size
        self.iterations = iterations
        self.report_fn = report_fn

    def fit(self, X_train, y_train_hot):
        optimizer = self.optimizer_builder(self.params)
        loss = self.loss_builder()
        for iteration in range(self.iterations):
            batch_index = torch.LongTensor(np.random.randint(0, len(y_train_hot), (self.batch_size,)))
            features_batch = X_train[batch_index]
            labels_batch = y_train_hot[batch_index]
            optimizer.zero_grad()
            predictions = self.classify_probabilities(features_batch)
            curr_cost = loss(predictions, labels_batch)
            curr_cost.backward()
            optimizer.step()
            if self.report_fn:
                self.report_fn(ClassifierContext(self, iteration, self.params,
                                                    features_batch, labels_batch))


class EpochTorchClassifier(BaseTorchClassifier):

    def __init__(self, model: QMLModel,
                 num_classes: int,
                 weights_shape: Tuple[int, int, int],
                 optimizer_builder=lambda params: optim.Adam(params, lr=0.1),
                 loss_builder=lambda: CrossEntropyLoss(),
                 batch_size=5,
                 epochs=10,
                 device="default.qubit",
                 report_fn=None):
        self.model = model
        self.optimizer_builder = optimizer_builder
        self.loss_builder = loss_builder
        self.circuit = qml.QNode(model.evaluate,
                                 qml.device(device, wires=model.num_qubits),
                                 interface="torch")
        weights = Variable(0.1 * torch.randn(weights_shape), requires_grad=True)
        bias = Variable(0.1 * torch.ones(num_classes), requires_grad=True)
        self.params = (weights, bias)
        self.shape = weights.shape
        self.batch_size = batch_size
        self.epochs = epochs
        self.report_fn = report_fn

    def fit(self, X_train, y_train_hot):
        data_loader = torch.utils.data.DataLoader(
            list(zip(X_train, y_train_hot)),
            batch_size=self.batch_size, shuffle=True
        )
        optimizer = self.optimizer_builder(self.params)
        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.
            epoch_accuracy = 0.
            batches = 0
            loss = self.loss_builder()
            for X_batch, y_hot_batch in data_loader:
                optimizer.zero_grad()
                predictions = self.classify_probabilities(X_batch)
                curr_cost = loss(predictions, y_hot_batch)
                curr_cost.backward()
                optimizer.step()
                # collect performance metrics
                epoch_loss += curr_cost.item()
                y_train = self.one_hot_to_label(y_hot_batch)
                epoch_accuracy += accuracy_score(y_train, torch.argmax(predictions, axis=1).detach().numpy())
                batches += 1

            epoch_loss = epoch_loss / batches

            if self.report_fn:
                self.report_fn(ClassifierContext(self, epoch, self.params,
                                                 current_batch_X = None,
                                                 current_batch_y = None,
                                                 current_cost=epoch_loss))

            epoch_accuracy = epoch_accuracy / batches
            print(f"Epoch: {epoch}. Avg loss = {epoch_loss}, avg training accuracy = {epoch_accuracy}")

