from sklearn.datasets import load_digits
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from penny.models import MultiClassQMLModel
from penny.torch_classifier import IterationTorchClassifier, EpochTorchClassifier
from utils.preprocess import tensorize
from utils.reporting import HistoryTracker


def record_progress(classifier_context):
    iteration = classifier_context.current_iteration
    classifier = classifier_context.classifier
    if (iteration + 1) % 10 == 0:
        y_pred = classifier.predict(X_train)
        accuracy = accuracy_score(y_train, y_pred)
        print(f"Iter: {iteration + 1:5d} | Accuracy: {accuracy:0.7f} ")

if __name__ == '__main__':

    random_seed = 777
    np.random.seed(random_seed)
    X, y = load_digits(return_X_y=True)

    selection = y < 5
    y = y[selection]
    X = X[selection]
    y_hot = OneHotEncoder(sparse_output=False).fit_transform(y.reshape(-1, 1))
    X = StandardScaler().fit_transform(X)
    X = PCA(n_components=8).fit_transform(X)
    X, y, y_hot = tensorize((X, y, y_hot))
    X_train, X_test, y_train, y_test, y_train_hot, y_test_hot = train_test_split(X, y, y_hot,
                                                                                 shuffle=True, test_size=0.2,
                                                                                 random_state=random_seed)

    qubits = 8
    num_classes = 5
    # classifier = IterationTorchClassifier(
    #     model=MultiClassQMLModel(qubits, num_classes=num_classes),
    #     num_classes=num_classes,
    #     weights_shape=(3, qubits, 3),
    #     iterations=100,
    #     report_fn=record_progress)
    classifier = EpochTorchClassifier(
        model=MultiClassQMLModel(qubits, num_classes=num_classes),
        num_classes=num_classes,
        weights_shape=(3, qubits, 3),
        epochs=10,
        report_fn=record_progress
    )

    classifier.fit(X_train, y_train_hot)
    predictions = classifier.predict(X_test)
    cr = classification_report(y_test, predictions)
    print(cr)
