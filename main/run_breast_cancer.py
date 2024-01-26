import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from penny.classifier import ClassifierContext, BinaryClassifier
from penny.models import StronglyEntangledBinaryModel
from utils.preprocess import scale_for_angle_encoding

if __name__ == '__main__':
    np.random.seed(9999)
    features, y = load_breast_cancer(return_X_y=True)
    X = StandardScaler().fit_transform(features)
    y[y == 0] = -1
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2)

    num_features = 5

    pca = PCA(n_components=num_features)
    X_train_pca = pca.fit_transform(X_train)
    X_train_scaled = scale_for_angle_encoding(X_train_pca)
    X_test_pca = pca.transform(X_test)
    X_test_scaled = scale_for_angle_encoding(X_test_pca)

    pca_5_accuracy = []


    def capture_progress(classifier_context: ClassifierContext):
        iteration = classifier_context.current_iteration
        classifier = classifier_context.classifier
        predictions = classifier.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, predictions)
        pca_5_accuracy.append(accuracy)
        if (iteration + 1) % 10 == 0:
            print(f"Iter: {iteration + 1:5d} | Accuracy: {accuracy:0.7f} ")


    num_qubits = num_features
    num_layers = 3
    classifier = BinaryClassifier(model=StronglyEntangledBinaryModel(num_qubits),
                                  weights_shape=(3, num_qubits, 3),
                                  device="lightning.qubit",
                                  iterations=100,
                                  report_fn=capture_progress)
    classifier.fit(X_train_scaled, y_train)
    final_weights, final_bias = classifier.get_parameters()
    predictions_test = classifier.predict(X_test_scaled)
    cr = classification_report(y_test, predictions_test)
    print(cr)
