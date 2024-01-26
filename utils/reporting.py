from sklearn.metrics import accuracy_score

from penny.classifier import ClassifierContext


class HistoryTracker:
    """
    Track the history of a classifier's accuracy on the test set
    """

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.history = []

    def __call__(self, classifier_context: ClassifierContext):
        classifier = classifier_context.classifier
        y_pred = classifier.predict(self.X)
        accuracy = accuracy_score(self.y, y_pred)
        self.history.append(accuracy)

