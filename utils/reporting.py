from sklearn.metrics import accuracy_score


def report_progress(classifier, iteration, features, labels):
    if (iteration + 1) % 10 == 0:
        predictions = classifier.predict(features)
        accuracy = accuracy_score(labels, predictions)
        # TODO: should cost be in interface?
        penalty = classifier.cost(classifier.weights, classifier.bias, features, labels)
        print(f"Iter: {iteration + 1:5d} | Cost: {penalty:0.7f} | Accuracy: {accuracy:0.7f} ")
        return penalty, accuracy