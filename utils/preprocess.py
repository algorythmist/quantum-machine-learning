import numpy as np
from sklearn.preprocessing import MinMaxScaler


def scale_for_angle_encoding(features):
    # scale the features to be between -pi and pi
    return MinMaxScaler((-np.pi, np.pi)).fit_transform(features)


def scale_for_amplitude_encoding(features):
    """
    Normalize features so that the sum of squares of each row is 1
    """
    return features / np.linalg.norm(features, axis=1)[:, np.newaxis]
