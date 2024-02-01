import numpy as np
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from penny.classifier import ClassifierContext, BinaryClassifier
from penny.models import StronglyEntangledBinaryModel
from utils.preprocess import scale_for_angle_encoding

if __name__ == '__main__':
    X, y = load_wine(return_X_y=True)
    print(y)