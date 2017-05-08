import time

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle


def generate_X(car_features, notcar_features, X_scaler=None):
    """Stack and normalize features"""
    X = np.vstack((car_features, notcar_features)).astype(np.float64)

    if X_scaler is None:
        X_scaler = StandardScaler().fit(X)

    scaled_X = X_scaler.transform(X)
    return scaled_X, X_scaler


def generate_y(car_features, notcar_features):
    """Generate (labels) for features"""
    return np.hstack((
        np.ones(len(car_features)),
        np.zeros(len(notcar_features))
    ))


def train_model(X, y, test_size=0.1):
    """
    1) Shuffles and splits data for training and validation.
    2) Trains on Linear SVC
    3) Prints validation error
    4) Returns model
    """
    # Shuffle and split data
    X, y = shuffle(X, y)
    rand_state = np.random.randint(0, 100)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=rand_state)

    # Train
    svc = LinearSVC()
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()

    # Print times & accuracy, return model
    print(round(t2-t, 2), 'Seconds to train SVC...')
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    return svc
