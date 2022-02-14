from svm import SupportVectorMachine
from nn import NeuralNetwork
import os
import cv2
import numpy as np
import pandas as pd

# reading the labels of training data
df = pd.read_csv("cifar-10/trainLabels.csv")
labels = dict(zip(range(0, 10), df['label'].unique()))

# TODO: The svm model works only for 2 labels, enhance to include all
print(labels)


def load_dataset(dataset, path, df):
    '''Loads the data into program'''
    X = []
    y = []

    for (label, label_text) in labels.items():

        for id in list(df[df['label'] == label_text]['id']):

            image = cv2.imread(os.path.join(
                path, dataset, f'{id}.png'), cv2.IMREAD_GRAYSCALE)

            X.append(image)
            y.append(label)

    return np.array(X), np.array(y).astype('uint8')


def create_data(path):
    '''Data pre-processing and also splitting the data between test and split data'''
    X0, y0 = load_dataset('train', path, df)

    X0 = (X0.astype(np.float32)-127.5)/127.5
    X0 = X0.reshape(X0.shape[0], -1)
    keys = np.array(range(X0.shape[0]))

    # It's necessary to shuffle the data for accurate training
    np.random.shuffle(keys)
    X0 = X0[keys]
    y0 = y0[keys]

    # Training data
    X = X0[:40000]
    y = y0[:40000]

    # Testing data
    X_test = X0[40000:]
    y_test = y0[40000:]

    return X, y, X_test, y_test


# Create testing and training data
X, y, X_test, y_test = create_data('cifar-10')


def run_svm():
    # Initializing the model with super-parameters
    model = SupportVectorMachine(
        learning_rate=0.0001, epochs=100, lambda_parameter=0.0002)

    # Training the model to fit cifar-10 data
    model.fit_plus(X, y)

    # Model Evaluation

    # accuracy on training data
    X_train_prediction = model.predict_plus(X)
    training_data_accuracy = np.mean(X_train_prediction == y)

    # accuracy on testing data
    X_test_prediction = model.predict_plus(X_test)
    testing_data_accuracy = np.mean(X_test_prediction == y_test)

    print("accuracy on training data ", training_data_accuracy)
    print("accuracy on testing data ", testing_data_accuracy)


def run_nn():
    model = NeuralNetwork(learning_rate=0.90, decay=1e-3, momentum=1.2,
                          epochs=20, batch_size=128, n_inputs=1024, n_neurons=64, n_outputs=10)
    model.fit(X, y)
    model.test(X_test, y_test)

run_svm()
run_nn()