from svm import SupportVectorMachine
import os
import cv2
import numpy as np
import pandas as pd

# reading the labels of training data
df = pd.read_csv("cifar-10/trainLabels.csv")
labels = dict(zip(range(0, 10), df['label'].unique()))

# TODO: The svm model works only for 2 labels, enhance to include all
print(labels)
labels = {0: 'frog', 1: 'truck'}


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

    # TODO:remove this when implementing for multiple labels
    y0 = np.where(y0 <= 0, -1, 1)

    # Training data
    X = X0[:8000]
    y = y0[:8000]

    # Testing data
    X_test = X0[8000:]
    y_test = y0[8000:]

    return X, y, X_test, y_test


# Create testing and training data
X, y, X_test, y_test = create_data('cifar-10')
print(np.count_nonzero(y))


# Initializing the model with super-parameters
model = SupportVectorMachine(
    learning_rate=0.0001, epochs=10, lambda_parameter=0.02)

# Training the model to fit cifar-10 data
model.fit(X, y)

# Model Evaluation

# accuracy on training data
X_train_prediction = model.predict(X)
training_data_accuracy = np.mean(X_train_prediction == y)

# accuracy on testing data
X_test_prediction = model.predict(X_test)
testing_data_accuracy = np.mean(X_test_prediction == y_test)

print("accuracy on training data ", training_data_accuracy)
print("accuracy on testing data ", testing_data_accuracy)
