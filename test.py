from svm import SupportVectorMachine
from nn import NeuralNetwork
import os
import cv2
import numpy as np
import pandas as pd


# reading the labels of data
df = pd.read_csv("cifar-10/trainLabels.csv")
labels = dict(zip(range(0, 10), df['label'].unique()))

# print an index of the labels with label_id for reference
print(labels)


def load_dataset(dataset, path, df):
    '''Loads the data into program'''

    X = []  # Stores the features of dataset(here,image)
    y = []  # Stores the label_id of every entry in dataset

    # iterate over all labels(10 here)
    for (label, label_text) in labels.items():

        # In the database,find the ids of item with label=our current label
        for id in list(df[df['label'] == label_text]['id']):

            # read the image corresponding to given id as grayscale image
            # (this reduces the number of features and also model will work if the color of object changes)
            image = cv2.imread(os.path.join(
                path, dataset, f'{id}.png'), cv2.IMREAD_GRAYSCALE)

            # add the image and label to our dataset
            X.append(image)
            y.append(label)

    # return X,y as np arrays to make operations faster
    return np.array(X), np.array(y).astype('uint8')


def pre_process_data(X0, y0):
    '''Pre-processing data'''

    # This normailzes all the features to values between -1 and 1
    X0 = (X0.astype(np.float32)-127.5)/127.5

    # The numpy array is flatened
    X0 = X0.reshape(X0.shape[0], -1)

    # It's necessary to shuffle the data for proper training
    keys = np.array(range(X0.shape[0]))
    np.random.shuffle(keys)
    X0 = X0[keys]
    y0 = y0[keys]

    return X0, y0


def train_test_split(X0, y0, split_point):
    '''Splitting the data into training and testing data'''
    # Training data
    X = X0[:split_point]
    y = y0[:split_point]

    # Testing data
    X_test = X0[split_point:]
    y_test = y0[split_point:]

    return X, y, X_test, y_test


def create_data(path):
    '''Returns the final data to be passed on to the models'''

    # extract the data from the required folder and labels file
    X0, y0 = load_dataset('train', path, df)

    # pre-process data to ensure model will be trained properly
    X0, y0 = pre_process_data(X0, y0)

    # return the data after splitting into testing and training set
    return train_test_split(X0, y0, 40000)


print('Loading dataset..')

# Create testing and training data
X, y, X_test, y_test = create_data('cifar-10')

print('Completed loading dataset')


choice = input("select a model to proceed-")
if choice == 'svm':
    # Initializing the SVM model with hyper-parameters
    model = SupportVectorMachine(
        learning_rate=0.0001, epochs=15, lambda_parameter=0.0001, decay=0.00104)
elif choice == 'nn':
    # Initializing the NN model with hyper-parameters
    model = NeuralNetwork(learning_rate=0.11, decay=0.00104, momentum=1.0,
                          epochs=15, batch_size=128, n_inputs=1024, n_neurons=64, n_outputs=10)

print(f'Training the {choice} model..')

# Training the model to fit cifar-10 data
model.fit_plus(X, y)

print(f'Completed training the {choice} model')


print(f'Evaluating the {choice} model..')

# accuracy on training data
X_train_prediction = model.predict_plus(X)
training_data_accuracy = np.mean(X_train_prediction == y)

# accuracy on testing data
X_test_prediction = model.predict_plus(X_test)
testing_data_accuracy = np.mean(X_test_prediction == y_test)

print(f'Completed evaluating the {choice} model..')


print("accuracy on training data ", training_data_accuracy)
print("accuracy on testing data ", testing_data_accuracy)
