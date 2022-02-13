import numpy as np


class SupportVectorMachine():

    def __init__(self, learning_rate, epochs, lambda_parameter):
        '''Initializing the classifier model with superparameters'''
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.lambda_parameter = lambda_parameter

    def fit(self, X, y):
        '''Fitting the model with training parameters'''

        # m  = number of Data points
        # n  = number of input features
        self.m, self.n = X.shape

        # initiating the weight value and bias value
        self.w = np.zeros(self.n)
        self.b = 0

        self.X = X
        self.y = y

        # implementing Gradient Descent algorithm for Optimization
        for epoch in range(self.epochs):
            self.update_params()
            prediction = self.predict(X)
            accuracy = np.mean(prediction == y)
            print(f"accuracy on epoch {epoch}= ", accuracy)

    def update_params(self):
        '''Updates weights and bias of the model'''

        # gradients ( dw, db)
        for index, x_i in enumerate(self.X):

            condition = self.y[index] * (np.dot(x_i, self.w) - self.b) >= 1
            if condition:
                dw = 2 * self.lambda_parameter * self.w
                db = 0

            else:
                dw = 2 * self.lambda_parameter * \
                    self.w - np.dot(x_i, self.y[index])
                db = self.y[index]

            self.w = self.w - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db

    def predict(self, X):
        '''predict the label for a given input'''

        output = np.dot(X, self.w) - self.b
        y = np.sign(output)

        return y