import numpy as np


class SupportVectorMachine():

    def __init__(self, learning_rate, epochs, lambda_parameter, decay=0.):
        '''Initializing the classifier model with superparameters'''

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.lambda_parameter = lambda_parameter
        self.decay = decay
        self.current_learning_rate = learning_rate
        self.iterations = 0

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
            #print(f"accuracy on epoch {epoch}= ", f'{accuracy:4f}')

    def update_params(self):
        '''Updates weights and bias of the model'''

        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1./(1+self.decay*self.iterations))

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

        self.iterations += 1

    def predict(self, X):
        '''predict the label for a given input'''

        output = np.dot(X, self.w) - self.b
        y = np.sign(output)

        return y

    def fit_plus(self, X, y):
        self.labels = np.unique(y)
        self.w_plus = []
        self.b_plus = []
        for label in self.labels:
            self.iterations=0
            self.current_learning_rate=self.learning_rate
            print('Training class:', label)
            y_maksed = np.where(y == label, 1, -1)
            X_balanced, y_balanced = self.balance_data(X, y_maksed)
            self.fit(X_balanced, y_balanced)
            self.w_plus.append(self.w)
            self.b_plus.append(self.b)
            self.w = np.zeros(0)
            self.b = 0
            self.X = None
            self.y = None
            y_masked = None

    def predict_plus(self, X):
        prediction_plus = []
        for x in X:
            x_detected = False
            x_probs = []
            for index, w in enumerate(self.w_plus):
                self.w = w
                self.b = self.b_plus[index]
                x_probs.append(np.dot(x, self.w) - self.b)
            prediction_plus.append(x_probs.index(max(x_probs)))

        return np.array(prediction_plus)

    def balance_data(self, X, y):
        size_minority = np.count_nonzero(y == 1)

        X_plus = X[y == 1]
        X_minus = X[y == -1]

        X_minus = X_minus[:size_minority]

        X_copy = np.concatenate((X_plus, X_minus))
        y_copy = np.concatenate((np.full(shape=(size_minority,), fill_value=1),
                                 np.full(shape=(size_minority,), fill_value=-1)))

        # It's necessary to shuffle the data for accurate training
        keys = np.array(range(X_copy.shape[0]))
        np.random.shuffle(keys)
        X_balanced = X_copy[keys]
        y_balanced = y_copy[keys]

        return X_balanced, y_balanced
