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

    def update_params(self):
        '''Updates weights and bias of the model'''

        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1./(1+self.decay*self.iterations))

        # gradients ( dw, db)
        for index, x_i in enumerate(self.X):

            condition = self.y[index] * (np.dot(x_i, self.w) - self.b) >= 1

            if condition:
                # dc_dw
                dw = 2 * self.lambda_parameter * self.w

                # dc_db
                db = 0

            else:
                # dc_dw
                dw = 2 * self.lambda_parameter * \
                    self.w - np.dot(x_i, self.y[index])

                # dc_db
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
        '''Training on more than 2 classes'''

        # all the labels,here 0-9
        self.labels = np.unique(y)

        # storing the weights and bias of 10 decision boudaries
        self.w_plus = []
        self.b_plus = []

        # Approach here user id to work on one label at a time.
        # Say when trainng for class 0,set them as 1 and others as -1, and calculate w and b for it.
        # Repeat over all labels

        for label in self.labels:

            # Re-initialize these parameters for each class
            self.iterations = 0
            self.current_learning_rate = self.learning_rate

            print('Training class:', label)

            # working class=1,others=-1
            y_maksed = np.where(y == label, 1, -1)

            # The working class has only 1/10th representation,whereas others have 9/10th, so it's necessary to balance the two
            X_balanced, y_balanced = self.balance_data(X, y_maksed)

            # train to identify the working class
            self.fit(X_balanced, y_balanced)

            # store the weights and biases for the working class
            self.w_plus.append(self.w)
            self.b_plus.append(self.b)

            # set the parameters again to zero, for training of next class
            self.w = np.zeros(0)
            self.b = 0
            self.X = None
            self.y = None
            y_masked = None

    def predict_plus(self, X):
        '''Predicting labels for more than one class'''
        
        prediction_plus = []
        
        for x in X:
            
            #probabilities that x has for being in each of the classes,depends on it's distance from decision boundary
            x_probs = []
            
            for index, w in enumerate(self.w_plus):
                
                self.w = w
                self.b = self.b_plus[index]
                
                #TODO: apply something like softmax to get actual probabilities
                x_probs.append(np.dot(x, self.w) - self.b)
                
            #store the predicted class for x in X
            prediction_plus.append(x_probs.index(max(x_probs)))

        return np.array(prediction_plus)

    def balance_data(self, X, y):
        '''To balance unbalanced data'''
        #Trims the size of excess class, here the -1 class
        
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
