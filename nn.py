import numpy as np
from sklearn.metrics import DetCurveDisplay


class NeuralNetwork:
    def __init__(self, learning_rate, decay, momentum, epochs, batch_size, n_inputs, n_neurons, n_outputs):
        '''Initializes the parameters of NN'''

        # these parameters are used by the optimizer
        self.learning_rate = learning_rate
        self.decay = decay
        self.momentum = momentum

        # these parameters are used by fit_plus
        self.epochs = epochs
        self.batch_size = batch_size

        # the neural layers
        self.dense1 = Layer_Dense(n_inputs, n_neurons)
        self.dense2 = Layer_Dense(n_neurons, n_outputs)

        # the activation functions
        self.activation1 = Activation_ReLU()
        self.activation2 = Activation_Softmax()

        # the loss function
        self.loss = Loss()

        # the optimizer
        self.optimizer = Optimizer_SGD(
            self.learning_rate, self.decay, self.momentum)

    def fit_plus(self, X, y):
        '''Train the model with input data'''

        self.X = X
        self.y = y

        #Calculate the number of train steps required for each epoch
        train_steps = X.shape[0]//self.batch_size
        if train_steps*self.batch_size < X.shape[0]:
            train_steps += 1

        for epoch in range(self.epochs):
            for step in range(train_steps):

                # Divide the data into batches,for faster processing
                batch_X = X[step*self.batch_size:(step+1)*self.batch_size]
                batch_y = y[step*self.batch_size:(step+1)*self.batch_size]

                # Perform forward pass
                self.dense1.forward(batch_X)
                self.activation1.forward(self.dense1.output)
                self.dense2.forward(self.activation1.output)
                self.activation2.forward(self.dense2.output)

                # there are two ways to pass true_labels,using one hot vectors and normally
                if len(batch_y.shape) == 2:
                    batch_y = np.argmax(y, axis=1)

                # Perform backward pass
                self.loss.backward(self.activation2.output, batch_y)
                self.activation2.backward(self.loss.dinputs)
                self.dense2.backward(self.activation2.dinputs)
                self.activation1.backward(self.dense2.dinputs)
                self.dense1.backward(self.activation1.dinputs)

                # update weights and biases
                self.optimizer.pre_update_params()
                self.optimizer.update_params(self.dense1)
                self.optimizer.update_params(self.dense2)
                self.optimizer.post_update_params()

            print(f'epoch: {epoch}')

    def predict_plus(self, X_test):
        '''Predict the label for a given set of input images'''

        self.dense1.forward(X_test)
        self.activation1.forward(self.dense1.output)
        self.dense2.forward(self.activation1.output)
        self.activation2.forward(self.dense2.output)

        predictions = np.argmax(self.activation2.output, axis=1)

        return predictions


class Layer_Dense:
    '''A layer of neurons'''

    def __init__(self, n_inputs, n_neurons):
        '''Initializes the parameters of a layer'''

        self.weights = 0.10*np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        '''Called during forward pass'''

        self.inputs = inputs
        self.output = np.dot(inputs, self.weights)+self.biases

    def backward(self, dvalues):
        '''Called during backward pass'''

        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)


class Activation_ReLU:
    '''ReLU activation funmction'''

    def forward(self, inputs):
        '''Called suring forward pass'''

        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        '''Called during backward pass'''

        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


class Activation_Softmax:
    '''SoftMax activation function'''

    def forward(self, inputs):
        '''Called during forward pass'''

        self.inputs = inputs
        exp_values = np.exp(inputs-np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values/np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        '''Called during backward pass'''

        self.dinputs = np.empty_like(dvalues)

        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(
                single_output)-np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)


class Loss():
    '''Loss Function'''

    def backward(self, dvalues, y_true):
        '''Call during backward pass'''

        samples = len(dvalues)
        labels = len(dvalues[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        self.dinputs = -y_true/dvalues
        self.dinputs = self.dinputs/samples


class Optimizer_SGD:
    ''''The optimizer updates the weights and biases in direction of reducing loss'''

    def __init__(self, learning_rate=1., decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self):
        '''Runs before updating parameters'''

        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1./(1+self.decay*self.iterations))

    def update_params(self, layer):
        '''Update the weights and biases of the given layer'''

        if self.momentum:
            if not hasattr(layer, 'weight_momentum'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

                weight_updates = self.momentum*layer.weight_momentums - \
                    self.current_learning_rate*layer.dweights
                layer.weight_momentums = weight_updates

                bias_updates = self.momentum*layer.bias_momentums - \
                    self.current_learning_rate*layer.dbiases
                layer.bias_momentums = bias_updates

            else:
                weight_updates = -self.current_learning_rate*layer.dweights
                bias_updates = -self.current_learning_rate*layer.dbiases

            layer.weights += weight_updates
            layer.biases += bias_updates

    def post_update_params(self):
        '''Runs after updating params'''
        self.iterations += 1
