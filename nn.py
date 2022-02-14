import numpy as np
from sklearn.metrics import DetCurveDisplay


class NeuralNetwork:
    def __init__(self, learning_rate, decay, momentum, epochs, batch_size, n_inputs, n_neurons, n_outputs):
        self.learning_rate = learning_rate
        self.decay = decay
        self.momentum = momentum
        self.epochs = epochs
        self.batch_size = batch_size
        self.dense1 = Layer_Dense(n_inputs, n_neurons)
        self.dense2 = Layer_Dense(n_neurons, n_outputs)
        self.activation1 = Activation_ReLU()
        self.loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
        self.optimizer = Optimizer_SGD(
            self.learning_rate, self.decay, self.momentum)

    def fit(self, X, y):
        self.X = X
        self.y = y

        train_steps = X.shape[0]//self.batch_size
        if train_steps*self.batch_size < X.shape[0]:
            train_steps += 1

        for epoch in range(self.epochs):
            for step in range(train_steps):
                batch_X = X[step*self.batch_size:(step+1)*self.batch_size]
                batch_y = y[step*self.batch_size:(step+1)*self.batch_size]
                self.dense1.forward(batch_X)

                self.activation1.forward(self.dense1.output)

                self.dense2.forward(self.activation1.output)

                loss = self.loss_activation.forward(
                    self.dense2.output, batch_y)

                predictions = np.argmax(self.loss_activation.output, axis=1)
                if len(batch_y.shape) == 2:
                    batch_y = np.argmax(y, axis=1)
                accuracy = np.mean(predictions == batch_y)

                self.loss_activation.backward(
                    self.loss_activation.output, batch_y)
                self.dense2.backward(self.loss_activation.dinputs)
                self.activation1.backward(self.dense2.dinputs)
                self.dense1.backward(self.activation1.dinputs)

                self.optimizer.pre_update_params()
                self.optimizer.update_params(self.dense1)
                self.optimizer.update_params(self.dense2)
                self.optimizer.post_update_params()

            print(f'epoch: {epoch}')
            print(f'acc :{accuracy:.3f}')
            print(f'loss :{loss}')
        
        print('Training Accuracy:',accuracy)

    def test(self, X_test, y_test):
        self.dense1.forward(X_test)
        self.activation1.forward(self.dense1.output)
        self.dense2.forward(self.activation1.output)
        self.loss_activation.forward(self.dense2.output,y_test)
        predictions = np.argmax(self.loss_activation.output, axis=1)
        if len(y_test.shape) == 2:
            y_test = np.argmax(y_test, axis=1)
        accuracy = np.mean(predictions == y_test)
        print(f'testing accuracy:{accuracy:3f}')


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10*np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights)+self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)


class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()

        self.dinputs[self.inputs <= 0] = 0


class Activation_Softmax:
    def forward(self, inputs):
        self.inputs = inputs
        exp_values = np.exp(inputs-np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values/np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)

        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(
                single_output)-np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)


class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss


class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]

        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)

        neg_log_likelihoods = -np.log(correct_confidences)
        return neg_log_likelihoods

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        self.dinputs = -y_true/dvalues
        self.dinputs = self.dinputs/samples


class Activation_Softmax_Loss_CategoricalCrossentropy():
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs/samples


class Optimizer_SGD:
    def __init__(self, learning_rate=1., decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1./(1+self.decay*self.iterations))

    def update_params(self, layer):
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
        self.iterations += 1
