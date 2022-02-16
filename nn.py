import numpy as np


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

        # Calculate the number of train steps required for each epoch
        train_steps = X.shape[0]//self.batch_size
        if train_steps*self.batch_size < X.shape[0]:
            train_steps += 1

        for epoch in range(self.epochs):

            for step in range(train_steps):

                # Divide the data into batches,for faster processing
                batch_X = X[step*self.batch_size:(step+1)*self.batch_size]
                batch_y = y[step*self.batch_size:(step+1)*self.batch_size]

                # Perform forward pass

                # output1=w*i+b
                self.dense1.forward(batch_X)

                # output2=ReLU(output1)
                self.activation1.forward(self.dense1.output)

                # output3=w*output1+b
                self.dense2.forward(self.activation1.output)

                # output4=SoftMax(output3)
                self.activation2.forward(self.dense2.output)

                # there are two ways to pass true_labels,using one hot vectors
                # and categorically,convert one-hot to categorical
                if len(batch_y.shape) == 2:
                    batch_y = np.argmax(y, axis=1)

                # Perform backward pass

                # dloss/doutput4
                self.loss.backward(self.activation2.output, batch_y)

                # doutput4/doutput3
                self.activation2.backward(self.loss.dinputs)

                # doutput3/doutput2
                self.dense2.backward(self.activation2.dinputs)

                # doutput2/doutput1
                self.activation1.backward(self.dense2.dinputs)

                # doutput1/d
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

        # dvalues is the derivative from the next layer
    
        #v=w*i+b
        #differentiate wrt weights
        #dv=i*dw   
        self.dweights = np.dot(self.inputs.T, dvalues)

        # v=w*i+b
        # partially differentiate wrt b
        # dv=db
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # v=w*i+b
        #partially differentiate wrt i
        # dv=w*di
        self.dinputs = np.dot(dvalues, self.weights.T)


class Activation_ReLU:
    '''ReLU activation funmction'''
    # if input is less than 0 output 0,
    # else output=input

    def forward(self, inputs):
        '''Called suring forward pass'''

        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        '''Called during backward pass'''
        # the derivative is equal to dvalues if input is greater than 0

        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


class Activation_Softmax:
    '''SoftMax activation function'''
    # This function converts the input data into probability distribution

    def forward(self, inputs):
        '''Called during forward pass'''

        self.inputs = inputs

        # convert inputs to e^(inputs-max(inputs)).The input is exponentialized
        # to avoid negative values which would'nt be converted to probabibilty
        # and this also increases the probability for +ve numbers,this makes
        # prediction accurate. We replace input by input-max(imputs) to
        # avoid very large numbers which will anywas be reduced to small values in the next step
        exp_values = np.exp(inputs-np.max(inputs, axis=1, keepdims=True))

        # calculate the probabilities value based on the exp_values
        probabilities = exp_values/np.sum(exp_values, axis=1, keepdims=True)

        # this probabilities value is the output of softMax function
        self.output = probabilities

    def backward(self, dvalues):
        '''Called during backward pass'''

        # This returns the derivative dvalue/dsoftMax,
        # the derivation of which is beyond my scope :(

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

        # number of samples
        samples = len(dvalues)

        # number of classes
        labels = len(dvalues[0])

        # convert categorical labels to one hot encoded labels
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # the derivative is calculated using categorical cross
        # entropy loss function, since forward pass to it is not needed it is ommited.
        # The derivative of this loss function with respect to its inputs (predicted values at the i-th sample,
        # since we are interested in a gradient with respect to the predicted values) equals the negative
        # ground-truth vector, divided by the vector of the predicted values (which is also the output vector
        # of the softmax function).
        self.dinputs = -y_true/dvalues

        # This is gradient normalization. The more bigger our sample set is,
        # the higher becomes the gradient,to avoid this we normalize the gradient.
        self.dinputs = self.dinputs/samples


class Optimizer_SGD:
    ''''The optimizer updates the weights and biases in direction of reducing loss'''
    '''The basic analogy used here is that of a balling rolling down a very 
        uneven rouded hill,consisting of small heaps.The end goal is to reach the bottom-most point,
        or the global minima'''

    def __init__(self, learning_rate=1., decay=0., momentum=0.):
        # learning-rate=the rate at which weights and bias should be changed
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate

        # decay=As we get closer to the global minima,it is necesary to reduce
        # the learning rate,so global minima is not overtaken
        self.decay = decay

        # the number of times optimizer changes the values
        self.iterations = 0

        # momentum=It may happen that the optimizer is stuck in a local minima
        # to escape this it is necesarry to increase the loss for some time
        # to get out of the local minima
        self.momentum = momentum

    def pre_update_params(self):
        '''Runs before updating parameters'''

        # reduce the learning rate with each passing iteration
        if self.decay:

            self.current_learning_rate = self.learning_rate * \
                (1./(1+self.decay*self.iterations))

    def update_params(self, layer):
        '''Update the weights and biases of the given layer'''

        if self.momentum:

            # if the layer has no weight/bias momentum(i.e during the first call)
            if not hasattr(layer, 'weight_momentum'):

                # start with 0 momentum
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

                # weight_change = momentum_factor*weight_momentum - learning_rate*dweights
                weight_updates = self.momentum*layer.weight_momentums - \
                    self.current_learning_rate*layer.dweights

                # momentum is proportional to speed analogy
                layer.weight_momentums = weight_updates

                # bias_change = momentum_factor*bias_momentum - learning_rate*dbiases
                bias_updates = self.momentum*layer.bias_momentums - \
                    self.current_learning_rate*layer.dbiases

                # momentum is proportional to speed analogy
                layer.bias_momentums = bias_updates

        else:
            weight_updates = -self.current_learning_rate*layer.dweights
            bias_updates = -self.current_learning_rate*layer.dbiases

        # update weights and biases
        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_params(self):
        '''Runs after updating params'''

        self.iterations += 1
