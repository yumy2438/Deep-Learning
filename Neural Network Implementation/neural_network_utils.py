import h5py
from dnn_utils import *


class NeuralNetwork:
    def __init__(self, X=None, Y=None, learning_rate=None, layer_dims=None, initialize_type="Xavier", l2_lambda=0, num_iterations=1500,
                 print_cost=False, model_name=None):
        self.X = X
        self.Y = Y
        self.m = None if Y is None else Y.shape[1]
        self.print_cost = print_cost
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.layer_dims = layer_dims
        self.l2_lambda = l2_lambda
        self.initialize_type = initialize_type
        self.parameters = {}
        self.model_name = model_name

    def train_model(self):
        if self.initialize_type == "Xavier":
            self.parameters = initialize_parameters_Xavier(self.layer_dims)
        elif self.initialize_type == "He":
            self.parameters = initialize_parameters_He(self.layer_dims)
        else:
            print("There is no initialization having the name %s. Please enter Xavier or He." % (self.initialize_type))
            return
        for i in range(0, self.num_iterations):
            A_last, caches = model_forward(self.X, self.parameters)
            cost = compute_cost_with_regularization(A_last, self.Y, self.parameters, self.l2_lambda)
            grads = model_backward(A_last, self.Y, caches, self.l2_lambda)
            update_parameters(self.parameters, grads, self.learning_rate)
            if self.print_cost and i % 100 == 0:
                print("Cost after iteration %i: %f" % (i, cost))
        if self.model_name is not None:
            self.save_parameters()

    def predict(self, X):
        A = X
        L = len(self.parameters) // 2  # 1 w, 1 b
        for l in range(1, L):
            A, _ = linear_activation_forward(A, self.parameters["W" + str(l)], self.parameters["b" + str(l)],
                                             "relu")

        result, _ = linear_activation_forward(A, self.parameters["W" + str(L)], self.parameters["b" + str(L)],
                                              "sigmoid")
        return result

    def load_trained_model(self, name):
        "Load the model with the parameters given in the h5 file."
        try:
            if '.h5' not in name:
                print("Please give a h5 file.")
                return
            f = h5py.File(name, 'r')
            for key in f.keys():
                self.parameters[key] = f[key][:]
            f.close()
        except OSError as oserror:
            print("No such file.")

    def save_parameters(self):
        """
        Create a hdf5 file that contains the coefficients of the parameters.
        :return:
        """
        try:
            if '.' in self.model_name:
                print("Please give a model name without extension.")
                return
            f = h5py.File(self.model_name + '.h5', 'w')
            for key in self.parameters:
                f.create_dataset(key, data=self.parameters[key])
            f.close()
        except Exception as e:
            print(e)

    def accuracy(self, test_X, test_Y):
        test_o = self.predict(test_X)
        test_result = (test_o > 0.5).astype(int)
        result = (test_result - test_Y == 0).astype(int)
        return np.sum(result) / result.shape[1]
