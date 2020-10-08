import h5py
from dnn_utils import *
from dataset_operations import shuffle_dataset


class NeuralNetwork:
    def __init__(self):
        self.parameters = None
        self.error = False


    def train_model_GD(self, X, Y, layer_dims, num_iterations, learning_rate=.075, l2_lambda=0, initialize_type="Xavier", print_cost=False, model_name=None):
        self.parameters = initialize_params(initialize_type, layer_dims)
        if not self.parameters:
            return
        for i in range(0, num_iterations):
            cost = one_iteration_of_network(X, Y, self.parameters, learning_rate, l2_lambda)
            if print_cost and i % 100 == 0:
                print("Cost after iteration %i: %f" % (i, cost))
        if model_name:
            self.save_parameters(model_name)

    def train_model_with_mini_batches(self, X, Y, layer_dims, learning_rate=0.075, l2_lambda=0, batch_size=32, num_epoch=100, initialize_type="Xavier", optimizer="gd", beta_momentum=.9, print_cost=False, model_name=None):
        self.parameters = initialize_params(initialize_type, layer_dims)
        if not self.parameters:
            return
        if optimizer == "momentum":
            velocities = initialize_velocities(self.parameters)
        shuffled_X, shuffled_Y = shuffle_dataset(X, Y)
        batch_indexes = get_batches_index(Y, batch_size)
        for i in range(num_epoch):
            for (start_ind, end_ind) in batch_indexes:
                mini_batch_X = shuffled_X[:, start_ind:end_ind]
                mini_batch_Y = shuffled_Y[:, start_ind:end_ind]
                cost = one_iteration_of_network(mini_batch_X, mini_batch_Y, self.parameters, learning_rate, l2_lambda, optimizer, velocities, beta_momentum)
            if print_cost and i % 100 == 0:
                print("Cost after iteration %i: %f" % (i, cost))
        if model_name:
            self.save_parameters(model_name)

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
        if self.error:
            return
        try:
            if '.h5' not in name:
                self.error = True
                raise Exception("Please give a h5 file.")
            f = h5py.File(name, 'r')
            self.parameters = {}
            for key in f.keys():
                self.parameters[key] = f[key][:]
            f.close()
        except Exception as exp:
            self.error = True
            print("No such file:"+str(exp))

    def save_parameters(self, model_name):
        """
        Create a hdf5 file that contains the coefficients of the parameters.
        :return:
        """
        if self.error:
            return
        try:
            if '.' in model_name:
                print("Please give a model name without extension.")
                self.error = True
                return
            f = h5py.File(model_name + '.h5', 'w')
            for key in self.parameters:
                f.create_dataset(key, data=self.parameters[key])
            f.close()
        except Exception as e:
            self.error = True
            print(e)

    def get_accuracy_on(self, test_X, test_Y):
        if self.error:
            return
        test_o = self.predict(test_X)
        test_result = (test_o > 0.5).astype(int)
        result = (test_result - test_Y == 0).astype(int)
        return np.sum(result) / result.shape[1]
