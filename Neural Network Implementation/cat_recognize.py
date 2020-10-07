import os
from neural_network_utils import *


def load_dataset():
    base_dir = 'datasets'
    train_dataset = h5py.File(os.path.join(base_dir, 'train_catvnoncat.h5'), 'r')
    train_x = np.array(train_dataset['train_set_x'][:])
    train_y = np.array(train_dataset['train_set_y'][:])

    test_dataset = h5py.File(os.path.join(base_dir, 'test_catvnoncat.h5'), 'r')
    test_x = np.array(test_dataset['test_set_x'][:])
    test_y = np.array(test_dataset['test_set_y'][:])

    classes = np.array(test_dataset['list_classes'][:])

    train_y = train_y.reshape((1, train_y.shape[0]))  # (1,m)
    test_y = test_y.reshape((1, test_y.shape[0]))

    return train_x, train_y, test_x, test_y, classes

train_x, train_y, test_x, test_y, classes = load_dataset()
# Reshape the training and test examples
train_x_flatten = train_x.reshape(train_x.shape[0], -1).T
test_x_flatten = test_x.reshape(test_x.shape[0], -1).T
# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten / 255.
test_x = test_x_flatten / 255.

#layers_dims = [12288, 20, 7, 5, 1]  # 4-layer model
#neuralNet = NeuralNetwork(train_x,train_y,0.0075,layers_dims,"Xavier",0.7,1000,True,"model_1000")
#neuralNet.train_model()
neuralNet = NeuralNetwork()
neuralNet.load_trained_model("model_1000.h5")
print(neuralNet.accuracy(test_x,test_y))
