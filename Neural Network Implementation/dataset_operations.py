import os
import numpy as np
import h5py

def load_dataset(train_path, test_path):
    base_dir = 'datasets'
    train_dataset = h5py.File(train_path, 'r')
    train_x = np.array(train_dataset['train_set_x'][:])
    train_y = np.array(train_dataset['train_set_y'][:])

    test_dataset = h5py.File(test_path, 'r')
    test_x = np.array(test_dataset['test_set_x'][:])
    test_y = np.array(test_dataset['test_set_y'][:])

    classes = np.array(test_dataset['list_classes'][:])

    train_y = train_y.reshape((1, train_y.shape[0]))  # (1,m)
    test_y = test_y.reshape((1, test_y.shape[0]))

    return train_x, train_y, test_x, test_y, classes

def shuffle_dataset(X,Y):
    m = Y.shape[1]
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1,m))
    return shuffled_X, shuffled_Y