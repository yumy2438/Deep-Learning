from neural_network_utils import *
from dataset_operations import load_dataset


train_x, train_y, test_x, test_y, classes = load_dataset()
# Reshape the training and test examples
train_x_flatten = train_x.reshape(train_x.shape[0], -1).T
test_x_flatten = test_x.reshape(test_x.shape[0], -1).T

train_x = train_x_flatten / 255.
test_x = test_x_flatten / 255.
layers_dims = [12288, 20, 7, 5, 1]  # 4-layer model

neuralNet = NeuralNetwork()
neuralNet.train_model_GD(train_x,train_y, layers_dims, 3000, 0.0075, 0.7, "Xavier", True,"model_gd2")
batchneural = NeuralNetwork()
batchneural.train_model_with_mini_batches(train_x,train_y,layers_dims,0.0075, 0.7, num_epoch=3000, optimizer="momentum", print_cost=True, model_name="momentummoel2")
"""
neuralNet = NeuralNetwork()
neuralNet.load_trained_model("model_gd2.h5")
print(neuralNet.get_accuracy_on(test_x,test_y))
neuralNet.load_trained_model("momentummoel2.h5")
print(neuralNet.get_accuracy_on(test_x,test_y))
"""