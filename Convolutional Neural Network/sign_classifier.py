import tensorflow as tf
from dataset_operations import *
from cnn_utils import *

X_train, Y_train, X_test,  Y_test, classes = load_dataset('datasets/train_signs.h5', 'datasets/test_signs.h5')
#Normalize them:
X_train = X_train / 255.
X_test = X_test / 255.


Y_train = convert_to_one_hot(Y_train, len(classes)).T
Y_test = convert_to_one_hot(Y_test, len(classes)).T

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16,(3,3), padding='SAME',strides=1, activation='relu'),
    tf.keras.layers.MaxPool2D((8,8),strides=8, padding='SAME'),
    tf.keras.layers.Conv2D(32,(3,3), padding='SAME', strides=1, activation='relu'),
    tf.keras.layers.MaxPool2D((4,4), padding='SAME'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(6, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.009), loss=tf.losses.categorical_crossentropy, metrics=['accuracy'])

model.fit(X_train, Y_train, 64, 100, 2)

model.evaluate(X_test, Y_test)
