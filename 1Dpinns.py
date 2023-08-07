import tensorflow as tf
import numpy as np

tf.random.set_seed(1234)


class PhysicsInformedNN:
    def __init__(self, layers):
        self.layers = layers
        self.weights, self.biases = self.initialize_weights()

    def initialize_weights(self):
        weights = []
        biases = []
        num_layers = len(self.layers)

        for l in range(0, num_layers - 1):
            W = tf.Variable(tf.random.normal([self.layers[l], self.layers[l + 1]]), dtype=tf.float32)
            b = tf.Variable(tf.zeros([1, self.layers[l + 1]]), dtype=tf.float32)
            weights.append(W)
            biases.append(b)

        return weights, biases


if __name__ == "__main__":
    layers = [2, 5, 5, 1]
    model = PhysicsInformedNN(layers)
    print(model.weights)


