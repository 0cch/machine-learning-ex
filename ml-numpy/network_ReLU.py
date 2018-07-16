# -*- coding: utf-8 -*-

import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
    
def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

def ReLU(z):
    return np.where(z > 0.0, z, 0.01 * z)

def ReLU_prime(z):
    return np.where(z > 0.0, 1.0, 0.01)

class Network(object):
    def __init__(self, sizes, active, active_prime):
        self.sizes = sizes
        self.layer_number = len(sizes)
        self.bias = [np.random.randn(x, 1) * 0.1 for x in sizes[1:]]
        self.weights = [np.random.randn(y,x) * 0.1 for x, y in zip(sizes[:-1], sizes[1:])]
        self.active = active
        self.active_prime = active_prime
        
    def feedforward(self, a):
        for w, b in zip(self.weights, self.bias):
            a = self.active(w.dot(a)+b)
        return a
    
    def cost_derivative(self, output_activations, y):
        return (output_activations-y)
    
    def evaluate(self, test_data):
        test_result = [(np.argmax(self.feedforward(x)), y) for x, y in test_data]
        return sum([int(x==y) for x, y in test_result])
    
    def backprop(self, x, y):
        nabla_w = [np.zeros(x.shape) for x in self.weights]
        nabla_b = [np.zeros(x.shape) for x in self.bias]
        activation = x
        activations = [x]
        zs = []
        for w,b in zip(self.weights, self.bias):
            z = w.dot(activation)+b
            zs.append(z)
            activation = self.active(z)
            activations.append(activation)
        delta = self.cost_derivative(activations[-1], y) * self.active_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = delta.dot(activations[-2].transpose())
        for l in range(2, self.layer_number):
            delta = self.active_prime(zs[-l]) * self.weights[-l+1].transpose().dot(delta)
            nabla_b[-l] = delta
            nabla_w[-l] = delta.dot(activations[-l-1].transpose())
        
        return nabla_w, nabla_b
    
    def update_mini_batch(self, mini_batch, eta):
        nabla_w = [np.zeros(x.shape) for x in self.weights]
        nabla_b = [np.zeros(x.shape) for x in self.bias]
        for x, y in mini_batch:
            delta_nabla_w, delta_nabla_b = self.backprop(x, y)
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            
        self.weights = [w - eta*nw/len(mini_batch) for w, nw in zip(self.weights, nabla_w)]
        self.bias = [b - eta*nb/len(mini_batch) for b, nb in zip(self.bias, nabla_b)]
        
        
    
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        np.random.shuffle(training_data)
        n = len(training_data)
        if test_data is not None:
            n_test = len(test_data)
        
        for i in range(epochs):
            mini_batchs = [training_data[j:j+mini_batch_size] for j in range(0, n, mini_batch_size)]
            for mini_batch in mini_batchs:
                self.update_mini_batch(mini_batch, eta)
            if test_data is not None:
                print("Epoch {} : {} / {}".format(i,self.evaluate(test_data),n_test));
            else:
                print("Epoch {} complete".format(i))