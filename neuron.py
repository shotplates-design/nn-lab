import numpy as np 

x = np.array([1.0, 2.0, 3.0])    # inputs
w = np.array([0.5, -0.2, 0.1])   # weights
b = 0.4                          # bias

# Weighted Sum of Inputs
z = ((x[0]*w[0]) + (x[1]*w[1]) + (x[2]*w[2])) + b

def sigmoid(z):
    return 1 / (1+np.exp(-z))


def neuron(x1, x2, w1, w2, b):
    z = (x1*w1) + (x2*w2) + b
    return sigmoid(z)

out = sigmoid(z)
print("z =", round(z, 4))
print("output =", round(out, 4))
print(round(neuron(34, 18, 0.05, 0.1, -3), 4))






























