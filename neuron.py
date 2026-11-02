import numpy as np 
import torch 

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

def relu(x):
    if x >=0:
        return x
    else:
        return 0

def forward(x1, x2):
    h1 = relu((0.5)*(x1) + (0.5*x2) - 1)
    h2 = relu((1*x1) - (1*x2) + 0)
    h1 = 0.5
    h2 = 0
    out = (1*h1 + 1*h2 - 1)

    print("h1 =", round(h1, 4), "h2 =", round(h2, 4), "out =", round(out, 4))

    return sigmoid(out)




out = sigmoid(z)
print("z =", round(z, 4))
print("output =", round(out, 4))
print(round(neuron(34, 18, 0.05, 0.1, -3), 2))


x1 = int(input("Enter x1: "))
x2 = int(input("Enter x2: ")) 

print(round(forward(x1, x2), 2))




























