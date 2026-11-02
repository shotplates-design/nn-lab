import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

x = np.array([34.0, 18.0])            # minutes, shots

# Layer 1: 3 hidden neurons, each with 2 weights
W1 = np.array([[0.05, 0.1], [-0.02, 0.04],[0.03, -0.05]])
b1 = np.array([-3.0, 0.5, 0.2])

# Layer 2: 1 output neuron with 3 weights (one per hidden neuron)
W2 = np.array([[1.5, -1.0, 2.0]])
b2 = np.array([-0.5])

# TODO 1: hidden layer = sigmoid of (W1 times x, plus b1)
h = sigmoid((W1@x) + b1)

# TODO 2: output = sigmoid of (W2 times h, plus b2)
out = sigmoid((W2@h) + b2)

print("hidden =", np.round(h, 4))
print("output =", np.round(out, 4))