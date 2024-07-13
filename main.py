from tqdm import tqdm
import torch
import torch.nn as nn
import random
import math

class FAModel(nn.Module):
    def __init__(self, shape):
        super(FAModel, self).__init__()
        self.shape = shape
        
        self.weights = [torch.randn(shape[i + 1], shape[i]) for i in range(len(shape) - 1)]
        self.alphas = [torch.randn(shape[i + 1], shape[i]) for i in range(len(shape) - 1)]
        self.resetTraces()

        self.feedback_connections = [torch.randn(shape[i + 1], shape[i]) for i in range(len(shape) - 1)]

        self.biases = [torch.randn(shape[i], 1) for i in range(1, len(shape))]
        self.activations = [None] * (len(self.weights) + 1)

        self.resetDerivatives()
        
        self.eta = 0.01
        self.lr = 0.1
    
    def forward(self, x):
        for i in range(len(self.weights)):
            self.activations[i] = x.clone()

            plasticWeights = self.weights[i] + self.alphas[i] * self.traces[i]
            x = torch.sigmoid(torch.matmul(plasticWeights, torch.where(x > 0.5, 1.0, 0.0)) + self.biases[i])

            self.traces[i] = (1 - self.eta) * self.traces[i] + self.eta * torch.matmul(x, self.activations[i].T)
        
        self.activations[-1] = x.clone()
        return x

    def backpropagate(self, error):
        def tanh_derivative(x):
            return (1-x) * x
        for i in reversed(range(len(self.weights))):
            activation = self.activations[i + 1]
            prev_activation = torch.where(self.activations[i] > 0.5, 1.0, 0.0)
            grad_activation = error * tanh_derivative(activation)


            grad_sum = self.alphas[i] * prev_activation.repeat(1, self.traces[i].size()[0]).T

            grad_weights = torch.matmul(grad_activation, prev_activation.T) + grad_sum * self.weight_derivatives[i] * (1 - self.eta)
            grad_alphas = grad_weights * self.traces[i] + grad_sum * self.alpha_derivatives[i] * (1 - self.eta)


            # plasticWeights = self.weights[i] + self.alphas[i] * self.traces[i]
            # feedback alignment
            error = torch.matmul(self.feedback_connections[i].T, grad_activation) 

            self.weights[i] -= self.lr * grad_weights
            self.alphas[i] -= self.lr * grad_alphas
            self.biases[i] -= self.lr * grad_activation

            self.weight_derivatives[i] += prev_activation.repeat(1, self.traces[i].size()[0]).T * grad_weights
            self.weight_derivatives[i] *= self.eta

            self.alpha_derivatives[i] += prev_activation.repeat(1, self.traces[i].size()[0]).T * grad_alphas
            self.alpha_derivatives[i] *= self.eta

    def resetTraces(self):
        self.traces = [torch.zeros(self.shape[i + 1], self.shape[i]) for i in range(len(self.shape) - 1)]

    def resetDerivatives(self):
        self.weight_derivatives = [torch.zeros(self.shape[i + 1], self.shape[i]) for i in range(len(self.shape) - 1)]
        self.alpha_derivatives = [torch.zeros(self.shape[i + 1], self.shape[i]) for i in range(len(self.shape) - 1)]

model = FAModel(shape=[2, 10, 1])

dataset = [
    ([[0.0], [0.0]], [0.0]),
    ([[0.0], [1.0]], [1.0]),
    ([[1.0], [0.0]], [1.0]),
    ([[1.0], [1.0]], [0.0])
]

for epoch in tqdm(range(10000)):
    sample = random.choice(dataset)
    input_tensor = torch.Tensor(sample[0])
    target_tensor = torch.Tensor(sample[1])

    output = model.forward(input_tensor)
    error = (output - target_tensor)
    model.backpropagate(error)

print("Predictions after training:")
print("Input: [0, 0], Output:", model.forward(torch.Tensor([[0.0], [0.0]])))
print("Input: [0, 1], Output:", model.forward(torch.Tensor([[0.0], [1.0]])))
print("Input: [1, 0], Output:", model.forward(torch.Tensor([[1.0], [0.0]])))
print("Input: [1, 1], Output:", model.forward(torch.Tensor([[1.0], [1.0]])))
