import torch
import torch.nn as nn


class ReLU:
    def forward(self, x):
        return x * (x > 0)

    def backward(self, x, dout):
        return 1.0 * (x > 0) * dout


class Linear:
    def __init__(self, nin, nout):
        self.w = torch.randn(nin, nout)

    def forward(self, x):
        return x @ self.w

    def backward(self, x, dout):
        dw = (dout @ x).T
        dx = self.w @ dout
        return dx, dw


class MLP:
    def __init__(self, layers):
        layers = list(zip(layers[:-1], layers[1:]))
        self.w1 = Linear(layers[0])
        self.relu = ReLU()
        self.w2 = Linear(layers[1])

    def forward(self, x):
        h1 = self.w1(x)
        act1 = self.relu(h1)
        h2 = self.w2(act1)
        return (h1, act1, h2)

    def backward(self, out, dloss_dh2, lr=0.001):
        h1, act1, h2 = out
        dloss_dact1, dloss_dw2 = self.w2.backward(act1, dloss_dh2)
        dloss_dh1 = self.relu.backward(h1, dloss_dact1)
        dloss_dx, dloss_dw1 = self.w1.backward(act1, dloss_dh1)

        # Also step.
        self.w1.w -= lr * dloss_dw1
        self.w2.w -= lr * dloss_dw2


if __name__ == "__main__":
    # MSE loss
    B, H = 3, 5
    D = 7
    x = torch.randn(B, H)
    y = torch.ones((B,))
    net = MLP([H, D, 1])

    # Training..
    total_steps = 100
    for i in range(total_steps):
        ypred = net(x)
        loss = torch.sum((ypred - y) ** 2)
        print(f"{loss=}")
        dloss_dypred = 2 * (ypred - y)
        net.backward(ypred, dloss_dypred)
