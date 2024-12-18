import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, layers):
        super().__init__()
        layers = list(zip(layers[:-1], layers[1:]))
        self.w1 = nn.Linear(*layers[0])
        self.relu = nn.ReLU()
        self.w2 = nn.Linear(*layers[1])

    def forward(self, x):
        out = self.w1(x)
        out = self.w2(self.relu(out))
        return out


class Classifier(nn.Module):
    def __init__(self, num_classes, layers):
        super().__init__()
        self.feature_ext = MLP(layers)
        self.classifier = nn.Linear(layers[-1], num_classes)

    def forward(self, x):
        out = self.feature_ext(x)
        return self.classifier(out)


def train():
    model = Classifier(num_classes=4, layers=[5, 7, 6])
    criterion = nn.CrossEntropyLoss()
    torch.manual_seed(0)
    x = torch.randn(3, 5)
    y = (torch.randn(3) > 0.5).long()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for i in range(100):
        pred = model(x)
        loss = criterion(pred, y)
        if i % 10 == 0:
            print(f"Step = {i:2} Loss = {loss.item():.2f}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    train()
