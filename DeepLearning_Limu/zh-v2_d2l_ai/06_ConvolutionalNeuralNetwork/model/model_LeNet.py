import torch
from torch import nn

class LeNet_my(nn.Module):
    def __init__(self):
        super(LeNet_my, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(16*5*5, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":

    net = LeNet_my()
    X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)

    output = net(X)

    print(output.shape)

    print(net)
