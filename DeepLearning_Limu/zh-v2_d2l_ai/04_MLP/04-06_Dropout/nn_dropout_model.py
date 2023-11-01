import torch
from torch import nn


class Dropout2Model(nn.Module):
    def __init__(self, dropout1, dropout2):
        super(Dropout2Model, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3072, 256),
            nn.ReLU(),
            # 第一个全连接层后加dropout
            nn.Dropout(dropout1),
            nn.Linear(256, 64),
            nn.ReLU(),
            # 第二个全连接层之后加dropout
            nn.Dropout(dropout2),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    
    dropout1, dropout2 = 0.2, 0.5

    model = Dropout2Model(dropout1, dropout2)
    input = torch.ones((64, 3, 32, 32))
    output = model(input)

    print(output.shape)