from torch import nn

# Code which defines the structure of the ANN
class ArtificialNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network_stack = nn.Sequential(
            nn.Linear(5, 128),
            nn.Softmax(dim = 1),
            nn.Linear(128, 128),
            nn.Softmax(dim = 1),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        logits = self.network_stack(x)
        return logits