import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
import numpy as np


class CnnEncoder(nn.Module):
    def __init__(self):
        super(CnnEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(4, 2),
            nn.Conv2d(8, 8, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(4, 2)
        )

    def forward(self, view_state):
        # [batch, 128]
        x = self.encoder(view_state)
        return x


class ContinuousActor(nn.Module):
    def __init__(
        self, encoder, device
    ):
        """Initialize."""
        super(ContinuousActor, self).__init__()

        self.device = device

        self.encoder = encoder
        self.cnn_net = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: [16, 4, 4]

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: [32, 2, 2]

            nn.Flatten()
        )

        self.hidden = nn.Linear(128, 32)

        self.mu_layer = nn.Linear(32, 2)
        self.log_std_layer = nn.Linear(32, 2)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        # for name, parameter in self.named_parameters():
        #     print(name, "actor parameter")

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""

        x_1 = self.encoder(state)
        x_2 = self.cnn_net(x_1)
        # print(x_2.shape, "!@$%$%#$%#$@")
        x_3 = self.hidden(x_2)
        x_4 = self.relu(x_3)

        x_mu = self.mu_layer(x_4)
        x_log = self.log_std_layer(x_4)

        mu = self.tanh(x_mu)
        log_std = self.tanh(x_log)

        std = torch.exp(log_std)
        dist = Normal(mu, std)
        action = dist.sample()

        return action, dist
        
        
class Critic(nn.Module):
    def __init__(self, encoder, device):
        """Initialize."""
        super(Critic, self).__init__()

        self.device = device

        self.encoder = encoder
        self.cnn_net = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: [16, 4, 4]

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: [32, 2, 2]

            nn.Flatten()
        )

        self.relu = nn.ReLU()

        self.hidden = nn.Linear(128, 32)
        self.out = nn.Linear(32, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""

        x_1 = self.encoder(state)
        x_2 = self.cnn_net(x_1)
        x_3 = self.hidden(x_2)
        x_4 = self.relu(x_3)
        value = self.out(x_4)

        return value
def init_weights(m):
    if type(m) in (nn.Linear, nn.Conv2d):
        nn.init.orthogonal_(m.weight.data, np.sqrt(float(2)))
        if m.bias is not None:
            m.bias.data.fill_(0)


class Memory:
    """Storing the memory of the trajectory (s, a, r ...)."""
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.is_terminals = []
        self.log_probs = []
        self.values = []

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.is_terminals = []
        self.log_probs = []
        self.values = []
