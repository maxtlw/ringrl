import torch
from torch import Tensor
from torch.nn import Conv2d, Linear, Module, ReLU
from torch.nn.functional import layer_norm


class DQN(Module):
    def __init__(self, observations_channels: int, action_space_dim: int):
        super(DQN, self).__init__()
        self.conv1 = Conv2d(observations_channels, 32, kernel_size=8, stride=4)
        self.conv2 = Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = Linear(64 * 7 * 7, 512)
        self.fc2 = Linear(512, action_space_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = layer_norm(x, normalized_shape=[x.shape[-1]])
        x = ReLU()(x)
        x = self.conv2(x)
        x = layer_norm(x, normalized_shape=[x.shape[-1]])
        x = ReLU()(x)
        x = self.conv3(x)
        x = layer_norm(x, normalized_shape=[x.shape[-1]])
        x = ReLU()(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = layer_norm(x, normalized_shape=[x.shape[-1]])
        x = ReLU()(x)
        x = self.fc2(x)
        return x
