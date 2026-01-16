import torch
from torch import Tensor
from torch.nn import Conv2d, Linear, Module, ReLU
from torch.nn.functional import layer_norm


class QNet(Module):
    def __init__(
        self, observations_channels: int, hidden_dim: int, action_space_dim: int
    ):
        super(QNet, self).__init__()
        self.conv1 = Conv2d(
            observations_channels, hidden_dim, kernel_size=3, stride=2, padding=1
        )  # 84x84 -> 42x42
        self.conv2 = Conv2d(
            hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1
        )  # 42x42 -> 21x21
        self.conv3 = Conv2d(
            hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1
        )  # 21x21 -> 11x11
        self.conv4 = Conv2d(
            hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1
        )  # 11x11 -> 6x6
        self.fc1 = Linear(hidden_dim * 6 * 6, hidden_dim)
        self.fc2 = Linear(hidden_dim, action_space_dim)

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
        x = self.conv4(x)
        x = layer_norm(x, normalized_shape=[x.shape[-1]])
        x = ReLU()(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = layer_norm(x, normalized_shape=[x.shape[-1]])
        x = ReLU()(x)
        x = self.fc2(x)
        return x
