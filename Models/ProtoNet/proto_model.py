import torch.nn.modules as nn
import torch


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, max_pool_factor=1.0):
        super().__init__()
        stride = (int(2 * max_pool_factor))
        self.max_pool = nn.MaxPool1d(kernel_size=stride, stride=stride, ceil_mode=False)
        self.normalize = nn.BatchNorm1d(out_channels, affine=True)
        self.relu = nn.ReLU()

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=1, bias=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.normalize(x)
        x = self.relu(x)
        x = self.max_pool(x)
        return x


class ConvBase(nn.Sequential):
    def __init__(self, hidden=64, channels=1, layers=4, max_pool_factor=1.0):
        core = [ConvBlock(channels, hidden, 3, max_pool_factor)]
        for _ in range(layers - 1):
            core.append(ConvBlock(hidden, hidden, 3, max_pool_factor))
        super(ConvBase, self).__init__(*core)


class CNN4Backbone(ConvBase):
    def forward(self, x):
        x = super(CNN4Backbone, self).forward(x)
        x = x.reshape(x.size(0), -1)
        return x


class Net4CNN(torch.nn.Module):
    def __init__(self, hidden_size, layers, channels):
        super().__init__()
        self.features = CNN4Backbone(hidden_size, channels, layers, max_pool_factor=1.)

    def forward(self, x):
        return self.features(x)


if __name__ == "__main__":
    model = Net4CNN(64, 4, 1)
    print(model)