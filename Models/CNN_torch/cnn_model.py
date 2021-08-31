import torch.nn as nn
from my_utils.train_utils import accuracy, MMD_loss


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2),
    )


class CNN(nn.Module):
    def __init__(self, in_chn, hidden_chn, cb_num, embedding_size, out_size):
        super().__init__()
        conv1 = conv_block(in_chn, hidden_chn)
        conv_more = [conv_block(hidden_chn, hidden_chn) for _ in range(cb_num - 1)]
        self.feature_net = nn.Sequential(conv1, *conv_more, nn.Flatten())
        self.classifier = nn.Linear(embedding_size, out_size)

    def forward(self, x):
        feat = self.feature_net(x)
        out = self.classifier(feat)
        return out


class CNN_MMD(nn.Module):
    def __init__(self, in_chn, hidden_chn, cb_num, embedding_size, out_size):
        super().__init__()
        conv1 = conv_block(in_chn, hidden_chn)
        conv_more = [conv_block(hidden_chn, hidden_chn) for _ in range(cb_num - 1)]
        self.feature_net = nn.Sequential(conv1, *conv_more, nn.Flatten())
        self.classifier = nn.Linear(embedding_size, out_size)
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean')
        self.mmd_loss = MMD_loss()

    def forward(self, x_s, y_s, x_t=None):
        f_s = self.feature_net(x_s)
        o_s = self.classifier(f_s)
        if self.training:
            f_t = self.feature_net(x_t)
            loss = self.ce_loss(o_s, y_s) + self.mmd_loss(f_s, f_t)
        else:
            loss = self.ce_loss(o_s, y_s)
        acc = accuracy(o_s, y_s)

        return loss, acc
