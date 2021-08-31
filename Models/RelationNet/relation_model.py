import torch.nn as nn

def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2),
    )


class encoder_net(nn.Module):
    def __init__(self, in_chn, hidden_chn, cb_num=4):
        super().__init__()
        conv1 = conv_block(in_chn, hidden_chn)
        conv1_more = [conv_block(hidden_chn, hidden_chn) for _ in range(cb_num - 1)]
        self.feature_net = nn.Sequential(conv1, *conv1_more)  # (None, 64, 1024/2^4)

    def forward(self, x):
        return self.feature_net(x)


class relation_net(nn.Module):
    def __init__(self, hidden_chn, embed_size, h_size):
        super().__init__()
        conv2 = conv_block(hidden_chn * 2, hidden_chn)
        conv3 = conv_block(hidden_chn, hidden_chn)
        self.relation_net = nn.Sequential(conv2, conv3, nn.Flatten(),
                                          nn.Linear(embed_size, h_size), nn.ReLU(),
                                          nn.Linear(h_size, 1), nn.Sigmoid())
        # self.relation_net = nn.Sequential(conv2, conv3, nn.Flatten(),
        #                                   nn.Linear(embed_size, 1), nn.Sigmoid())

    def forward(self, x):  # (N, CHN, DIM) => (N, 1)
        return self.relation_net(x)
