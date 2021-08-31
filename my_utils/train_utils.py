"""
By yancy F. 2021-8-29.
"""
import torch
import torch.nn as nn


def accuracy(predictions, targets):
    # predictions: (n, nc), targets: (n,)
    predictions = predictions.argmax(dim=-1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.shape[0]


class MMD_loss(nn.Module):
    """
    Code source: https://github.com/ZongxianLee/MMD_Loss.Pytorch
    """

    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        """

        :param source:(n_sample, n_feature)
        :param target:(n_sample, n_feature)
        :param kernel_mul:多核MMD，以bandwidth为中心，两边扩展的基数，比如bandwidth/kernel_mul, bandwidth, bandwidth*kernel_mul
        :param kernel_num:取不同高斯核的数量
        :param fix_sigma:是否固定，如果固定，则为单核MMD
        :return:
        """
        n_samples = int(source.shape[0]) + int(target.shape[0])
        # 一般source和target的尺度是一样的，这样便于计算
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(total.shape[0], total.shape[0], total.shape[1])
        total1 = total.unsqueeze(1).expand(total.shape[0], total.shape[0], total.shape[1])
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = source.shape[0]
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul,
                                       kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY - YX)
        return loss
