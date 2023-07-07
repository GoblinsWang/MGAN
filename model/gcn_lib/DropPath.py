import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    # 如果丢弃的概率为0或者模型处于推理状态,那么drop_path相当于Identity操作
    if drop_prob == 0. or not training:
        return x
    # 假设drop_prob=0.2,那么keep_prob=0.8
    keep_prob = 1 - drop_prob
    # x.shape[0]为数据的batch_size, 假设x的形状为(batch_size, C, H, W)那么产生的01矩阵的形状为(batch_size, 1, 1, 1)
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    # 产生01分布的随机数, random_tensor有keep_prob的概率为1
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        # 这里的缩放主要是为了控制期望相同, 在Dropout中我们可以看到相似的操作
        random_tensor.div_(keep_prob)
    # 缩放完成之后对数据进行01的mask操作
    # E(x * random_tensor / keep_prob) = E(x) * E(random_tensor) / keep_prob = E(x) * keep_prob / keep_prob = E(x)
    return x * random_tensor

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)