import torch
import torchvision
import math
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, densenet121, resnet50
from torchvision.models import densenet201, resnet101, resnet152
from torchvision.models import vgg19_bn
#import  efficientnet_pytorch as efficientnet
from torch.autograd import Variable
from torch_geometric.nn import GraphConv, TopKPooling,GCNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import argparse
from torch_geometric.nn import DataParallel
from torch_geometric.nn import radius_graph
from torch.autograd import Function

class GradReverse(Function):
    # 重写父类方法的时候，最好添加默认参数，不然会有warning（为了好看。。）
    @staticmethod
    def forward(ctx, x, lambd, **kwargs: None):
        # 　其实就是传入dict{'lambd' = lambd}
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, *grad_output):
        # 传入的是tuple，我们只需要第一个
        return grad_output[0] * -ctx.lambd, None

    # 这样写是没有warning，看起来很舒服，但是显然是多此一举咯，所以也可以改写成

    def backward(ctx, grad_output):
        # 直接传入一格数
        return grad_output * -ctx.lambd, None


def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)
