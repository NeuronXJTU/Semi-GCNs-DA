import torch
import torchvision
import math
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, densenet121, resnet50
from torchvision.models import densenet201, resnet101, resnet152
from torchvision.models import vgg19_bn
from torch.autograd import Variable
from torch_geometric.nn import GraphConv, TopKPooling,GCNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import argparse
from torch_geometric.nn import DataParallel
from torch_geometric.nn import radius_graph
from functions import grad_reverse



class All_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = Resnet18(3,256)
        self.G = Mgcnlinear(256, 2)
        self.classModel = Predictor_deep(2, 256, 0.05)

    def forward(self, x):
        feature = self.feature(x)
        GPre = self.G(feature)
        ClassPre = self.classModel(feature)
        return GPre,ClassPre,feature


class Predictor_deep(nn.Module):
    def __init__(self, num_class=2, inc=256, temp=0.05):
        super(Predictor_deep, self).__init__()
        self.fc1 = nn.Linear(inc, 124)
        self.fc2 = nn.Linear(124, num_class, bias=False)
        self.num_class = num_class
        self.temp = temp

    def forward(self, x, eta=0.1):
        x = self.fc1(x)
        x = grad_reverse(x, eta)
        x = F.normalize(x)
        x_out = self.fc2(x) / self.temp
#        x_out = F.sigmoid(x_out)
        return x_out

class Resnet18(nn.Module):
    '''
    ClassifyNet feature
    '''
    def __init__(self, channel_size=3, num_classes=2):
        super().__init__()
        self.resnet = resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(
            channel_size, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.linear = torch.nn.Linear(512, num_classes)
        self.resnet.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        for m in self.resnet.conv1.modules():
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu'
            )
        for m in self.resnet.fc.modules():
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.resnet.avgpool(x)
        x = torch.flatten(x,start_dim=1)
        x = self.linear(x)
        return x


class Mgcnlinear(nn.Module):
    '''
    Mgcnlinear
    '''
    def L2_dist(self,x,y):
        dist = torch.reshape( torch.sum(x*x,1), (-1,1) )
        dist = dist.expand( dist.shape[0], dist.shape[0] )
        dist2 = torch.reshape(torch.sum(y*y,1), (1,-1))
        dist2 = dist2.expand( dist2.shape[1], dist2.shape[1] )
        dist = dist + dist2
        dist -= 2.0*torch.mm(x, torch.transpose(y,1,0))
        return dist
    
    def __init__(self, channel_size=256, num_classes=2):
        super().__init__()
        self.linear1 = torch.nn.Linear(channel_size, 256)
        self.linear2 = torch.nn.Linear(256, num_classes)
        self.conv1 = GraphConv(256, 256)
        self.conv2 = GraphConv(256, 128)
        self.conv3 = GraphConv(256, 256)
    
    def forward(self, x):
        '''
        if self.training:
           #print('yes!!!!!!!!!!!this way')
           #F.softmax(x,dim=-1).shape=[32,256]
           edge_index = radius_graph(F.softmax(x,dim=-1), 3, None, True, 3)
        else:
           edge_index = radius_graph(x, 0.000001, None, True, 2)
        '''
        edge_index = radius_graph(F.softmax(x,dim=-1), 3, None, True, 3)
        x = F.relu(self.linear1(x))
        x1 = F.relu(self.conv1(x,edge_index))          
        x = self.linear2(x1)
#        x = F.sigmoid(x)
        return x,x1


