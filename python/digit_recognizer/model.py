# -*- coding: utf-8 -*-
# import torch
from torch import nn
import torch.nn.functional as F


class DigitRecognizer(nn.Module):
    def __init__(self):
        super(DigitRecognizer, self).__init__()

        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        nn.init.xavier_uniform_(self.conv1.weight)

        # self.conv2 = nn.Conv2d(in_channels=self.conv1.out_channels, out_channels=16, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=self.conv1.out_channels, out_channels=64, kernel_size=3)
        nn.init.xavier_uniform_(self.conv2.weight)

        self.fc1 = nn.Linear(in_features=self.conv2.out_channels * 5 * 5, out_features=800)
        nn.init.xavier_uniform_(self.fc1.weight)

        self.fc2 = nn.Linear(in_features=self.fc1.out_features, out_features=800)
        nn.init.xavier_uniform_(self.fc2.weight)

        self.fc3 = nn.Linear(in_features=self.fc2.out_features, out_features=9)
        nn.init.xavier_uniform_(self.fc3.weight)

        self.freezable = [self.conv1, self.conv2]

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def freeze(self):
        for layer in self.freezable:
            for param in layer.parameters():
                param.requires_grad = False
