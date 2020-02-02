# -*- coding: utf-8 -*-
from torch import nn, Tensor
import torch.nn.functional as F


class DigitRecognizer2(nn.Module):
    def __init__(self):
        super(DigitRecognizer2, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=12, kernel_size=5, padding=1)
        # self.dropout_conv1 = nn.Dropout2d(0.1)
        self.bn1 = nn.BatchNorm2d(self.conv1.out_channels)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=self.conv1.out_channels, out_channels=36, kernel_size=5)
        # self.dropout_conv2= nn.Dropout2d(0.1)
        self.bn2 = nn.BatchNorm2d(self.conv2.out_channels)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        img_size_after_conv = 4
        self.fc1 = nn.Linear(in_features=self.conv2.out_channels * img_size_after_conv * img_size_after_conv,
                             out_features=180)
        # self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(in_features=self.fc1.out_features, out_features=100)
        # self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(in_features=self.fc2.out_features, out_features=9)

        for m in self.modules():
            if m == self \
                    or isinstance(m, nn.Dropout) \
                    or isinstance(m, nn.Dropout2d)\
                    or isinstance(m, nn.MaxPool2d):
                continue

            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain("leaky_relu"))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain("leaky_relu"))
                nn.init.normal_(m.bias)
            else:
                raise Exception("Invalid module: " + repr(m))
            # print(m)

    def forward(self, x: Tensor):
        x = self.conv1(x)
        # x = self.dropout_conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu_(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        # x = self.dropout_conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu_(x)
        x = self.maxpool2(x)

        x = x.view(-1, self.num_flat_features(x))

        x = self.fc1(x)
        # x = self.dropout1(x)
        x = F.leaky_relu_(x)

        x = self.fc2(x)
        # x = self.dropout2(x)
        x = F.leaky_relu_(x)
        x = self.fc3(x)
        return x

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
