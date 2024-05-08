import torch.nn as nn
from functions import ReverseLayerF
from torchvision import models


class CNNModel(nn.Module):

    def __init__(self):
        super(CNNModel, self).__init__()
        base_model = models.resnet50(pretrained=True)
        self.conv1 = nn.Conv2d(1, 3, 3, padding=1, bias=False)
        self.feature = nn.Sequential(*list(base_model.children())[:-2])
        self.GAvgPool = nn.AvgPool2d(kernel_size=5)
        
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(2048, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout2d())
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, 8))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax())

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(2048, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))
        # self.float()

    def forward(self, input_data, alpha):
        # input_data = input_data.expand(input_data.data.shape[0], 1, 256, 256)
        # feature = self.feature(input_data)
        feature = self.conv1(input_data)
        feature = self.feature(feature)
        feature = self.GAvgPool(feature)
        # print(feature.shape)
        feature = feature.view(feature.size(0), feature.size(1))
        # print(feature.shape)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output