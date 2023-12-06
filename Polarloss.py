import numpy as np
import torch

class PolarLoss():

    def __init__(self, weight=None, size_average=True):
        """
        初始化参数，因为要实现 torch.nn.CrossEntropyLoss 的两个比较重要的参数

        :param weight: 给予每个类别不同的权重
        :param size_average: 是否要对 loss 求平均
        """

        self.weight = weight
        self.size_average = size_average


    def __call__(self, input, target, confi):
        """
        计算损失
        这个方法让类的实例表现的像函数一样，像函数一样可以调用

        :param input: (batch_size, C)，C是类别的总数
        :param target: (batch_size, 1)
        :return: 损失
        """

        batch_loss = 0.
        polar_input = torch.zeros(input.shape[0])
        polar_target = torch.zeros(input.shape[0])

        for i in range(input.shape[0]):
            # 计算单个损失
            if -0.5 * np.pi < input[i] < 0.5 * np.pi:
                polar_input[i] = 0
            else:
                polar_input[i] = 1
            if -0.5 * np.pi < target[i] < 0.5 * np.pi:
                polar_target[i] = 0
            else:
                polar_target[i] = 1
            loss = confi[i] * (polar_input[i] - polar_target[i]) ** 2
            # loss = (polar_input[i] - polar_target[i]) ** 2

            if self.weight:
                loss = self.weight[target[i]] * loss
            # print("单个损失： ", loss)

            # 损失累加
            batch_loss += loss

        # 整个 batch 的总损失是否要求平均
        if self.size_average == True:
            batch_loss /= input.shape[0]

        return batch_loss
