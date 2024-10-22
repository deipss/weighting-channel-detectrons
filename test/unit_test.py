import unittest
import torch
import torch.nn as nn


class TestPytorch(unittest.TestCase):
    def test_cross_loss(self):

        criterion = nn.CrossEntropyLoss()
        output=torch.randn(3,5,requires_grad=True)
        target = torch.tensor([1,0,4])
        loss = criterion(output,target);
        print(output.item())
        print("loss=",loss.item())

    def test_l1(self):
        # 假设 batch_size = 3
        # 创建带有不同 reduction 参数的 L1Loss 实例
        criterion_mean = nn.L1Loss(reduction='mean')
        criterion_sum = nn.L1Loss(reduction='sum')
        criterion_none = nn.L1Loss(reduction='none')

        # 假设 batch_size = 3
        predictions = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        targets = torch.tensor([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5], [7.5, 8.5, 9.5]])

        # 计算损失
        loss_mean = criterion_mean(predictions, targets)
        loss_sum = criterion_sum(predictions, targets)
        loss_none = criterion_none(predictions, targets)

        # 打印损失值
        print("Loss (mean):", loss_mean.item())
        print("Loss (sum):", loss_sum.item())
        print("Loss (none):", loss_none)