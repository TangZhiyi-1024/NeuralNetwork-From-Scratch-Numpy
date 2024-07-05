import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        self.prediction_tensor = None
        self.label_tensor = None

    def forward(self, prediction_tensor, label_tensor):
        self.prediction_tensor = prediction_tensor
        self.label_tensor = label_tensor
        # 确保预测值不小于0，也不大于1
        self.prediction_tensor = np.clip(self.prediction_tensor, 0, 1)
        # 计算每个样本的交叉熵损失
        epsilon = np.finfo(float).eps  # 使用浮点数的最小正数
        # 只计算正确类别对应的损失
        loss = -np.sum(np.log(self.prediction_tensor[self.label_tensor == 1] + epsilon))
        return loss

    def backward(self, label_tensor):
        error_tensor = -(label_tensor / (self.prediction_tensor + np.finfo(float).eps))
        return error_tensor