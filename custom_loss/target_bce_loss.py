import torch
from torch.nn import BCELoss


class TargetBCELoss:
    def __init__(self):
        self.bce_loss = BCELoss()

    def __call__(self, output, label, first_class, second_class):
        first_output = output[:, first_class]  # batch_size x 1
        second_output = output[:, second_class]  # batch_size x 1

        binary_output = torch.hstack([first_output, second_output])
        binary_softmax_output = torch.softmax(binary_output, dim=1)
        binary_first_class_output = binary_softmax_output[:, 0]

        ones = torch.ones_like(label)
        zeros = torch.zeros_like(label)
        binary_target = torch.where(label == int(first_class), ones, zeros).float()

        loss = self.bce_loss(binary_first_class_output, binary_target)
        return loss


# binary linear loss
class TargetBLLoss:
    def __call__(self, output, label, first_class, second_class):
        first_output = output[:, first_class]  # batch_size x 1
        second_output = output[:, second_class]  # batch_size x 1

        binary_output = torch.hstack([first_output, second_output])
        binary_softmax_output = torch.softmax(binary_output, dim=1)
        # binary_first_class_output = binary_softmax_output[:, 0]

        # ones = torch.ones_like(label)
        # zeros = torch.zeros_like(label)
        # binary_target = torch.where(label == int(first_class), ones, zeros).float()

        loss = torch.mean(binary_softmax_output[:, 0] - binary_softmax_output[:, 1])
        return loss