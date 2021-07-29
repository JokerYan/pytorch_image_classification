import torch
import torch.nn as nn


class LocalLipschitzValueLoss:
    def __init__(self, base_loss_func):
        self.base_loss_func = base_loss_func

    def __call__(self, output, target, model_input):
        base_loss = self.base_loss_func(output, target)

        assert model_input.requires_grad
        max_output = torch.sum(torch.max(output, dim=1).values)
        max_output.backward(retain_graph=True)
        input_grad = model_input.grad.data
        input_grad_norm = torch.norm(input_grad, p=2)  # l2 norm
        print("base_loss: {}\tgrad_norm: {}".format(base_loss, input_grad_norm))

        return base_loss
