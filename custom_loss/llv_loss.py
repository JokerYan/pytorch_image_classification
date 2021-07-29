import torch
import torch.nn as nn


class LocalLipschitzValueLoss:
    def __init__(self, base_loss_func, logger=None):
        self.base_loss_func = base_loss_func
        self.norm_ratio = 0.1
        self.logger = logger

    def __call__(self, output, target, model_input):
        base_loss = self.base_loss_func(output, target)

        assert model_input.requires_grad
        max_output = torch.sum(torch.max(output, dim=1).values)
        # max_output.backward(retain_graph=False, create_graph=True)
        # input_grad = model_input.grad.data
        input_grad = torch.autograd.grad(max_output, model_input, retain_graph=True, create_graph=True)[0]
        input_grad_norm = torch.norm(input_grad, p=2)  # l2 norm
        msg = "base_loss: {}\tgrad_norm: {}".format(base_loss, torch.max(input_grad))
        if self.logger is not None:
            self.logger.info(msg)
        else:
            print(msg)

        total_loss = base_loss + self.norm_ratio * input_grad_norm

        # return base_loss
        return total_loss
