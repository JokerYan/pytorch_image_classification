import torch
import torch.nn as nn


class LocalLipschitzValueLoss:
    def __init__(self, base_loss_func, logger=None):
        self.base_loss_func = base_loss_func
        self.norm_ratio = 0.1
        self.logger = logger

        self.llv_thresh = 0

    def __call__(self, output, target, model_input):
        base_loss = self.base_loss_func(output, target)

        # max_output.backward(retain_graph=False, create_graph=True)
        # input_grad = model_input.grad.data
        # input_grad_norm = self.get_input_grad_norm(output, model_input)
        input_grad_norm = self.get_input_loss_grad_norm(model_input, base_loss)
        total_loss = base_loss + self.norm_ratio * torch.relu(input_grad_norm - self.llv_thresh)

        msg = "base_loss: {}\tgrad_norm: {}\ttotal_loss: {}".format(base_loss, input_grad_norm, total_loss)
        if self.logger is not None:
            self.logger.info(msg)
        else:
            print(msg)

        # return base_loss
        return total_loss

    @staticmethod
    def get_input_grad_norm(output, model_input, is_train=True):
        assert model_input.requires_grad
        max_output = torch.sum(torch.max(output, dim=1).values)
        input_grad = torch.autograd.grad(max_output, model_input, retain_graph=is_train, create_graph=is_train)[0]
        input_grad_norm = torch.norm(input_grad, p=2)  # l2 norm

        return input_grad_norm

    @staticmethod
    def get_input_loss_grad_norm(model_input, base_loss):
        assert model_input.requires_grad
        input_grad = torch.autograd.grad(base_loss, model_input, retain_graph=True, create_graph=True)[0]
        input_grad_norm = torch.norm(input_grad, p=2)

        return input_grad_norm

