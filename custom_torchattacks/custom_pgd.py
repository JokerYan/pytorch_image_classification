import numpy as np
import torch
import torch.nn as nn

import torchvision.transforms as torch_transforms
from torchattacks import PGD

from pytorch_image_classification.transforms import _get_dataset_stats


class CustomPGD(PGD):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 0.3)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 40)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=40, random_start=True)
        >>> adv_images = attack(images, labels)

    """
    def __init__(self, config, model, eps=0.3,
                 alpha=2/255, steps=40, random_start=True):
        super(CustomPGD, self).__init__(model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self._supported_mode = ['default', 'targeted']
        self.config = config
        self.mean, self.std = _get_dataset_stats(config)
        self.Normalize = torch_transforms.Normalize(
            self.mean, self.std
        )

    def denormalize(self, images, is_tensor=True):
        if is_tensor:
            images = images.clone().detach().cpu().numpy()
        # image = np.squeeze(images)
        std = np.expand_dims(self.std, [0, 2, 3])
        mean = np.expand_dims(self.mean, [0, 2, 3])

        images = np.multiply(images, std)
        mean = np.multiply(np.ones_like(images), mean)
        images = images + mean
        # images = np.expand_dims(image, 0)
        if is_tensor:
            images = torch.Tensor(images).to(self.device)
        return images

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        # denormalize
        images = self.denormalize(images)

        if self._targeted:
            target_labels = self._get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(self.Normalize(adv_images))

            # Calculate loss
            if self._targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
