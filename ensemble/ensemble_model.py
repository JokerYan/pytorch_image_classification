import torch
import torch.nn as nn


class EnsembleModel(nn.Module):
    def __init__(self, model_list):
        super(EnsembleModel, self).__init__()
        self.model_list = nn.ModuleList(model_list)

    def forward(self, x):
        output_list = []
        for model in self.model_list:
            output = model(x)
            # output = torch.softmax(output, dim=1)
            output_list.append(output)
        output_list = torch.stack(output_list)
        output_mean = torch.mean(output_list, dim=0)
        output_variance = torch.var(output_list, dim=0)

        # calculate output_variance after sigmoid
        variance_thresh = 0.03
        variance_temp = 0.01
        output_variance_sigmoid = torch.sigmoid((output_variance - variance_thresh) / variance_temp)

        # concatenate output with one extra fake output
        # if the original output is of dim n, the new output is of dim (n + 1)
        output_c = torch.min(output_mean, (1 - output_variance_sigmoid))
        output_fake = torch.max(output_variance_sigmoid, dim=1, keepdim=True).values
        output_final = torch.cat([output_c, output_fake], 1)

        # return output_final
        return output_mean[0]



