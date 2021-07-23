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
            print("output before softmax", output)
            output = torch.softmax(output, dim=1)
            print("output after softmax", output)
            output_list.append(output)
        output_list = torch.stack(output_list)
        output_mean = torch.mean(output_list, dim=0)
        output_variance = torch.var(output_list, dim=0)

        # calculate output_variance after sigmoid
        variance_thresh = 0.03
        variance_temp = 0.01
        print("variance before sigmoid", output_variance)
        output_variance_sigmoid = torch.sigmoid((output_variance - variance_thresh) / variance_temp)
        print("variance after sigmoid", output_variance_sigmoid)

        # concatenate output with one extra fake output
        # if the original output is of dim n, the new output is of dim (n + 1)
        output_c = torch.min(output_mean, (1 - output_variance_sigmoid))
        output_fake = torch.max(output_variance_sigmoid, dim=0, keepdim=True).values
        output_final = torch.cat([output_c, output_fake], 0)
        print("output_mean", output_mean)
        print("output_c", output_c)
        print("output_fake", output_fake)
        print("output_final", output_final)

        return output_final




