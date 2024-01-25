import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch import Tensor

import copy
######################################################## NON TMC MODELS ########################################################
class NetWISDM(nn.Module):
    def __init__(self):
        super(NetWISDM, self).__init__()
        self.der = False
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 64, (9,), stride = 1, padding = 4),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True))
        self.MaxPool = nn.MaxPool1d(2,stride = 2)
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, (5,), stride = 1, padding = 2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True))
        self.AvgPool = nn.AvgPool1d(2,stride = 2)
        
        self.classifier = nn.Sequential(
            nn.Linear(1536, 6)
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = F.dropout(x, 0.3, training=self.training)
        x = self.MaxPool(x)
        x = self.conv2(x)
        x = F.dropout(x, 0.3, training=self.training)
        x = self.AvgPool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        if self.der:
            x = F.softmax(x, dim=1)
        return x
    
    #add n new neurons to the classification layer while keeping the rest of the network unchanged
    def add_new_neurons(self, n):
        #get the size of the last layer
        inp_features = self.classifier[-1].in_features
        out_features = self.classifier[-1].out_features
        weight = copy.deepcopy(self.classifier[-1].weight.data)
        bias = copy.deepcopy(self.classifier[-1].bias.data)
        
        #create a new layer with the same parameters except for the number of features
        self.classifier[-1] = nn.Linear(inp_features, out_features + n)
        
        #initialize the new layer with the same weights and biases as the old one
        self.classifier[-1].weight.data[:out_features] = weight
        self.classifier[-1].bias.data[:out_features] = bias

class NetUCIHAR(nn.Module):
    def __init__(self):
        super(NetUCIHAR, self).__init__()
        self.der = False
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 64, (9,), stride = 1, padding = 4),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True))
        self.MaxPool = nn.MaxPool1d(2,stride = 2)
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, (5,), stride = 1, padding = 2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True))
        self.AvgPool = nn.AvgPool1d(2,stride = 2)
        
        self.classifier = nn.Sequential(
            nn.Linear(17920, 6)
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = F.dropout(x, 0.3, training=self.training)
        x = self.MaxPool(x)
        x = self.conv2(x)
        x = F.dropout(x, 0.3, training=self.training)
        x = self.AvgPool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        if self.der:
            x = F.softmax(x, dim=1)
        return x
    
    #add n new neurons to the classification layer while keeping the rest of the network unchanged
    def add_new_neurons(self, n):
        #get the size of the last layer
        inp_features = self.classifier[-1].in_features
        out_features = self.classifier[-1].out_features
        weight = copy.deepcopy(self.classifier[-1].weight.data)
        bias = copy.deepcopy(self.classifier[-1].bias.data)
        
        #create a new layer with the same parameters except for the number of features
        self.classifier[-1] = nn.Linear(inp_features, out_features + n)
        
        #initialize the new layer with the same weights and biases as the old one
        self.classifier[-1].weight.data[:out_features] = weight
        self.classifier[-1].bias.data[:out_features] = bias

class NetUSCHAD(nn.Module):
    def __init__(self):
        super(NetUSCHAD, self).__init__()
        self.der = False
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 64, (9,), stride = 1, padding = 4),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True))
        self.MaxPool = nn.MaxPool1d(2,stride = 2)
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, (5,), stride = 1, padding = 2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True))
        self.AvgPool = nn.AvgPool1d(2,stride = 2)
        
        self.classifier = nn.Sequential(
            nn.Linear(3200, 12)
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = F.dropout(x, 0.3, training=self.training)
        x = self.MaxPool(x)
        x = self.conv2(x)
        x = F.dropout(x, 0.3, training=self.training)
        x = self.AvgPool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        if self.der:
            x = F.softmax(x, dim=1)
        return x
    
    #add n new neurons to the classification layer while keeping the rest of the network unchanged
    def add_new_neurons(self, n):
        #get the size of the last layer
        inp_features = self.classifier[-1].in_features
        out_features = self.classifier[-1].out_features
        weight = copy.deepcopy(self.classifier[-1].weight.data)
        bias = copy.deepcopy(self.classifier[-1].bias.data)
        
        #create a new layer with the same parameters except for the number of features
        self.classifier[-1] = nn.Linear(inp_features, out_features + n)
        
        #initialize the new layer with the same weights and biases as the old one
        self.classifier[-1].weight.data[:out_features] = weight
        self.classifier[-1].bias.data[:out_features] = bias


def obtain_logits(model, data): #for DER and DER++
    initial_state = model.der

    if initial_state == True:
        model.der = False

    logit = model(data)

    if model.der != initial_state:
        model.der = initial_state
        
    return logit
####################################################################################################################################




######################################################## TMC MODELS ########################################################
class NetWISDM_TMC(nn.Module):
    def __init__(self):
        super(NetWISDM_TMC, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, (9,), stride = 1, padding = 4),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True))
        self.MaxPool = nn.MaxPool1d(2,stride = 2)
        self.layertmc1 = LinearSequential(
            LinearConv1D(64, 128, (5,), stride = 1, padding = 2),
            LinearBatchNorm1d(128),
            LinearReLU(inplace=True))
        self.layertmc2 = LinearDropout(0.3)
        self.layertmc3 = LinearAvgPool1d(2,stride = 2)
        
        self.classifier = LinearSequential(
            LinearLinear(1536, 6)
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = F.dropout(x, 0.3, training=self.training)
        x = self.MaxPool(x)

        x_jvp = torch.zeros_like(x)

        x, x_jvp = self.layertmc1(x, x_jvp)
        x, x_jvp = self.layertmc2(x, x_jvp)
        x, x_jvp = self.layertmc3(x, x_jvp)

        x, x_jvp = torch.flatten(x, 1), torch.flatten(x_jvp, 1)
        x, x_jvp = self.classifier(x, x_jvp)
        return x, x_jvp
    
    #add n new neurons to the classification layer while keeping the rest of the network unchanged
    def add_new_neurons(self, n):
        #get the size of the last layer
        inp_features = self.classifier[-1].in_features
        out_features = self.classifier[-1].out_features
        weight = copy.deepcopy(self.classifier[-1].weight.data)
        bias = copy.deepcopy(self.classifier[-1].bias.data)
        
        #create a new layer with the same parameters except for the number of features
        self.classifier[-1] = nn.Linear(inp_features, out_features + n)
        
        #initialize the new layer with the same weights and biases as the old one
        self.classifier[-1].weight.data[:out_features] = weight
        self.classifier[-1].bias.data[:out_features] = bias

class NetUCIHAR_TMC(nn.Module):
    def __init__(self):
        super(NetUCIHAR_TMC, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, (9,), stride = 1, padding = 4),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True))
        self.MaxPool = nn.MaxPool1d(2,stride = 2)
        self.layertmc1 = LinearSequential(
            LinearConv1D(64, 128, (5,), stride = 1, padding = 2),
            LinearBatchNorm1d(128),
            LinearReLU(inplace=True))
        self.layertmc2 = LinearDropout(0.3)
        self.layertmc3 = LinearAvgPool1d(2,stride = 2)
        
        self.classifier = LinearSequential(
            LinearLinear(17920, 6)
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = F.dropout(x, 0.3, training=self.training)
        x = self.MaxPool(x)

        x_jvp = torch.zeros_like(x)

        x, x_jvp = self.layertmc1(x, x_jvp)
        x, x_jvp = self.layertmc2(x, x_jvp)
        x, x_jvp = self.layertmc3(x, x_jvp)
        
        x, x_jvp = torch.flatten(x, 1), torch.flatten(x_jvp, 1)
        x, x_jvp = self.classifier(x, x_jvp)
        return x, x_jvp
    
    #add n new neurons to the classification layer while keeping the rest of the network unchanged
    def add_new_neurons(self, n):
        #get the size of the last layer
        inp_features = self.classifier[-1].in_features
        out_features = self.classifier[-1].out_features
        weight = copy.deepcopy(self.classifier[-1].weight.data)
        bias = copy.deepcopy(self.classifier[-1].bias.data)
        
        #create a new layer with the same parameters except for the number of features
        self.classifier[-1] = nn.Linear(inp_features, out_features + n)
        
        #initialize the new layer with the same weights and biases as the old one
        self.classifier[-1].weight.data[:out_features] = weight
        self.classifier[-1].bias.data[:out_features] = bias

class NetUSCHAD_TMC(nn.Module):
    def __init__(self):
        super(NetUSCHAD_TMC, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, (9,), stride = 1, padding = 4),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True))
        self.MaxPool = nn.MaxPool1d(2,stride = 2)
        self.layertmc1 = LinearSequential(
            LinearConv1D(64, 128, (5,), stride = 1, padding = 2),
            LinearBatchNorm1d(128),
            LinearReLU(inplace=True))
        self.layertmc2 = LinearDropout(0.3)
        self.layertmc3 = LinearAvgPool1d(2,stride = 2)
        
        self.classifier = LinearSequential(
            LinearLinear(3200, 12)
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = F.dropout(x, 0.3, training=self.training)
        x = self.MaxPool(x)

        x_jvp = torch.zeros_like(x)

        x, x_jvp = self.layertmc1(x, x_jvp)
        x, x_jvp = self.layertmc2(x, x_jvp)
        x, x_jvp = self.layertmc3(x, x_jvp)
        
        x, x_jvp = torch.flatten(x, 1), torch.flatten(x_jvp, 1)
        x, x_jvp = self.classifier(x, x_jvp)
        return x, x_jvp
    
    #add n new neurons to the classification layer while keeping the rest of the network unchanged
    def add_new_neurons(self, n):
        #get the size of the last layer
        inp_features = self.classifier[-1].in_features
        out_features = self.classifier[-1].out_features
        weight = copy.deepcopy(self.classifier[-1].weight.data)
        bias = copy.deepcopy(self.classifier[-1].bias.data)
        
        #create a new layer with the same parameters except for the number of features
        self.classifier[-1] = LinearLinear(inp_features, out_features + n)
        
        #initialize the new layer with the same weights and biases as the old one
        self.classifier[-1].weight.data[:out_features] = weight
        self.classifier[-1].bias.data[:out_features] = bias

####################################################################################################################################


######################################################## TMC MODIFIED LAYERS #######################################################
NOISE_SCALE = 1e-12

class LinearConv2D(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros'  # TODO: refine this type
    ):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation
            ,groups, bias, padding_mode)

        self.linear_weight = Parameter(torch.zeros_like(self.weight, requires_grad=True) + NOISE_SCALE)
        if bias:
            self.linear_bias = Parameter(torch.zeros_like(self.bias, requires_grad=True) + NOISE_SCALE)
        else:
            self.linear_bias = None

    def _linear_forward(self, input: Tensor) -> Tensor:
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            self.linear_weight, self.linear_bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, self.linear_weight, self.linear_bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def _conv_forward(self, input: Tensor, add_bias: bool = True) -> Tensor:
        bias = self.bias if add_bias else None
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            self.weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, self.weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input: Tensor, input_jvp: Tensor) -> Tensor:
        output = self._conv_forward(input)
        output_jvp = self._linear_forward(input) + self._conv_forward(input_jvp, add_bias=False)
        return output, output_jvp
    
class LinearConv1D(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros'  # TODO: refine this type
    ):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
                         groups, bias, padding_mode)
        
        self.linear_weight = Parameter(torch.zeros_like(self.weight, requires_grad=True) + NOISE_SCALE)
        if bias:
            self.linear_bias = Parameter(torch.zeros_like(self.bias, requires_grad=True) + NOISE_SCALE)
        else:
            self.linear_bias = None

    def _linear_forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.padding_mode != 'zeros':
            return F.conv1d(F.pad(input, self._reversed_padding_repeated_once, mode=self.padding_mode),
                            self.linear_weight, self.linear_bias, self.stride,
                            0, self.dilation, self.groups)
        return F.conv1d(input, self.linear_weight, self.linear_bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def _conv_forward(self, input: torch.Tensor, add_bias: bool = True) -> torch.Tensor:
        bias = self.bias if add_bias else None
        if self.padding_mode != 'zeros':
            return F.conv1d(F.pad(input, self._reversed_padding_repeated_once, mode=self.padding_mode),
                            self.weight, bias, self.stride,
                            0, self.dilation, self.groups)
        return F.conv1d(input, self.weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input: torch.Tensor, input_jvp: torch.Tensor) -> torch.Tensor:
        output = self._conv_forward(input)
        output_jvp = self._linear_forward(input) + self._conv_forward(input_jvp, add_bias=False)
        return output, output_jvp

class LinearLinear(nn.Linear):

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__(in_features, out_features, bias)
        self.linear_weight = Parameter(torch.zeros_like(self.weight, requires_grad=True) + NOISE_SCALE)
        if bias:
            self.linear_bias = Parameter(torch.zeros_like(self.bias, requires_grad=True) + NOISE_SCALE)
        else:
            self.linear_bias = None

    def forward(self, input: Tensor, input_jvp: Tensor) -> Tensor:
        output = super().forward(input)
        output_jvp = F.linear(input, self.linear_weight, self.linear_bias) + F.linear(input_jvp, self.weight, None)
        return output, output_jvp

class LinearReLU(nn.ReLU):

    def forward(self, input: Tensor, input_jvp: Tensor) -> Tensor:
        output = super().forward(input)
        output_jvp = input_jvp * (output > 0).float()
        return output, output_jvp
    
class LinearDropout(nn.Module):
    def __init__(self, p=0.5):
        super(LinearDropout, self).__init__()
        self.p = p

    def forward(self, input: Tensor, input_jvp: Tensor) -> Tensor:
        if self.training:
            device = input.device
            mask = torch.rand(input.size(), device = device) > self.p
            masked_input = input * mask.float()
            masked_input_jvp = input_jvp * mask.float()
        else:
            masked_input = input
            masked_input_jvp = input_jvp

        return masked_input, masked_input_jvp

class LinearSequential(nn.Sequential):

    def forward(self, *input):
        for module in self:
            input = module(*input)
        return input

class LinearBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine=True,
                 track_running_stats=True) -> None:
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        if affine:
            self.linear_weight = Parameter(torch.zeros_like(self.weight, requires_grad=True) + NOISE_SCALE)
            self.linear_bias = Parameter(torch.zeros_like(self.bias, requires_grad=True) + NOISE_SCALE)

    def _linear_forward(self, input: Tensor) -> Tensor:
        return F.batch_norm(input, self.running_mean, self.running_var,
                            self.linear_weight, self.linear_bias, training=False, momentum=self.momentum, eps=self.eps)

    def _jvp_forward(self, input: Tensor) -> Tensor:
        return F.batch_norm(input,
                            torch.zeros_like(self.running_mean), self.running_var, self.weight, None, training=False,
                            momentum=self.momentum, eps=self.eps)

    def forward(self, input: Tensor, input_jvp:Tensor) -> Tensor:
        output = super().forward(input)
        output_jvp = self._linear_forward(input) + self._jvp_forward(input_jvp)
        return output, output_jvp
    
class LinearBatchNorm1d(nn.BatchNorm1d):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine=True,
                 track_running_stats=True) -> None:
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        if affine:
            self.linear_weight = Parameter(torch.zeros_like(self.weight, requires_grad=True) + NOISE_SCALE)
            self.linear_bias = Parameter(torch.zeros_like(self.bias, requires_grad=True) + NOISE_SCALE)

    def _linear_forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.batch_norm(input, self.running_mean, self.running_var,
                            self.linear_weight, self.linear_bias, training=False,
                            momentum=self.momentum, eps=self.eps)

    def _jvp_forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.batch_norm(input, torch.zeros_like(self.running_mean), self.running_var,
                            self.weight, None, training=False,
                            momentum=self.momentum, eps=self.eps)

    def forward(self, input: torch.Tensor, input_jvp: torch.Tensor) -> torch.Tensor:
        output = super().forward(input)
        output_jvp = self._linear_forward(input) + self._jvp_forward(input_jvp)
        return output, output_jvp

class LinearAdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):

    def __init__(self, output_size) -> None:
        super().__init__(output_size)

    def forward(self, input, input_jvp):
        return super().forward(input), super().forward(input_jvp)

class LinearAvgPool1d(nn.AvgPool1d):
    def __init__(self, kernel_size, stride=None, padding=0) -> None:
        super().__init__(kernel_size, stride=stride, padding=padding)

    def forward(self, input, input_jvp):
        return super().forward(input), super().forward(input_jvp)

class LinearMaxPool1d(nn.MaxPool1d):
    def __init__(self, kernel_size, stride=None, padding=0) -> None:
        super().__init__(kernel_size, stride=stride, padding=padding)

    def forward(self, input, input_jvp):
        return super().forward(input), super().forward(input_jvp)
    
####################################################################################################################################
