from torch import nn
import torch.nn.functional as F
import torchvision
import torch
import numpy as np
import math

class _ConvNd(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias, padding_mode):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.padding_mode = padding_mode
        self.groups = groups
        if transposed:
            self.weight = nn.parameter.Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = nn.parameter.Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.parameter.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

class Conv2dSamePadding(_ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        kernel_size = torch.nn.modules.utils._pair(kernel_size)
        stride = torch.nn.modules.utils._pair(stride)
        padding = torch.nn.modules.utils._pair(padding)
        dilation = torch.nn.modules.utils._pair(dilation)
        super(Conv2dSamePadding, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, torch.nn.modules.utils._pair(0), groups, bias, padding_mode)

    def forward(self, input, stride=1, dilation=1):
        return conv2d_same_padding(input, self.weight, self.bias, stride,
                        self.padding, dilation, self.groups, padding_mode=self.padding_mode)

def conv2d_same_padding(input, weight, bias=None, stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros'):
    dilation = torch.nn.modules.utils._pair(dilation)
    #print("stride ->", stride)
    input_rows = input.size(2)
    filter_rows = weight.size(2)
    effective_filter_size_rows = (filter_rows - 1) * dilation[1] + 1
    out_rows = (input_rows + stride - 1) // stride
    padding_needed = max(0, (out_rows - 1) * stride + effective_filter_size_rows -
                  input_rows)
    padding_rows = max(0, (out_rows - 1) * stride +
                        (filter_rows - 1) * dilation[0] + 1 - input_rows)
    rows_odd = (padding_rows % 2 != 0)
    padding_cols = max(0, (out_rows - 1) * stride +
                        (filter_rows - 1) * dilation[1] + 1 - input_rows)
    cols_odd = (padding_rows % 2 != 0)

    if rows_odd or cols_odd:
        input = torch.nn.functional.pad(input, [0, int(cols_odd), 0, int(rows_odd)], mode=padding_mode)

    return F.conv2d(input, weight, bias, stride,
                  padding=(padding_rows // 2, padding_cols // 2),
                  dilation=dilation, groups=groups)


class SDCLayerSharedWeights(nn.Module):
    def __init__(self, input_size, n_conv, kernel_size, n_kernels, stride, padding): # image_size
        super(SDCLayerSharedWeights, self).__init__()
        self.input_size = input_size
        self.n_conv = n_conv
        self.kernel_size = kernel_size
        self.n_kernels = n_kernels
        #self.dilations = dilations
        self.stride = stride
        self.padding = padding
        
        self.dilated_conv = Conv2dSamePadding(self.input_size, 
                                              self.n_kernels, 
                                              self.kernel_size, 
                                              stride, 
                                              padding, 
                                              padding_mode='circular')
        
        self.activation = nn.ReLU() # ELU
        
        self.reduction = nn.Conv2d(self.input_size, self.n_kernels, kernel_size=1, stride=2, padding=0)
        
        self.softmax = nn.Softmax(dim=1)
       

    def forward(self, x):
        dilated_outputs = []
        dilated_outputs_test = []
        
        self_dilations = [(1,1), (1,2), (1,3), (1,4)]

        if self.stride==2 or self.stride==(2,2): 
            h, w = int(x.shape[2]/2), int(x.shape[3]/2)
            x = self.reduction(x)
        else: 
            h, w = x.shape[2], x.shape[3]

        n_convs = self.n_conv        
        hardcoding_weights = np.zeros((h, n_convs))           
        M = np.zeros((h,w))                                                             
        phi = np.linspace(-np.pi/2, np.pi/2, num=h, endpoint=True)
        for idx in range(h):
            M[idx, :] = 1/np.cos(phi[idx])
        
        ideal_dilations = torch.from_numpy(M[:, 0]).to("cuda")                 
        ideal_dilations = ideal_dilations.reshape(ideal_dilations.shape[0],1) 

        dilation_rates=[1,2,3,4]  
        
        for i in range(h):
            value = ideal_dilations[i,0]          

            if value >= 4:
                hardcoding_weights[i,3] = 1.0
                hardcoding_weights[i,:3] = 0.0
                hardcoding_weights[-1] = hardcoding_weights[0]           
            else:
                differences = [abs(value - dilations) for dilations in dilation_rates] 
                differences = torch.tensor(differences)                                 
                idx = np.argsort(differences)[:2]                                       
                smallest_differences = differences[idx]                           
		
                inverse_differences = 1 / smallest_differences
                
                if math.isinf(inverse_differences[0]): 
                    inverse_differences[0] = 10 
                weights_sum = inverse_differences.sum()
                normalized_weights = inverse_differences / weights_sum

                hardcoding_weights[i, idx[0]] = normalized_weights[0]
                hardcoding_weights[i, idx[1]] = normalized_weights[1]

                for j in range(1, 4):
                    hardcoding_weights[-j] = hardcoding_weights[j-1]

        final_weights = torch.from_numpy(hardcoding_weights).to("cuda") 
        final_weights = final_weights.to(torch.float32)   
        
        for i in range(self.n_conv):           
            x_d = self.dilated_conv(x, dilation=self_dilations[i]).to("cuda")
                
            '''applying hardcoding weights to each for in the dilated feature maps'''
            m = final_weights[:,i].unsqueeze(1)
            s = m * x_d # learnable weights * feature maps
            dilated_outputs.append(s)   

        stacked = torch.stack(dilated_outputs, dim=0)    
        final = torch.sum(stacked, dim=0)
        
        return final 

