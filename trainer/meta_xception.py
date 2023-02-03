import numbers
from copy import copy

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np



def extract_top_level_dict(current_dict):
    output_dict = dict()
    for key in current_dict.keys():
        name = key.replace("layer_dict.", "")
        name = name.replace("layer_dict.", "")
        name = name.replace("block_dict.", "")
        name = name.replace("module-", "")
        top_level = name.split(".")[0]
        sub_level = ".".join(name.split(".")[1:])

        if top_level not in output_dict:
            if sub_level == "":
                output_dict[top_level] = current_dict[key]
            else:
                output_dict[top_level] = {sub_level: current_dict[key]}
        else:
            new_item = {key: value for key, value in output_dict[top_level].items()}
            new_item[sub_level] = current_dict[key]
            output_dict[top_level] = new_item

    #print(current_dict.keys(), output_dict.keys())
    return output_dict

class MetaConv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_bias, groups=1, dilation_rate=1):
        super(MetaConv2dLayer, self).__init__()
        num_filters = out_channels
        self.stride = stride
        self.padding = padding
        self.dilation_rate = dilation_rate
        self.use_bias = use_bias
        self.groups = groups
        self.weight = nn.Parameter(torch.empty(num_filters, in_channels//self.groups, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight)

        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(num_filters))

    def forward(self, x, params=None):
        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            if self.use_bias:
                (weight, bias) = params["weight"], params["bias"]
            else:
                (weight) = params["weight"]
                bias = None
        else:
            if self.use_bias:
                weight, bias = self.weight, self.bias
            else:
                weight = self.weight
                bias = None

        out = F.conv2d(input=x, weight=weight, bias=bias, stride=self.stride,
                       padding=self.padding, dilation=self.dilation_rate, groups=self.groups)
        return out

class MetaLinearLayer(nn.Module):
    def __init__(self, in_channels, out_channels, use_bias):
        super(MetaLinearLayer, self).__init__()

        self.use_bias = use_bias
        self.weights = nn.Parameter(torch.ones(out_channels, in_channels))
        nn.init.xavier_uniform_(self.weights)
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x, params=None):
        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            if self.use_bias:
                (weight, bias) = params["weights"], params["bias"]
            else:
                (weight) = params["weights"]
                bias = None
        else:
            pass
            #print('no inner loop params', self)

            if self.use_bias:
                weight, bias = self.weights, self.bias
            else:
                weight = self.weights
                bias = None
        # print(x.shape)
        out = F.linear(input=x, weight=weight, bias=bias)
        return out

class MetaBatchNormLayer(nn.Module):
    def __init__(self, num_features, device, args, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, meta_batch_norm=True, no_learnable_params=False,
                 use_per_step_bn_statistics=False):
        super(MetaBatchNormLayer, self).__init__()
        self.num_features = num_features
        self.eps = eps

        self.affine = affine
        self.track_running_stats = track_running_stats
        self.meta_batch_norm = meta_batch_norm
        self.num_features = num_features
        self.device = device
        self.use_per_step_bn_statistics = use_per_step_bn_statistics
        self.args = args
        self.learnable_gamma = self.args.learnable_bn_gamma
        self.learnable_beta = self.args.learnable_bn_beta

        if use_per_step_bn_statistics:
            self.running_mean = nn.Parameter(torch.zeros(args.number_of_training_steps_per_iter, num_features),
                                             requires_grad=False)
            self.running_var = nn.Parameter(torch.ones(args.number_of_training_steps_per_iter, num_features),
                                            requires_grad=False)
            self.bias = nn.Parameter(torch.zeros(args.number_of_training_steps_per_iter, num_features),
                                     requires_grad=self.learnable_beta)
            self.weight = nn.Parameter(torch.ones(args.number_of_training_steps_per_iter, num_features),
                                       requires_grad=self.learnable_gamma)
        else:
            self.running_mean = nn.Parameter(torch.zeros(num_features), requires_grad=False)
            self.running_var = nn.Parameter(torch.ones(num_features), requires_grad=False)
            self.bias = nn.Parameter(torch.zeros(num_features),
                                     requires_grad=self.learnable_beta)
            self.weight = nn.Parameter(torch.ones(num_features),
                                       requires_grad=self.learnable_gamma)

        if self.args.enable_inner_loop_optimizable_bn_params:
            self.bias = nn.Parameter(torch.zeros(num_features),
                                     requires_grad=self.learnable_beta)
            self.weight = nn.Parameter(torch.ones(num_features),
                                       requires_grad=self.learnable_gamma)

        self.backup_running_mean = torch.zeros(self.running_mean.shape)
        self.backup_running_var = torch.ones(self.running_var.shape)

        self.momentum = momentum

    def forward(self, input, num_step, params=None, training=False, backup_running_statistics=False):
        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            (weight, bias) = params["weight"], params["bias"]
            #print(num_step, params['weight'])
        else:
            #print(num_step, "no params")
            weight, bias = self.weight, self.bias

        if self.use_per_step_bn_statistics:
            running_mean = self.running_mean[num_step]
            running_var = self.running_var[num_step]
            if params is None:
                if not self.args.enable_inner_loop_optimizable_bn_params:
                    bias = self.bias[num_step]
                    weight = self.weight[num_step]
        else:
            running_mean = self.running_mean
            running_var = self.running_var


        if backup_running_statistics and self.use_per_step_bn_statistics:
            self.backup_running_mean.data = copy(self.running_mean.data)
            self.backup_running_var.data = copy(self.running_var.data)

        momentum = self.momentum

        output = F.batch_norm(input, running_mean, running_var, weight, bias,
                              training=True, momentum=momentum, eps=self.eps)

        return output

    def restore_backup_stats(self):
        """
        Resets batch statistics to their backup values which are collected after each forward pass.
        """
        if self.use_per_step_bn_statistics:
            self.running_mean = nn.Parameter(self.backup_running_mean.to(device=self.device), requires_grad=False)
            self.running_var = nn.Parameter(self.backup_running_var.to(device=self.device), requires_grad=False)

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

class MetaConvBatchNormLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_bias, args, normalization=True,
                 meta_layer=True, no_bn_learnable_params=False, device=None):
        super(MetaConvBatchNormLayer, self).__init__()
        self.normalization = normalization
        self.use_per_step_bn_statistics = args.per_step_bn_statistics
        self.in_channels = in_channels
        self.args = args
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = use_bias
        self.meta_layer = meta_layer
        self.no_bn_learnable_params = no_bn_learnable_params
        self.device = device
        self.layer_dict = nn.ModuleDict()
        self.conv = MetaConv2dLayer(in_channels=self.in_channels, out_channels=self.out_channels,
                                    kernel_size=self.kernel_size,
                                    stride=self.stride, padding=self.padding, use_bias=self.use_bias)

        if self.normalization:
            self.norm_layer = MetaBatchNormLayer(self.out_channels, track_running_stats=True,
                                                 meta_batch_norm=self.meta_layer,
                                                 no_learnable_params=self.no_bn_learnable_params,
                                                 device=self.device,
                                                 use_per_step_bn_statistics=self.use_per_step_bn_statistics,
                                                 args=self.args)

    def forward(self, x, num_step, params=None, training=False, backup_running_statistics=False):
        batch_norm_params = None
        conv_params = None
        if params is not None:
            params = extract_top_level_dict(current_dict=params)

            if self.normalization:
                if 'norm_layer' in params:
                    batch_norm_params = params['norm_layer']

                if 'activation_function_pre' in params:
                    activation_function_pre_params = params['activation_function_pre']

            conv_params = params['conv']

        out = x

        out = self.conv.forward(out, params=conv_params)

        if self.normalization:
            out = self.norm_layer.forward(out, num_step=num_step,
                                          params=batch_norm_params, training=training,
                                          backup_running_statistics=backup_running_statistics)
        return out

    def restore_backup_stats(self):
        """
        Restore stored statistics from the backup, replacing the current ones.
        """
        if self.normalization:
            self.norm_layer.restore_backup_stats()

class MetaSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_bias):
        super(MetaSeparableConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = use_bias
        self.conv = MetaConv2dLayer(in_channels=self.in_channels, out_channels=self.in_channels,
                                    kernel_size=self.kernel_size, groups=self.in_channels,
                                    stride=self.stride, padding=self.padding, use_bias=self.use_bias)

        self.pointwise = MetaConv2dLayer(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1,
                                    stride=1, padding=0, groups=1, dilation_rate=1, use_bias=self.use_bias)

    def forward(self, x, params=None):
        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            conv_params = params['conv']
            pointwise_params = params['pointwise']
        out = x
        out = self.conv.forward(out, params=conv_params)
        out = self.pointwise.forward(out, params=pointwise_params)

        return out

class MetaSeparableConv2dBatchNormLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_bias, args, normalization=True,
                 meta_layer=True, no_bn_learnable_params=False, device=None):
        super(MetaSeparableConv2dBatchNormLayer, self).__init__()
        self.normalization = normalization
        self.use_per_step_bn_statistics = args.per_step_bn_statistics
        self.in_channels = in_channels
        self.args = args
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = use_bias
        self.meta_layer = meta_layer
        self.no_bn_learnable_params = no_bn_learnable_params
        self.device = device
        self.layer_dict = nn.ModuleDict()
        self.conv = MetaSeparableConv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                        kernel_size=self.kernel_size, stride=self.stride,
                                        padding=self.padding, use_bias=self.use_bias)

        if self.normalization:
            self.norm_layer = MetaBatchNormLayer(self.out_channels, track_running_stats=True,
                                                 meta_batch_norm=self.meta_layer,
                                                 no_learnable_params=self.no_bn_learnable_params,
                                                 device=self.device,
                                                 use_per_step_bn_statistics=self.use_per_step_bn_statistics,
                                                 args=self.args)

    def forward(self, x, num_step, params=None, training=False, backup_running_statistics=False):
        batch_norm_params = None
        conv_params = None
        if params is not None:
            params = extract_top_level_dict(current_dict=params)

            if self.normalization:
                if 'norm_layer' in params:
                    batch_norm_params = params['norm_layer']

                if 'activation_function_pre' in params:
                    activation_function_pre_params = params['activation_function_pre']

            conv_params = params['conv']

        out = x


        out = self.conv.forward(out, params=conv_params)

        if self.normalization:
            out = self.norm_layer.forward(out, num_step=num_step,
                                          params=batch_norm_params, training=training,
                                          backup_running_statistics=backup_running_statistics)
        return out

    def restore_backup_stats(self):
        """
        Restore stored statistics from the backup, replacing the current ones.
        """
        if self.normalization:
            self.norm_layer.restore_backup_stats()

class MetaBlock(nn.Module):
    def __init__(self, args, device, in_channels, out_channels, reps, strides=1, start_with_relu=True, grow_first=True):
        super(MetaBlock, self).__init__()
        self.args = args
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reps = reps
        self.strides = strides
        self.start_with_relu = start_with_relu
        self.grow_first = grow_first
        self.device = device
        self.layer_dict = nn.ModuleDict()

        filters = self.in_channels
        if self.grow_first:
            self.sepb1 = MetaSeparableConv2dBatchNormLayer(in_channels=self.in_channels, out_channels=self.out_channels,
                                                           kernel_size=3, stride=1, padding=1,use_bias=False,
                                                           args=self.args, normalization=True, meta_layer=True,
                                                           no_bn_learnable_params=False, device=self.device)
            filters = self.out_channels
        else:
            self.sepb1 = None
        for i in range(self.reps -1):
            self.layer_dict['conv{}'.format(i)] = MetaSeparableConv2dBatchNormLayer(in_channels=filters,
                                                                                    out_channels=filters,
                                                                                    kernel_size=3, stride=1, padding=1,
                                                                                    use_bias=False, args=self.args,
                                                                                    normalization=True, meta_layer=True,
                                                                                    no_bn_learnable_params=False,
                                                                                    device=self.device)
        if not self.grow_first:
            self.sepb2 = MetaSeparableConv2dBatchNormLayer(in_channels=self.in_channels, out_channels=self.out_channels,
                                                           kernel_size=3, stride=1, padding=1, use_bias=False,
                                                           args=self.args, normalization=True, meta_layer=True,
                                                           no_bn_learnable_params=False, device=self.device)
        else:
            self.sepb2 = None


        if self.out_channels != self.in_channels or self.strides != 1:
            self.skip = MetaConvBatchNormLayer(in_channels=self.in_channels, out_channels=self.out_channels,
                                               kernel_size=1, stride=self.strides, padding=0,
                                               use_bias=False, args=self.args, normalization=True,
                                               meta_layer=True, no_bn_learnable_params=False, device=self.device)
        else:
            self.skip = None

    def forward(self, x, num_step, params=None, training=False, backup_running_statistics=False):
        param_dict = dict()
        if params is not None:
            param_dict = extract_top_level_dict(current_dict=params)

        for name, param in self.layer_dict.named_parameters():
            path_bits = name.split(".")
            layer_name = path_bits[0]
            if layer_name not in param_dict:
                param_dict[layer_name] = None

        out = x
        if self.grow_first:
            if self.start_with_relu:
                out = F.leaky_relu(out)
            out = self.sepb1.forward(out, params=param_dict['sepb1'], training=training, num_step=num_step,
                                     backup_running_statistics=backup_running_statistics)
        for i in range(self.reps -1):
            out = F.leaky_relu(out)
            out = self.layer_dict['conv{}'.format(i)](out, params=param_dict['conv{}'.format(i)], training=training,
                                                      backup_running_statistics=backup_running_statistics,
                                                      num_step=num_step)
        if  not self.grow_first:
            out = F.leaky_relu(out)
            out = self.sepb2.forward(out, params=param_dict['sepb2'], training=training,num_step=num_step,
                                     backup_running_statistics=backup_running_statistics)
        if self.strides != 1:
            out = F.max_pool2d(input=out, kernel_size=(3, 3), stride=self.strides, padding=1)

        if self.skip is not None:
            out2 = self.skip(x, params=param_dict['skip'], training=training, num_step=num_step,
                             backup_running_statistics=backup_running_statistics)
        else:
            out2 = x
        out += out2

        return out

    def restore_backup_stats(self):
        if self.sepb1 is not None:
            self.sepb1.restore_backup_stats()
        if self.sepb2 is not None:
            self.sepb2.restore_backup_stats()
        if self.skip is not None:
            self.skip.restore_backup_stats()
        for i in range(self.reps -1):
            self.layer_dict['conv{}'.format(i)].restore_backup_stats()

class MetaXception(nn.Module):
    def __init__(self, args, device, inc, num_output_classes, direct=False):
        super(MetaXception, self).__init__()

        self.args = args
        self.device = device
        self.direct = direct
        self.inc = inc
        self.num_block = 12
        self.num_output_classes = num_output_classes
        self.layer_dict = nn.ModuleDict()
        ### Entry flow
        self.convbn1 = MetaConvBatchNormLayer(in_channels=self.inc, out_channels=32,
                                              kernel_size=3, stride=2, padding=0,use_bias=False,
                                              args=self.args, normalization=True,meta_layer=True,
                                              no_bn_learnable_params=False, device=self.device)
        self.convbn2 = MetaConvBatchNormLayer(in_channels=32, out_channels=64, kernel_size=3, stride=1,
                                              padding=0,use_bias=False, args=self.args, normalization=True,
                                              meta_layer=True, no_bn_learnable_params=False, device=self.device)

        self.layer_dict['block1'] = MetaBlock(args=self.args, device=self.device, in_channels=64, out_channels=128,
                                              reps=2, strides=2, start_with_relu=False, grow_first=True)
        self.layer_dict['block2'] = MetaBlock(args=self.args, device=self.device, in_channels=128, out_channels=256,
                                              reps=2, strides=2, start_with_relu=True, grow_first=True)
        self.layer_dict['block3'] = MetaBlock(args=self.args, device=self.device, in_channels=256, out_channels=728,
                                              reps=2, strides=2, start_with_relu=True, grow_first=True)

        #### Middle flow
        for i in range(4,self.num_block):
            self.layer_dict['block{}'.format(i)] = MetaBlock(args=self.args, device=self.device,
                                                             in_channels=728, out_channels=728, reps=3,
                                                             strides=1, start_with_relu=True, grow_first=True)

        ### Exit flow
        self.layer_dict['block{}'.format(self.num_block)] = MetaBlock(args=self.args, device=self.device, in_channels=728, out_channels=1024,
                                               reps=2, strides=2, start_with_relu=True, grow_first=False)

        self.convbn3 = MetaSeparableConv2dBatchNormLayer(in_channels=1024, out_channels=1536,kernel_size=3,
                                                         stride=1, padding=1, use_bias=False,args=self.args,
                                                         normalization=True, meta_layer=True,
                                                         no_bn_learnable_params=False, device=self.device)
        self.convbn4 = MetaSeparableConv2dBatchNormLayer(in_channels=1536, out_channels=2048, kernel_size=3,
                                                         stride=1, padding=1, use_bias=False, args=self.args,
                                                         normalization=True, meta_layer=True,
                                                         no_bn_learnable_params=False, device=self.device)
        self.fc = MetaLinearLayer(in_channels=2048, out_channels=self.num_output_classes, use_bias=True)

    def forward(self, x, num_step, params=None, training=False, backup_running_statistics=False):
        param_dict = dict()
        if params is not None:
            if self.direct:
                params = {key: value[0] for key, value in params.items()}
            param_dict = extract_top_level_dict(current_dict=params)

        for name, param in self.layer_dict.named_parameters():
            path_bits = name.split(".")
            layer_name = path_bits[0]
            if layer_name not in param_dict:
                param_dict[layer_name] = None

        out = x

        ### features
        out = self.convbn1.forward(out, params=param_dict['convbn1'], training=training, num_step=num_step,
                                   backup_running_statistics=backup_running_statistics)
        out = F.leaky_relu(out)
        out = self.convbn2.forward(out, params=param_dict['convbn2'], training=training, num_step=num_step,
                                   backup_running_statistics=backup_running_statistics)
        out = F.leaky_relu(out)

        for i in range(1, self.num_block+1):
            out = self.layer_dict['block{}'.format(i)].forward(out, params=param_dict['block{}'.format(i)],
                                                               training=training, num_step=num_step,
                                                               backup_running_statistics=backup_running_statistics)

        out = self.convbn3.forward(out, params=param_dict['convbn3'], training=training, num_step=num_step,
                                   backup_running_statistics=backup_running_statistics)
        out = F.leaky_relu(out)
        out = self.convbn4.forward(out, params=param_dict['convbn4'], training=training, num_step=num_step,
                                   backup_running_statistics=backup_running_statistics)
        ### Exit
        out = F.leaky_relu(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        features = out
        out = F.dropout(out, 0.5, training)
        out = self.fc.forward(out, params=param_dict['fc'])

        return out, features

    def restore_backup_stats(self):
        self.convbn1.restore_backup_stats()
        self.convbn2.restore_backup_stats()
        self.convbn3.restore_backup_stats()
        self.convbn4.restore_backup_stats()
        for i in range(1,self.num_block+1):
            self.layer_dict['block{}'.format(i)].restore_backup_stats()

    def zero_grad(self, params=None):
        if params is None:
            for param in self.parameters():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            param.grad.zero_()
        else:
            for name, param in params.items():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            param.grad.zero_()
                            params[name].grad = None

