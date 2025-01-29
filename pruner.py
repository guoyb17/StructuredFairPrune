import torch
import torch.nn as nn
import torch_pruning as tp
from torch_pruning.pruner import function


class FairPruneImportance(tp.importance.MagnitudeImportance):
    def __init__(self,
                 light_weight,
                 dark_weight,
                 group_reduction:str="mean", 
                 normalizer:str='mean', 
                 bias=False,
                 target_types:list=[nn.modules.conv._ConvNd, nn.Linear, nn.modules.batchnorm._BatchNorm]):
        self.group_reduction = group_reduction
        self.normalizer = normalizer
        self.target_types = target_types
        self.bias = bias
        self._accu_grad_light = {}
        self._counter_light = {}
        self._accu_grad_dark = {}
        self._counter_dark = {}
        self.light_weight = light_weight
        self.dark_weight = dark_weight

    def zero_grad_light(self):
        self._accu_grad_light = {}
        self._counter_light = {}

    def zero_grad_dark(self):
        self._accu_grad_dark = {}
        self._counter_dark = {}

    def accumulate_grad_light(self, model):
        for name, param in model.named_parameters():
            if param.grad is not None:
                if name not in self._accu_grad_light:
                    self._accu_grad_light[param] = param.grad.data.clone().pow(2)
                else:
                    self._accu_grad_light[param] += param.grad.data.clone().pow(2)
                
                if name not in self._counter_light:
                    self._counter_light[param] = 1
                else:
                    self._counter_light[param] += 1

    def accumulate_grad_dark(self, model):
        for name, param in model.named_parameters():
            if param.grad is not None:
                if name not in self._accu_grad_dark:
                    self._accu_grad_dark[param] = param.grad.data.clone().pow(2)
                else:
                    self._accu_grad_dark[param] += param.grad.data.clone().pow(2)
                
                if name not in self._counter_dark:
                    self._counter_dark[param] = 1
                else:
                    self._counter_dark[param] += 1
    
    @torch.no_grad()
    def __call__(self, group, ch_groups=1):
        group_imp = []
        group_idxs = []

        if len(self._accu_grad_light) > 0: # fill gradients so that we can re-use the implementation for Taylor
            for p, g in self._accu_grad_light.items():
                p.grad.data = self.light_weight * g / self._counter_light[p]
            self.zero_grad_light()

        if len(self._accu_grad_dark) > 0:
            for p, g in self._accu_grad_dark.items():
                p.grad.data += self.dark_weight * g / self._counter_dark[p]
            self.zero_grad_dark()

        for i, (dep, idxs) in enumerate(group):
            idxs.sort()
            layer = dep.target.module
            prune_fn = dep.handler
            root_idxs = group[i].root_idxs

            if not isinstance(layer, tuple(self.target_types)):
                continue

            if prune_fn in [
                function.prune_conv_out_channels,
                function.prune_linear_out_channels,
            ]:
                if layer.weight.grad is not None:
                    if hasattr(layer, "transposed") and layer.transposed:
                        w = layer.weight.data.transpose(1, 0)[idxs].flatten(1)
                        h = layer.weight.grad.data.transpose(1, 0)[idxs].flatten(1)
                    else:
                        w = layer.weight.data[idxs].flatten(1)
                        h = layer.weight.grad.data[idxs].flatten(1)

                    local_imp = (w**2 * h).sum(1)
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)
                
                if self.bias and layer.bias is not None and layer.bias.grad is not None:
                    b = layer.bias.data[idxs]
                    h = layer.bias.grad.data[idxs]
                    local_imp = (b**2 * h)
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)
                    
            # Conv in_channels
            elif prune_fn in [
                function.prune_conv_in_channels,
                function.prune_linear_in_channels,
            ]:
                if layer.weight.grad is not None:
                    if hasattr(layer, "transposed") and layer.transposed:
                        w = (layer.weight).flatten(1)[idxs]
                        h = (layer.weight.grad).flatten(1)[idxs]
                    else:
                        w = (layer.weight).transpose(0, 1).flatten(1)[idxs]
                        h = (layer.weight.grad).transpose(0, 1).flatten(1)[idxs]

                    local_imp = (w**2 * h).sum(1)
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)

            # BN
            elif prune_fn == function.prune_groupnorm_out_channels:
                # regularize BN
                if layer.affine:

                    if layer.weight.grad is not None:
                        w = layer.weight.data[idxs]
                        h = layer.weight.grad.data[idxs]
                        local_imp = (w**2 * h)
                        group_imp.append(local_imp)
                        group_idxs.append(root_idxs)

                    if self.bias and layer.bias is not None and layer.bias.grad is None:
                        b = layer.bias.data[idxs]
                        h = layer.bias.grad.data[idxs]
                        local_imp = (b**2 * h).abs()
                        group_imp.append(local_imp)
                        group_idxs.append(root_idxs)

        if len(group_imp) == 0: # skip groups without parameterized layers
            return None
        group_imp = self._reduce(group_imp, group_idxs)
        group_imp = self._normalize(group_imp, self.normalizer)
        return group_imp
