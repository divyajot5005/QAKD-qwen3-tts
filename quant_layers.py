import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any


class STEQuantizer(torch.autograd.Function):
    """
    Straight-Through Estimator for Quantization.
    Forward: Quantizes weights/activations to INT4.
    Backward: Passes gradients unchanged (Identity).
    """
    
    @staticmethod
    def forward(
        ctx, 
        input: torch.Tensor, 
        scale: torch.Tensor, 
        zero_point: torch.Tensor,
        bit_width: int = 4,
        symmetric: bool = True
    ) -> torch.Tensor:
        q_min = -(2 ** (bit_width - 1))
        q_max = (2 ** (bit_width - 1)) - 1
        
        input_div = input / scale
        q_input = input_div.round() + zero_point
        q_input = torch.clamp(q_input, q_min, q_max)
        
        output = (q_input - zero_point) * scale
        return output
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None, None, None]:
        return grad_output, None, None, None, None


class STEQuantizerGroupwise(torch.autograd.Function):
    """
    Group-wise Straight-Through Estimator for Quantization.
    Supports block-wise quantization with per-group scale and zero-point.
    """
    
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        bit_width: int = 4,
        group_size: int = 128
    ) -> torch.Tensor:
        q_min = -(2 ** (bit_width - 1))
        q_max = (2 ** (bit_width - 1)) - 1
        
        out_features, in_features = input.shape
        num_groups = in_features // group_size
        
        input_grouped = input.view(out_features, num_groups, group_size)
        scale_expanded = scale.view(out_features, num_groups, 1)
        zero_point_expanded = zero_point.view(out_features, num_groups, 1)
        
        input_div = input_grouped / scale_expanded
        q_input = input_div.round() + zero_point_expanded
        q_input = torch.clamp(q_input, q_min, q_max)
        
        output_grouped = (q_input - zero_point_expanded) * scale_expanded
        output = output_grouped.view(out_features, in_features)
        
        return output.contiguous()
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None, None, None]:
        return grad_output, None, None, None, None


def fake_quantize_int4(
    x: torch.Tensor, 
    scale: torch.Tensor, 
    zero_point: torch.Tensor = None,
    symmetric: bool = True
) -> torch.Tensor:
    if zero_point is None:
        zero_point = torch.zeros_like(scale)
    return STEQuantizer.apply(x, scale, zero_point, 4, symmetric)


def fake_quantize_int4_groupwise(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    group_size: int = 128
) -> torch.Tensor:
    return STEQuantizerGroupwise.apply(x, scale, zero_point, 4, group_size)


class QuantizedLinearINT4(nn.Module):
    """
    A Linear layer that simulates INT4 quantization for weights during training (QAT).
    Supports:
    - Group-wise (block) quantization for weights
    - Activation quantization with learnable scale
    - Learnable zero-points for asymmetric quantization
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        group_size: int = 128,
        quantize_input: bool = True,
        symmetric_weights: bool = True,
        symmetric_activations: bool = False
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.quantize_input = quantize_input
        self.symmetric_weights = symmetric_weights
        self.symmetric_activations = symmetric_activations
        
        assert in_features % group_size == 0, \
            f"in_features ({in_features}) must be divisible by group_size ({group_size})"
        
        self.num_groups = in_features // group_size
        
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.weight_scale = nn.Parameter(torch.ones(out_features, self.num_groups))
        
        if symmetric_weights:
            self.register_buffer('weight_zero_point', torch.zeros(out_features, self.num_groups))
        else:
            self.weight_zero_point = nn.Parameter(torch.zeros(out_features, self.num_groups))
        
        if quantize_input:
            self.input_scale = nn.Parameter(torch.ones(1))
            if symmetric_activations:
                self.register_buffer('input_zero_point', torch.zeros(1))
            else:
                self.input_zero_point = nn.Parameter(torch.zeros(1))
        else:
            self.input_scale = None
            self.input_zero_point = None
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in**0.5)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def get_weight_scale_positive(self) -> torch.Tensor:
        return self.weight_scale.abs() + 1e-8
    
    def get_input_scale_positive(self) -> torch.Tensor:
        if self.input_scale is None:
            return None
        return self.input_scale.abs() + 1e-8
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.quantize_input and self.input_scale is not None:
            input_scale = self.get_input_scale_positive()
            input = fake_quantize_int4(
                input, 
                input_scale, 
                self.input_zero_point,
                symmetric=self.symmetric_activations
            )
        
        weight_scale = self.get_weight_scale_positive()
        w_quant = fake_quantize_int4_groupwise(
            self.weight,
            weight_scale,
            self.weight_zero_point,
            self.group_size
        )
        
        w_quant = w_quant.contiguous()
        
        return F.linear(input, w_quant, self.bias)
    
    @classmethod
    def from_float(
        cls,
        mod: nn.Linear,
        group_size: int = 128,
        quantize_input: bool = True,
        symmetric_weights: bool = True,
        symmetric_activations: bool = False
    ) -> 'QuantizedLinearINT4':
        """
        Convert a standard nn.Linear module to QuantizedLinearINT4.
        Initializes scales based on weight statistics.
        """
        assert isinstance(mod, nn.Linear)
        
        device = mod.weight.device
        dtype = mod.weight.dtype
        
        q_mod = cls(
            mod.in_features,
            mod.out_features,
            bias=mod.bias is not None,
            group_size=group_size,
            quantize_input=quantize_input,
            symmetric_weights=symmetric_weights,
            symmetric_activations=symmetric_activations
        )
        
        q_mod.weight.data = mod.weight.data.clone()
        if mod.bias is not None:
            q_mod.bias.data = mod.bias.data.clone()
        
        weight = mod.weight.data
        out_features, in_features = weight.shape
        num_groups = in_features // group_size
        
        weight_grouped = weight.view(out_features, num_groups, group_size)
        
        if symmetric_weights:
            max_val = weight_grouped.abs().amax(dim=2, keepdim=True).squeeze(-1)
            q_mod.weight_scale.data = max_val / 7.0
        else:
            min_val = weight_grouped.amin(dim=2, keepdim=True).squeeze(-1)
            max_val = weight_grouped.amax(dim=2, keepdim=True).squeeze(-1)
            
            range_val = max_val - min_val
            scale = range_val / 15.0
            zero_point = -min_val / scale - 8.0
            
            q_mod.weight_scale.data = scale
            q_mod.weight_zero_point.data = zero_point.round()
        
        q_mod = q_mod.to(device=device, dtype=dtype)
        
        if hasattr(q_mod, 'weight_zero_point'):
            q_mod.weight_zero_point = q_mod.weight_zero_point.to(device=device, dtype=dtype)
        if hasattr(q_mod, 'input_zero_point') and q_mod.input_zero_point is not None:
            q_mod.input_zero_point = q_mod.input_zero_point.to(device=device, dtype=dtype)
        
        return q_mod
    
    def set_calibration_stats(
        self,
        input_min: float = None,
        input_max: float = None,
        input_scale: float = None,
        input_zero_point: float = None
    ) -> None:
        """
        Set input scale/zero_point from calibration statistics.
        """
        if self.quantize_input and input_scale is not None:
            self.input_scale.data.fill_(input_scale)
            if not self.symmetric_activations and input_zero_point is not None:
                self.input_zero_point.data.fill_(input_zero_point)
        elif self.quantize_input and input_min is not None and input_max is not None:
            if self.symmetric_activations:
                max_abs = max(abs(input_min), abs(input_max))
                self.input_scale.data.fill_(max_abs / 7.0)
            else:
                scale = (input_max - input_min) / 15.0
                zero_point = -input_min / scale - 8.0
                self.input_scale.data.fill_(scale)
                if isinstance(self.input_zero_point, nn.Parameter):
                    self.input_zero_point.data.fill_(zero_point)
    
    def extra_repr(self) -> str:
        return (
            f'in_features={self.in_features}, out_features={self.out_features}, '
            f'bias={self.bias is not None}, group_size={self.group_size}, '
            f'quantize_input={self.quantize_input}'
        )


class QuantizedLinearINT4PerChannel(nn.Module):
    """
    Simpler per-channel (per-output) quantization variant.
    Lower overhead but potentially lower accuracy than group-wise.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        quantize_input: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quantize_input = quantize_input
        
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.weight_scale = nn.Parameter(torch.ones(out_features, 1))
        
        if quantize_input:
            self.input_scale = nn.Parameter(torch.ones(1))
        else:
            self.input_scale = None
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in**0.5)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.quantize_input and self.input_scale is not None:
            s = self.input_scale.abs() + 1e-8
            input = fake_quantize_int4(input, s, torch.zeros_like(s))
        
        s = self.weight_scale.abs() + 1e-8
        w_quant = fake_quantize_int4(self.weight, s, torch.zeros_like(s))
        
        return F.linear(input, w_quant, self.bias)
    
    @classmethod
    def from_float(cls, mod: nn.Linear, quantize_input: bool = True) -> 'QuantizedLinearINT4PerChannel':
        assert isinstance(mod, nn.Linear)
        q_mod = cls(mod.in_features, mod.out_features, bias=mod.bias is not None, 
                    quantize_input=quantize_input)
        q_mod.weight.data = mod.weight.data.clone()
        if mod.bias is not None:
            q_mod.bias.data = mod.bias.data.clone()
        
        max_val = mod.weight.data.abs().amax(dim=1, keepdim=True)
        q_mod.weight_scale.data = max_val / 7.0
        
        return q_mod
