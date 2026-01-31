import torch
import torch.nn as nn
import torch.nn.functional as F

class STEQuantizer(torch.autograd.Function):
    """
    Straight-Through Estimator for Quantization.
    Forward: Quantizes weights/activations to INT4.
    Backward: Passes gradients unchanged (Identity).
    """
    @staticmethod
    def forward(ctx, input, scale, zero_point, bit_width=4):
        q_min = -(2 ** (bit_width - 1))
        q_max = (2 ** (bit_width - 1)) - 1
        
        # Quantize
        input_div = input / scale
        q_input = input_div.round() + zero_point
        q_input = torch.clamp(q_input, q_min, q_max)
        
        # Dequantize (Simulated Quantization)
        output = (q_input - zero_point) * scale
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-Through Estimator: returns grad_output for input, others are None
        return grad_output, None, None, None

def fake_quantize_int4(x, scale, zero_point=0):
    return STEQuantizer.apply(x, scale, zero_point)

class QuantizedLinearINT4(nn.Module):
    """
    A Linear layer that simulates INT4 quantization for weights during training (QAT).
    Includes a learnable scale factor (if desired) or static calibration.
    """
    def __init__(self, in_features, out_features, bias=True, groups=128):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.groups = groups # Block-wise quantization granularity
        
        # Initialize weights and bias as normal FP16/FP32
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        # Scale parameter for quantization (learnable)
        # Shape depends on grouping. If groups=1, per-tensor. If groups=in_features/128, block-wise.
        # For simplicity in this base implementation, we'll start with per-channel (out_features) scaling for weights
        self.weight_scale = nn.Parameter(torch.ones(out_features, 1)) 
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in**0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # 1. Quantize Weights (Simulated)
        # Dynamically calculate scale if not learned, or use learned scale
        # Here we use the learned scale `self.weight_scale`
        
        # Simple symmetric quantization for weights usually works best for INT4
        # Range: [-8, 7]
        
        # Update scale on the fly during training? Or keep fixed? 
        # For QAT/Distillation, standard practice is to update scales or learn them.
        
        # Let's ensure scale is positive
        s = self.weight_scale.abs() + 1e-8
        
        # Quantize weight
        w_quant = fake_quantize_int4(self.weight, s, zero_point=0)
        
        # 2. Perform Linear Operation
        return F.linear(input, w_quant, self.bias)

    @classmethod
    def from_float(cls, mod, groups=128):
        """
        Convert a standard nn.Linear module to QuantizedLinearINT4
        """
        assert isinstance(mod, nn.Linear)
        q_mod = cls(mod.in_features, mod.out_features, bias=mod.bias is not None, groups=groups)
        q_mod.weight.data = mod.weight.data.clone()
        if mod.bias is not None:
            q_mod.bias.data = mod.bias.data.clone()
        
        # Initialize scales based on current weights (Per-channel max-abs)
        # out_channels x 1
        max_val = mod.weight.data.abs().amax(dim=1, keepdim=True)
        # INT4 range is approx [-8, 7], so scale = max_val / 7 or 8. Let's use 7 for safety.
        q_mod.weight_scale.data = max_val / 7.0
        
        return q_mod
