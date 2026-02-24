import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
from collections import defaultdict


class CalibrationHook:
    """
    Forward hook to collect activation statistics for calibration.
    Attaches to all QuantizedLinearINT4 layers and records min/max/percentile statistics.
    """
    
    def __init__(self, percentile: float = 99.99):
        self.percentile = percentile
        self.stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'min': float('inf'),
            'max': float('-inf'),
            'abs_max': 0.0,
            'count': 0,
            'sum': 0.0,
            'sum_sq': 0.0
        })
        self.handles: List = []
    
    def __call__(self, name: str):
        def hook(module, input, output):
            if isinstance(input, tuple):
                input = input[0]
            
            input_float = input.detach().float()
            
            current_min = input_float.min().item()
            current_max = input_float.max().item()
            current_abs_max = input_float.abs().max().item()
            
            self.stats[name]['min'] = min(self.stats[name]['min'], current_min)
            self.stats[name]['max'] = max(self.stats[name]['max'], current_max)
            self.stats[name]['abs_max'] = max(self.stats[name]['abs_max'], current_abs_max)
            self.stats[name]['count'] += input_float.numel()
            self.stats[name]['sum'] += input_float.sum().item()
            self.stats[name]['sum_sq'] += (input_float ** 2).sum().item()
        
        return hook
    
    def register(self, model: nn.Module, target_class: type = None) -> None:
        """Register hooks on all quantized linear layers."""
        if target_class is None:
            from quant_layers import QuantizedLinearINT4
            target_class = QuantizedLinearINT4
        
        for name, module in model.named_modules():
            if isinstance(module, target_class):
                handle = module.register_forward_hook(self(name))
                self.handles.append(handle)
    
    def remove(self) -> None:
        """Remove all registered hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
    
    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Return collected statistics."""
        return dict(self.stats)
    
    def get_mean_std(self, name: str) -> tuple:
        """Calculate mean and std from collected statistics."""
        stats = self.stats[name]
        if stats['count'] == 0:
            return 0.0, 1.0
        
        mean = stats['sum'] / stats['count']
        variance = (stats['sum_sq'] / stats['count']) - (mean ** 2)
        std = max(variance ** 0.5, 1e-8)
        return mean, std
    
    def get_percentile_range(self, name: str, percentile: float = None) -> tuple:
        """
        Estimate percentile range from min/max and distribution.
        Uses a simplified approximation based on assumed Gaussian distribution.
        """
        if percentile is None:
            percentile = self.percentile
        
        stats = self.stats[name]
        mean, std = self.get_mean_std(name)
        
        from scipy.stats import norm
        z_score = norm.ppf(percentile / 100.0)
        
        p_min = mean - z_score * std
        p_max = mean + z_score * std
        
        actual_min = max(stats['min'], p_min)
        actual_max = min(stats['max'], p_max)
        
        return actual_min, actual_max


def run_calibration(
    model: nn.Module,
    dataloader,
    device: str = 'cuda',
    num_samples: int = 512,
    target_class: type = None
) -> Dict[str, Dict[str, Any]]:
    """
    Run calibration on model using data from dataloader.
    
    Args:
        model: Model with QuantizedLinearINT4 layers
        dataloader: DataLoader providing calibration data
        device: Device to run on
        num_samples: Number of samples to use for calibration
        target_class: Class of quantized layers to calibrate
    
    Returns:
        Dictionary of per-layer activation statistics
    """
    if target_class is None:
        from quant_layers import QuantizedLinearINT4
        target_class = QuantizedLinearINT4
    
    model.eval()
    model.to(device)
    
    hook = CalibrationHook()
    hook.register(model, target_class)
    
    sample_count = 0
    with torch.no_grad():
        for batch in dataloader:
            if sample_count >= num_samples:
                break
            
            if isinstance(batch, dict):
                input_ids = batch.get('input_ids')
                attention_mask = batch.get('attention_mask')
                
                if input_ids is None:
                    continue
                
                input_ids = input_ids.to(device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                
                try:
                    if attention_mask is not None:
                        model(input_ids=input_ids, attention_mask=attention_mask)
                    else:
                        model(input_ids=input_ids)
                except Exception as e:
                    try:
                        model(input_ids)
                    except:
                        pass
            else:
                batch = batch.to(device) if isinstance(batch, torch.Tensor) else batch[0].to(device)
                try:
                    model(batch)
                except:
                    pass
            
            batch_size = input_ids.shape[0] if isinstance(input_ids, torch.Tensor) else 1
            sample_count += batch_size
    
    hook.remove()
    stats = hook.get_stats()
    
    model.cpu()
    torch.cuda.empty_cache()
    
    return stats


def compute_optimal_scale_mse(
    weight: torch.Tensor,
    bit_width: int = 4,
    num_steps: int = 100,
    group_size: int = None
) -> tuple:
    """
    Compute MSE-optimal scale for quantization.
    
    Args:
        weight: Weight tensor to quantize
        bit_width: Bit width for quantization
        num_steps: Number of scale candidates to try
        group_size: If not None, compute per-group scales
    
    Returns:
        Tuple of (scale, zero_point)
    """
    q_min = -(2 ** (bit_width - 1))
    q_max = (2 ** (bit_width - 1)) - 1
    
    if group_size is not None:
        out_features, in_features = weight.shape
        num_groups = in_features // group_size
        weight_grouped = weight.view(out_features, num_groups, group_size)
        
        scales = []
        zero_points = []
        
        for g in range(num_groups):
            w_g = weight_grouped[:, g, :].flatten()
            scale_g, zp_g = _compute_mse_scale_1d(w_g, q_min, q_max, num_steps)
            scales.append(scale_g)
            zero_points.append(zp_g)
        
        scale = torch.tensor(scales).view(1, num_groups).expand(out_features, -1)
        zero_point = torch.tensor(zero_points).view(1, num_groups).expand(out_features, -1)
        
        return scale, zero_point
    else:
        return _compute_mse_scale_1d(weight.flatten(), q_min, q_max, num_steps)


def _compute_mse_scale_1d(
    w: torch.Tensor,
    q_min: float,
    q_max: float,
    num_steps: int
) -> tuple:
    """Helper to compute MSE-optimal scale for 1D tensor."""
    w_abs_max = w.abs().max().item()
    
    best_scale = w_abs_max / q_max
    best_mse = float('inf')
    
    for factor in torch.linspace(0.5, 1.5, num_steps):
        scale = (w_abs_max * factor / q_max).item()
        if scale < 1e-8:
            continue
        
        w_q = torch.clamp((w / scale).round(), q_min, q_max)
        w_deq = w_q * scale
        mse = ((w - w_deq) ** 2).mean().item()
        
        if mse < best_mse:
            best_mse = mse
            best_scale = scale
    
    return best_scale, 0.0


def initialize_scales_from_calibration(
    model: nn.Module,
    stats: Dict[str, Dict[str, Any]],
    target_class: type = None,
    use_percentile: bool = True,
    percentile: float = 99.99
) -> None:
    """
    Initialize input scales of QuantizedLinearINT4 layers from calibration statistics.
    
    Args:
        model: Model with QuantizedLinearINT4 layers
        stats: Calibration statistics from run_calibration
        target_class: Class of quantized layers
        use_percentile: Whether to use percentile range vs absolute min/max
        percentile: Percentile to use if use_percentile is True
    """
    if target_class is None:
        from quant_layers import QuantizedLinearINT4
        target_class = QuantizedLinearINT4
    
    for name, module in model.named_modules():
        if isinstance(module, target_class) and module.quantize_input:
            if name in stats:
                layer_stats = stats[name]
                
                if use_percentile:
                    try:
                        from scipy.stats import norm
                        if layer_stats['count'] > 0:
                            mean = layer_stats['sum'] / layer_stats['count']
                            variance = (layer_stats['sum_sq'] / layer_stats['count']) - (mean ** 2)
                            std = max(variance ** 0.5, 1e-8)
                            
                            z_score = norm.ppf(min(percentile, 99.999) / 100.0)
                            input_min = mean - z_score * std
                            input_max = mean + z_score * std
                        else:
                            input_min = layer_stats['min']
                            input_max = layer_stats['max']
                    except ImportError:
                        input_min = layer_stats['min']
                        input_max = layer_stats['max']
                else:
                    input_min = layer_stats['min']
                    input_max = layer_stats['max']
                
                if module.symmetric_activations:
                    max_abs = max(abs(input_min), abs(input_max))
                    scale = max_abs / 7.0
                    module.input_scale.data.fill_(scale)
                else:
                    scale = (input_max - input_min) / 15.0
                    zero_point = -input_min / scale - 8.0
                    
                    module.input_scale.data.fill_(scale)
                    if isinstance(module.input_zero_point, nn.Parameter):
                        module.input_zero_point.data.fill_(zero_point)


def refine_weight_scales_mse(
    model: nn.Module,
    target_class: type = None
) -> None:
    """
    Refine weight scales using MSE optimization.
    
    Args:
        model: Model with QuantizedLinearINT4 layers
        target_class: Class of quantized layers
    """
    if target_class is None:
        from quant_layers import QuantizedLinearINT4
        target_class = QuantizedLinearINT4
    
    for name, module in model.named_modules():
        if isinstance(module, target_class):
            scale, zero_point = compute_optimal_scale_mse(
                module.weight.data,
                bit_width=4,
                num_steps=100,
                group_size=module.group_size
            )
            
            module.weight_scale.data = torch.log(torch.abs(scale) + 1e-8)
            if isinstance(module.weight_zero_point, nn.Parameter):
                module.weight_zero_point.data = zero_point


def get_calibration_summary(stats: Dict[str, Dict[str, Any]]) -> str:
    """
    Generate a summary string of calibration statistics.
    
    Args:
        stats: Calibration statistics dictionary
    
    Returns:
        Formatted summary string
    """
    lines = ["Calibration Statistics Summary", "=" * 50]
    
    for name, layer_stats in sorted(stats.items()):
        lines.append(f"\n{name}:")
        lines.append(f"  Min: {layer_stats['min']:.6f}")
        lines.append(f"  Max: {layer_stats['max']:.6f}")
        lines.append(f"  Abs Max: {layer_stats['abs_max']:.6f}")
        
        if layer_stats['count'] > 0:
            mean = layer_stats['sum'] / layer_stats['count']
            variance = (layer_stats['sum_sq'] / layer_stats['count']) - (mean ** 2)
            std = max(variance ** 0.5, 1e-8)
            lines.append(f"  Mean: {mean:.6f}")
            lines.append(f"  Std: {std:.6f}")
        
        lines.append(f"  Sample count: {layer_stats['count']}")
    
    return "\n".join(lines)
