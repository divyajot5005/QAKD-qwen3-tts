import torch
import torch.nn as nn
from quant_layers import QuantizedLinearINT4

def replace_linear_layers(model, target_class=nn.Linear, quantized_class=QuantizedLinearINT4, allowed_names=None, ignored_names=None, device=None):
    """
    Recursively replaces all instances of target_class with quantized_class in the model.
    """
    if allowed_names is None:
        allowed_names = [] # Convert all if empty? No, usually list. If None, allow all.
    
    # We iterate manually to modify the model in-place
    for name, module in model.named_children():
        
        # Check if we should skip this module
        if ignored_names and any(ign in name for ign in ignored_names):
            continue
            
        if isinstance(module, target_class):
            # Verify if we should convert this specific layer (if whitelist provided)
            # This is a simplifiction; usually we filter by full path. 
            # ideally passing full names to this recursive function would be better.
            
            # Create quantized layer
            print(f"Quantizing layer: {name}")
            quantized_layer = quantized_class.from_float(module)
            
            # Move to device if specified
            if device is not None:
                quantized_layer = quantized_layer.to(device)
            
            # Replace
            setattr(model, name, quantized_layer)
        else:
            # Recurse
            replace_linear_layers(module, target_class, quantized_class, allowed_names, ignored_names, device)

def get_named_linears(model):
    """
    Helper to list all linear layers to help user decide what to quantize.
    """
    return {name: m for name, m in model.named_modules() if isinstance(m, nn.Linear)}
