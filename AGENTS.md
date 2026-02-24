# AGENTS.md

This document provides guidelines for agentic coding agents working in this repository.

## Project Overview

This project implements Quantization Aware Knowledge Distillation (QAKD) to distill a Qwen 3 TTS model into an INT4 native structure. The codebase uses PyTorch and HuggingFace Transformers for model manipulation and training.

## Build/Lint/Test Commands

### Setup
```bash
pip install -r requirements.txt
```

### Running Scripts
```bash
# Download the teacher model from HuggingFace
python download_model.py

# Create the student model with INT4 quantized layers
python create_student.py

# Run knowledge distillation training
python distill.py

# Inspection utilities
python inspect_model.py
python inspect_structure.py
python inspect_layers.py
python inspect_forward.py
python check_head.py
python check_vocab.py
```

### Testing
This project does not have a formal test suite. When adding new functionality, create standalone scripts to verify behavior before integrating into the main pipeline.

### Linting
No linting configuration is present. If adding linting, use:
```bash
# Recommended: ruff for fast linting
ruff check .

# Or flake8
flake8 .
```

## Code Style Guidelines

### Imports
Group imports in the following order, separated by blank lines:
1. Standard library imports (alphabetically sorted)
2. Third-party imports (alphabetically sorted)
3. Local/application imports

```python
import os
import sys

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from model_utils import replace_linear_layers
from quant_layers import QuantizedLinearINT4
```

### Naming Conventions
- **Functions**: `snake_case` (e.g., `distillation_loss`, `replace_linear_layers`)
- **Classes**: `PascalCase` (e.g., `QuantizedLinearINT4`, `STEQuantizer`)
- **Variables**: `snake_case` (e.g., `teacher_model`, `weight_scale`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MODEL_PATH`, `DEVICE`)
- **Private methods**: Prefix with underscore (e.g., `_helper_function`)

### Formatting
- Maximum line length: 100 characters
- Use 4 spaces for indentation (no tabs)
- Blank lines between class methods
- Single blank line between function definitions

### Docstrings
Use triple-quoted docstrings for classes and public functions:

```python
class QuantizedLinearINT4(nn.Module):
    """
    A Linear layer that simulates INT4 quantization for weights during training (QAT).
    Includes a learnable scale factor or static calibration.
    """
    
def replace_linear_layers(model, target_class=nn.Linear, quantized_class=QuantizedLinearINT4):
    """
    Recursively replaces all instances of target_class with quantized_class in the model.
    """
```

### Type Hints
Type hints are optional but encouraged for function signatures:

```python
def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0
) -> torch.Tensor:
```

### Error Handling
- Use try/except for imports that may fail due to missing optional dependencies
- Provide clear error messages for user-facing issues

```python
try:
    from qwen_tts import Qwen3TTSModel as ModelClass
except ImportError:
    print("Could not import Qwen3TTSModel")
    sys.exit(1)
```

- Use `assert` for internal invariants and preconditions

```python
assert isinstance(mod, nn.Linear)
```

### PyTorch Conventions
- Use `torch.float16` for model weights when targeting GPU
- Use `device_map` parameter when loading models from HuggingFace
- Always use `trust_remote_code=True` when loading Qwen TTS models
- Clear GPU memory explicitly with `del` when handling large tensors

```python
model_kwargs = {"device_map": DEVICE, "trust_remote_code": True, "torch_dtype": torch.float16}
```

### File Structure
- Each script should be executable standalone with `if __name__ == "__main__":`
- Utility functions should be placed in appropriate modules (e.g., `model_utils.py`, `quant_layers.py`)
- Main training/execution logic goes in dedicated scripts (e.g., `distill.py`, `create_student.py`)

### Quantization Layer Guidelines
- Quantized layers inherit from `nn.Module`
- Implement `from_float` class method for converting standard layers
- Use Straight-Through Estimator (STE) for gradient computation in quantization

### Model Loading Pattern
```python
def get_talker(path, **kwargs):
    wrapper = ModelClass.from_pretrained(path, **kwargs)
    if hasattr(wrapper, "model") and hasattr(wrapper.model, "talker"):
        return wrapper.model.talker
    elif hasattr(wrapper, "talker"):
        return wrapper.talker
    return wrapper.model
```

### GPU Memory Optimization
- Use gradient checkpointing for large models: `model.gradient_checkpointing_enable()`
- Use 8-bit optimizers when available: `bnb.optim.AdamW8bit`
- Move models to appropriate devices explicitly

### Logging
- Use `print()` statements for progress logging
- Use `tqdm` for progress bars during training loops
- Include relevant metrics in progress bar postfix

```python
progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
progress_bar.set_postfix({'loss': loss.item()})
```

## Project-Specific Notes

### Model Architecture
The Qwen 3 TTS model hierarchy is:
```
Qwen3TTSModel (wrapper)
└── Qwen3TTSForConditionalGeneration (.model)
    └── Talker (.talker) - The main LLM backbone for quantization
```

### Key Files
- `quant_layers.py`: Custom INT4 Linear layer with STE, group-wise quantization, activation quantization
- `model_utils.py`: Helper functions for layer replacement
- `distill.py`: Main training loop with knowledge distillation, multi-objective loss
- `create_student.py`: Creates INT4 student architecture with calibration
- `calibration.py`: Calibration utilities for scale initialization

### Dependencies
- PyTorch (torch)
- Transformers (transformers)
- Accelerate (accelerate)
- BitsAndBytes (bitsandbytes) - for 8-bit optimizers
- Datasets (datasets) - for calibration data
- HuggingFace Hub (huggingface_hub)