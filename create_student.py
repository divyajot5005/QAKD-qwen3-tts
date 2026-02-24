import os
import sys
import torch
from torch.utils.data import DataLoader

from model_utils import replace_linear_layers
from quant_layers import QuantizedLinearINT4
from calibration import run_calibration, initialize_scales_from_calibration, get_calibration_summary

try:
    from qwen_tts import Qwen3TTSModel as ModelClass
except ImportError:
    print("Could not import Qwen3TTSModel from qwen_tts.")
    sys.exit(1)


def create_student_model(
    model_path: str = "./model_files",
    save_path: str = "./student_model_int4",
    use_calibration: bool = True,
    calibration_samples: int = 512,
    group_size: int = 128,
    quantize_input: bool = True
):
    """
    Create a student model with INT4 quantized layers.
    
    Args:
        model_path: Path to the teacher model
        save_path: Path to save the student model
        use_calibration: Whether to run calibration for input scale initialization
        calibration_samples: Number of samples for calibration
        group_size: Group size for block-wise quantization
        quantize_input: Whether to enable activation quantization
    """
    print(f"Loading Teacher model from {model_path}...")
    
    teacher_wrapper = ModelClass.from_pretrained(model_path, device_map="cpu", trust_remote_code=True)
    teacher_model = teacher_wrapper.model
    
    print("Teacher model loaded (Unwrapped).")
    
    print("Loading specialized Student Copy from disk...")
    student_wrapper = ModelClass.from_pretrained(model_path, device_map="cpu", trust_remote_code=True)
    student_model = student_wrapper.model
    
    student_model.to(dtype=torch.float16)
    
    print("Replacing Linear layers with QuantizedLinearINT4...")
    print(f"  Group size: {group_size}")
    print(f"  Quantize input: {quantize_input}")
    
    def replace_with_config(model, group_size=128, quantize_input=True):
        """Replace linear layers with configured QuantizedLinearINT4."""
        from quant_layers import QuantizedLinearINT4
        for name, module in model.named_children():
            if isinstance(module, torch.nn.Linear):
                print(f"Quantizing layer: {name}")
                quantized_layer = QuantizedLinearINT4.from_float(
                    module,
                    group_size=group_size,
                    quantize_input=quantize_input
                )
                setattr(model, name, quantized_layer)
            else:
                replace_with_config(module, group_size, quantize_input)
    
    replace_with_config(student_model, group_size, quantize_input)
    
    print("Student model structure created.")
    
    if use_calibration:
        print(f"\nRunning calibration with {calibration_samples} samples...")
        print("Note: Calibration requires a dataset. Using dummy data if not available.")
        
        try:
            from transformers import AutoTokenizer
            from datasets import load_dataset
            
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, fix_mistral_regex=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            
            def tokenize_function(examples):
                return tokenizer(examples["text"], truncation=True, max_length=256, padding="max_length")
            
            tokenized_datasets = dataset.map(
                tokenize_function, 
                batched=True, 
                remove_columns=["text"],
                load_from_cache_file=False
            )
            tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask"])
            
            calib_loader = DataLoader(
                tokenized_datasets, 
                batch_size=8, 
                shuffle=True
            )
            
            stats = run_calibration(
                student_model,
                calib_loader,
                device="cuda" if torch.cuda.is_available() else "cpu",
                num_samples=calibration_samples
            )
            
            print("\nCalibration statistics:")
            print(get_calibration_summary(stats))
            
            print("\nInitializing input scales from calibration...")
            initialize_scales_from_calibration(student_model, stats)
            
            print("Calibration complete.")
            
        except Exception as e:
            print(f"Calibration skipped due to error: {e}")
            print("Using default scale initialization.")
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    print(f"\nSaving student model to {save_path}...")
    torch.save(student_model.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
    
    try:
        if hasattr(teacher_model, 'config'):
            teacher_model.config.save_pretrained(save_path)
    except Exception:
        pass
    
    config = {
        'group_size': group_size,
        'quantize_input': quantize_input,
        'use_calibration': use_calibration,
        'calibration_samples': calibration_samples if use_calibration else 0,
    }
    torch.save(config, os.path.join(save_path, "quant_config.bin"))
    
    print("Done.")
    print(f"\nStudent model saved to: {save_path}")
    print(f"  - pytorch_model.bin (weights)")
    print(f"  - quant_config.bin (quantization configuration)")


def load_student_model(
    model_path: str = "./student_model_int4",
    device: str = "cuda"
) -> torch.nn.Module:
    """
    Load a saved student model.
    
    Args:
        model_path: Path to the saved student model
        device: Device to load the model onto
    
    Returns:
        Loaded student model
    """
    config = torch.load(os.path.join(model_path, "quant_config.bin"))
    
    print(f"Loading student model with config:")
    print(f"  Group size: {config['group_size']}")
    print(f"  Quantize input: {config['quantize_input']}")
    
    base_wrapper = ModelClass.from_pretrained(
        model_path, 
        device_map="cpu", 
        trust_remote_code=True
    )
    model = base_wrapper.model
    
    from quant_layers import QuantizedLinearINT4
    
    def replace_with_config(m, group_size=128, quantize_input=True):
        for name, module in m.named_children():
            if isinstance(module, torch.nn.Linear):
                quantized_layer = QuantizedLinearINT4.from_float(
                    module,
                    group_size=group_size,
                    quantize_input=quantize_input
                )
                setattr(m, name, quantized_layer)
            else:
                replace_with_config(module, group_size, quantize_input)
    
    replace_with_config(model, config['group_size'], config['quantize_input'])
    
    state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location=device)
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.to(dtype=torch.float16)
    
    print("Student model loaded successfully.")
    return model


if __name__ == "__main__":
    create_student_model(
        use_calibration=True,
        calibration_samples=512,
        group_size=128,
        quantize_input=True
    )
