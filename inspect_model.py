import torch
import sys

# Import from library
try:
    from qwen_tts import Qwen3TTSModel
except ImportError:
    print("Could not import Qwen3TTSModel")
    sys.exit(1)

def inspect_model(model_path="./model_files"):
    print(f"Loading model from {model_path}...")
    model = Qwen3TTSModel.from_pretrained(model_path, device_map="cpu", trust_remote_code=True)
    
    print("\n--- Direct Attributes ---")
    print(dir(model))
    
    print("\n--- Type ---")
    print(type(model))
    
    print("\n--- Isninstance nn.Module? ---")
    print(isinstance(model, torch.nn.Module))
    
    if hasattr(model, 'model'):
        print("\n--- model.model detected! ---")
        print(type(model.model))
        print(isinstance(model.model, torch.nn.Module))

if __name__ == "__main__":
    inspect_model()
