import torch
from transformers import AutoModel
import torch.nn as nn

try:
    from qwen_tts import Qwen3TTSModel as ModelClass
except ImportError:
    pass

MODEL_PATH = "./model_files"

def check_head():
    def get_talker(path, **kwargs):
        wrapper = ModelClass.from_pretrained(path, **kwargs)
        if hasattr(wrapper, "model") and hasattr(wrapper.model, "talker"):
            return wrapper.model.talker
        elif hasattr(wrapper, "talker"):
            return wrapper.talker
        return wrapper.model

    model = get_talker(MODEL_PATH, device_map="cpu", trust_remote_code=True)
    
    # Usually lm_head or output_projection
    if hasattr(model, "lm_head"):
        print(f"lm_head: {model.lm_head}")
        print(f"lm_head weight shape: {model.lm_head.weight.shape}")
    else:
        print("No lm_head attribute found. Searching modules for last linear...")
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                print(f"Found Linear: {name} | Size: {module.weight.shape}")

if __name__ == "__main__":
    check_head()
