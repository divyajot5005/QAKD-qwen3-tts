import torch
from transformers import AutoModel
import torch.nn as nn

# Mock import
try:
    from qwen_tts import Qwen3TTSModel as ModelClass
except ImportError:
    print("Error importing qwen_tts")
    exit()

MODEL_PATH = "./model_files"

def inspect_layers():
    print("Loading model...")
    # helper function to drill down to the LLM backbone
    def get_talker(path, **kwargs):
        wrapper = ModelClass.from_pretrained(path, **kwargs)
        if hasattr(wrapper, "model") and hasattr(wrapper.model, "talker"):
            return wrapper.model.talker
        elif hasattr(wrapper, "talker"):
            return wrapper.talker
        return wrapper.model

    model = get_talker(MODEL_PATH, device_map="cpu", trust_remote_code=True)
    
    print(f"Traversing modules of {type(model)}...")
    
    found_text_embed = False
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            print(f"Found Embedding: {name} | Size: {module.weight.shape}")
            if module.weight.shape[0] > 50000:
                print("  -> Likely Text Embeddings (Vocab)")
                found_text_embed = True
            elif module.weight.shape[0] < 5000:
                print("  -> Likely VQ/Audio Embeddings")
                
    if not found_text_embed:
        print("\nWARNING: No large embedding table found. This model might not accept raw text tokens.")

if __name__ == "__main__":
    inspect_layers()
