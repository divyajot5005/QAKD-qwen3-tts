import torch
from transformers import AutoModel
import inspect

# Mock import
try:
    from qwen_tts import Qwen3TTSModel as ModelClass
except ImportError:
    print("Error importing qwen_tts")
    exit()

MODEL_PATH = "./model_files"

def inspect_forward():
    print("Loading model for inspection...")
    # Load wrapper
    wrapper = ModelClass.from_pretrained(MODEL_PATH, device_map="cpu", trust_remote_code=True)
    
    talker = None
    if hasattr(wrapper, "model") and hasattr(wrapper.model, "talker"):
        talker = wrapper.model.talker
    else:
        print("Could not locate talker. Dumping wrapper.model keys:")
        if hasattr(wrapper, "model"):
             print(dir(wrapper.model))
        return

    print(f"\nTalker Type: {type(talker)}")
    
    # Inspect Forward
    sig = inspect.signature(talker.forward)
    print(f"\nForward Signature: {sig}")
    
    print("\nDocstring:")
    print(talker.forward.__doc__)

if __name__ == "__main__":
    inspect_forward()
