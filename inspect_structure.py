import torch
from transformers import AutoModel
import sys
import os

# Mock import if library not installed (but user has it)
try:
    from qwen_tts import Qwen3TTSModel as ModelClass
except ImportError:
    print("Could not import Qwen3TTSModel. Ensure qwen_tts is installed.")
    # Fallback to standard HF loading to check if that works
    ModelClass = None

MODEL_PATH = "./model_files"

def inspect():
    print("Loading model wrapper...")
    if ModelClass:
        try:
            model = ModelClass.from_pretrained(MODEL_PATH, device_map="cpu", trust_remote_code=True)
            print(f"Wrapper type: {type(model)}")
            print(f"Wrapper attributes: {dir(model)}")
            
            if hasattr(model, 'model'):
                print(f"\n.model type: {type(model.model)}")
                print(f".model attributes: {dir(model.model)}")
                
                # Check for other likely submodules
                if hasattr(model.model, 'llm'):
                    print(f"\n.model.llm type: {type(model.model.llm)}")
                if hasattr(model.model, 'transformer'):
                    print(f"\n.model.transformer type: {type(model.model.transformer)}")
                    
        except Exception as e:
            print(f"Error loading via ModelClass: {e}")
            
    print("\nAttempting raw AutoModel load...")
    try:
        raw_model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True)
        print(f"AutoModel type: {type(raw_model)}")
    except Exception as e:
        print(f"Error loading via AutoModel: {e}")

if __name__ == "__main__":
    inspect()
