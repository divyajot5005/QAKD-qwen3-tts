import torch
from transformers import AutoTokenizer, AutoConfig
import os

# Mock import
try:
    from qwen_tts import Qwen3TTSModel as ModelClass
except ImportError:
    print("Error importing qwen_tts")
    exit()

MODEL_PATH = "./model_files"

def check_vocab():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"Tokenizer len: {len(tokenizer)}")
    
    print("\nLoading model config (skipped AutoConfig)...")
    # config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    # print(f"Config vocab size: {getattr(config, 'vocab_size', 'Unknown')}")
    
    # helper function to drill down to the LLM backbone
    def get_talker(path, **kwargs):
        wrapper = ModelClass.from_pretrained(path, **kwargs)
        if hasattr(wrapper, "model") and hasattr(wrapper.model, "talker"):
            return wrapper.model.talker
        elif hasattr(wrapper, "talker"):
            return wrapper.talker
        return wrapper.model

    print("\nLoading model on CPU to verify embeddings...")
    model = get_talker(MODEL_PATH, device_map="cpu", trust_remote_code=True)
    
    embed = model.get_input_embeddings()
    print(f"Embedding layer: {embed}")
    print(f"Embedding num_embeddings: {embed.num_embeddings}")
    
    # Check bounds
    if len(tokenizer) > embed.num_embeddings:
        print(f"\nCRITICAL WARNING: Tokenizer has {len(tokenizer)} tokens but model has {embed.num_embeddings} embeddings.")
        print("This causes device-side asserts when high-ID tokens are used.")
    
    # Try a forward pass on CPU with high-value tokens
    print("\nTesting forward pass on CPU...")
    try:
        dummy_input = torch.tensor([[embed.num_embeddings - 1]], dtype=torch.long)
        dummy_past = torch.zeros(1, 1, model.config.hidden_size)
        
        model(input_ids=dummy_input, past_hidden=dummy_past)
        print("Forward pass successful with max ID.")
    except Exception as e:
        print(f"Forward pass failed: {e}")

if __name__ == "__main__":
    check_vocab()
