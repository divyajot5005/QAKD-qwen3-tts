import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import os
import gc

from transformers import AutoTokenizer
from datasets import load_dataset
from model_utils import replace_linear_layers
from quant_layers import QuantizedLinearINT4

# Import from library
try:
    from qwen_tts import Qwen3TTSModel as ModelClass
except ImportError:
    print("Could not import Qwen3TTSModel")
    exit(1)

def distillation_loss(student_logits, teacher_logits, temperature=2.0):
    soft_targets = F.softmax(teacher_logits / temperature, dim=-1)
    soft_prob = F.log_softmax(student_logits / temperature, dim=-1)
    return F.kl_div(soft_prob, soft_targets, reduction='batchmean') * (temperature ** 2)

def train_distill(teacher_wrapper, student_wrapper, dataloader, optimizer, device, epochs=1):
    # Handle Wrappers vs Modules
    # Teacher
    if hasattr(teacher_wrapper, "eval"):
        teacher_wrapper.eval()
    elif hasattr(teacher_wrapper, "model"):
        teacher_wrapper.model.eval()
        
    # Student
    if hasattr(student_wrapper, "train"):
        student_wrapper.train()
    elif hasattr(student_wrapper, "model"):
        student_wrapper.model.train()
    
    print("Starting Distillation (H100 Optimization)...")
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Input to CPU for teacher, GPU for Student
            input_ids_cpu = batch['input_ids']
            attention_mask_cpu = batch['attention_mask']
            
            input_ids_gpu = input_ids_cpu.to(device)
            attention_mask_gpu = attention_mask_cpu.to(device)
            
                # 1. Teacher Forward (GPU)
            with torch.no_grad():
                teacher_outputs = teacher_wrapper(input_ids=input_ids_gpu, attention_mask=attention_mask_gpu)
                
                if hasattr(teacher_outputs, 'logits'):
                    teacher_logits = teacher_outputs.logits.to(device)
                else:
                    teacher_logits = teacher_outputs[0].to(device)

            # 2. Student Forward (GPU)
            student_outputs = student_wrapper(input_ids=input_ids_gpu, attention_mask=attention_mask_gpu)
            if hasattr(student_outputs, 'logits'):
                student_logits = student_outputs.logits
            else:
                student_logits = student_outputs[0]
            
            # Calculate Loss
            loss = distillation_loss(student_logits, teacher_logits)
            
            # Optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
            # Cleanup
            del teacher_logits, student_logits, loss
            
        avg_loss = total_loss / len(dataloader)
        print(f"Average Loss: {avg_loss:.4f}")
        
        if hasattr(student_wrapper, "state_dict"):
            state_dict = student_wrapper.state_dict()
        else:
            state_dict = student_wrapper.model.state_dict()
            
        torch.save(state_dict, f"qwen3_tts_int4_ep{epoch}.pt")

def main():
    # 6GB VRAM Optimization Config
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = "./model_files" 
    
    print(f"Using device: {DEVICE} for Student. Teacher will stay on CPU.")
    
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Run download_model.py")
        return

    # 1. Load Teacher Model (GPU)
    print("Loading Teacher Model (GPU)...")
    model_kwargs = {"device_map": DEVICE, "trust_remote_code": True, "torch_dtype": torch.float16}
    teacher_wrapper = ModelClass.from_pretrained(MODEL_PATH, **model_kwargs)
    print(f"Teacher wrapper type: {type(teacher_wrapper)}")
    # We use the wrapper for forward passes
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Prepare Student Model
    print("Creating Student Model (INT4 on GPU)...")
    student_wrapper = ModelClass.from_pretrained(MODEL_PATH, **model_kwargs)
    
    # helper to access inner module
    if hasattr(student_wrapper, "model"):
        student_inner = student_wrapper.model
    else:
        # Fallback if .model doesn't exist (inspect to be sure)
        print(f"DEBUG: Attributes of wrapper: {dir(student_wrapper)}")
        # Try to guess - maybe it IS the module?
        # But user reported 'no attribute to', so it's not a module.
        # Let's assume .model exists based on previous code.
        student_inner = student_wrapper.model 
        
    student_inner.to(dtype=torch.float16) # Half precision
    
    replace_linear_layers(student_inner, quantized_class=QuantizedLinearINT4)
    
    # Move Student to GPU (if not already handled by device_map)
    # student_wrapper usually manages device, but we can ensure inner is right
    student_inner.to(DEVICE)
    
    # Enable Gradient Checkpointing (Saves VRAM)
    if hasattr(student_inner, "gradient_checkpointing_enable"):
        print("Enabling Gradient Checkpointing...")
        student_inner.gradient_checkpointing_enable()

    # 3. Load Calibration Dataset
    print("Loading Dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:500]") 
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=256, padding="max_length")

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask"])
    
    # H100 has 80GB, so we can use a larger batch size
    dataloader = DataLoader(tokenized_datasets, batch_size=16, shuffle=True) 
    
    # 4. Optimizer - Use 8-bit Adam via bitsandbytes
    import bitsandbytes as bnb
    print("Using 8-bit AdamW...")
    optimizer = bnb.optim.AdamW8bit(student_inner.parameters(), lr=1e-5)
    
    # 5. Run
    # Pass WRAPPERS to training loop for forward calls
    train_distill(teacher_wrapper, student_wrapper, dataloader, optimizer, DEVICE, epochs=1)

if __name__ == "__main__":
    main()
