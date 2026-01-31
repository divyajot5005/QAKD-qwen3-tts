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

def train_distill(teacher_model, student_model, dataloader, optimizer, device, epochs=1):
    # Teacher is on CPU, Student is on GPU
    teacher_model.eval()
    student_model.train()
    
    print("Starting Distillation (Low VRAM Mode)...")
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
            
            # 1. Teacher Forward (CPU)
            with torch.no_grad():
                # Note: Qwen3TTS inner model might return a tuple or specialized output
                # We usually expect CausalLMOutputWithPast or similar
                teacher_outputs = teacher_model(input_ids=input_ids_cpu, attention_mask=attention_mask_cpu)
                
                if hasattr(teacher_outputs, 'logits'):
                    teacher_logits = teacher_outputs.logits.to(device)
                else:
                    # Fallback if output is tuple (logits are usually 0th element)
                    teacher_logits = teacher_outputs[0].to(device)

            # 2. Student Forward (GPU)
            student_outputs = student_model(input_ids=input_ids_gpu, attention_mask=attention_mask_gpu)
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
        
        torch.save(student_model.state_dict(), f"qwen3_tts_int4_ep{epoch}.pt")

def main():
    # 6GB VRAM Optimization Config
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = "./model_files" 
    
    print(f"Using device: {DEVICE} for Student. Teacher will stay on CPU.")
    
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Run download_model.py")
        return

    # 1. Load Teacher Model (CPU ONLY)
    print("Loading Teacher Model (CPU)...")
    teacher_wrapper = ModelClass.from_pretrained(MODEL_PATH, device_map="cpu", trust_remote_code=True)
    teacher_model = teacher_wrapper.model # Unwrap
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Prepare Student Model
    print("Creating Student Model (INT4 on GPU)...")
    student_wrapper = ModelClass.from_pretrained(MODEL_PATH, device_map="cpu", trust_remote_code=True)
    student_model = student_wrapper.model # Unwrap
    
    student_model.to(dtype=torch.float16) # Half precision
    
    replace_linear_layers(student_model, quantized_class=QuantizedLinearINT4)
    
    # Move Student to GPU
    student_model.to(DEVICE)
    
    # Enable Gradient Checkpointing (Saves VRAM)
    if hasattr(student_model, "gradient_checkpointing_enable"):
        print("Enabling Gradient Checkpointing...")
        student_model.gradient_checkpointing_enable()

    # 3. Load Calibration Dataset
    print("Loading Dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:500]") 
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=256, padding="max_length")

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask"])
    
    dataloader = DataLoader(tokenized_datasets, batch_size=1, shuffle=True) 
    
    # 4. Optimizer - Use 8-bit Adam via bitsandbytes
    import bitsandbytes as bnb
    print("Using 8-bit AdamW...")
    optimizer = bnb.optim.AdamW8bit(student_model.parameters(), lr=1e-5)
    
    # 5. Run
    train_distill(teacher_model, student_model, dataloader, optimizer, DEVICE, epochs=1)

if __name__ == "__main__":
    main()
