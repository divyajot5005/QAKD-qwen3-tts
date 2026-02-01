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

def train_distill(teacher_model, student_model, dataloader, optimizer, device, epochs=5):
    # Teacher and Student are standard nn.Module (Talkers)
    teacher_model.eval()
    student_model.train()
    
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
            
            # Prepare dummy past_hidden
            batch_size = input_ids_gpu.shape[0]
            hidden_size = teacher_model.config.hidden_size
            dtype = teacher_model.dtype
            
            # Using zeros acts as a "neutral" prompt
            dummy_past_hidden = torch.zeros(batch_size, 1, hidden_size, device=device, dtype=dtype)

            # --- MANUALLY EMBED TEXT INPUTS ---
            # The model default forward(...) maps input_ids -> codec_embedding (Audio Codes).
            # We must map input_ids -> text_embedding manually to use Text.
            
            # Locate text_embedding module (usually talker.model.text_embedding)
            # Inspection showed: model.text_embedding
            if hasattr(teacher_model, "model") and hasattr(teacher_model.model, "text_embedding"):
                text_embed_layer = teacher_model.model.text_embedding
            elif hasattr(teacher_model, "text_embedding"):
                text_embed_layer = teacher_model.text_embedding
            else:
                raise AttributeError("Could not find text_embedding layer in teacher model")
                
            # Embed inputs (No Gen, just embeddings)
            with torch.no_grad():
                inputs_embeds_teacher = text_embed_layer(input_ids_gpu)
            
            # For Student, we assume same structure or share weights logic
            # If student is INT4, we must use its own embedding layer (which might be FP16 still)
            if hasattr(student_model, "model") and hasattr(student_model.model, "text_embedding"):
                student_embed_layer = student_model.model.text_embedding
            elif hasattr(student_model, "text_embedding"):
                student_embed_layer = student_model.text_embedding
            else:
                student_embed_layer = text_embed_layer # Fallback? No, student has its own parameters.
                
            inputs_embeds_student = student_embed_layer(input_ids_gpu)

            # 1. Teacher Forward (GPU)
            with torch.no_grad():
                teacher_outputs = teacher_model(
                    inputs_embeds=inputs_embeds_teacher, 
                    attention_mask=attention_mask_gpu,
                    past_hidden=dummy_past_hidden
                )
                
                if hasattr(teacher_outputs, 'logits'):
                    teacher_logits = teacher_outputs.logits
                else:
                    teacher_logits = teacher_outputs[0]

            # 2. Student Forward (GPU)
            student_outputs = student_model(
                inputs_embeds=inputs_embeds_student, 
                attention_mask=attention_mask_gpu,
                past_hidden=dummy_past_hidden
            )
            
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
        
        if hasattr(student_model, "state_dict"):
            state_dict = student_model.state_dict()
        else:
            state_dict = student_model.model.state_dict()
            
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
    
    # helper function to drill down to the LLM backbone
    def get_talker(path, **kwargs):
        wrapper = ModelClass.from_pretrained(path, **kwargs)
        # Hierarchy: Qwen3TTSModel (Wrapper) -> Qwen3TTSForConditionalGeneration (.model) -> Talker (.talker)
        if hasattr(wrapper, "model") and hasattr(wrapper.model, "talker"):
            return wrapper.model.talker
        elif hasattr(wrapper, "talker"):
            return wrapper.talker
        else:
            # Fallback or maybe wrapper.model IS the talker?
            print(f"Structure lookup failed. Wrapper keys: {dir(wrapper)}")
            if hasattr(wrapper, "model"):
                 print(f"Inner keys: {dir(wrapper.model)}")
            return wrapper.model

    teacher_model = get_talker(MODEL_PATH, **model_kwargs)
    print(f"Teacher Backbone type: {type(teacher_model)}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Prepare Student Model
    print("Creating Student Model (INT4 on GPU)...")
    student_model = get_talker(MODEL_PATH, **model_kwargs)
    print(f"Student Backbone type: {type(student_model)}")
    
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
    # Use full training split
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train") 
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=256, padding="max_length")

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask"])
    
    # H100 has 80GB, so we can use a larger batch size
    dataloader = DataLoader(tokenized_datasets, batch_size=16, shuffle=True) 
    
    # 4. Optimizer - Use 8-bit Adam via bitsandbytes
    import bitsandbytes as bnb
    print("Using 8-bit AdamW...")
    optimizer = bnb.optim.AdamW8bit(student_model.parameters(), lr=1e-5)
    
    # 5. Run
    # Pass inner models (talkers) to training loop
    train_distill(teacher_model, student_model, dataloader, optimizer, DEVICE, epochs=1)

if __name__ == "__main__":
    main()
