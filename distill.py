import gc
import os
import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from transformers import AutoTokenizer
from datasets import load_dataset
from model_utils import replace_linear_layers
from quant_layers import QuantizedLinearINT4

try:
    from qwen_tts import Qwen3TTSModel as ModelClass
except ImportError:
    print("Could not import Qwen3TTSModel")
    exit(1)


DISTILLATION_CONFIG = {
    'temperature': 2.0,
    'learning_rate': 1e-4,
    'batch_size': 1,
    'gradient_accumulation_steps': 16,
    'max_seq_length': 32,
    'epochs': 3,
    'gradient_clip_max_norm': 1.0,
    'warmup_ratio': 0.1,
    'validation_split': 0.05,
}


def distillation_loss(student_logits, teacher_logits, temperature=2.0):
    soft_targets = F.softmax(teacher_logits / temperature, dim=-1)
    soft_prob = F.log_softmax(student_logits / temperature, dim=-1)
    return F.kl_div(soft_prob, soft_targets, reduction='batchmean') * (temperature ** 2)


def get_talker(path, **kwargs):
    wrapper = ModelClass.from_pretrained(path, **kwargs)
    if hasattr(wrapper, "model") and hasattr(wrapper.model, "talker"):
        return wrapper.model.talker
    elif hasattr(wrapper, "talker"):
        return wrapper.talker
    return wrapper.model


def get_text_embedding_layer(model):
    if hasattr(model, "model") and hasattr(model.model, "text_embedding"):
        return model.model.text_embedding
    elif hasattr(model, "text_embedding"):
        return model.text_embedding
    return None


def train_epoch_simple(
    teacher_model,
    student_model,
    dataloader,
    optimizer,
    scheduler,
    device,
    epoch,
    total_epochs,
    config
):
    teacher_model.eval()
    student_model.train()
    
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{total_epochs}")
    
    text_embed_teacher = get_text_embedding_layer(teacher_model)
    text_embed_student = get_text_embedding_layer(student_model)
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        max_seq_len = config.get('max_seq_length', 64)
        if input_ids.shape[1] > max_seq_len:
            input_ids = input_ids[:, :max_seq_len]
            attention_mask = attention_mask[:, :max_seq_len]
        
        batch_size = input_ids.shape[0]
        hidden_size = 2048
        dtype = torch.float16
        
        dummy_past_hidden = torch.zeros(batch_size, 1, hidden_size, device=device, dtype=dtype)
        
        with torch.no_grad():
            inputs_embeds_teacher = text_embed_teacher(input_ids) if text_embed_teacher else input_ids
            teacher_outputs = teacher_model(
                inputs_embeds=inputs_embeds_teacher,
                attention_mask=attention_mask,
                past_hidden=dummy_past_hidden
            )
            
            if hasattr(teacher_outputs, 'logits'):
                teacher_logits = teacher_outputs.logits
            else:
                teacher_logits = teacher_outputs[0]
        
        inputs_embeds_student = text_embed_student(input_ids) if text_embed_student else input_ids
        student_outputs = student_model(
            inputs_embeds=inputs_embeds_student,
            attention_mask=attention_mask,
            past_hidden=dummy_past_hidden
        )
        
        if hasattr(student_outputs, 'logits'):
            student_logits = student_outputs.logits
        else:
            student_logits = student_outputs[0]
        
        loss = distillation_loss(student_logits, teacher_logits, config.get('temperature', 2.0))
        
        accumulation_steps = config.get('gradient_accumulation_steps', 1)
        loss_value = loss.item()
        loss = loss / accumulation_steps
        loss.backward()
        
        del teacher_logits, student_logits, loss
        
        if (batch_idx + 1) % accumulation_steps == 0:
            clip_grad_norm_(student_model.parameters(), max_norm=config.get('gradient_clip_max_norm', 1.0))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()
        
        total_loss += loss_value
        num_batches += 1
        
        progress_bar.set_postfix({'loss': f"{total_loss/num_batches:.4f}", 'lr': f"{scheduler.get_last_lr()[0]:.2e}"})
    
    if num_batches % accumulation_steps != 0:
        clip_grad_norm_(student_model.parameters(), max_norm=config.get('gradient_clip_max_norm', 1.0))
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    
    return {'loss': total_loss / num_batches}


def validate_simple(teacher_model, student_model, dataloader, device, config):
    teacher_model.eval()
    student_model.eval()
    
    total_loss = 0
    num_batches = 0
    
    text_embed_teacher = get_text_embedding_layer(teacher_model)
    text_embed_student = get_text_embedding_layer(student_model)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            max_seq_len = config.get('max_seq_length', 64)
            if input_ids.shape[1] > max_seq_len:
                input_ids = input_ids[:, :max_seq_len]
                attention_mask = attention_mask[:, :max_seq_len]
            
            batch_size = input_ids.shape[0]
            hidden_size = 2048
            dtype = torch.float16
            
            dummy_past_hidden = torch.zeros(batch_size, 1, hidden_size, device=device, dtype=dtype)
            
            inputs_embeds_teacher = text_embed_teacher(input_ids) if text_embed_teacher else input_ids
            teacher_outputs = teacher_model(
                inputs_embeds=inputs_embeds_teacher,
                attention_mask=attention_mask,
                past_hidden=dummy_past_hidden
            )
            
            if hasattr(teacher_outputs, 'logits'):
                teacher_logits = teacher_outputs.logits
            else:
                teacher_logits = teacher_outputs[0]
            
            inputs_embeds_student = text_embed_student(input_ids) if text_embed_student else input_ids
            student_outputs = student_model(
                inputs_embeds=inputs_embeds_student,
                attention_mask=attention_mask,
                past_hidden=dummy_past_hidden
            )
            
            if hasattr(student_outputs, 'logits'):
                student_logits = student_outputs.logits
            else:
                student_logits = student_outputs[0]
            
            loss = distillation_loss(student_logits, teacher_logits, config.get('temperature', 2.0))
            total_loss += loss.item()
            num_batches += 1
    
    return {'loss': total_loss / num_batches}


def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = "./model_files"
    SAVE_PATH = "./checkpoints"
    
    print(f"Using device: {DEVICE}")
    print(f"Config: {DISTILLATION_CONFIG}")
    
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Run download_model.py")
        return
    
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    
    print("Loading Teacher Model on GPU...")
    teacher_model = get_talker(MODEL_PATH, device_map=DEVICE, trust_remote_code=True, torch_dtype=torch.float16)
    print(f"Teacher type: {type(teacher_model)}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, fix_mistral_regex=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Creating Student Model on GPU...")
    student_model = get_talker(MODEL_PATH, device_map=DEVICE, trust_remote_code=True, torch_dtype=torch.float16)
    print(f"Student type: {type(student_model)}")
    
    # Disable activation quantization and gradient checkpointing for simpler setup
    from quant_layers import QuantizedLinearINT4
    from model_utils import replace_linear_layers
    
    def replace_with_config(model):
        for name, module in model.named_children():
            if isinstance(module, torch.nn.Linear):
                print(f"Quantizing layer: {name}")
                quantized_layer = QuantizedLinearINT4.from_float(
                    module,
                    group_size=128,
                    quantize_input=False  # Disable for simplicity
                )
                setattr(model, name, quantized_layer)
            else:
                replace_with_config(module)
    
    replace_with_config(student_model)
    # Skip gradient checkpointing for now
    
    print("Loading Dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=DISTILLATION_CONFIG['max_seq_length'], padding="max_length")
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"], load_from_cache_file=False)
    tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask"])
    
    val_split = DISTILLATION_CONFIG['validation_split']
    train_size = int((1 - val_split) * len(tokenized_datasets))
    val_size = len(tokenized_datasets) - train_size
    
    train_dataset, val_dataset = random_split(tokenized_datasets, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_dataset, batch_size=DISTILLATION_CONFIG['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=DISTILLATION_CONFIG['batch_size'], shuffle=False, num_workers=0)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    print("Setting up optimizer...")
    optimizer = optim.AdamW(student_model.parameters(), lr=DISTILLATION_CONFIG['learning_rate'], weight_decay=0.01)
    
    epochs = DISTILLATION_CONFIG['epochs']
    total_steps = len(train_loader) * epochs // DISTILLATION_CONFIG['gradient_accumulation_steps']
    warmup_steps = int(total_steps * DISTILLATION_CONFIG['warmup_ratio'])
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=warmup_steps, T_mult=2)
    
    print(f"\nStarting training for {epochs} epochs...")
    print(f"Effective batch size: {DISTILLATION_CONFIG['batch_size'] * DISTILLATION_CONFIG['gradient_accumulation_steps']}")
    print(f"Total optimization steps: {total_steps}")
    print()
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"{'='*50}")
        
        train_metrics = train_epoch_simple(
            teacher_model, student_model, train_loader,
            optimizer, scheduler, DEVICE, epoch, epochs, DISTILLATION_CONFIG
        )
        
        val_metrics = validate_simple(teacher_model, student_model, val_loader, DEVICE, DISTILLATION_CONFIG)
        
        print(f"\nTrain Loss: {train_metrics['loss']:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            checkpoint_path = os.path.join(SAVE_PATH, "best_model.pt")
            if hasattr(student_model, "state_dict"):
                torch.save(student_model.state_dict(), checkpoint_path)
            print(f"New best model saved! Val loss: {best_val_loss:.4f}")
    
    print(f"\n{'='*50}")
    print("Training Complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
