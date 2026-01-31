import torch
import os
import sys
from model_utils import replace_linear_layers

# Import from library
try:
    from qwen_tts import Qwen3TTSModel as ModelClass
except ImportError:
    print("Could not import Qwen3TTSModel from qwen_tts.")
    sys.exit(1)

def create_student_model(model_path="./model_files", save_path="./student_model_int4"):
    print(f"Loading Teacher model from {model_path}...")
    
    # Load Wrapper
    teacher_wrapper = ModelClass.from_pretrained(model_path, device_map="cpu", trust_remote_code=True)
    # UNWRAP: Get the underlying nn.Module
    teacher_model = teacher_wrapper.model 
    
    print("Teacher model loaded (Unwrapped).")
    
    print("Loading specialized Student Copy from disk...")
    student_wrapper = ModelClass.from_pretrained(model_path, device_map="cpu", trust_remote_code=True)
    student_model = student_wrapper.model
    
    # Ensure precision (float16)
    student_model.to(dtype=torch.float16)
    
    # Replace Linear Layers with Quantized Layers
    print("Replacing Linear layers with QuantizedLinearINT4...")
    replace_linear_layers(student_model)
    
    print("Student model structure created.")
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    print(f"Saving student model to {save_path}...")
    torch.save(student_model.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
    
    # Save config
    try:
        if hasattr(teacher_model, 'config'):
            teacher_model.config.save_pretrained(save_path)
    except:
        pass
        
    print("Done.")

if __name__ == "__main__":
    create_student_model()
