# Quantization Aware Knowledge Distillation (QAKD) Setup

This project implements QAKD to distill a Qwen 3 TTS model into an INT4 native structure.

## Prerequisites

1.  **Python Environment**: Ensure you have Python 3.10+ installed.
2.  **Dependencies**: Run the following command to install required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Workflow

### 1. Download the Teacher Model
Since the local files were deleted, we must download the original model from HuggingFace to use as the "Teacher".
Run:
```bash
python download_model.py
```
*This will download `Qwen/Qwen3-TTS-12Hz-1.7B-Base` to `./model_files`.*

### 2. Create the Student Model
We create a structural copy of the model and replace its linear layers with our custom `QuantizedLinearINT4` layers.
Run:
```bash
python create_student.py
```
*This will verify the architecture can be loaded and quantized.*

### 3. Run Distillation
Train the Student using the Teacher's outputs.
Run:
```bash
python distill.py
```
*Note: You may need to adjust `distill.py` to point to your specific dataset.*

## Files structure
*   `quant_layers.py`: Custom INT4 Linear layer with STE.
*   `model_utils.py`: Helper to swap layers.
*   `download_model.py`: Fetches model from HF.
*   `create_student.py`: Prepares the INT4 student architecture.
*   `distill.py`: The training loop.
