import os
from huggingface_hub import snapshot_download

def download_model(repo_id="Qwen/Qwen3-TTS-12Hz-1.7B-Base", local_dir="./model_files"):
    """
    Downloads the specified model from HuggingFace Hub.
    """
    print(f"Downloading {repo_id} to {local_dir}...")
    snapshot_download(repo_id=repo_id, local_dir=local_dir, local_dir_use_symlinks=False)
    print("Download complete.")

if __name__ == "__main__":
    # You can change the repo_id if you are using a different variant (e.g. VoiceDesign)
    download_model()
