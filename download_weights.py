import os
import sys
import tarfile
from huggingface_hub import hf_hub_download

# ==============================================================================
# Description: Automatically downloads pretrained weights from Hugging Face Hub 
#              and extracts them while preserving the original directory structure.
# License: Apache License 2.0
# ==============================================================================

# 1. Configuration (Please update with your own repository details)
REPO_ID = "ahoka76/rs45327"   # Hugging Face Repository ID
FILENAME = "weights.tar"                # Target uncompressed tar file name
TARGET_DIR = "./"                       # Extraction destination directory

def main():
    print("=" * 80)
    print(" [MDPI Remote Sensing] Model Parameters Download & Initialization")
    print(f" Repository ID : {REPO_ID}")
    print(f" Artifact Name : {FILENAME}")
    print("=" * 80)

    # 2. Download the uncompressed tar file from Hugging Face Hub
    print("\n📦 Step 1: Downloading the tar archive from Hugging Face...")
    
    # Automatically reads the token from the environment variable if the repo is Private
    hf_token = os.getenv("HF_TOKEN")
    
    try:
        downloaded_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
            repo_type="model",
            token=hf_token
        )
        print(f"✅ Download completed successfully!")
        print(f"   Cached path: {downloaded_path}")
        
    except Exception as e:
        print(f"❌ Error: Failed to download artifacts from Hugging Face Hub.")
        print(f"   If the repository is private, please ensure 'HF_TOKEN' is set correctly.")
        print(f"   Details: {e}")
        sys.exit(1)

    # 3. Extract the tar file and restore the original folder structure
    print(f"\n📂 Step 2: Extracting tar archive to preserve original directory structure...")
    
    try:
        # Changed mode from "r:gz" to "r:" for uncompressed tar archives
        with tarfile.open(downloaded_path, "r:") as tar:
            # Extract all files safely
            tar.extractall(path=TARGET_DIR)
        print(f"🎉 Success! All .pth files and directory structures have been fully restored.")
        print(f"   The workspace is now ready for evaluation and training replication.")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Error: Failed to extract the tar archive.")
        print(f"   Details: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

