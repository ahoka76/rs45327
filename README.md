# Model Registry and Replication Guide

This repository contains the official configuration files and replication instructions for our study submitted to **MDPI Remote Sensing**. To ensure full reproducibility, we provide the complete evaluation matrix across multiple remote sensing datasets, splits, and state-of-the-art architectures.

---

### Prerequisites and Environment

To replicate the experimental environment and prevent version mismatches, ensure your system satisfies the following core framework specifications:

* **PyTorch:** `v2.10.0`
* **MMSegmentation:** `v1.2.2`
* **MMSegmentation Extension:** [mmseg-extension](https://github.com/chenller/mmseg-extension) (chenller/mmseg-extension)

Please make sure that both `mmseg` and the `mmseg-extension` repository are properly installed and registered in your Python environment before running the evaluation pipelines.

---

### Benchmark Directory Structure and Model Matrix

The complete model registry is structured hierarchically across 4 datasets (`flair`, `oem`, `cld`, `unified`) and 3 evaluation splits (`holdout`, `partA`, `partB`). 

Following the successful execution of the retrieval script, your workspace will be populated into the following structure:

```text
model/
├── [flair | oem | cld | unified]/      -- 4 Remote Sensing Datasets
│   ├── [holdout | partA | partB]/      -- 3 Evaluation Splits
│   │   ├── configs/
│   │   │   ├── knet-swinlarge.py
│   │   │   ├── segformer-mitb5.py
│   │   │   └── ... (15 architectures per split)
│   │   └── weights/
│   │       ├── knet-swinlarge/
│   │       │   └── best_mIoU_iter_XXXXX.pth
│   │       └── ...
```

#### Supported Model Architectures per Dataset/Split
Every single split contains the exact configuration files (.py) and pre-trained checkpoint parameters (.pth) for the following state-of-the-art architectures:

| Framework / Backbone | Configuration File Name | Model Category |
| :--- | :--- | :--- |
| K-Net | knet-swinlarge.py, knet-swinbase.py, knet-vitcomerbase.py | Transformer-based Query Segmentation |
| UperNet | upernet-swinbase.py, upernet-swintiny.py, upernet-convnextbase.py, upernet-unireplknetbase.py, upernet-resnet101.py | Unified Perceptual Parsing / Multi-scale |
| SegFormer | segformer-mitb5.py, segformer-mitb2.py | Lightweight & Efficient Semantic ViT |
| HRNet + OCR | hrnetocr-hrnetw48.py, hrnetocr-hrnetw18.py | High-Resolution Dense Prediction |
| DeepLabV3+ | deeplabv3plus-resnet101.py | Atrous Spatial Pyramid Pooling Baseline |
| PSPNet | pspnet-resnet101.py | Pyramid Pooling Context Baseline |

---

### Weights Retrieval via Hugging Face Hub

To replicate the experimental results without inflating the GitHub repository size, all trained model checkpoints (.pth) are hosted on the Hugging Face Hub. A Python execution script, `download_weights.py`, is provided to automate the file acquisition and restore the comprehensive directory tree structure.

#### Prerequisites
The script requires the Hugging Face Hub utility library. Install the package using pip:
```bash
pip install huggingface_hub
```

#### Authentication for Private Repositories (Optional)
If your Hugging Face repository status is configured as Private, you must generate an Access Token with Read permissions from your Hugging Face account settings and export it as an environment variable before executing the script:
```bash
export HF_TOKEN="your_huggingface_access_token"
```
*Note: If the repository is configured as Public, this authentication step can be skipped.*

#### Execution
Run the initialization script from the root directory of the project:
```bash
python download_weights.py
```
The script will sequentially download the uncompressed `weights.tar` archive from the hub and unpack all checkpoints into their respective `model/[dataset]/[split]/weights/[architecture]/` subdirectories automatically.

---

### Evaluation and Replication Command Template

To replicate the evaluation scores reported in the paper for any specific combination, use the standard MMSegmentation test execution pipeline. 

**Command Template:**
```bash
python tools/test.py \
  model/<DATASET_NAME>/<SPLIT_NAME>/configs/<ARCHITECTURE>.py \
  model/<DATASET_NAME>/<SPLIT_NAME>/weights/<ARCHITECTURE>/<CHECKPOINT_FILE>.pth \
```

**Example 1: Evaluating K-Net Swin-Large on the flair dataset (holdout split)**
```bash
python tools/test.py \
  model/flair/holdout/configs/knet-swinlarge.py \
  model/flair/holdout/weights/knet-swinlarge/best_mIoU_iter_183750.pth \
```

**Example 2: Evaluating SegFormer Mit-B5 on the unified dataset (partB split)**
```bash
python tools/test.py \
  model/unified/partB/configs/segformer-mitb5.py \
  model/unified/partB/weights/segformer-mitb5/best_average_mIoU_iter_924956.pth \
```

---

### Citation and License

The codebase and metadata in this repository are licensed under the **Apache License 2.0**. If you use these configuration files or pretrained weights in your research, please cite our paper:
