# CLIP Project: Vision-Language Model Training and Evaluation

A comprehensive implementation for training and evaluating CLIP and related vision-language models on image-text retrieval tasks.

## Overview

This project implements training and evaluation pipelines for various vision-language models including:
- **CLIP** (OpenAI): Contrastive Language-Image Pre-training
- **SigLIP** (Google): Sigmoid Loss for Language Image Pre-training  
- **BLIP2** (Salesforce): Bootstrapping Language-Image Pre-training with Frozen Image Encoders

The project focuses on image-text retrieval tasks using the Flickr8k dataset with comprehensive evaluation metrics.

## Project Structure

```
course_project/
├── README.md           # This file
├── configs.py          # Configuration settings for models and training
├── data.py            # Data loading and preprocessing utilities
├── eval.py            # Model evaluation script
├── gallery.py         # Visualization and results gallery
├── metrics.py         # Retrieval evaluation metrics (R@1, R@5, R@10, MedR)
├── models.py          # Model loading utilities (CLIP, SigLIP, BLIP2)
├── train.py           # Training script with custom trainer
└── utils.py           # Utility functions
```

## Features

- **Multi-model Support**: Easily switch between CLIP, SigLIP, and BLIP2 architectures
- **Flexible Training**: Custom trainer with support for LoRA fine-tuning
- **Comprehensive Evaluation**: Image-to-text and text-to-image retrieval metrics
- **Dataset Integration**: Built-in support for Flickr8k and Winoground datasets
- **Visualization Tools**: Gallery generation for qualitative analysis

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd course_project
```

2. Install required dependencies:
```bash
pip install torch transformers datasets peft accelerate pillow numpy scikit-learn matplotlib
```

## Quick Start

### Training

To train a CLIP model on Flickr8k:
```bash
python train.py
```

The training will:
- Load the specified model (default: CLIP ViT-Base-Patch32)
- Prepare the Flickr8k dataset
- Train using contrastive loss
- Evaluate retrieval metrics after each epoch
- Save checkpoints to `./outputs/`

### Evaluation

To evaluate a trained model:
```bash
python eval.py
```

### Generate Results Gallery

To create visualizations of retrieval results:
```bash
python gallery.py
```

## Configuration

Modify `configs.py` to adjust training parameters:

```python
@dataclass
class CLIPConfig:
    # Model selection
    model_name: str = "openai/clip-vit-base-patch32"  # or "google/siglip2-base-patch16-224"
    
    # Training hyperparameters
    batch_size: int = 32
    learning_rate: float = 1e-5
    num_epochs: int = 10
    warmup_ratio: float = 0.1
    weight_decay: float = 1e-2
    
    # Dataset settings
    flickr8k_dataset: str = "jxie/flickr8k"
    
    # Output directories
    output_dir: str = "./outputs"
    model_save_path: str = "./models/clip_finetuned"
```

## Supported Models

### CLIP
- Model: `openai/clip-vit-base-patch32`
- Architecture: ViT-Base + Text Transformer
- Use case: General image-text retrieval

### SigLIP
- Model: `google/siglip2-base-patch16-224`
- Architecture: Improved contrastive learning with sigmoid loss
- Use case: Enhanced image-text retrieval performance

### BLIP2
- Model: `Salesforce/blip2-flan-t5-xl`
- Architecture: Vision encoder + language model with LoRA
- Use case: Advanced vision-language understanding

## Evaluation Metrics

The project evaluates retrieval performance using:
- **Recall@K (R@1, R@5, R@10)**: Fraction of correct items in top-K results
- **Median Rank (MedR)**: Median rank of correct retrieval
- **Image-to-Text (i2t)**: Retrieving text for given images
- **Text-to-Image (t2i)**: Retrieving images for given text

Example output:
```
[Validation retrieval @ epoch 1]
{'i2t_R@1': 0.4234, 'i2t_R@5': 0.7891, 'i2t_R@10': 0.8456,
 't2i_R@1': 0.3892, 't2i_R@5': 0.7234, 't2i_R@10': 0.8123,
 'i2t_MedR': 2.1, 't2i_MedR': 2.8}
```

## Datasets

### Flickr8k
- **Description**: 8,000 images with 5 captions each
- **Task**: Image-text retrieval
- **Split**: Train (6,000), Validation (1,000), Test (1,000)

### Winoground
- **Description**: Challenging vision-language reasoning dataset
- **Task**: Understanding compositional relationships

## Advanced Features

### LoRA Fine-tuning
For BLIP2 models, LoRA (Low-Rank Adaptation) is enabled:
```python
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["qformer.query_tokens", "language_model"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

### Custom Callbacks
The training pipeline includes:
- Periodic evaluation during training
- Metric logging and visualization
- Model checkpointing

## Requirements

- Python 3.8+
- PyTorch 1.12+
- Transformers 4.20+
- Datasets
- PEFT (for LoRA)
- PIL, NumPy, scikit-learn
- Matplotlib (for visualization)
