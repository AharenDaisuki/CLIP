import numpy as np

from matplotlib import pyplot as plt
from datasets import load_dataset

from utils import resize_image

def visualize_topk_images(dataset, image_indices, top_k:int=5, savefig:str='topk.png'):
    assert top_k > 1
    samples = dataset.select(image_indices)

    # Create figure
    fig, axes = plt.subplots(1, top_k, figsize=(20, 4))
    
    fig.suptitle("retrieval results", fontsize=16)
    
    # Plot each sample
    for _, (ax, sample) in enumerate(zip(axes, samples)):
        image = sample["image"]
        image = resize_image(image)
        ax.imshow(image)
        ax.axis("off")
    
    plt.tight_layout()
    plt.savefig(savefig)
    plt.show()

def visualize_flickr8k(sample_size=5):
    """Visualize samples from Flickr8k dataset"""
    dataset = load_dataset("jxie/flickr8k")
    
    # Select random samples
    indices = np.random.choice(len(dataset["train"]), sample_size, replace=False)
    samples = dataset["train"].select(indices)
    
    # Create figure
    fig, axes = plt.subplots(1, sample_size, figsize=(20, 4))
    if sample_size == 1:
        axes = [axes]
    
    fig.suptitle("Flickr8k Dataset Samples", fontsize=16)
    
    # Plot each sample
    for i, (ax, sample) in enumerate(zip(axes, samples)):
        image = sample["image"]
        captions = sample["caption"] if isinstance(sample["caption"], list) else [sample["caption"]]
        
        ax.imshow(image)
        ax.set_title(f"Sample {i+1}\nCaptions: {len(captions)}", fontsize=10)
        ax.axis("off")
    
    plt.tight_layout()
    plt.show()

def visualize_winoground(sample_size=5):
    """Visualize samples from Winoground dataset"""
    dataset = load_dataset("facebook/winoground")
    
    # Select random samples
    indices = np.random.choice(len(dataset["test"]), min(sample_size, len(dataset["test"])), replace=False)
    samples = dataset["test"].select(indices)
    
    # Create figure
    fig, axes = plt.subplots(2, sample_size, figsize=(20, 8))
    if sample_size == 1:
        axes = [[axes[0]], [axes[1]]]
    
    fig.suptitle("Winoground Dataset Samples", fontsize=16)
    
    # Plot each sample (image pairs)
    for i, sample in enumerate(samples):
        # First image-caption pair
        axes[0][i].imshow(sample["image_0"])
        axes[0][i].set_title(f"Pair {i+1}\nCaption 0: {sample['caption_0']}", fontsize=8)
        axes[0][i].axis("off")
        
        # Second image-caption pair
        axes[1][i].imshow(sample["image_1"])
        axes[1][i].set_title(f"Caption 1: {sample['caption_1']}", fontsize=8)
        axes[1][i].axis("off")
    
    plt.tight_layout()
    plt.show()