import torch
import random

from typing import Any, Dict, List
from datasets import load_dataset, Image as HFImage
from dataclasses import dataclass
from transformers import AutoTokenizer

@dataclass
class CLIPCollator:
    processor: Any
    pad_to_max_for_siglip: bool

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        images = [f["image"] for f in features]
        texts  = [random.choice(f["captions"]) for f in features]

        image_proc = getattr(self.processor, "image_processor", self.processor)
        tokenizer  = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            name_or_path = getattr(self.processor, "name_or_path", None) or "openai/clip-vit-base-patch32"
            tokenizer = AutoTokenizer.from_pretrained(name_or_path, use_fast=True)

        pixel = image_proc(images=images, return_tensors="pt")
        text  = tokenizer(
            text=texts,
            truncation=True,
            padding=("max_length" if self.pad_to_max_for_siglip else True),
            return_tensors="pt",
        )
        return {**pixel, **text}
    
def prepare_flickr8k_dataset():
    dataset = load_dataset("jxie/flickr8k")
    def collect_caps(ex):
        return {"image": ex["image"], "captions": [ex[f"caption_{i}"] for i in range(5)]}
    dataset = dataset.map(collect_caps, remove_columns=[c for c in dataset["train"].column_names if c != "image"])
    for split in dataset.keys():
        dataset[split] = dataset[split].cast_column("image", HFImage())
    return dataset