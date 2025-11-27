import torch 
import torch.nn.functional as F

# from tqdm import tqdm
from typing import Tuple, Any, List
from transformers import AutoTokenizer

def seed_everything(seed: int):
    """
    Set random seed for reproducibility.
    """
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# device
def set_model_device() -> torch.device:
    """
    Set the device for model training based on availability of CUDA.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:   
        device = torch.device("cpu")
    return device

def _device_is_cuda(device) -> bool:
    if isinstance(device, torch.device):
        return device.type == "cuda"
    if isinstance(device, str):
        return device.startswith("cuda")
    return False

def get_model_device_dtype(model) -> Tuple[torch.device, torch.dtype]:
    """
    Return the device and dtype of a model by inspecting the first parameter.
    """
    p = next(model.parameters())
    return p.device, p.dtype

# embedding
def embed_images(
    model, processor, images: List[Any], device: torch.device, batch_size: int = 128
) -> torch.Tensor:
    """
    Compute L2-normalized image embeddings using model.get_image_features.
    Returns a float32 tensor on CPU/GPU with shape (N, D).
    Defaults preserved: batch_size=128.
    """
    model.eval()
    _, model_dtype = get_model_device_dtype(model)
    feats = []
    image_proc = getattr(processor, "image_processor", processor)

    with torch.inference_mode():
        for i in range(0, len(images), batch_size):
            batch_imgs = images[i : i + batch_size]
            inputs = image_proc(images=batch_imgs, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device, dtype=model_dtype)
            f = model.get_image_features(pixel_values=pixel_values)
            feats.append(f.detach())

    feats = torch.cat(feats, dim=0)  # device == model device
    feats = F.normalize(feats.float(), dim=-1)
    return feats

def embed_texts(
    model, processor, texts: List[str], device: torch.device,
    batch_size: int = 256, is_siglip: bool = False
) -> torch.Tensor:
    """
    Compute L2-normalized text embeddings using model.get_text_features.
    Defaults preserved: batch_size=256, is_siglip=False.
    """
    model.eval()
    feats = []
    # pad = "max_length" if is_siglip else True

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        # Fallback: try model.name_or_path; use CLIP tokenizer by default.
        name_or_path = getattr(model, "name_or_path", None) or "openai/clip-vit-base-patch16"
        tokenizer = AutoTokenizer.from_pretrained(name_or_path, use_fast=True)

    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            batch_txts = texts[i : i + batch_size]
            if is_siglip:
                # SigLIP requires max_length padding
                inputs = tokenizer(
                    text=batch_txts,
                    truncation=True,
                    padding="max_length",
                    max_length=64,
                    return_tensors="pt",
                )
            else:   
                inputs = tokenizer(
                    text=batch_txts, 
                    truncation=True, 
                    padding=True, 
                    return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)
            attn = inputs.get("attention_mask", None)
            attn = attn.to(device) if attn is not None else None
            f = model.get_text_features(input_ids=input_ids, attention_mask=attn)
            feats.append(f.detach())

    feats = torch.cat(feats, dim=0)
    feats = F.normalize(feats.float(), dim=-1)
    return feats

def build_eval_index(hf_dataset) -> Tuple[list, list, List[List[int]], List[int]]:
    """
    Convert a split with columns:
      - image: PIL image
      - captions: List[str] (>=1)
    into:
      - images: list of images (N)
      - texts: list of all captions (M)
      - img_to_txt: list of lists; img_to_txt[i] -> indices of texts for image i
      - txt_to_img: list of ints; txt_to_img[j] -> image index for text j
    """
    images = []
    texts = []
    img_to_txt = []
    txt_to_img = []

    for i in range(len(hf_dataset)):
        ex = hf_dataset[i]
        images.append(ex["image"])
        caps = ex["captions"]
        cur_txt_ids = []
        for c in caps:
            cur_txt_ids.append(len(texts))
            texts.append(c)
            txt_to_img.append(i)
        img_to_txt.append(cur_txt_ids)

    return images, texts, img_to_txt, txt_to_img