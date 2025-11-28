import os

from configs import CLIPConfig
from models import load_model, load_siglip, load_clip
from metrics import compute_metrics
from utils import seed_everything, set_model_device
from data import prepare_flickr8k_dataset, CLIPCollator

def eval_split(split, model, processor, device,
    eval_bs_images: int, eval_bs_texts: int, is_siglip: bool
):
    """
    Run retrieval metrics on a dataset split and pretty print a concise subset.
    """
    # print(f"\n[Retrieval evaluation on {name}]")
    metrics = compute_metrics(
        model=model,
        processor=processor,
        split_ds=split,
        device=device,
        eval_image_bs=eval_bs_images,
        eval_text_bs=eval_bs_texts,
        is_siglip=is_siglip,
    )
    keys = ["i2t_R@1", "i2t_R@5", "i2t_R@10", "t2i_R@1", "t2i_R@5", "t2i_R@10",
            "i2t_MedR", "t2i_MedR", 
            # "avg_best_cosine", 
            "n_images", "n_texts"]
    shown = {k: (round(metrics[k], 4) if isinstance(metrics[k], float) else metrics[k]) for k in keys}
    print(shown)
    return metrics

def eval_clip(checkpoint: str, save_path: str, pad_to_max: bool = False):
    # config
    config = CLIPConfig()
    # reproducibility
    seed_everything(5489)
    # load model and processor
    # if pad_to_max: 
    #     model, processor = load_siglip(checkpoint)
    # else: 
    #     model, processor = load_clip(checkpoint)
    model, processor = load_model(checkpoint)
    # load dataset
    flickr8k = prepare_flickr8k_dataset()
    flickr8k["test"] = flickr8k["test"].shuffle(seed=5489)
    # collator = CLIPCollator(processor=processor, pad_to_max_for_siglip=pad_to_max)

    # evaluation
    device = set_model_device()
    model.to(device).eval()
    test_metric = eval_split(flickr8k["test"], model, processor, device,
        eval_bs_images=config.eval_batch_size, 
        eval_bs_texts=max(128, config.eval_batch_size), 
        is_siglip=pad_to_max,
    )
    import json
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, "test_metric.json"), "w") as f:
        json.dump({k: (None if v is None else float(v)) for k, v in test_metric.items()}, f, indent=2)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the model checkpoint to evaluate.")
    parser.add_argument("--save_path", type=str, required=True,
                        help="Directory to save evaluation results.")
    parser.add_argument("--pad_to_max", action="store_true",
                        help="Whether to pad texts to max length (for SigLIP).")
    args = parser.parse_args()

    eval_clip(args.checkpoint, args.save_path, pad_to_max=args.pad_to_max)