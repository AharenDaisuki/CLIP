from dataclasses import dataclass

@dataclass
class CLIPConfig:
    # Model configuration
    model_name: str = "openai/clip-vit-base-patch32"
    # model_name: str = "google/siglip2-base-patch16-224"
    # model_name: str = "Salesforce/blip2-flan-t5-xl"

    # Dataset configuration
    flickr8k_dataset: str = "jxie/flickr8k"
    winoground_dataset: str = "facebook/winoground"

    # Training configuration
    batch_size: int = 32
    learning_rate: float = 1e-5
    num_epochs: int = 10
    warmup_ratio: float = 0.1
    weight_decay: float = 1e-2

    # Device configuration
    # device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Output configuration
    output_dir: str = "./outputs"
    model_save_path: str = "./models/clip_finetuned"

    # Evaluation configuration
    eval_batch_size: int = 16