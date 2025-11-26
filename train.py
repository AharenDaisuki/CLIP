from torch.optim import AdamW
from transformers import Trainer, TrainerCallback, TrainingArguments

from models import load_model, load_blip
from data import prepare_flickr8k_dataset, CLIPCollator
from configs import CLIPConfig
from metrics import compute_metrics

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class CLIPTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        outputs = model(**inputs, return_loss=True)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]
        return (loss, outputs) if return_outputs else loss

class EvalCallback(TrainerCallback):
    """
    Periodically run retrieval evaluation on a provided split.
    Defaults preserved from your original callback wiring.
    """
    def __init__(self, processor, eval_dataset, eval_batch_size=128, is_siglip=False, every_n_epochs=1):
        self.processor = processor
        self.split_ds = eval_dataset
        self.eval_bs = eval_batch_size
        self.is_siglip = is_siglip
        self.every = every_n_epochs

    def on_epoch_end(self, args, state, control, **kwargs):
        if (state.epoch or 0) % self.every != 0:
            return
        model = kwargs["model"]
        device = model.device
        metrics = compute_metrics(
            model, self.processor, self.split_ds, device,
            eval_image_bs=self.eval_bs, eval_text_bs=max(128, self.eval_bs),
            is_siglip=self.is_siglip
        )
        print(f"\n[Validation retrieval @ epoch {int(state.epoch)}]")
        # keys = ["i2t_R@1", "i2t_R@5", "i2t_R@10", "t2i_R@1", "t2i_R@5", "t2i_R@10",
        #         "i2t_MedR", "t2i_MedR", "avg_best_cosine"]
        keys = ["i2t_R@1", "i2t_R@5", "i2t_R@10", "t2i_R@1", "t2i_R@5", "t2i_R@10", "i2t_MedR", "t2i_MedR"]
        print({k: (round(metrics[k], 4) if isinstance(metrics[k], float) else metrics[k]) for k in keys})

def train_clip():
    # config
    config = CLIPConfig()
    training_arguments = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="cosine",
        weight_decay=config.weight_decay,
        eval_strategy='epoch',
        save_strategy='epoch',
        logging_strategy='epoch',
        save_total_limit=3,
        logging_dir=os.path.join(config.output_dir, "logs"),
        remove_unused_columns=False,
        seed = 5489,
    )
    is_siglip = True if config.model_name.startswith("google/siglip") else False
    # model
    model, processor = load_model(config.model_name)
    # model, processor = load_blip(config.model_name)
    # Dataset
    flickr8k = prepare_flickr8k_dataset()
    flickr8k["train"] = flickr8k["train"].shuffle(seed=5489)
    # Collator
    collator = CLIPCollator(processor=processor, pad_to_max_for_siglip=is_siglip)
    # optimizer
    optimizer = AdamW(model.parameters(), 
                      lr=config.learning_rate, 
                      weight_decay=config.weight_decay)
    # trainer
    trainer = CLIPTrainer(
        model=model,
        args=training_arguments,
        train_dataset=flickr8k["train"],
        eval_dataset=flickr8k["validation"],
        data_collator=collator,
        optimizers=(optimizer, None),
    )

    # Register callback if we have a validation split
    if "validation" in flickr8k:
        trainer.add_callback(EvalCallback(
            processor = processor,
            eval_dataset = flickr8k["validation"],
            eval_batch_size = config.eval_batch_size,
            every_n_epochs = 1,
            is_siglip = is_siglip,
        ))

    # Train
    trainer.train()

    # Save
    trainer.save_model(config.output_dir)
    processor.save_pretrained(config.output_dir)

if __name__ == "__main__":
    train_clip()