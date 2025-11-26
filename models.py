from transformers import AutoModel, AutoProcessor
from transformers import CLIPModel, CLIPProcessor

from transformers import Blip2Processor, Blip2ForConditionalGeneration
from peft import LoraConfig, get_peft_model

def load_model(model_name: str):
    model = AutoModel.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
    return model, processor

def load_clip():
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name, use_fast=True)
    return model, processor

def load_blip():
    model_name = "Salesforce/blip2-flan-t5-xl"
    processor = Blip2Processor.from_pretrained(model_name, use_fast=True)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name,
        device_map="auto",
        load_in_8bit=True,  # Optional for low memory
    )
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["qformer.query_tokens", "language_model"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    return model, processor



# def load_clip(model_name: str):
#     model = AutoModel.from_pretrained(model_name)
#     processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
#     return model, processor

# def load_blip():
#     model_name = "Salesforce/blip-image-captioning-base"
#     model = AutoModel.from_pretrained(model_name)
#     processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
#     return model, processor