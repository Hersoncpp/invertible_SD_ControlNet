from transformers import CLIPProcessor, CLIPModel
from diffusers import CLIPTextModel, CLIPTokenizer
import torch

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_text_embedding(text_prompts, clip_model, clip_processor, device="cuda"):
    clip_model = clip_model.to(device)
    if isinstance(text_prompts, str):
        text_prompts = [text_prompts]
    
    inputs = clip_processor(text=text_prompts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        text_embeds = clip_model.get_text_features(**inputs)  
    
    return text_embeds