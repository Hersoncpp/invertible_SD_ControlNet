from huggingface_hub import HfApi
from transformers import T5Tokenizer, T5EncoderModel, CLIPTokenizer, CLIPTextModel

api = HfApi()#endpoint="https://hf-mirror.com")

# 检查模型是否存在
print(api.model_info(repo_id="openai/clip-vit-large-patch14"))

# 然后正常加载
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
# hf download https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11e_sd15_ip2p.pth
# model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")