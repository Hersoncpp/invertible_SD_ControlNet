from huggingface_hub import snapshot_download

# rm -rf ~/disk1/huggingface_cache/models--lllyasviel--ControlNet-v1-1
# mkdir -p ~/disk1/huggingface_cache/models--lllyasviel--ControlNet-v1-1
# chmod -R 755 ~/disk1/huggingface_cache  # 确保可读写

# 下载不同的任务模型
# huggingface-cli download lllyasviel/ControlNet-v1-1 --local-dir ~/disk1/huggingface_cache/

# 下载微调数据
# nohup hf download TIGER-Lab/OmniEdit-Filtered-1.2M --repo-type=dataset --local-dir ~/disk1/DF_INV/raw_data/OmniEdit-Filtered-1.2M >~/disk1/DF_INV/raw_data/nohup.out &
# 指定自定义下载路径（例如：/mnt/data/controlnet）
download_path = snapshot_download(
    repo_id="lllyasviel/ControlNet-v1-1",
    cache_dir="~/disk1/huggingface_cache",  # 你的自定义路径
    local_dir_use_symlinks=False,  # 禁用符号链接，直接复制文件
    force_download=True,           # 忽略现有缓存
    resume_download=False          # 不从断点续传
)
print(f"模型已下载到：{download_path}")
# 下载stable-diffusion 1.5 模型
# wget --header="Authorization: Bearer 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" https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt -O v1-5-pruned.ckpt

# 下载lineart anime 模型
# wget --header="Authorization: Bearer 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" https://huggingface.co/botp/Anything-Preservation/resolve/main/anything-v3-full.safetensors -O anything-v3-full.safetensors