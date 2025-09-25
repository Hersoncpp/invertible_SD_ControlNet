from cldm.model import create_model, load_state_dict
from train.data_loader import MyDataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch

from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
import os
#reduce fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
# 检查 是否有可用的GPU
print("CUDA available: ", torch.cuda.is_available())


batch_size = 2
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False
max_epochs = 5  # 设置训练轮数
resolution = 256

model_dir = './models'
model_name = 'control_v11e_sd15_ip2p'
model = create_model(f'{model_dir}/{model_name}.yaml').cpu()
model.load_state_dict(load_state_dict(f'{model_dir}/v1-5-pruned.ckpt', location='cuda'), strict=False)
model.load_state_dict(load_state_dict(f'{model_dir}/{model_name}.pth', location='cuda'), strict=False)

model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

# dataset preparation (parquet file)
data_dir = "/home/hesong/disk1/DF_INV/code/ControlNet-v1-1-nightly/dataset/OmniEdit-Filtered-1.2M_train_filtered"
dataloader = DataLoader(MyDataset(data_json_file=f'{data_dir}/prompts.json'), batch_size=batch_size, shuffle=True, num_workers=4)

logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(
    gpus=1,
    precision=16,
    accumulate_grad_batches=4,
    callbacks=[logger],
    max_epochs=max_epochs  # 设置训练轮数
)

trainer.fit(model, dataloader)

checkpoints_dir = './train/checkpoints'
os.makedirs(checkpoints_dir, exist_ok=True)
save_path = f'{checkpoints_dir}/{model_name}-finetuned.ckpt'
idx = 0
while save_path is not None and os.path.exists(save_path):
    save_path = save_path.replace('.ckpt', f'_{idx}.ckpt')
    idx += 1
trainer.save_checkpoint(save_path)
