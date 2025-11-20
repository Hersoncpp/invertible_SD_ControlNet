# Super parameters
quantization = True
clamp = 2.0
channels_in = 3
log10_lr = -4.5
lr = 10 ** log10_lr
epochs = 1000
weight_decay = 1e-5
init_scale = 0.01

lamda_reconstruction = 5
lamda_guide = 1
lamda_low_frequency = 1
device_ids = [0, 1, 2, 3]

# Train:
batch_size = 8
cropsize = 256
betas = (0.5, 0.999)
weight_step = 1000
gamma = 0.5

# Val:
cropsize_val = 256
batchsize_val = 1
shuffle_val = False
val_freq = 50


# Dataset
# TRAIN_JSON_PATH  =  "/home/hesong/disk1/DF_INV/code/ControlNet-v1-1-nightly/dataset/OmniEdit-Filtered-1.2M_train_filtered/prompts.json"
TRAIN_JSON_PATH  =  "/home/hesong/disk1/DF_INV/code/ControlNet-v1-1-nightly/inv_modules/IVOP/codes/data/dataset/ControlNet_ST_full/prompts.json"
# TRAIN_JSON_PATH  = "/home/hesong/disk1/DF_INV/code/ControlNet-v1-1-nightly/inv_modules/IVOP/codes/data/dataset/ControlNet_ST/prompts.json"
VAL_JSON_PATH  = "/home/hesong/disk1/DF_INV/code/ControlNet-v1-1-nightly/inv_modules/IVOP/codes/data/dataset/ControlNet_ST/prompts.json"
# TRAIN_PATH = '/home/jjp/Dataset/DIV2K/DIV2K_train_HR/'
# VAL_PATH = '/home/jjp/Dataset/DIV2K/DIV2K_valid_HR/'
format_train = 'png'
format_val = 'png'

# Display and logging:
loss_display_cutoff = 2.0
loss_names = ['L', 'lr']
silent = False
live_visualization = False
progress_bar = False


# Saving checkpoints:

MODEL_PATH = '/home/hesong/disk1/DF_INV/code/ControlNet-v1-1-nightly/inv_modules/HiNet/model/'
checkpoint_on_error = True
SAVE_freq = 50

IMAGE_PATH = '/home/yukai/disk1/invertible_SD_ControlNet/inv_modules/HiNet/image/'
IMAGE_PATH_cover = IMAGE_PATH + 'cover/'
IMAGE_PATH_secret = IMAGE_PATH + 'secret/'
IMAGE_PATH_steg = IMAGE_PATH + 'steg/'
IMAGE_PATH_secret_rev = IMAGE_PATH + 'secret-rev/'

# Load:
suffix = 'model_checkpoint_00100_full_control_quantization.pt' # model_checkpoint_00200_full_control_.pt
tain_next = False
trained_epoch = 0
save_suffix = f"full_control_{'compressed' if quantization else ''}" # "partial" or "full"