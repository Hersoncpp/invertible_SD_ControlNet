images_dir = "/home/hesong/disk1/DF_INV/code/ControlNet-v1-1-nightly/dataset/OmniEdit-Filtered-1.2M_train/source_images"

import os

# find how many images in images_dir
image_ids = os.listdir(images_dir)
print(f"Found {len(image_ids)} images in {images_dir}")