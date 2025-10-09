images_dir = "/home/hesong/disk1/DF_INV/code/ControlNet-v1-1-nightly/dataset/OmniEdit-Filtered-1.2M_train/source_images"

import os
import cv2
# find how many images in images_dir
# image_ids = os.listdir(images_dir)
# print(f"Found {len(image_ids)} images in {images_dir}")

# get image byte size:
def get_image_size(image_path):
    return os.path.getsize(image_path)

def get_image_shape(image_path):
    import cv2
    img = cv2.imread(image_path)
    return img.shape  # (H, W, C)

if __name__ == "__main__":

    img1_path = f"/home/hesong/disk1/DF_INV/code/ControlNet-v1-1-nightly/inv_modules/IVOP/experiments/style_inv_archived_250927-194353/val_images/0/0_46100.jpg"
    save_img2_path = 'prac_imgs/img2.jpg'
    img2_path = f"/home/hesong/disk1/DF_INV/code/ControlNet-v1-1-nightly/inv_modules/HiNet/image/secret-rev/00000.png"
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    cv2.imwrite(save_img2_path, img2)
    img1_size = get_image_size(img1_path)
    img2_size = get_image_size(img2_path)
    img1_shape = get_image_shape(img1_path)
    img2_shape = get_image_shape(img2_path)

    print(f"Image 1 size: {img1_size} bytes, shape: {img1_shape}")
    print(f"Image 2 size: {img2_size} bytes, shape: {img2_shape}")
    img2 = cv2.imread(save_img2_path)
    img2_shape = get_image_shape(save_img2_path)
    img2_size = get_image_size(save_img2_path)
    img2_shape = get_image_shape(save_img2_path)
    print(f"Re-read Image 2 size: {img2_size} bytes, shape: {img2_shape}")
    # divide img2_size by (img2_shape[0] / img2_shape[1])**2
    # ratio = img2_size / (img2_shape[0] / img2_shape[1])**2
    # print(f"Image 2 size / (H/W)^2: {ratio}")


    # Save as JPEG with default quality (95)
    save_img1_path = 'prac_imgs/img1.jpg'
    save_img2_path = 'prac_imgs/img2.jpg'
    cv2.imwrite(save_img1_path, img1, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
    cv2.imwrite(save_img2_path, img2, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
    # compare new sizes
    new_img1_size = get_image_size(save_img1_path)
    new_img2_size = get_image_size(save_img2_path)
    new_image1_shape = get_image_shape(save_img1_path)
    new_image2_shape = get_image_shape(save_img2_path)
    print(f"New Image 1 size: {new_img1_size} bytes, shape: {new_image1_shape}")
    print(f"New Image 2 size: {new_img2_size} bytes, shape: {new_image2_shape}")
    
