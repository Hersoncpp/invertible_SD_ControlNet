


def work(image_id):
    img_dir = "/home/hesong/disk1/DF_INV/code/ControlNet-v1-1-nightly/inv_modules/IVOP/experiments/style_inv/val_images"
    #image_id = "2"
    image_read_dir = f"{img_dir}/{image_id}"
    training_steps = 3400
    GT_img_path = f"{image_read_dir}/{image_id}_GT_100.jpg"
    LR_ref_img_path = f"{image_read_dir}/{image_id}_LR_ref_100.jpg"
    forwLR_img_path = f"{image_read_dir}/{image_id}_forwLR_{training_steps}.jpg"
    rec_img_path = f"{image_read_dir}/{image_id}_{training_steps}.jpg"

    import os
    import cv2
    import numpy as np
    from matplotlib import pyplot as plt

    GT_img = cv2.imread(GT_img_path)
    GT_img = cv2.cvtColor(GT_img, cv2.COLOR_BGR2RGB)
    LR_ref_img = cv2.imread(LR_ref_img_path)
    LR_ref_img = cv2.cvtColor(LR_ref_img, cv2.COLOR_BGR2RGB)
    forwLR_img = cv2.imread(forwLR_img_path)
    forwLR_img = cv2.cvtColor(forwLR_img, cv2.COLOR_BGR2RGB)
    rec_img = cv2.imread(rec_img_path)
    rec_img = cv2.cvtColor(rec_img, cv2.COLOR_BGR2RGB)

    # changes between GT and rec
    diff_GT_rec = cv2.absdiff(GT_img, rec_img)
    diff_GT_rec = cv2.cvtColor(diff_GT_rec, cv2.COLOR_RGB2GRAY)
    # _, diff_GT_rec = cv2.threshold(diff_GT_rec, 30, 255, cv2.THRESH_BINARY)
    # diff_GT_rec = cv2.cvtColor(diff_GT_rec, cv2.COLOR_GRAY2RGB)

    # changes between forwLR and GT
    diff_forwLR_GT = cv2.absdiff(forwLR_img, GT_img)
    diff_forwLR_GT = cv2.cvtColor(diff_forwLR_GT, cv2.COLOR_RGB2GRAY)
    # _, diff_forwLR_GT = cv2.threshold(diff_forwLR_GT, 30, 255, cv2.THRESH_BINARY)
    # diff_forwLR_GT = cv2.cvtColor(diff_forwLR_GT, cv2.COLOR_GRAY2RGB)

    # changes between forwLR and rec
    diff_forwLR_rec = cv2.absdiff(forwLR_img, rec_img)
    diff_forwLR_rec = cv2.cvtColor(diff_forwLR_rec, cv2.COLOR_RGB2GRAY)
    # _, diff_forwLR_rec = cv2.threshold(diff_forwLR_rec, 30, 255, cv2.THRESH_BINARY)
    # diff_forwLR_rec = cv2.cvtColor(diff_forwLR_rec, cv2.COLOR_GRAY2RGB)

    # save images
    save_dir = f"{image_read_dir}"
    GT_rec_diff_path = f"{save_dir}/{image_id}_GT_rec_diff_{training_steps}.jpg"
    forwLR_GT_diff_path = f"{save_dir}/{image_id}_forwLR_GT_diff_{training_steps}.jpg"
    forwLR_rec_diff_path = f"{save_dir}/{image_id}_forwLR_rec_diff_{training_steps}.jpg"

    cv2.imwrite(GT_rec_diff_path, diff_GT_rec)
    cv2.imwrite(forwLR_GT_diff_path, diff_forwLR_GT)
    cv2.imwrite(forwLR_rec_diff_path, diff_forwLR_rec)
    print(f"Saved difference images to {save_dir}")

if __name__ == "__main__":
    image_ids = ["2", "3", "4", "5", "6", "7", "8", "9"]
    for image_id in image_ids:
        work(image_id)