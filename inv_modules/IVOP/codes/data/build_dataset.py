import os
import json
import cv2
from PIL import Image
import sys
sys.path.append('/home/hesong/disk1/DF_INV/code/ControlNet-v1-1-nightly')
from ip2p_predict import predict

def fun(image, prompt, transform):
    input_dict = {
        'input_image': image,
        'prompt': prompt,
        'a_prompt': 'best quality',
        'n_prompt': 'lowres, bad anatomy, bad hands, cropped, worst quality',
        'num_samples': 1,
        'image_resolution': 512,
        'ddim_steps': 20,
        'guess_mode': False,
        'strength': 1.0,
        'scale': 9.0,
        'seed': 12345,
        'eta': 0.0
    }
    output_image = transform(**input_dict)
    if isinstance(output_image, list):
        output_image = output_image[0]
    return output_image

if __name__ == "__main__":
    dataset_dir = "/home/hesong/disk1/DF_INV/code/ControlNet-v1-1-nightly/inv_modules/IVOP/codes/data/dataset"
    dataset_name = "ControlNet_ST"
    save_dir = f"{dataset_dir}/{dataset_name}"
    src_img_dir = f"{save_dir}/source_images"
    tar_img_dir = f"{save_dir}/target_images"
    prompt_json_path = f"{save_dir}/prompts.json"
    prompts_read_path = "/home/hesong/disk1/DF_INV/code/ControlNet-v1-1-nightly/dataset/OmniEdit-Filtered-1.2M/prompts.json"
    predict_trans = predict(cuda_id=1)
    transform_func = fun

    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)
    # if os.path.exists(src_img_dir) is False:
    #     os.makedirs(src_img_dir)
    if os.path.exists(tar_img_dir) is False:
        os.makedirs(tar_img_dir)

    # check prompts_read_path exists
    if os.path.exists(prompts_read_path) is False:
        raise ValueError(f"prompts_read_path {prompts_read_path} does not exist")
    # clear the prompt_json_path file
    if os.path.exists(prompt_json_path):
        os.remove(prompt_json_path)
    
    with open(prompts_read_path, 'rt') as f:
        for line in f:
            _dict = json.loads(line)
            source_fpth = _dict['source']
            src_name = os.path.basename(source_fpth)
            # target_fpth = _dict['target']
            prompt = _dict['prompt']
            source_img = cv2.imread(source_fpth)
            source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
            target_img = transform_func(source_img, prompt, predict_trans)
            target_fpth = f"{tar_img_dir}/{src_name}"
            # save target_img
            target_img_pil = Image.fromarray(target_img)
            target_img_pil.save(target_fpth)

            _dict['target'] = target_fpth

            prompt_txt = "{\"source\": \"" + str(source_fpth) + "\", \"target\": \"" + str(target_fpth) + "\", \"prompt\": \"" + prompt.replace("'", "\'") + "\"}\n"

            with open(prompt_json_path, "a") as f:
                f.write(prompt_txt)
