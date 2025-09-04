from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler



model_name = 'control_v11e_sd15_ip2p' 
model = create_model(f'./models/{model_name}.yaml').cpu()
model.load_state_dict(load_state_dict('./models/v1-5-pruned.ckpt', location='cuda'), strict=False)
model_name = "control_v11e_sd15_ip2p-finetuned"
model.load_state_dict(load_state_dict(f'./models/{model_name}.pth', location='cuda'), strict=False)
model = model.cuda()
ddim_sampler = DDIMSampler(model)


def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta):
    # input_image, prompt, image_resolution=256, ddim_steps20
    with torch.no_grad():
        input_image = HWC3(input_image)

        detected_map = input_image.copy()

        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()
        
        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
        # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01

        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return [detected_map] + results

def save_files(images, file_name, resolution=256):
    from PIL import Image
    import math
    import cv2
    import torch
    """
    Combine a list of images into a single large image in a grid.
    Args:
        images: List of images (numpy.ndarray, PIL.Image, or torch.Tensor).
        resolution: Size of each image (assumed square, e.g., 256).
    Returns:
        numpy.ndarray: Combined grid image (RGB, uint8).
    """
    # Convert all images to NumPy uint8 (RGB, [H, W, 3])
    # print(type(images))
    np_images = []
    for img in images:
        # print(type(img))
        # for key, val in img.items():
        #     print(f"{key}: {val}")
        if isinstance(img, torch.Tensor):
            # Tensor: [C, H, W] or [H, W, C], possibly [-1, 1] or [0, 1]
            img = img.cpu()
            if img.shape[0] in [1, 3]:  # CHW format
                img = img.permute(1, 2, 0)  # To HWC
            # Denormalize
            if img.min() < 0:
                img = (img + 1.0) * 127.5  # [-1, 1] to [0, 255]
            else:
                img = img * 255.0  # [0, 1] to [0, 255]
            img = img.clamp(0, 255).numpy().astype(np.uint8)
        elif isinstance(img, Image.Image):
            img = np.array(img)  # Convert PIL.Image to NumPy
        elif isinstance(img, np.ndarray):
            img = img
        elif isinstance(img, dict):
            img_path = img.get('name', None)
            if img_path is not None:
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                raise ValueError("Dictionary input must contain 'name' key with image path.")
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")
        # Resize to resolution
        img = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_LINEAR)
        np_images.append(img)

    # Calculate grid dimensions
    n_images = len(np_images)
    cols = math.ceil(n_images / math.sqrt(n_images))  # Images per row
    rows = math.ceil(n_images / cols)  # Images per column

    # Create blank canvas (white background)
    grid_width = cols * resolution
    grid_height = rows * resolution
    grid_image = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255

    # Place images in grid
    for i, img in enumerate(np_images):
        row = i // cols
        col = i % cols
        y_start = row * resolution
        x_start = col * resolution
        grid_image[y_start:y_start+resolution, x_start:x_start+resolution] = img
    # Save combined image
    try:
        im = Image.fromarray(grid_image)
        save_path = f"./test_outputs/{model_name}/{file_name.name.split('/')[-1]}"
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        im.save(save_path)
        print(f"Saved to {save_path}")
        return "output.png"
    except Exception as e:
        print(f"Error saving image: {e}")
    return None
        

block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## Control Stable Diffusion with Instruct Pix2Pix")
        gr.Markdown(f"### Checkpoint used: {model_name}")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source='upload', type="numpy") 
            file_name = gr.File(label="Upload Image name")
            prompt = gr.Textbox(label="Prompt")
            run_button = gr.Button(label="Run")

            num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
            seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, value=12345)
            det = gr.Radio(choices=["None"], type="value", value="None", label="Preprocessor")
            with gr.Accordion("Advanced options", open=False):
                image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
                strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                guess_mode = gr.Checkbox(label='Guess Mode', value=False)
                ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
                scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
                eta = gr.Slider(label="DDIM ETA", minimum=0.0, maximum=1.0, value=1.0, step=0.01)
                a_prompt = gr.Textbox(label="Added Prompt", value='best quality')
                n_prompt = gr.Textbox(label="Negative Prompt", value='lowres, bad anatomy, bad hands, cropped, worst quality')
        with gr.Column():
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
            save_button = gr.Button(label="SaveImages")
            
    ips = [input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta]
    run_button.click(fn=process, inputs=ips, outputs=[result_gallery])
    save_button.click(fn=save_files, inputs=[result_gallery, file_name], outputs=[])


block.launch(server_name='0.0.0.0')
