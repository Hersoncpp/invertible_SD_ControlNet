import pandas as pd

def read_image_from_bytes(img_data):
    from PIL import Image
    import io
    """
    从包含字节数据的字典中读取图像
    
    Args:
        img_data: {'bytes': b'\x89PNG\r\n\x1a\n\x00\x00...'} 或 {'path': 'image.png'}
    
    Returns:
        PIL Image 对象
    """
    if 'bytes' in img_data and img_data['bytes'] is not None:
        # 从字节数据读取
        image_bytes = img_data['bytes']
        image = Image.open(io.BytesIO(image_bytes))
        return image
    elif 'path' in img_data:
        # 从文件路径读取
        image = Image.open(img_data['path'])
        return image
    else:
        raise ValueError("图像数据必须包含 'bytes' 或 'path' 字段")


def build_dataset_from_parquet_file(parquet_file, src_img_dir, tar_img_dir, prompts_json_path):
    """
    读取 Parquet 文件并构建数据集
    
    Args:
        parquet_file: Parquet 文件的路径
        data_dir: 数据集保存的目录
    """
    # 保证data_dir存在
    from tqdm import tqdm
    
    # read the parquet file
    df = pd.read_parquet(parquet_file)
    print(df.shape)
    # show the first rows
    # print(df.head(1))
    # # show the columns
    # print(df.columns)
    # # show the data types of the columns
    # print(df.dtypes)

    # read out the src_img and edited_img and edited_prompt_list with task as 'style'
    df_style = df[df['task'] == 'style'][['omni_edit_id', 'src_img', 'edited_img', 'edited_prompt_list']]

    # print(df_style.head(1))
    # print(df_style.iloc[0])
    # print(df_style.iloc[0].src_img.keys())

    # read src_img and edited_img using PIL
    # print(type(df_style.iloc[0].src_img))
    dataset  = df_style.iloc[:]
    
    for index, row in tqdm(dataset.iterrows(), total=len(dataset), desc="Processing dataset"):
        edit_id = row['omni_edit_id']
        src_img = read_image_from_bytes(row['src_img'])
        edited_img = read_image_from_bytes(row['edited_img'])

        src_img_fpth = src_img_dir / f"{edit_id}.jpg"
        src_img.save(src_img_fpth)
        # convert path to absolute path
        src_img_fpth = src_img_fpth.resolve()
        edited_img_fpth = tar_img_dir / f"{edit_id}.jpg"
        edited_img.save(edited_img_fpth)
        # convert path to absolute path
        edited_img_fpth = edited_img_fpth.resolve()

        edited_prompt = row['edited_prompt_list'][0]
        
        # prompt_dict = {
        #     'source': src_img_fpth,
        #     'target': edited_img_fpth,
        #     'prompt': edited_prompt
        # }
        # print(f"Omni Edit ID: {row['omni_edit_id']}")
        # print(f"Source Image Size: {src_img.size}, Edited Image Size: {edited_img.size}")
        # print(f"Edited Prompts: {edited_prompt}")

        prompt_txt = "{\"source\": \"" + str(src_img_fpth) + "\", \"target\": \"" + str(edited_img_fpth) + "\", \"prompt\": \"" + edited_prompt.replace("'", "\'") + "\"}\n"
        with open(prompts_json_path, "a") as f:
            f.write(prompt_txt)


    # src_img = Image.open(df_style.iloc[0].src_img.get('bytes'))
    # # save the image to path
    # path = "/home/hesong/disk1/DF_INV/raw_data/OmniEdit-Filtered-1.2M/test.png"
    # src_img.save(path)


def clear_directory(dir_path):
    import shutil
    from pathlib import Path
    dir_path = Path(dir_path)
    if dir_path.exists() and dir_path.is_dir():
        shutil.rmtree(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    from pathlib import Path

    parquet_files = ['/home/hesong/disk1/DF_INV/raw_data/OmniEdit-Filtered-1.2M/data/dev-00000-of-00001.parquet']
    data_dir = './dataset/OmniEdit-Filtered-1.2M'
    data_dir = Path(data_dir)
    if data_dir.exists() is False:
        data_dir.mkdir(parents=True, exist_ok=True)
    src_img_dir = data_dir / "source_images"
    if src_img_dir.exists() is False:
        src_img_dir.mkdir(parents=True, exist_ok=True)
    else: # clear the directory
        clear_directory(src_img_dir)
    tar_img_dir = data_dir / "target_images"
    if tar_img_dir.exists() is False:
        tar_img_dir.mkdir(parents=True, exist_ok=True)
    else: # clear the directory
        clear_directory(tar_img_dir)

    prompts_json_path = data_dir / "prompts.json"
    with open(prompts_json_path, "wt") as f:
        f.write("") # clear the file if exists
    for parquet_file in parquet_files:
        build_dataset_from_parquet_file(parquet_file=parquet_file, src_img_dir=src_img_dir, tar_img_dir=tar_img_dir, prompts_json_path=prompts_json_path)
