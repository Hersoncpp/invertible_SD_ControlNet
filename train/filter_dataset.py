import json
import os
from tqdm import tqdm
import random
prompts_json_path = "/home/hesong/disk1/DF_INV/code/ControlNet-v1-1-nightly/dataset/OmniEdit-Filtered-1.2M_train/prompts.json"
filtered_prompts_json_path = "/home/hesong/disk1/DF_INV/code/ControlNet-v1-1-nightly/dataset/OmniEdit-Filtered-1.2M_train_filtered/prompts.json"
limit_num = 1500
anime_limit = 200
lines = []
filter_list = []
with open(prompts_json_path, "rt") as f:
    for i, line in tqdm(enumerate(f)):
        data = json.loads(line)
        # print(data)
        # print(i)
        lines.append(data)
        if 'source' not in data or 'target' not in data or 'prompt' not in data:
            print(f"Error in line {i}: {line}")
            continue
        if os.path.exists(data['source']) is False:
            print(f"Source image not found: {data['source']}")
            continue
        if os.path.exists(data['target']) is False:
            print(f"Target image not found: {data['target']}")
            continue
        if data['prompt'] is None or len(data['prompt'].strip()) == 0:
            print(f"Prompt is empty in line {i}: {line}")
            continue
        if data['prompt'].lower().find('anime') >= 0 or data['prompt'].lower().find('cute') >= 0:
            filter_list.append(i)

if len(filter_list) > anime_limit:
    # randomly choose anime_limit from filter_list
    filter_list = random.sample(filter_list, anime_limit)

chioce_num = limit_num - len(filter_list)
print(f"Filter out {len(filter_list)} prompts, need to choice {chioce_num} more prompts")

import random
choice_list = []
while len(choice_list) < chioce_num:
    idx = random.randint(0, len(lines)-1)
    if idx in filter_list or idx in choice_list:
        continue
    choice_list.append(idx)

final_list = filter_list + choice_list
final_list = sorted(final_list)

# save the final_list to json file
os.makedirs(os.path.dirname(filtered_prompts_json_path), exist_ok=True)
with open(filtered_prompts_json_path, "wt") as f:
    for idx in final_list:
        line = lines[idx]
        src_img_fpth = line['source']
        edited_img_fpth = line['target']
        edited_prompt = line['prompt']
        prompt_txt = "{\"source\": \"" + str(src_img_fpth) + "\", \"target\": \"" + str(edited_img_fpth) + "\", \"prompt\": \"" + edited_prompt.replace("'", "\'").replace('"', '\"') + "\"}\n"
        f.write(prompt_txt)

print(f"Saved filtered prompts to {filtered_prompts_json_path}, total {len(final_list)} prompts")

