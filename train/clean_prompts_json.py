import os
import json

# prompt_json_path = "/home/hesong/disk1/DF_INV/code/ControlNet-v1-1-nightly/dataset/OmniEdit-Filtered-1.2M_train/prompts.json"
# cleaned_list = []
# with open(prompt_json_path, "rt") as f:
#     for line in f:
#         data = json.loads(line)
#         data['prompt'] = data['prompt'].replace('"', '\"').replace("'", "\'")
#         cleaned_list.append(data)
# print(f"Cleaned {len(cleaned_list)} prompts")
# with open(prompt_json_path, "wt") as f:
#     for data in cleaned_list:
#         src_img_fpth = data['source']
#         edited_img_fpth = data['target']
#         edited_prompt = data['prompt']
#         prompt_txt = "{\"source\": \"" + str(src_img_fpth) + "\", \"target\": \"" + str(edited_img_fpth) + "\", \"prompt\": \"" + edited_prompt.replace("'", "\'").replace('"', '\"') + "\"}\n"

