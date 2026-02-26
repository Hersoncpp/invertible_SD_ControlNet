import os
import shutil
import json

# ----------------------------- CONFIGURATION -----------------------------
# Replace this with the actual path to your JSON/JSONL file
json_file_path = "/home/hesong/disk1/DF_INV/code/ControlNet-v1-1-nightly/inv_modules/IVOP/codes/data/dataset/ControlNet_ST/prompts.json"  # <-- CHANGE THIS TO YOUR FILE NAME/PATH, e.g., "OmniEdit-Filtered-1.2M_train.jsonl"

# The extracted folder will be created in the current working directory
extracted_dir = "/home/yukai/disk1/invertible_SD_ControlNet/dataset/val_imgs"
source_dir = os.path.join(extracted_dir, "source")
target_dir = os.path.join(extracted_dir, "target")
# -------------------------------------------------------------------------

# Create the directories (exist_ok=True avoids errors if they already exist)
os.makedirs(source_dir, exist_ok=True)
os.makedirs(target_dir, exist_ok=True)

count = 0

with open(json_file_path, "r", encoding="utf-8") as f:
    for line_num, line in enumerate(f, start=1):
        line = line.strip()
        if not line:
            continue  # skip empty lines
        
        try:
            data = json.loads(line)
            
            # Extract source and target paths
            source_path = data.get("source")
            target_path = data.get("target")
            
            if not source_path or not target_path:
                print(f"Warning: Line {line_num} missing 'source' or 'target' key. Skipping.")
                continue
            
            if "73015" in os.path.basename(source_path) or "73015" in os.path.basename(target_path):
                print(f"Skipping pair on line {line_num} because filename contains '73015'.")
                continue
            
            # Get filenames (preserving original names)
            source_filename = os.path.basename(source_path)
            target_filename = os.path.basename(target_path)
            
            # Copy source image
            dest_source = os.path.join(source_dir, source_filename)
            shutil.copy2(source_path, dest_source)  # copy2 preserves metadata
            
            # Copy target image
            dest_target = os.path.join(target_dir, target_filename)
            shutil.copy2(target_path, dest_target)
            
            count += 1
            
            # Optional: progress update every 1000 images
            if count % 1000 == 0:
                print(f"Copied {count} image pairs so far...")
                
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON on line {line_num}: {e}")
            continue
        except FileNotFoundError as e:
            print(f"Error: File not found on line {line_num}: {e}")
            continue
        except Exception as e:
            print(f"Error: Unexpected issue on line {line_num}: {e}")
            continue

print("\n=== Summary ===")
print(f"Successfully processed and copied {count} image pairs.")
print(f"Source images → {source_dir}")
print(f"Target images → {target_dir}")
print(f"Total number of images (entries) in the JSON file: {count}")