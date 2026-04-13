import json
import os
import shutil

DATASET_JSON = "dataset/WLASL_v0.3.json"
VIDEO_DIR = "dataset/WLASL/videos"
OUTPUT_DIR = "dataset/WLASL/wlasl100"

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(DATASET_JSON, "r") as f:
    data = json.load(f)

# sort by number of video instances
data_sorted = sorted(data, key=lambda x: len(x["instances"]), reverse=True)

top100 = data_sorted[:100]

print("Selected words:")
for item in top100:
    print(item["gloss"])

for item in top100:

    label = item["gloss"]
    label_dir = os.path.join(OUTPUT_DIR, label)

    os.makedirs(label_dir, exist_ok=True)

    for inst in item["instances"]:

        video_id = inst["video_id"] + ".mp4"

        src = os.path.join(VIDEO_DIR, video_id)

        if os.path.exists(src):
            shutil.copy(src, label_dir)