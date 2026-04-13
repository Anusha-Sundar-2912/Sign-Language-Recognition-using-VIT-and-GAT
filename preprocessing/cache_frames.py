import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

FRAME_DIR = "dataset/WLASL/frames"
CACHE_DIR = "dataset/WLASL/frame_cache"

os.makedirs(CACHE_DIR, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])


labels = sorted(os.listdir(FRAME_DIR))

for label in labels:

    label_path = os.path.join(FRAME_DIR, label)
    cache_label = os.path.join(CACHE_DIR, label)

    os.makedirs(cache_label, exist_ok=True)

    videos = os.listdir(label_path)

    for vid in tqdm(videos, desc=label):

        frame_path = os.path.join(label_path, vid)

        images = sorted(os.listdir(frame_path))

        # ensure 16 frames
        if len(images) >= 16:
            images = images[:16]
        else:
            while len(images) < 16:
                images.append(images[-1])

        frames = []

        for img in images:

            img_path = os.path.join(frame_path, img)

            image = Image.open(img_path).convert("RGB")

            image = transform(image)

            frames.append(image)

        frames = torch.stack(frames)

        save_path = os.path.join(cache_label, vid + ".pt")

        torch.save(frames, save_path)

print("Frame caching complete.")