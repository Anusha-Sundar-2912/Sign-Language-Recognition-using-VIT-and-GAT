import cv2
import os
from pathlib import Path

DATASET_DIR = "dataset/WLASL/wlasl100"
FRAME_DIR = "dataset/WLASL/frames"

NUM_FRAMES = 16

for label in os.listdir(DATASET_DIR):

    label_path = os.path.join(DATASET_DIR, label)

    for video in os.listdir(label_path):

        video_path = os.path.join(label_path, video)

        save_dir = os.path.join(FRAME_DIR, label, video[:-4])

        os.makedirs(save_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        step = max(total // NUM_FRAMES, 1)

        frame_id = 0
        count = 0

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            if count % step == 0 and frame_id < NUM_FRAMES:

                frame_path = os.path.join(save_dir, f"{frame_id}.jpg")

                cv2.imwrite(frame_path, frame)

                frame_id += 1

            count += 1

        cap.release()

print("Frame extraction complete.")