import cv2
import mediapipe as mp
import numpy as np
import os
from tqdm import tqdm

FRAME_DIR = "dataset/WLASL/frames"
KEYPOINT_DIR = "dataset/WLASL/keypoints"

mp_hands = mp.solutions.hands.Hands(
    static_image_mode=True,
    max_num_hands=1
)

os.makedirs(KEYPOINT_DIR, exist_ok=True)

for label in os.listdir(FRAME_DIR):

    label_path = os.path.join(FRAME_DIR, label)

    for video in os.listdir(label_path):

        video_path = os.path.join(label_path, video)

        save_path = os.path.join(KEYPOINT_DIR, label)

        os.makedirs(save_path, exist_ok=True)

        keypoints = []

        for img in sorted(os.listdir(video_path)):

            img_path = os.path.join(video_path, img)

            frame = cv2.imread(img_path)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = mp_hands.process(frame)

            if results.multi_hand_landmarks:

                hand = results.multi_hand_landmarks[0]

                points = []

                for lm in hand.landmark:
                    points.append([lm.x, lm.y, lm.z])

                keypoints.append(points)

            else:
                keypoints.append(np.zeros((21,3)))

        keypoints = np.array(keypoints)

        np.save(
            os.path.join(save_path, video + ".npy"),
            keypoints
        )

print("Keypoint extraction complete.")