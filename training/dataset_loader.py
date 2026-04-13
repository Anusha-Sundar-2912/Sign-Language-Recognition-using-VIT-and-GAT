import os
import torch
import numpy as np
from torch.utils.data import Dataset


class WLASLDataset(Dataset):

    def __init__(self, frame_dir, keypoint_dir):

        self.frame_dir = frame_dir
        self.keypoint_dir = keypoint_dir

        self.labels = sorted(os.listdir(frame_dir))
        self.samples = []

        for label_idx, label in enumerate(self.labels):

            label_path = os.path.join(frame_dir, label)

            if not os.path.isdir(label_path):
                continue

            files = os.listdir(label_path)

            for f in files:

                if not f.endswith(".pt"):
                    continue

                frame_path = os.path.join(label_path, f)

                kp_file = f.replace(".pt", ".npy")
                kp_path = os.path.join(keypoint_dir, label, kp_file)

                if os.path.exists(kp_path):
                    self.samples.append((frame_path, kp_path, label_idx))

        print("Total samples:", len(self.samples))


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):

        frame_path, kp_path, label = self.samples[idx]

        # -------- LOAD CACHED FRAMES --------
        frames = torch.load(frame_path)

        num_frames = frames.shape[0]
        target_frames = 8

        # Temporal sampling across video
        if num_frames >= target_frames:

            indices = np.linspace(
                0,
                num_frames - 1,
                target_frames
            ).astype(int)

            frames = frames[indices]

        else:

            pad = frames[-1:].repeat(
                target_frames - num_frames,
                1,
                1,
                1
            )

            frames = torch.cat([frames, pad], dim=0)


        # -------- LOAD KEYPOINTS --------
        keypoints = np.load(kp_path)

        kp_frames = keypoints.shape[0]

        if kp_frames >= target_frames:

            indices = np.linspace(
                0,
                kp_frames - 1,
                target_frames
            ).astype(int)

            keypoints = keypoints[indices]

        else:

            pad = np.repeat(
                keypoints[-1:],
                target_frames - kp_frames,
                axis=0
            )

            keypoints = np.concatenate(
                [keypoints, pad],
                axis=0
            )

        keypoints = torch.tensor(keypoints).float()

        return frames, keypoints, label