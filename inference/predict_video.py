import cv2
import torch
import numpy as np
from models.fusion_model import SignLanguageModel
from training.dataset_loader import WLASLDataset

FRAME_DIR = "dataset/WLASL/frame_cache"
KEYPOINT_DIR = "dataset/WLASL/keypoints"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = WLASLDataset(FRAME_DIR, KEYPOINT_DIR)
labels = dataset.labels

model = SignLanguageModel(len(labels)).to(device)
model.load_state_dict(torch.load("weights/best_model.pth", map_location=device))
model.eval()


def predict_video(video_path):

    cap = cv2.VideoCapture(video_path)

    frames = []

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (224, 224))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frames.append(frame)

    cap.release()

    if len(frames) < 8:
        return "Video too short"

    indices = np.linspace(0, len(frames)-1, 8).astype(int)
    frames = [frames[i] for i in indices]

    frames = np.array(frames)
    frames = torch.tensor(frames).permute(0,3,1,2).float()/255
    frames = frames.unsqueeze(0).to(device)

    keypoints = torch.zeros((1,8,33,3)).to(device)

    with torch.no_grad():

        outputs = model(frames, keypoints)
        pred = torch.argmax(outputs,1).item()

    return labels[pred]