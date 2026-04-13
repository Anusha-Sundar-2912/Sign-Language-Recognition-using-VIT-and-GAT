import cv2
import torch
import numpy as np
import time

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


def draw_probabilities(frame, probs):

    top3 = torch.topk(probs, 3)

    for i,(conf,idx) in enumerate(zip(top3.values[0], top3.indices[0])):

        label = labels[idx]
        confidence = conf.item()

        bar_width = int(confidence * 250)
        y = 120 + i*40

        cv2.putText(
            frame,
            f"{label} {confidence:.2f}",
            (40,y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255,255,255),
            2
        )

        cv2.rectangle(
            frame,
            (200,y-20),
            (200+bar_width,y-5),
            (0,255,0),
            -1
        )


def center_crop(frame):

    h, w = frame.shape[:2]

    x1 = w//4
    y1 = h//4
    x2 = 3*w//4
    y2 = 3*h//4

    return frame[y1:y2, x1:x2]


def preprocess_frames(frames_buffer):

    frames = np.array(frames_buffer)
    frames = torch.tensor(frames).permute(0,3,1,2).float()

    frames = frames / 255.0

    mean = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)
    std  = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)

    frames = (frames - mean) / std

    frames = frames.unsqueeze(0)

    return frames


def predict_live():

    cap = cv2.VideoCapture(0)

    # camera settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
    cap.set(cv2.CAP_PROP_FPS,30)

    frames_buffer = []

    prediction = "Detecting"
    confidence = 0

    frame_counter = 0

    fps_time = time.time()

    cv2.namedWindow("Sign Language Recognition", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Sign Language Recognition",1000,750)

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        display_frame = frame.copy()

        # center crop (helps match WLASL framing)
        frame = center_crop(frame)

        model_frame = cv2.resize(frame,(224,224))
        rgb = cv2.cvtColor(model_frame, cv2.COLOR_BGR2RGB)

        frames_buffer.append(rgb)

        if len(frames_buffer) > 8:
            frames_buffer.pop(0)

        frame_counter += 1

        if len(frames_buffer) == 8 and frame_counter % 2 == 0:

            frames = preprocess_frames(frames_buffer).to(device)

            keypoints = torch.zeros((1,8,33,3)).to(device)

            with torch.no_grad():

                outputs = model(frames,keypoints)

                probs = torch.softmax(outputs,dim=1)

                conf,pred = torch.max(probs,1)

                confidence = conf.item()
                pred = pred.item()

                prediction = labels[pred]

        cv2.putText(
            display_frame,
            f"Prediction: {prediction.upper()} ({confidence:.2f})",
            (30,60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0,255,0),
            3
        )

        if 'probs' in locals():
            draw_probabilities(display_frame,probs)

        new_time = time.time()
        fps = 1/(new_time - fps_time)
        fps_time = new_time

        cv2.putText(
            display_frame,
            f"FPS: {int(fps)}",
            (860,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255,255,255),
            2
        )

        cv2.imshow("Sign Language Recognition",display_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    return prediction