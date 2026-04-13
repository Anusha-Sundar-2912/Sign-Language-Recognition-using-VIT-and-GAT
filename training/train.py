import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from torch.cuda.amp import autocast, GradScaler

from training.dataset_loader import WLASLDataset
from models.fusion_model import SignLanguageModel


FRAME_DIR = "dataset/WLASL/frame_cache"
KEYPOINT_DIR = "dataset/WLASL/keypoints"

BATCH_SIZE = 4
EPOCHS = 25
LR = 1e-4


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


def main():

    dataset = WLASLDataset(FRAME_DIR, KEYPOINT_DIR)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    num_classes = len(dataset.labels)
    print("Classes:", num_classes)

    model = SignLanguageModel(num_classes).to(device)

    # Label smoothing improves generalization
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = optim.Adam(
        model.parameters(),
        lr=LR
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    # Mixed precision scaler
    scaler = GradScaler()

    best_acc = 0

    for epoch in range(EPOCHS):

        model.train()

        total = 0
        correct = 0

        loop = tqdm(loader)

        for frames, keypoints, labels in loop:

            frames = frames.to(device)
            keypoints = keypoints.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Mixed precision forward pass
            with autocast():

                outputs = model(frames, keypoints)

                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            loop.set_description(f"Epoch {epoch+1}/{EPOCHS}")
            loop.set_postfix(loss=loss.item())

        acc = correct / total

        print("Accuracy:", acc)

        scheduler.step()

        if acc > best_acc:

            best_acc = acc

            os.makedirs("weights", exist_ok=True)

            torch.save(
                model.state_dict(),
                "weights/best_model.pth"
            )

    print("Training complete")
    print("Best accuracy:", best_acc)


if __name__ == "__main__":
    main()