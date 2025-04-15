import numpy as np
import h5py
import pandas as pd
import argparse
import json
import logging
import os
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.utils.data.distributed
from torchvision import datasets, transforms

class SpectrogramDataset(Dataset):
    """
    Custom dataset for spectrogram data.

    When the Spectrogram dataset is used to create a dataloader object, the
    dataloader consists of batches of spectrograms and their corresponding labels.
    Here is info on the shape of the spectrogram and label objects in each batch:

    Spectrogram Tensor Dimensions in Batch - (32, 1, 128, 626)
        Batch Size: 32
        Channels: 1 - Think of it as a grayscale image, rather than RGB
        Mel Bands (Height): 128 - 128 Mel filter banks (typical for Mel spectrograms)
        Time Steps (Width): 626 - Number of frames

    Label Tensor Dimensions in Batch - (32, 12)
        Batch Size: 32
        Number of Labels: 12 - Multi-hot encoded vector of the 12 effects. This
            would increase if we added additional effects.
    """

    def __init__(self, hdf5_file, csv_file):
        self.hdf5_file_path = hdf5_file
        self.labels = pd.read_csv(csv_file)
        self.label_map = self.labels.columns[1:].tolist() # Get effect label names
        self.hdf5_file = None   # File will be opened for each worker

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Open HDF5 file once per worker
        if self.hdf5_file is None:
            self.hdf5_file = h5py.File(self.hdf5_file_path, "r", swmr=True) # SWMR ensures multi-thread safe

        key = self.labels.iloc[idx]['key']
        spectrogram = torch.tensor(self.hdf5_file[key][()], dtype=torch.float32).unsqueeze(0)
        label_values = self.labels.iloc[idx][1:].infer_objects().fillna(0).astype(float).values  # Convert all label columns to float
        label = torch.tensor(label_values, dtype=torch.float32)  # Convert to tensor


        return spectrogram, label

    def __del__(self):
        if self.hdf5_file is not None:
            self.hdf5_file.close()

class spectrogramCNN(nn.Module):
    def __init__(self, num_classes):
        super(spectrogramCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten_size = (512 * (128 // 32) * (626 // 32))

        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))
        x = self.pool(torch.relu(self.bn5(self.conv5(x))))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x) # Sigmoid is redundant if using BCEWithLogitcsLoss
        return x

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    training_dir = args.training
    validation_dir = args.validation

    h5_train_path = os.path.join(training_dir,'final_train.h5')
    csv_train_path = os.path.join(training_dir,'final_train.csv')

    h5_val_path = os.path.join(validation_dir,'final_validate.h5')
    csv_val_path = os.path.join(validation_dir, 'final_validate.csv')

    train_dataset = SpectrogramDataset(h5_train_path, csv_train_path)
    val_dataset = SpectrogramDataset(h5_val_path, csv_val_path)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=12, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=6, pin_memory=True)

    num_classes = len(train_dataset.label_map)

    model = spectrogramCNN().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.398)  # 0.0001 → 0.00001 over 5 epochs

    # Training loop
    num_epochs = args.epochs
    print_freq = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (spectrograms, labels) in enumerate(train_loader):
            spectrograms, labels = spectrograms.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(spectrograms)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (batch_idx + 1) % print_freq == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}")

        # Update learning rate
        scheduler.step()
        print(f"Updated Learning Rate: {scheduler.get_last_lr()}")

        # Validation step
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for spectrograms, labels in val_loader:
                spectrograms, labels = spectrograms.to(device), labels.to(device)
                outputs = model(spectrograms)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Compute accuracy
                predicted = (torch.sigmoid(outputs) > 0.5).float()  # Convert logits to binary predictions

                # Store for metric computation
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)

        # Convert lists to numpy arrays for metric calculations
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Compute metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
        recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        # Print classification report
        class_names = train_dataset.label_map
        print(classification_report(all_labels, all_preds, target_names=class_names))

        print(f"\nValidation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}\n")
    save_model(model, args.model_dir)

def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.DataParallel(spectrogramCNN())
    with open(os.path.join(model_dir, args.model-name), "rb") as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)

def save_model(model, model_dir):
    path = os.path.join(model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )

     parser.add_argument(
        "--model-name",
        type=str,
        default='model.pth',
    )

    # Container environment
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--training", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--validation", type=str, default=os.environ["SM_CHANNEL_VALIDATION"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])

    train(parser.parse_args())

