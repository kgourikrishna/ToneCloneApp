import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
from io import BytesIO

SAMPLE_LENGTH = 10  # In seconds
OVERLAP = 0  # Percentage overlap
N_FFT = 2048
HOP_LENGTH = 512
NUM_MELS = 128
NUM_CLASSES = 13
TARGET_SAMPLE_RATE = 32000

EFFECT_LABELS = {
    "ODV": "overdrive", "DST": "distortion", "FUZ": "fuzz", "TRM": "tremolo",
    "PHZ": "phaser", "FLG": "flanger", "CHR": "chorus", "DLY": "delay", "HLL": "hall_reverb",
    "PLT": "plate_reverb", "OCT": "octaver", "FLT": "auto_filter"
}
LABEL_NAMES = list(EFFECT_LABELS.values())  # Ordered list of effect names
NUM_CLASSES = len(LABEL_NAMES)


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

        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layers
        self.fc1 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)  # Max pooling

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn5(self.conv5(x)))
        x = F.max_pool2d(x, 2)

        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Dropout
        x = self.fc2(x)
        return x


def model_fn(model_dir):
    print("MYDEBUG: Inside Model_fn")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = spectrogramCNN(NUM_CLASSES).to(device)
    print("MYDEBUG: model variable")
    with open(os.path.join(model_dir, 'final_multi_effects_alt7.mod'), "rb") as f:
        model.load_state_dict(torch.load(f, map_location=device))
    print("MYDEBUG: Model Loaded")
    return model.to(device)


def input_fn(request_body, request_content_type):
    """An input_fn that loads numpy object of spectrograms"""
    print("MYDEBUG: Input function called")
    if request_content_type == 'application/x-npy':
        print("MYDEBUG: Made it to if statement")
        spectrograms = np.load(BytesIO(request_body))
        print("MYDEBUG: Data loaded")
    else:
        print("MYDEBUG: Did not make if statement")
        raise ValueError(f"Unsupported content type: {request_content_type}")

    print("MYDEBUG: Return tensor")
    spectrograms = torch.tensor(spectrograms, dtype=torch.float32)
    return spectrograms


def predict_fn(input_data, model):
    print("MYDEBUG: Predict function")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("MYDEBUG: Set Device")
    model.to(device)
    model.eval()
    print("MYDEBUG: Model Evaluation ")
    with torch.no_grad():
        print("MYDEBUG: Inside torch no grad ")
        try:
            outputs = model(input_data)
        except Exception as e:
            print(e)
        print("MYDEBUG: Got Outputs ")
        try:
            predictions = torch.sigmoid(outputs)
        except Exception as e:
            print(e)
        print("MYDEBUG: Got Predictions and return")
        return [predictions, outputs]


def output_fn(prediction, content_type):
    output = {}
    if content_type == 'application/json':
        binary_preds = (prediction[0] > 0.5).cpu().numpy()
        logits = prediction[1].cpu().numpy()

        step_size = int(TARGET_SAMPLE_RATE * SAMPLE_LENGTH * (1 - OVERLAP / 100))

        for i, (segment_logits, segment_pred) in enumerate(zip(logits, binary_preds)):

            sigmoid_probs = torch.sigmoid(torch.tensor(segment_logits)).numpy().astype(float)
            sigmoid_probs_dict = dict(zip(LABEL_NAMES, sigmoid_probs))
            print("MYDEBUG: Probs")

            # Calculate start and end time of the segment
            start_time = (i * step_size) / TARGET_SAMPLE_RATE
            end_time = start_time + (SAMPLE_LENGTH)

            timestamp_range = f"{format_time(start_time)} - {format_time(end_time)}"

            print(f"\nSegment {i+1} ({timestamp_range}):")

            detected_effects = [LABEL_NAMES[j] for j in range(NUM_CLASSES) if segment_pred[j] == 1]

            dict_index = "Segment " + str(i+1) + " (" + str(timestamp_range) + ")"
            try:
                output[dict_index] = [detected_effects, sigmoid_probs_dict]
            except Exception as e:
                print(e)
        print("MYDEBUG: Output")
        print(output)
        try:
            return json.dumps(output)
        except Exception as e:
            print(e)
    else:
       raise ValueError(f"Unsupported content type: {content_type}") 


# format seconds to MM:SS
def format_time(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))[2:]  # Removes hours if <1hr