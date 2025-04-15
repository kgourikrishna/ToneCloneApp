import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime

import subprocess
import sys

subprocess.call([sys.executable, '-m', 'pip', 'install', 'panns-inference'])

from panns_inference.models import Cnn14
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


class spectrogramPANN(nn.Module):
    def __init__(self, pretrained_model, num_classes, freeze_backbone=False):
        super().__init__()

        self.cnn14 = pretrained_model

        # Remove audio extractors since we have already converted to spectrograms
        del self.cnn14.spectrogram_extractor
        del self.cnn14.logmel_extractor

        self.fc2 = nn.Linear(2048, num_classes)

        if freeze_backbone:
            for name, param in self.cnn14.named_parameters():
                if 'conv' in name:
                    param.requires_grad = False

    def forward(self, x):
        x = self.cnn14.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.cnn14.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.cnn14.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.cnn14.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.cnn14.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.cnn14.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2

        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.cnn14.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)

        return x


def model_fn(model_dir):
    print("MYDEBUG: Inside Model_fn")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_cnn14 = Cnn14(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000, classes_num=527)
    model = spectrogramPANN(pretrained_cnn14, NUM_CLASSES).to(device)
    print("MYDEBUG: model variable")
    with open(os.path.join(model_dir, 'last_PANN_model.mod'), "rb") as f:
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