import os
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torchvision import models


## Model and Preprocessing

def load_model_and_preprocess():
    model = models.mobilenet_v3_small(weights='IMAGENET1K_V1').eval()
    preprocess = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, preprocess


## Frame Processing

def extract_embedding(frame, model, preprocess):
    input_tensor = preprocess(frame).unsqueeze(0)
    with torch.no_grad():
        embedding = model(input_tensor)
    return embedding.squeeze().numpy()


def calculate_entropy(frame):
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([grayscale], [0], None, [256], [0, 256])
    hist = hist / hist.sum()
    return -np.sum(hist * np.log2(hist + 1e-7))


## Frame Filtering

def filter_frames(frames, embeddings, **kwargs):
    filtered_frames, filtered_embeddings = [], []

    for (index, frame), embedding in zip(frames, embeddings):
        if should_keep_frame(frame, embedding, **kwargs):
            filtered_frames.append((index, frame))
            filtered_embeddings.append(embedding)

    return filtered_frames, filtered_embeddings


def should_keep_frame(frame, embedding, **kwargs):
    if kwargs.get('variance_threshold') and np.var(frame) <= kwargs['variance_threshold']:
        return False
    if kwargs.get('entropy_threshold') and calculate_entropy(frame) <= kwargs['entropy_threshold']:
        return False
    if kwargs.get('embedding_magnitude_threshold') and np.linalg.norm(embedding) <= kwargs[
        'embedding_magnitude_threshold']:
        return False
    return True


## Video Processing

def process_video(video_path, model, preprocess, sampling_interval, **kwargs):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return None, None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = range(0, total_frames, sampling_interval)
    initial_frames, embeddings = [], []

    for frame_count in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        ret, frame = cap.read()
        if not ret:
            break
        embedding = extract_embedding(frame, model, preprocess)
        initial_frames.append((frame_count, frame))
        embeddings.append(embedding)

    cap.release()
    return filter_frames(initial_frames, embeddings, **kwargs)


def save_frames(frames, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for frame_index, frame in frames:
        frame_file = os.path.join(output_folder, f"frame_{frame_index}.jpg")
        cv2.imwrite(frame_file, frame)


## Main Function

def filter_frames_from_videos(video_folder, output_folder, num_videos, sampling_interval, **kwargs):
    model, preprocess = load_model_and_preprocess()
    videos = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi', '.mkv'))][:num_videos]

    for video_file in videos:
        video_path = os.path.join(video_folder, video_file)
        video_name = os.path.splitext(video_file)[0]
        frames_folder = os.path.join(output_folder, f"filtered_frames_{video_name}")

        frames, _ = process_video(video_path, model, preprocess, sampling_interval, **kwargs)
        if frames:
            save_frames(frames, frames_folder)
            print(f"Processed {video_file}: {len(frames)} frames saved.")
        else:
            print(f"Skipping video {video_file}: No frames left after filtering.")

    print("Frame filtering complete!")


## Configuration and Execution

if __name__ == "__main__":
    config = {
        "video_folder": "downloaded_videos",
        "output_folder": "primary_filtered_frames",
        "num_videos": 5,
        "sampling_interval": 100,
        "variance_threshold": None,
        "entropy_threshold": 5.0,
        "embedding_magnitude_threshold": None
    }

    filter_frames_from_videos(**config)
