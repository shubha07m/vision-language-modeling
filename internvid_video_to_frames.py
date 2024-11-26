import os
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torchvision import models

# Load the MobileNetV3 model and preprocessing pipeline
model = models.mobilenet_v3_small(weights='IMAGENET1K_V1').eval()
preprocess = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def extract_embedding(frame, model, preprocess):
    """
    Extracts the feature embedding of a frame using MobileNetV3.

    Args:
        frame (ndarray): The input video frame.
        model (torch.nn.Module): Pre-trained MobileNetV3 model.
        preprocess (torchvision.transforms.Compose): Preprocessing steps for the model.

    Returns:
        ndarray: The embedding vector for the frame.
    """
    input_tensor = preprocess(frame).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        embedding = model(input_tensor)
    return embedding.squeeze().numpy()

def calculate_entropy(frame):
    """
    Calculates the entropy of an image based on pixel intensity distribution.
    """
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([grayscale], [0], None, [256], [0, 256])
    hist = hist / hist.sum()  # Normalize histogram
    entropy = -np.sum(hist * np.log2(hist + 1e-7))  # Avoid log(0)
    return entropy

def filter_low_information_frames(frames, embeddings, variance_threshold=0, entropy_threshold=5.0,
                                     embedding_magnitude_threshold=0):
    """
    Filters out frames with low information content based on pixel variance, entropy, and embedding magnitude.

    Args:
        frames (list): List of (index, frame) tuples.
        embeddings (list): Corresponding embeddings for the frames.
        variance_threshold (float): Minimum variance threshold to consider a frame as informative.
        entropy_threshold (float): Minimum entropy threshold for image content richness.
        embedding_magnitude_threshold (float): Minimum magnitude threshold for embedding vector.

    Returns:
        tuple: Filtered frames and corresponding embeddings.
    """
    filtered_frames = []
    filtered_embeddings = []

    for (index, frame), embedding in zip(frames, embeddings):
        # Filter based on pixel variance
        if np.var(frame) > variance_threshold:
            # Filter based on entropy
            entropy = calculate_entropy(frame)
            if entropy > entropy_threshold:
                # Filter based on embedding magnitude
                if np.linalg.norm(embedding) > embedding_magnitude_threshold:
                    filtered_frames.append((index, frame))
                    filtered_embeddings.append(embedding)

    return filtered_frames, filtered_embeddings


def select_most_dissimilar(embeddings, num_final_frames, distance_metric='cosine'):
    """
    Selects the most dissimilar frames based on their distance from the mean embedding.
    Args:
        embeddings (list): List of frame embeddings.
        num_final_frames (int): Number of most dissimilar frames to select.
        distance_metric (str): Metric for calculating dissimilarity ('euclidean' or 'cosine').
    Returns:
        list: Indices of the selected frames.
    """
    # Convert embeddings to a 2D numpy array
    embeddings = np.array(embeddings)

    if embeddings.ndim == 1:
        # If only one embedding, expand dimensions to match the expected shape
        embeddings = embeddings.reshape(1, -1)

    mean_embedding = np.mean(embeddings, axis=0)

    if distance_metric == 'euclidean':
        # Euclidean distance from mean
        distances = np.linalg.norm(embeddings - mean_embedding, axis=1)
    elif distance_metric == 'cosine':
        # Cosine distance from mean
        distances = 1 - np.dot(embeddings, mean_embedding) / (
                np.linalg.norm(embeddings, axis=1) * np.linalg.norm(mean_embedding)
        )

    # Get indices of top `num_final_frames` most dissimilar embeddings
    top_indices = np.argsort(-distances)[:num_final_frames]
    return top_indices


def sample_frames_from_videos(video_folder, output_folder, num_videos, sampling_interval):
    """
    Samples frames from videos at fixed intervals and saves them.

    Args:
        video_folder (str): Path to the folder containing videos.
        output_folder (str): Path to save extracted frames.
        num_videos (int): Number of videos to process.
        sampling_interval (int): Interval for frame sampling.
    """
    os.makedirs(output_folder, exist_ok=True)
    videos = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi', '.mkv'))]
    sampled_videos = videos[:min(num_videos, len(videos))]

    for video_file in sampled_videos:
        video_path = os.path.join(video_folder, video_file)
        video_name = os.path.splitext(video_file)[0]
        frames_folder = os.path.join(output_folder, f"frames_{video_name}")
        os.makedirs(frames_folder, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file: {video_file}")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = list(range(0, total_frames, sampling_interval))
        frame_count = 0
        saved_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count in frame_indices:
                frame_file = os.path.join(frames_folder, f"frame_{frame_count}.jpg")
                cv2.imwrite(frame_file, frame)
                saved_count += 1

            frame_count += 1

        cap.release()
        print(f"Processed {video_file}: {saved_count} frames saved.")

    print("Fixed interval sampling complete!")


def dissimilar_frames_from_videos(
        video_folder, output_folder, num_videos, sampling_interval, num_final_frames, model, preprocess,
        distance_metric='cosine'
):
    os.makedirs(output_folder, exist_ok=True)
    videos = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi', '.mkv'))]
    sampled_videos = videos[:min(num_videos, len(videos))]

    for video_file in sampled_videos:
        video_path = os.path.join(video_folder, video_file)
        video_name = os.path.splitext(video_file)[0]
        frames_folder = os.path.join(output_folder, f"dissimilar_frames_{video_name}")

        # Skip the video if there are no frames to save, and don't create a folder
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file: {video_file}")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = list(range(0, total_frames, sampling_interval))
        initial_frames = []
        embeddings = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count in frame_indices:
                embedding = extract_embedding(frame, model, preprocess)
                initial_frames.append((frame_count, frame))
                embeddings.append(embedding)

            frame_count += 1

        cap.release()

        # Apply advanced filtering
        initial_frames, embeddings = filter_low_information_frames(initial_frames, embeddings)

        # Check if there are any frames left after filtering
        num_available_frames = len(initial_frames)

        if num_available_frames == 0:
            print(f"Skipping video {video_file}: No frames left after filtering.")
            continue  # Skip this video entirely if no frames left after filtering

        # Create the frames folder only if there are frames to save
        os.makedirs(frames_folder, exist_ok=True)

        if num_available_frames < num_final_frames:
            print(f"Warning: Only {num_available_frames} frames available after filtering for {video_file}. Saving all frames.")
            # Save all frames if there are fewer than the number of desired final frames
            selected_frames = initial_frames
        else:
            # Select the most dissimilar frames
            selected_indices = select_most_dissimilar(embeddings, num_final_frames, distance_metric)

            # Ensure indices are within bounds (if the number of frames was reduced)
            selected_indices = [i for i in selected_indices if i < num_available_frames]

            selected_frames = [initial_frames[i] for i in selected_indices]

        # Save the selected frames
        for frame_index, frame in selected_frames:
            frame_file = os.path.join(frames_folder, f"frame_{frame_index}.jpg")
            cv2.imwrite(frame_file, frame)

        print(f"Processed {video_file}: {len(selected_frames)} dissimilar frames saved.")

    print("Dissimilar sampling complete!")



# Configuration
video_folder = "downloaded_videos"
output_folder = "frame_folders"
num_videos = 25  # Number of videos to process
sampling_interval = 100  # Frame sampling interval
num_final_frames = 5  # Number of most dissimilar frames to save

# Uncomment to use the desired function:
# sample_frames_from_videos(video_folder, output_folder, num_videos, sampling_interval)
dissimilar_frames_from_videos(video_folder, output_folder, num_videos, sampling_interval, num_final_frames, model, preprocess)
