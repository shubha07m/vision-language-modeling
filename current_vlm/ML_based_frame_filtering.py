import os
import cv2
import numpy as np
import torch
from torchvision import models
import torchvision.transforms as T
from sklearn.metrics.pairwise import cosine_similarity


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


## Embedding Extraction

def extract_embedding(frame, model, preprocess):
    input_tensor = preprocess(frame).unsqueeze(0)
    with torch.no_grad():
        embedding = model(input_tensor)
    return embedding.squeeze().numpy()


## Frame Similarity Calculation

def calculate_similarity(embeddings):
    # Calculate cosine similarity between each pair of embeddings
    return cosine_similarity(embeddings)


## Frame Pruning (Removing Similar Frames)

def prune_similar_frames(frames_with_ids, embeddings, similarity_threshold=0.8):
    selected_frames_with_ids = []
    selected_embeddings = []

    for (index, frame), embedding in zip(frames_with_ids, embeddings):
        # Compute cosine similarity between the current frame and all selected frames
        if len(selected_embeddings) == 0:
            selected_frames_with_ids.append((index, frame))
            selected_embeddings.append(embedding)
        else:
            similarities = cosine_similarity([embedding], selected_embeddings)[0]
            if all(sim < similarity_threshold for sim in similarities):
                selected_frames_with_ids.append((index, frame))
                selected_embeddings.append(embedding)

    return selected_frames_with_ids


## Process Filtered Frames for Pruning

def process_filtered_frames_prune(filtered_folder, model, preprocess, similarity_threshold=0.8):
    frame_files = sorted([f for f in os.listdir(filtered_folder) if f.endswith(('.jpg', '.png'))])
    if not frame_files:
        print(f"No frames found in {filtered_folder}")
        return []

    # Load frames and extract embeddings
    frames_with_ids = [(int(f.split('_')[1].split('.')[0]), cv2.imread(os.path.join(filtered_folder, f)))
                       for f in frame_files]
    embeddings = [extract_embedding(frame, model, preprocess) for _, frame in frames_with_ids]

    # Prune similar frames
    selected_frames_with_ids = prune_similar_frames(frames_with_ids, embeddings, similarity_threshold)

    return selected_frames_with_ids


## Save Selected Frames

def save_selected_frames(selected_frames_with_ids, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for index, frame in selected_frames_with_ids:
        frame_path = os.path.join(output_folder, f"frame_{index}.jpg")
        cv2.imwrite(frame_path, frame)


## Main Function

def frame_pruning_from_videos(filtered_frame_folder, selected_frame_folder, similarity_threshold=0.8):
    model, preprocess = load_model_and_preprocess()

    filtered_videos = [f for f in os.listdir(filtered_frame_folder) if
                       os.path.isdir(os.path.join(filtered_frame_folder, f))]

    for video_folder in filtered_videos:
        input_folder = os.path.join(filtered_frame_folder, video_folder)
        output_folder = os.path.join(selected_frame_folder,
                                     video_folder.replace("filtered_frames_", "selected_frames_"))

        print(f"Processing {video_folder}...")

        selected_frames_with_ids = process_filtered_frames_prune(input_folder, model, preprocess, similarity_threshold)

        if selected_frames_with_ids:
            save_selected_frames(selected_frames_with_ids, output_folder)
            print(f"Selected frames saved for {video_folder} in {output_folder}.")
        else:
            print(f"No frames selected for {video_folder}.")

    print("Frame pruning complete!")


## Configuration and Execution

if __name__ == "__main__":
    config = {
        "filtered_frame_folder": "primary_filtered_frames",  # Folder containing filtered frames for each video
        "selected_frame_folder": "selected_frame_folders",  # Folder to save the selected frames
        "similarity_threshold": 0.8,  # Cosine similarity threshold to discard similar frames
    }
    frame_pruning_from_videos(**config)
