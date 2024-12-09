import os
import torch
import pandas as pd
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import shutil  # Added to handle copying files

# Check if MPS is available
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Load BLIP processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(DEVICE)


def generate_caption(image_path):
    """
    Generate a caption for an image using the BLIP model.

    Args:
        image_path (str): Path to the image.

    Returns:
        str: Generated caption.
    """
    try:
        # Open the image
        image = Image.open(image_path).convert("RGB")

        # Preprocess the image and prepare inputs
        inputs = processor(images=image, return_tensors="pt").to(DEVICE)

        # Generate captions
        caption_ids = model.generate(**inputs)
        caption = processor.decode(caption_ids[0], skip_special_tokens=True)
        return caption

    except Exception as e:
        print(f"Error generating caption for {image_path}: {e}")
        return None


def calculate_caption_similarity(captions):
    """
    Calculate cosine similarity between captions to find similarity scores.

    Args:
        captions (list): List of captions.

    Returns:
        list: Pairwise cosine similarity scores.
    """
    # Ensure captions are not empty
    if not captions:
        return []

    # Tokenize the captions using BLIP's processor
    inputs = processor.tokenizer(captions, return_tensors="pt", padding=True, truncation=True)
    embeddings = inputs['input_ids']

    # Calculate the cosine similarity
    similarity_matrix = cosine_similarity(embeddings.numpy())
    return similarity_matrix


def filter_similar_captions(captions, frames, similarity_threshold=0.9):
    """
    Filter out captions that are too similar based on a cosine similarity threshold.

    Args:
        captions (list): List of captions.
        frames (list): Corresponding list of frames.
        similarity_threshold (float): Threshold for cosine similarity.

    Returns:
        filtered_captions (list): List of filtered captions.
        filtered_frames (list): Corresponding list of filtered frames.
    """
    similarity_matrix = calculate_caption_similarity(captions)

    filtered_captions = []
    filtered_frames = []
    seen_indices = set()

    for i, caption in enumerate(captions):
        if i not in seen_indices:
            filtered_captions.append(caption)
            filtered_frames.append(frames[i])

            # Mark similar captions to be discarded
            for j in range(i + 1, len(captions)):
                if similarity_matrix[i][j] >= similarity_threshold:
                    seen_indices.add(j)

    return filtered_captions, filtered_frames


def process_filtered_frames(filtered_frame_folder, output_csv, advanced_filtering_enabled=True,
                            similarity_threshold=0.9):
    """
    Process filtered frames for each video, generate captions, filter by similarity, and save to a CSV file.

    Args:
        filtered_frame_folder (str): Path to the folder containing filtered frame folders for videos.
        output_csv (str): Path to save the resulting CSV file.
        advanced_filtering_enabled (bool): Flag to enable/disable advanced filtering based on caption similarity.
        similarity_threshold (float): Threshold for cosine similarity for advanced filtering.
    """
    data = []

    # Create further_filtered_frames folder only when advanced filtering is enabled
    if advanced_filtering_enabled:
        further_filtered_frames_folder = "further_filtered_frames"
        os.makedirs(further_filtered_frames_folder, exist_ok=True)

    # Iterate through each filtered frame folder
    for video_folder in os.listdir(filtered_frame_folder):
        video_path = os.path.join(filtered_frame_folder, video_folder)
        if not os.path.isdir(video_path):
            continue

        print(f"Processing filtered frames for video: {video_folder}")

        frames = []
        captions = []
        frame_files = sorted([f for f in os.listdir(video_path) if f.endswith(('.jpg', '.png', '.jpeg'))])

        # Iterate through each frame and generate captions
        for frame_file in frame_files:
            frame_path = os.path.join(video_path, frame_file)
            if frame_file.endswith(('.jpg', '.png', '.jpeg')):
                caption = generate_caption(frame_path)
                if caption is not None:
                    frames.append(frame_path)
                    captions.append(caption)

        # Apply advanced filtering if enabled
        if advanced_filtering_enabled:
            filtered_captions, filtered_frames = filter_similar_captions(captions, frames, similarity_threshold)
            print(f"Applied advanced filtering for {video_folder}. Remaining frames: {len(filtered_frames)}")
        else:
            filtered_captions, filtered_frames = captions, frames

        # Save filtered frames to further_filtered_frames folder (only if advanced filtering is enabled)
        if advanced_filtering_enabled:
            video_output_folder = os.path.join(further_filtered_frames_folder, video_folder)
            os.makedirs(video_output_folder, exist_ok=True)

            for i, frame_path in enumerate(filtered_frames):
                frame_filename = os.path.basename(frame_path)
                new_frame_path = os.path.join(video_output_folder, frame_filename)
                # Copy frame to the further filtered frames folder (keeping the structure intact)
                shutil.copy(frame_path, new_frame_path)  # Use copy instead of rename to keep the source unchanged

        # Collect the data for CSV
        for caption, frame in zip(filtered_captions, filtered_frames):
            video_name = video_folder
            frame_name = os.path.basename(frame)
            data.append({"video_name": video_name, "frame_name": frame_name, "caption": caption})

    # Define CSV output name based on filtering status
    if advanced_filtering_enabled:
        output_csv = "further_filtered_captions.csv"  # Rename to indicate advanced filtering
    else:
        output_csv = "selected_captions.csv"

    # Save results to a CSV file
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Captions saved to {output_csv}")


# Paths
filtered_frame_folders = "selected_frame_folders"  # Folder containing filtered frame folders
output_csv = "selected_captions.csv"  # Default CSV file name

# Run the function
process_filtered_frames(filtered_frame_folders, output_csv,
                        advanced_filtering_enabled=True)  # Set flag to False for caption generation only
