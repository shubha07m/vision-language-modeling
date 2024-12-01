import os
import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel, T5Tokenizer, T5ForConditionalGeneration
import pandas as pd
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

# Directories
VIDEOS_DIR = "downloaded_videos"
KEY_FRAMES_DIR = "further_filtered_frames"
CAPTIONS_CSV = "further_filtered_captions.csv"
SAVE_DIR = "trained_models"
os.makedirs(SAVE_DIR, exist_ok=True)

# Hyperparameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FRAME_INTERVAL = 0.5  # Seconds
EMBEDDING_DIM = 512

# Loss Weights
ALPHA = 0.5  # Weight for key frame classification loss
BETA = 0.7   # Weight for frame-level caption loss
GAMMA = 0.3  # Weight for video-level caption loss

# Initialize Models
def initialize_models():
    try:
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    except OSError as e:
        print(f"Error loading CLIP model: {e}")
        return None, None, None, None

    try:
        caption_model = T5ForConditionalGeneration.from_pretrained("t5-small").to(DEVICE)
        caption_tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=False)
    except ImportError as e:
        print(f"Error loading T5 model: {e}")
        print("Please install SentencePiece: pip install sentencepiece")
        return None, None, None, None

    return clip_model, clip_processor, caption_model, caption_tokenizer

# Utility Functions
def extract_frames(video_path, interval=FRAME_INTERVAL):
    """Extract frames from a video at regular intervals."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval)
    frames = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        frame_idx += 1
    cap.release()
    return frames

def get_clip_embeddings(frames, clip_model, clip_processor):
    """Get CLIP embeddings for frames."""
    inputs = clip_processor(images=frames, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        embeddings = clip_model.get_image_features(**inputs)
    return embeddings

def match_video_id(key_frames_dir, captions_csv, labels_csv, videos_dir):
    """Match key frames with their corresponding original videos and captions."""
    captions_df = pd.read_csv(captions_csv)
    labels_df = pd.read_csv(labels_csv)

    captions_df['video_id'] = captions_df['video_name'].str.replace('selected_frames_', '')
    target_video_ids = set(captions_df["video_id"].unique())

    matches = []
    for key_frame_folder in os.listdir(key_frames_dir):
        video_id = key_frame_folder.replace("selected_frames_", "")
        if video_id in target_video_ids:
            video_path = os.path.join(videos_dir, f"{video_id}.mkv")
            if os.path.exists(video_path):
                # Match the main caption for the video from labels.csv
                main_caption_row = labels_df[labels_df["YoutubeID"] == video_id]
                if not main_caption_row.empty:
                    main_caption = main_caption_row["Caption"].values[0]
                    caption_rows = captions_df[captions_df["video_id"] == video_id]
                    if not caption_rows.empty:
                        matches.append((video_path, os.path.join(key_frames_dir, key_frame_folder), caption_rows, main_caption))
    return matches

def train_transformer(clip_model, clip_processor, caption_model, caption_tokenizer, labels_csv, num_epochs=20):
    """Main training loop for the transformer."""
    matches = match_video_id(KEY_FRAMES_DIR, CAPTIONS_CSV, labels_csv, VIDEOS_DIR)

    # Define linear layer for binary classification
    linear_layer = nn.Linear(EMBEDDING_DIM, 1).to(DEVICE)

    # Define optimizer for both linear layer and caption model
    optimizer = torch.optim.Adam(
        list(linear_layer.parameters()) + list(caption_model.parameters()), lr=1e-4
    )

    # For tracking loss over epochs
    epoch_losses = []

    # Training Loop
    for epoch in range(num_epochs):
        epoch_loss = 0
        for video_path, key_frame_folder, caption_row, main_caption in tqdm(matches):
            # Step 1: Extract frames from the original video
            original_frames = extract_frames(video_path)
            original_embeddings = get_clip_embeddings(original_frames, clip_model, clip_processor)

            # Step 2: Load key frames and captions
            key_frame_paths = [os.path.join(key_frame_folder, f) for f in os.listdir(key_frame_folder)]
            key_frames = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in key_frame_paths]
            key_frame_embeddings = get_clip_embeddings(key_frames, clip_model, clip_processor)

            # Step 3: Prepare data for transformer
            input_embeddings = torch.cat([original_embeddings, key_frame_embeddings])
            input_labels = torch.cat([torch.zeros(len(original_embeddings)), torch.ones(len(key_frame_embeddings))])

            # Tokenize and encode the captions
            main_caption_tokenized = caption_tokenizer(main_caption, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
            key_frame_captions = caption_row["caption"].values[0]
            key_frame_captions_tokenized = caption_tokenizer(key_frame_captions, return_tensors="pt", padding=True, truncation=True).to(DEVICE)

            # Step 4: Define losses
            criterion_key_frame = nn.BCEWithLogitsLoss()
            criterion_caption = nn.CrossEntropyLoss()

            # Step 5: Forward pass
            # Project embeddings to logits for key frame classification
            key_frame_logits = linear_layer(input_embeddings).squeeze(-1)  # Shape: (N,)
            key_frame_loss = criterion_key_frame(key_frame_logits, input_labels)

            # Caption generation
            outputs_main_caption = caption_model(input_ids=main_caption_tokenized.input_ids, labels=main_caption_tokenized.input_ids)
            main_caption_loss = outputs_main_caption.loss

            outputs_key_frame_caption = caption_model(input_ids=key_frame_captions_tokenized.input_ids, labels=key_frame_captions_tokenized.input_ids)
            key_frame_caption_loss = outputs_key_frame_caption.loss

            # Combine losses
            loss = (
                ALPHA * key_frame_loss +
                BETA * key_frame_caption_loss +
                GAMMA * main_caption_loss
            )

            # Backpropagation and Optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Track loss for plotting later
        epoch_losses.append(epoch_loss / len(matches))

        # Save model every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(SAVE_DIR, f"model_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'clip_model_state_dict': clip_model.state_dict(),
                'caption_model_state_dict': caption_model.state_dict(),
                'linear_layer_state_dict': linear_layer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss / len(matches),
            }, save_path)
            print(f"Saved model checkpoint to {save_path}")

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss / len(matches)}")

    # Plot training loss
    plt.plot(range(1, num_epochs + 1), epoch_losses, label="Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss over Epochs")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    clip_model, clip_processor, caption_model, caption_tokenizer = initialize_models()

    if all([clip_model, clip_processor, caption_model, caption_tokenizer]):
        train_transformer(clip_model, clip_processor, caption_model, caption_tokenizer, labels_csv="labels.csv", num_epochs=20)
    else:
        print("Model initialization failed. Please check the error messages above.")
