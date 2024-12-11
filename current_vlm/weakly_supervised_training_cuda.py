import os
import torch
import torch.nn as nn
from nltk.translate.bleu_score import sentence_bleu
from transformers import CLIPProcessor, CLIPModel, T5Tokenizer, T5ForConditionalGeneration
import pandas as pd
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import dask
from dask import delayed, compute
import wandb  # Import W&B for monitoring

# Directories
VIDEOS_DIR = "downloaded_videos"
KEY_FRAMES_DIR = "further_filtered_frames"
CAPTIONS_CSV = "further_filtered_captions.csv"
LABELS_CSV = "labels.csv"
SAVE_DIR = "trained_models"
os.makedirs(SAVE_DIR, exist_ok=True)

# Hyperparameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FRAME_INTERVAL = 0.5  # Seconds
EMBEDDING_DIM = 512
BATCH_SIZE = 8

# Loss Weights (to be tuned)
ALPHA = 0.5  # Weight for key frame classification loss
BETA = 0.7   # Weight for frame-level caption loss
GAMMA = 0.3  # Weight for video-level caption loss

# Initialize W&B
wandb.init(project="video-captioning", entity="shubha07m")

class VideoCaptioningModel(nn.Module):
    def __init__(self, clip_model, caption_model, key_frame_classifier):
        super(VideoCaptioningModel, self).__init__()
        self.clip_model = clip_model
        self.caption_model = caption_model
        self.key_frame_classifier = key_frame_classifier

    def forward(self, frames, captions):
        # Encode frames and captions
        frame_embeddings = self.clip_model.get_image_features(frames)
        caption_inputs = self.caption_tokenizer(captions, return_tensors="pt", padding=True, truncation=True).to(DEVICE)

        # Classify frames
        frame_scores = self.key_frame_classifier(frame_embeddings)
        key_frame_labels = torch.tensor([1 if i < len(key_frames) else 0 for i in range(len(frames))]).to(DEVICE)
        frame_loss = nn.BCEWithLogitsLoss()(frame_scores, key_frame_labels.float())

        # Generate captions and compute loss
        generated_captions = self.caption_model.generate(input_ids=caption_inputs['input_ids'])
        generated_caption = self.caption_tokenizer.decode(generated_captions[0], skip_special_tokens=True)

        # BLEU score for evaluation
        bleu_scores = sentence_bleu([captions.split()], generated_caption.split())

        return frame_loss, bleu_scores

def initialize_models():
    # Initialize CLIP for visual features
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Initialize T5 for caption generation
    caption_model = T5ForConditionalGeneration.from_pretrained("t5-small").to(DEVICE)
    caption_tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=False)

    # Initialize key frame classifier
    key_frame_classifier = KeyFrameClassifier(EMBEDDING_DIM).to(DEVICE)

    return clip_model, clip_processor, caption_model, caption_tokenizer, key_frame_classifier

def extract_frames(video_path, sampling_interval=100):
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Select frames based on sampling interval (every nth frame)
        if frame_idx % sampling_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        frame_idx += 1
    cap.release()
    return frames

def get_clip_embeddings(frames, clip_model, clip_processor):
    inputs = clip_processor(images=frames, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        embeddings = clip_model.get_image_features(**inputs)
    return embeddings

def match_video_id(key_frames_dir, captions_csv, labels_csv, videos_dir):
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
                main_caption_row = labels_df[labels_df["YoutubeID"] == video_id]
                if not main_caption_row.empty:
                    main_caption = main_caption_row["Caption"].values[0]
                    caption_rows = captions_df[captions_df["video_id"] == video_id]
                    if not caption_rows.empty:
                        matches.append((video_path, os.path.join(key_frames_dir, key_frame_folder), caption_rows, main_caption))
    return matches

def distribute_caption_to_keyframes(video_caption, key_frame_embeddings):
    # Normalize key_frame embeddings to calculate attention
    attention_weights = torch.softmax(torch.norm(key_frame_embeddings, dim=1), dim=0)
    distributed_captions = [video_caption] * len(attention_weights)
    return distributed_captions, attention_weights

class KeyFrameClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super(KeyFrameClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return self.sigmoid(x)

def train_transformer(clip_model, clip_processor, caption_model, caption_tokenizer, key_frame_classifier, labels_csv, num_epochs):
    matches = match_video_id(KEY_FRAMES_DIR, CAPTIONS_CSV, labels_csv, VIDEOS_DIR)
    optimizer = torch.optim.Adam(
        list(key_frame_classifier.parameters()) + list(caption_model.parameters()),
        lr=1e-4
    )
    epoch_losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        for video_path, key_frame_folder, caption_rows, main_caption in tqdm(matches):
            original_frames = extract_frames(video_path)
            original_embeddings = get_clip_embeddings(original_frames, clip_model, clip_processor)

            key_frame_paths = [os.path.join(key_frame_folder, f) for f in os.listdir(key_frame_folder)]
            key_frames = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in key_frame_paths]
            key_frame_embeddings = get_clip_embeddings(key_frames, clip_model, clip_processor)

            input_embeddings = torch.cat([original_embeddings, key_frame_embeddings]).to(DEVICE)
            input_labels = torch.cat([torch.zeros(len(original_embeddings)), torch.ones(len(key_frame_embeddings))]).to(DEVICE)

            distributed_captions, attention_weights = distribute_caption_to_keyframes(main_caption, key_frame_embeddings)

            main_caption_tokenized = caption_tokenizer(main_caption, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
            key_frame_captions_tokenized = caption_tokenizer(distributed_captions, return_tensors="pt", padding=True, truncation=True).to(DEVICE)

            # Key frame classification loss
            key_frame_logits = key_frame_classifier(input_embeddings).squeeze(-1)
            key_frame_loss = nn.BCEWithLogitsLoss()(key_frame_logits, input_labels)

            # Caption generation losses (main and key frame-based)
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

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # Log loss to W&B
            wandb.log({"epoch_loss": epoch_loss / len(matches), "epoch": epoch + 1})

        epoch_losses.append(epoch_loss / len(matches))

        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(SAVE_DIR, f"model_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'clip_model_state_dict': clip_model.state_dict(),
                'caption_model_state_dict': caption_model.state_dict(),
                'key_frame_classifier_state_dict': key_frame_classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, save_path)

    # Save the loss plot only once after training is complete
    plt.plot(range(num_epochs), epoch_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss ({num_epochs} epochs)')
    plt.savefig("training_loss_plot.png")
    plt.clf()

    return clip_model, caption_model, key_frame_classifier

if __name__ == '__main__':
    clip_model, clip_processor, caption_model, caption_tokenizer, key_frame_classifier = initialize_models()
    clip_model.to(DEVICE)
    caption_model.to(DEVICE)
    key_frame_classifier.to(DEVICE)

    # Train model
    trained_clip_model, trained_caption_model, trained_key_frame_classifier = train_transformer(
        clip_model, clip_processor, caption_model, caption_tokenizer, key_frame_classifier, LABELS_CSV, num_epochs=10
    )

    wandb.finish()
