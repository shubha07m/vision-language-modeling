import torch
import cv2
import numpy as np
from transformers import CLIPProcessor, CLIPModel, T5Tokenizer, T5ForConditionalGeneration
import torch.nn.functional as F


# Hyperparameters
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
EMBEDDING_DIM = 512  # CLIP embedding size
T5_EMBEDDING_DIM = 768  # T5 embedding size
BATCH_SIZE = 8


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
        return None, None, None, None

    return clip_model, clip_processor, caption_model, caption_tokenizer


# Utility Functions
def extract_frames(video_path, sampling_interval=100):
    """Extract frames from a video at a fixed frame count interval."""
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sampling_interval == 0:  # Select frame based on frame count interval
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


def generate_caption(frame_embedding, caption_model, caption_tokenizer):
    """Generate caption for a given frame embedding."""
    # Prepare the input for T5
    text_input = "Generate caption for this image:"
    inputs = caption_tokenizer(text_input, return_tensors="pt").to(DEVICE)

    # Define the projection layer
    projection_layer = torch.nn.Linear(EMBEDDING_DIM, T5_EMBEDDING_DIM).to(DEVICE)

    # Project the CLIP embedding into a sequence (using a simple linear transformation)
    projected_embedding = F.linear(frame_embedding.unsqueeze(0), weight=projection_layer.weight, bias=projection_layer.bias)

    # Reshape the embedding to be compatible with T5 (batch_size, seq_length, embedding_size)
    projected_embedding = projected_embedding.unsqueeze(1)  # Shape: (1, 1, 768)

    # Feed the projected embedding as input to the T5 encoder
    # Ensure that the embedding is passed correctly into the encoder
    encoder_outputs = caption_model.encoder(inputs_embeds=projected_embedding)

    # Generate caption
    outputs = caption_model.generate(
        input_ids=inputs.input_ids,
        encoder_outputs=encoder_outputs,
        max_length=50
    )

    return caption_tokenizer.decode(outputs[0], skip_special_tokens=True)


def process_video(video_path, clip_model, clip_processor, caption_model, caption_tokenizer):
    """Process a video and generate captions for selected keyframes."""
    frames = extract_frames(video_path)
    frame_embeddings = get_clip_embeddings(frames, clip_model, clip_processor)

    keyframe_caption_pairs = []

    for idx, embedding in enumerate(frame_embeddings):
        caption = generate_caption(embedding, caption_model, caption_tokenizer)
        keyframe_caption_pairs.append((idx, caption))

    return keyframe_caption_pairs


def main():
    video_path = "new_testing_videos/Tgr7TFr5Lrg.mp4"
    clip_model, clip_processor, caption_model, caption_tokenizer = initialize_models()

    if all([clip_model, clip_processor, caption_model, caption_tokenizer]):
        keyframe_caption_pairs = process_video(video_path, clip_model, clip_processor, caption_model, caption_tokenizer)

        # Print the keyframe-caption pairs
        for idx, caption in keyframe_caption_pairs:
            print(f"Keyframe {idx}: {caption}")
    else:
        print("Model initialization failed. Please check the error messages above.")


if __name__ == "__main__":
    main()