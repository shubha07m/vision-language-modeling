from datasets import load_dataset
import csv
import os
from yt_dlp import YoutubeDL

# Load dataset
ds = load_dataset("OpenGVLab/InternVid", "InternVid-10M")

# Extract the data (Dataset object)
data = ds['FLT']

n = 111

# Extract the first n entries as dictionaries
first_n_videos = [entry for entry in data.select(range(n))]

# Save to labels.csv
output_file = 'labels.csv'

# Directory for downloaded videos
download_folder = 'downloaded_videos'
os.makedirs(download_folder, exist_ok=True)

# Initialize yt-dlp options
ydl_opts = {
    'outtmpl': os.path.join(download_folder, '%(id)s.%(ext)s'),  # Save as video_id.ext in the folder
    'format': 'bestvideo+bestaudio/best',  # Best quality
}

# Keep track of successfully downloaded videos
successful_videos = []

# Download videos
with YoutubeDL(ydl_opts) as ydl:
    for entry in first_n_videos:
        video_url = f"https://www.youtube.com/watch?v={entry['YoutubeID']}"
        try:
            ydl.download([video_url])
            print(f"Downloaded: {video_url}")
            successful_videos.append(entry)  # Add to successful list
        except Exception as e:
            print(f"Failed to download {video_url}: {e}")

# Save only successful entries to labels.csv
if successful_videos:
    with open(output_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=successful_videos[0].keys())
        writer.writeheader()
        writer.writerows(successful_videos)

print(f"Labels for successfully downloaded videos saved to {output_file}")
