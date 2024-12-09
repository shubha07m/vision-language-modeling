import subprocess
import sys

scripts = [
    'primary_frame_filtering.py',
    'ML_based_frame_filtering.py',
    'caption_generation.py'
]

for script in scripts:
    try:
        subprocess.run([sys.executable, script])
    except Exception as e:
        print(f"Error running {script}: {e}")