import os
import shutil
import imageio_ffmpeg
import audioread

ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
ffmpeg_dir = os.path.dirname(ffmpeg_path)
target_path = os.path.join(ffmpeg_dir, "ffmpeg.exe")

print(f"Source: {ffmpeg_path}")
print(f"Target: {target_path}")

if not os.path.exists(target_path):
    print("Copying ffmpeg to ffmpeg.exe...")
    try:
        shutil.copy(ffmpeg_path, target_path)
        print("Copy successful.")
    except Exception as e:
        print(f"Copy failed: {e}")
else:
    print("ffmpeg.exe already exists.")

os.environ["PATH"] += os.pathsep + ffmpeg_dir

print("Checking audioread backends again...")
# Force reload of backends if possible, or just check
import importlib
import audioread.ffdec
importlib.reload(audioread.ffdec)

if audioread.ffdec.available():
    print("FFmpeg backend IS available via ffdec!")
else:
    print("FFmpeg backend is STILL NOT available.")
