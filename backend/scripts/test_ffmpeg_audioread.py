import os
import sys
import imageio_ffmpeg
import audioread

print(f"Original PATH: {os.environ['PATH']}")

ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
print(f"FFmpeg exe path: {ffmpeg_path}")
ffmpeg_dir = os.path.dirname(ffmpeg_path)
print(f"FFmpeg dir: {ffmpeg_dir}")

os.environ["PATH"] += os.pathsep + ffmpeg_dir
print(f"Updated PATH: {os.environ['PATH']}")

# Check if audioread can find it
try:
    # We don't have a file to open, but we can check available backends
    print("Available backends:")
    for backend in audioread.available_backends():
        print(f" - {backend}")
        
    # specifically check for ffmpeg backend
    if 'ffmpeg' in audioread.available_backends():
        print("FFmpeg backend is available!")
    else:
        print("FFmpeg backend is NOT available.")

except Exception as e:
    print(f"Error: {e}")
