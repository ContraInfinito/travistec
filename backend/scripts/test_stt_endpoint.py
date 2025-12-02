import requests
import os

def test_stt():
    url = "http://127.0.0.1:8000/api/v1/transcribe"
    file_path = os.path.join(os.path.dirname(__file__), "../datasets/audio/wavs/audio_001.wav")
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print(f"Sending {file_path} to {url}...")
    with open(file_path, "rb") as f:
        files = {"audio": ("test.wav", f, "audio/wav")}
        try:
            response = requests.post(url, files=files)
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.text}")
            print("!!! TEST COMPLETE !!!")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_stt()
