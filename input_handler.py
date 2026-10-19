import tkinter as tk
from tkinter import filedialog
import requests
import os
from urllib.parse import urlparse


def select_local_audio():
    """Open a file-selection window and return the selected audio file path."""

    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Select an Audio File",
        filetypes=[
            ("Audio Files", "*.mp3 *.wav *.flac *.ogg"),
            ("MP3 Files", "*.mp3"),
            ("WAV Files", "*.wav"),
            ("FLAC Files", "*.flac"),
            ("OGG Files", "*.ogg"),
            ("All Files", "*.*")
        ]
    )

    root.destroy()

    return file_path


def download_audio_from_url(url):
    """Download an audio file from a direct URL."""

    os.makedirs("data/temp", exist_ok=True)

    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path)

    if not filename:
        filename = "downloaded_audio.mp3"

    file_path = os.path.join("data", "temp", filename)

    print("\nDownloading audio...")

    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()

    with open(file_path, "wb") as audio_file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                audio_file.write(chunk)

    print("Download successful!")

    return file_path