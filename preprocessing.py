import numpy as np
from scipy.signal import butter, filtfilt
import soundfile as sf
import os


def normalize_audio(audio):
    """
    Normalize audio amplitude between -1 and +1.
    """

    max_value = np.max(np.abs(audio))

    if max_value == 0:
        return audio

    normalized_audio = audio / max_value

    return normalized_audio


def high_pass_filter(audio, sample_rate, cutoff=80):
    """
    Remove very low-frequency components below the cutoff frequency.
    """

    nyquist = sample_rate / 2
    normalized_cutoff = cutoff / nyquist

    b, a = butter(
        N=4,
        Wn=normalized_cutoff,
        btype="high"
    )

    filtered_audio = filtfilt(b, a, audio)

    return filtered_audio


def preprocess_audio(audio, sample_rate):
    """
    Complete basic DSP preprocessing pipeline.
    """

    print("\nStarting audio preprocessing...")

    # Step 1: Normalize
    print("1. Normalizing audio...")
    audio = normalize_audio(audio)

    # Step 2: High-pass filtering
    print("2. Applying high-pass filter...")
    audio = high_pass_filter(
        audio,
        sample_rate,
        cutoff=80
    )

    # Step 3: Final normalization
    print("3. Final normalization...")
    audio = normalize_audio(audio)

    # Step 4: Save processed audio
    os.makedirs("data/output", exist_ok=True)

    output_path = os.path.join(
        "data",
        "output",
        "preprocessed_audio.wav"
    )

    sf.write(
        output_path,
        audio,
        sample_rate
    )

    print("4. Preprocessed audio saved successfully!")
    print(f"Saved file: {output_path}")

    print("Audio preprocessing completed successfully!")

    return audio