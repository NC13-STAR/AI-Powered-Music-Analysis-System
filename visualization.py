import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display


def plot_waveform(audio, sample_rate):
    """
    Create and save a waveform plot of the audio signal.
    """

    print("\nGenerating waveform...")

    # Create time axis
    time = np.arange(len(audio)) / sample_rate

    # Create graphs folder
    os.makedirs("graphs", exist_ok=True)

    # Create figure
    plt.figure(figsize=(12, 5))

    # Plot waveform
    plt.plot(time, audio)

    # Add labels
    plt.title("Audio Waveform")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")

    # Add grid
    plt.grid(True)

    # Improve layout
    plt.tight_layout()

    # Save graph
    output_path = os.path.join(
        "graphs",
        "waveform.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    # Display graph
    plt.show()

    # Close figure
    plt.close()

    print("Waveform generated successfully!")
    print(f"Waveform saved to: {output_path}")


def plot_fft(audio, sample_rate):
    """
    Perform Fast Fourier Transform (FFT)
    and generate the frequency spectrum.
    """

    print("\nPerforming FFT analysis...")

    # Use first 10 seconds for FFT
    max_samples = min(
        len(audio),
        sample_rate * 10
    )

    audio_segment = audio[:max_samples]

    # Apply Hann window
    window = np.hanning(
        len(audio_segment)
    )

    windowed_audio = (
        audio_segment * window
    )

    # Perform FFT
    fft_result = np.fft.rfft(
        windowed_audio
    )

    # Calculate magnitude
    magnitude = np.abs(
        fft_result
    )

    # Normalize magnitude
    magnitude = (
        magnitude / len(windowed_audio)
    )

    # Generate frequency axis
    frequencies = np.fft.rfftfreq(
        len(windowed_audio),
        d=1 / sample_rate
    )

    # Limit frequency range
    max_frequency = 20000

    valid_indices = (
        frequencies <= max_frequency
    )

    frequencies = frequencies[
        valid_indices
    ]

    magnitude = magnitude[
        valid_indices
    ]

    # Create graphs folder
    os.makedirs(
        "graphs",
        exist_ok=True
    )

    # Create FFT graph
    plt.figure(figsize=(12, 5))

    plt.plot(
        frequencies,
        magnitude
    )

    # Add labels
    plt.title(
        "FFT Frequency Spectrum"
    )

    plt.xlabel(
        "Frequency (Hz)"
    )

    plt.ylabel(
        "Magnitude"
    )

    plt.grid(True)

    plt.tight_layout()

    # Save graph
    output_path = os.path.join(
        "graphs",
        "fft_spectrum.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.show()

    plt.close()

    print(
        "FFT analysis completed successfully!"
    )

    print(
        f"FFT graph saved to: {output_path}"
    )


def plot_spectrogram(audio, sample_rate):
    """
    Generate a spectrogram using STFT
    (Short-Time Fourier Transform).
    """

    print("\nPerforming STFT analysis...")

    # ------------------------------------------------
    # Step 1: Select audio
    # ------------------------------------------------

    # Use the complete audio for spectrogram.
    # If the audio is very long, this can be changed
    # later for performance optimization.

    audio_segment = audio

    # ------------------------------------------------
    # Step 2: STFT
    # ------------------------------------------------

    stft_result = librosa.stft(
        audio_segment,
        n_fft=2048,
        hop_length=512,
        window="hann"
    )

    # ------------------------------------------------
    # Step 3: Convert amplitude to decibels
    # ------------------------------------------------

    magnitude = np.abs(
        stft_result
    )

    magnitude_db = librosa.amplitude_to_db(
        magnitude,
        ref=np.max
    )

    # ------------------------------------------------
    # Step 4: Create graphs folder
    # ------------------------------------------------

    os.makedirs(
        "graphs",
        exist_ok=True
    )

    # ------------------------------------------------
    # Step 5: Create spectrogram
    # ------------------------------------------------

    plt.figure(figsize=(12, 6))

    librosa.display.specshow(
        magnitude_db,
        sr=sample_rate,
        hop_length=512,
        x_axis="time",
        y_axis="hz"
    )

    # Add title
    plt.title(
        "Audio Spectrogram (STFT)"
    )

    # Add labels
    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Frequency (Hz)"
    )

    # Add color scale
    plt.colorbar(
        format="%+2.0f dB"
    )

    # Improve layout
    plt.tight_layout()

    # ------------------------------------------------
    # Step 6: Save spectrogram
    # ------------------------------------------------

    output_path = os.path.join(
        "graphs",
        "spectrogram.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    # Display graph
    plt.show()

    # Close figure
    plt.close()

    print(
        "STFT analysis completed successfully!"
    )

    print(
        f"Spectrogram saved to: {output_path}"
    )