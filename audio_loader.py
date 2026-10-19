import librosa


def load_audio(audio_path):
    """
    Load an audio file using the project's existing librosa setup.

    The audio is converted to mono and the original sample rate is preserved.

    Returns:
        audio: Audio signal
        sample_rate: Sampling rate
    """

    print("\nLoading audio...")

    try:
        audio, sample_rate = librosa.load(
            audio_path,
            sr=None,
            mono=True
        )

    except Exception as exc:
        raise RuntimeError(
            f"Could not load audio file:\n"
            f"{audio_path}\n\n"
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc

    duration = len(audio) / sample_rate if sample_rate else 0.0

    print("Audio loaded successfully!")
    print(f"Sampling rate: {sample_rate} Hz")
    print(f"Number of samples: {len(audio)}")
    print(f"Duration: {duration:.2f} seconds")

    return audio, sample_rate
