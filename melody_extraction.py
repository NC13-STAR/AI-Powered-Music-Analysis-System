import os
import numpy as np
import librosa
import soundfile as sf


def extract_melodic_component(audio, sample_rate):
    """
    Extract the harmonic/melodic component of an audio signal
    using Harmonic-Percussive Source Separation (HPSS).
    """

    print("\n" + "=" * 60)
    print("PHASE 6.1 - MELODY-FOCUSED AUDIO EXTRACTION")
    print("=" * 60)

    print("\nApplying Harmonic-Percussive Source Separation (HPSS)...")
    print("Separating harmonic and percussive components...")

    # HPSS
    harmonic, percussive = librosa.effects.hpss(audio)

    print("HPSS completed successfully!")

    # Calculate energy
    harmonic_energy = np.mean(harmonic ** 2)
    percussive_energy = np.mean(percussive ** 2)

    print("\nAudio component analysis:")
    print(f"Harmonic energy   : {harmonic_energy:.6f}")
    print(f"Percussive energy : {percussive_energy:.6f}")

    # Create output directory
    os.makedirs("data/output", exist_ok=True)

    # Save harmonic component
    output_path = os.path.join(
        "data",
        "output",
        "melody_component.wav"
    )

    sf.write(output_path, harmonic, sample_rate)

    print("\nMelody-focused component saved successfully!")
    print(f"Saved file: {output_path}")

    print("\n" + "=" * 60)
    print("PHASE 6.1 COMPLETED SUCCESSFULLY")
    print("=" * 60)

    return harmonic


if __name__ == "__main__":

    # Input file generated during preprocessing
    input_path = os.path.join(
        "data",
        "output",
        "preprocessed_audio.wav"
    )

    print("\nLoading preprocessed audio...")

    if not os.path.exists(input_path):
        print("\nERROR: Preprocessed audio file not found!")
        print(f"Expected file: {input_path}")
        print("\nPlease complete the preprocessing phase first.")
        exit()

    # Load audio
    audio, sample_rate = librosa.load(
        input_path,
        sr=None,
        mono=True
    )

    print("Preprocessed audio loaded successfully!")
    print(f"Sampling rate: {sample_rate} Hz")
    print(f"Duration: {len(audio) / sample_rate:.2f} seconds")

    # Run Phase 6.1
    extract_melodic_component(audio, sample_rate)