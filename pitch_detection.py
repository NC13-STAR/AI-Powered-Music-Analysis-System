import os
import numpy as np
import librosa
import matplotlib.pyplot as plt


def detect_pitch(audio, sample_rate):
    """
    Phase 6.2:
    Detect fundamental frequency (F0) from the
    melody-focused audio using PYIN.
    """

    print("\n" + "=" * 60)
    print("PHASE 6.2 - PITCH / F0 DETECTION")
    print("=" * 60)

    print("\nDetecting fundamental frequency (F0)...")
    print("Using PYIN pitch detection...")

    # PYIN pitch detection
    f0, voiced_flag, voiced_prob = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sample_rate,
        frame_length=2048,
        hop_length=512
    )

    print("Pitch detection completed successfully!")

    # Create time axis
    times = librosa.times_like(
        f0,
        sr=sample_rate,
        hop_length=512
    )

    # Find valid pitch values
    valid_pitch = ~np.isnan(f0)

    total_frames = len(f0)
    detected_frames = np.sum(valid_pitch)

    print("\nPitch detection statistics:")
    print(f"Total frames       : {total_frames}")
    print(f"Detected pitch     : {detected_frames}")
    print(
        f"Undetected frames : "
        f"{total_frames - detected_frames}"
    )

    # Display pitch statistics
    if detected_frames > 0:

        minimum_f0 = np.nanmin(f0)
        maximum_f0 = np.nanmax(f0)
        mean_f0 = np.nanmean(f0)

        print(f"Minimum F0        : {minimum_f0:.2f} Hz")
        print(f"Maximum F0        : {maximum_f0:.2f} Hz")
        print(f"Mean F0           : {mean_f0:.2f} Hz")

    # Create required directories
    os.makedirs("data/output", exist_ok=True)
    os.makedirs("graphs", exist_ok=True)

    # Save numerical pitch data
    output_file = "data/output/pitch_data.npz"

    np.savez(
        output_file,
        f0=f0,
        times=times,
        voiced_flag=voiced_flag,
        voiced_prob=voiced_prob
    )

    print(f"\nPitch data saved successfully!")
    print(f"Saved file: {output_file}")

    # --------------------------------------------------
    # Generate pitch graph
    # --------------------------------------------------

    plt.figure(figsize=(14, 5))

    plt.plot(
        times,
        f0,
        linewidth=1
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Phase 6.2 - Detected Pitch / Fundamental Frequency")

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    graph_path = "graphs/pitch_detection.png"

    plt.savefig(
        graph_path,
        dpi=300
    )

    plt.close()

    print(f"Pitch graph saved successfully!")
    print(f"Saved graph: {graph_path}")

    print("\n" + "=" * 60)
    print("PHASE 6.2 COMPLETED SUCCESSFULLY")
    print("=" * 60)

    return f0, times, voiced_flag, voiced_prob