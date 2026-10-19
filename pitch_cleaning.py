import os
import numpy as np
import matplotlib.pyplot as plt


def clean_pitch(
    f0,
    times,
    voiced_flag,
    voiced_prob,
    confidence_threshold=0.05
):
    """
    Phase 6.3:
    Clean the raw PYIN pitch contour.

    Operations:
    1. Remove invalid pitch values.
    2. Remove very-low-confidence pitch detections.
    3. Preserve genuine gaps.
    4. Do not perform smoothing or octave correction.

    Smoothing and octave correction are handled in Phase 6.4.
    """

    print("\n" + "=" * 60)
    print("PHASE 6.3 - PITCH CLEANING")
    print("=" * 60)

    print("\nCleaning raw PYIN pitch data...")

    # ----------------------------------------------------------
    # Make a copy
    # ----------------------------------------------------------

    cleaned_f0 = np.array(f0, dtype=float).copy()

    # ----------------------------------------------------------
    # Step 1 - Remove invalid pitch values
    # ----------------------------------------------------------

    invalid_pitch = (
        ~np.isfinite(cleaned_f0) |
        (cleaned_f0 <= 0)
    )

    invalid_count = np.sum(invalid_pitch)

    cleaned_f0[invalid_pitch] = np.nan

    print("\nStep 1 - Invalid pitch removal")
    print(f"Invalid pitch values removed: {invalid_count}")

    # ----------------------------------------------------------
    # Step 2 - Confidence filtering
    # ----------------------------------------------------------

    low_confidence = (
        np.isfinite(cleaned_f0) &
        (voiced_prob < confidence_threshold)
    )

    low_confidence_count = np.sum(low_confidence)

    cleaned_f0[low_confidence] = np.nan

    print("\nStep 2 - Confidence filtering")
    print(f"Confidence threshold: {confidence_threshold:.2f}")
    print(f"Low-confidence values removed: {low_confidence_count}")

    # ----------------------------------------------------------
    # Step 3 - Preserve remaining valid pitch
    # ----------------------------------------------------------

    # We intentionally DO NOT fill gaps.
    # We also DO NOT remove short valid segments.
    #
    # Real vocal pauses and uncertain regions should remain gaps.

    valid_pitch = np.isfinite(cleaned_f0)

    total_frames = len(cleaned_f0)
    detected_frames = np.sum(valid_pitch)
    missing_frames = total_frames - detected_frames

    print("\nStep 3 - Clean pitch statistics")
    print(f"Total frames       : {total_frames}")
    print(f"Clean pitch frames : {detected_frames}")
    print(f"Missing frames     : {missing_frames}")

    if detected_frames > 0:

        minimum_f0 = np.nanmin(cleaned_f0)
        maximum_f0 = np.nanmax(cleaned_f0)
        mean_f0 = np.nanmean(cleaned_f0)

        print(f"Minimum clean F0   : {minimum_f0:.2f} Hz")
        print(f"Maximum clean F0   : {maximum_f0:.2f} Hz")
        print(f"Mean clean F0      : {mean_f0:.2f} Hz")

    # ----------------------------------------------------------
    # Create directories
    # ----------------------------------------------------------

    os.makedirs("data/output", exist_ok=True)
    os.makedirs("graphs", exist_ok=True)

    # ----------------------------------------------------------
    # Save cleaned pitch
    # ----------------------------------------------------------

    output_file = "data/output/clean_pitch_data.npz"

    np.savez(
        output_file,
        f0=cleaned_f0,
        times=times,
        voiced_flag=voiced_flag,
        voiced_prob=voiced_prob
    )

    print("\nClean pitch data saved successfully!")
    print(f"Saved file: {output_file}")

    # ----------------------------------------------------------
    # Generate graph
    # ----------------------------------------------------------

    plt.figure(figsize=(14, 5))

    plt.plot(
        times,
        cleaned_f0,
        linewidth=1
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency (Hz)")
    plt.title(
        "Phase 6.3 - Cleaned Pitch / Fundamental Frequency"
    )

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    graph_path = "graphs/clean_pitch.png"

    plt.savefig(
        graph_path,
        dpi=300
    )

    plt.close()

    print("Clean pitch graph saved successfully!")
    print(f"Saved graph: {graph_path}")

    print("\n" + "=" * 60)
    print("PHASE 6.3 COMPLETED SUCCESSFULLY")
    print("=" * 60)

    return cleaned_f0, times