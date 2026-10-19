import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt, savgol_filter


def find_continuous_segments(valid):
    """
    Find continuous regions of valid pitch.
    Missing pitch values remain as genuine gaps.
    """
    segments = []
    start = None

    for i in range(len(valid)):
        if valid[i] and start is None:
            start = i

        elif not valid[i] and start is not None:
            segments.append((start, i))
            start = None

    if start is not None:
        segments.append((start, len(valid)))

    return segments


def octave_correct_segment(
    segment,
    max_octave_jump_semitones=7.0,
    local_window=15
):
    """
    Correct obvious octave/subharmonic errors inside one continuous
    pitch segment.

    The algorithm compares each pitch with a local median reference.
    Candidate pitches are:

        original
        original / 2
        original * 2

    Only corrections that produce a clearly closer local pitch are made.
    """

    corrected = segment.copy()
    corrections = 0

    half_window = local_window // 2

    for i in range(len(corrected)):

        current = corrected[i]

        if not np.isfinite(current) or current <= 0:
            continue

        left = max(0, i - half_window)
        right = min(len(corrected), i + half_window + 1)

        neighborhood = corrected[left:right]

        neighborhood = neighborhood[
            np.isfinite(neighborhood) & (neighborhood > 0)
        ]

        if len(neighborhood) < 5:
            continue

        local_reference = np.median(neighborhood)

        if local_reference <= 0:
            continue

        # Compare pitch in semitone distance.
        original_distance = abs(
            12 * np.log2(current / local_reference)
        )

        half_pitch = current / 2.0
        double_pitch = current * 2.0

        half_distance = abs(
            12 * np.log2(half_pitch / local_reference)
        )

        double_distance = abs(
            12 * np.log2(double_pitch / local_reference)
        )

        candidates = [
            (current, original_distance),
            (half_pitch, half_distance),
            (double_pitch, double_distance)
        ]

        best_pitch, best_distance = min(
            candidates,
            key=lambda x: x[1]
        )

        # Only correct when:
        # 1. the current pitch is substantially far from the local reference
        # 2. an octave-shifted candidate is much closer
        # 3. the improvement is significant
        if (
            best_pitch != current
            and original_distance > max_octave_jump_semitones
            and best_distance < 2.5
            and (original_distance - best_distance) > 5.0
        ):
            corrected[i] = best_pitch
            corrections += 1

    return corrected, corrections


def smooth_and_correct_pitch(
    f0,
    times,
    median_kernel=7,
    local_window=15,
    smooth_window=11,
    smooth_polyorder=2
):
    print("\n" + "=" * 65)
    print("PHASE 6.4 - ROBUST PITCH SMOOTHING + OCTAVE CORRECTION")
    print("=" * 65)

    print("\nProcessing cleaned pitch contour...")

    corrected_f0 = np.array(f0, dtype=float).copy()
    times = np.array(times, dtype=float)

    # ------------------------------------------------------------
    # STEP 1 - Identify valid pitch
    # ------------------------------------------------------------

    valid = np.isfinite(corrected_f0) & (corrected_f0 > 0)

    segments = find_continuous_segments(valid)

    print(f"\nContinuous pitch segments found: {len(segments)}")

    # ------------------------------------------------------------
    # STEP 2 - Median filtering
    # ------------------------------------------------------------

    median_changes = 0

    for start, end in segments:

        segment = corrected_f0[start:end].copy()

        if len(segment) < 3:
            continue

        kernel = median_kernel

        if kernel % 2 == 0:
            kernel += 1

        if kernel >= len(segment):
            kernel = (
                len(segment)
                if len(segment) % 2 == 1
                else len(segment) - 1
            )

        if kernel >= 3:

            filtered_segment = medfilt(
                segment,
                kernel_size=kernel
            )

            median_changes += np.sum(
                np.abs(filtered_segment - segment) > 0.01
            )

            segment = filtered_segment

            corrected_f0[start:end] = segment

    # ------------------------------------------------------------
    # STEP 3 - Octave correction
    # ------------------------------------------------------------

    octave_corrections = 0

    valid_after_median = (
        np.isfinite(corrected_f0) &
        (corrected_f0 > 0)
    )

    segments_after_median = find_continuous_segments(
        valid_after_median
    )

    for start, end in segments_after_median:

        segment = corrected_f0[start:end].copy()

        if len(segment) < 5:
            continue

        corrected_segment, corrections = octave_correct_segment(
            segment,
            max_octave_jump_semitones=7.0,
            local_window=local_window
        )

        corrected_f0[start:end] = corrected_segment

        octave_corrections += corrections

    # ------------------------------------------------------------
    # STEP 4 - Gentle Savitzky-Golay smoothing
    # ------------------------------------------------------------

    valid_after_correction = (
        np.isfinite(corrected_f0) &
        (corrected_f0 > 0)
    )

    final_segments = find_continuous_segments(
        valid_after_correction
    )

    for start, end in final_segments:

        segment = corrected_f0[start:end].copy()

        if len(segment) < 5:
            continue

        window = smooth_window

        if window % 2 == 0:
            window += 1

        if window >= len(segment):
            window = (
                len(segment)
                if len(segment) % 2 == 1
                else len(segment) - 1
            )

        if (
            window >= 5
            and window > smooth_polyorder
        ):
            segment = savgol_filter(
                segment,
                window_length=window,
                polyorder=smooth_polyorder
            )

        corrected_f0[start:end] = segment

    # ------------------------------------------------------------
    # STEP 5 - Final statistics
    # ------------------------------------------------------------

    valid_final = (
        np.isfinite(corrected_f0) &
        (corrected_f0 > 0)
    )

    total_frames = len(corrected_f0)

    detected_frames = np.sum(valid_final)

    coverage = (
        detected_frames / total_frames * 100
        if total_frames > 0
        else 0
    )

    print("\nPitch correction statistics:")

    print(f"Total frames            : {total_frames}")

    print(f"Valid pitch frames      : {detected_frames}")

    print(f"Pitch coverage          : {coverage:.2f}%")

    print(f"Median-filter changes   : {median_changes}")

    print(
        f"Octave corrections made: {octave_corrections}"
    )

    if detected_frames > 0:

        print(
            f"Minimum corrected F0    : "
            f"{np.nanmin(corrected_f0):.2f} Hz"
        )

        print(
            f"Maximum corrected F0    : "
            f"{np.nanmax(corrected_f0):.2f} Hz"
        )

        print(
            f"Mean corrected F0       : "
            f"{np.nanmean(corrected_f0):.2f} Hz"
        )

    # ------------------------------------------------------------
    # STEP 6 - Save data
    # ------------------------------------------------------------

    os.makedirs("data/output", exist_ok=True)
    os.makedirs("graphs", exist_ok=True)

    output_file = (
        "data/output/smoothed_pitch_data.npz"
    )

    np.savez(
        output_file,
        f0=corrected_f0,
        times=times
    )

    print("\nSmoothed pitch data saved successfully!")

    print(f"Saved file: {output_file}")

    # ------------------------------------------------------------
    # STEP 7 - Save graph
    # ------------------------------------------------------------

    plt.figure(figsize=(14, 5))

    plt.plot(
        times,
        corrected_f0,
        linewidth=1
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency (Hz)")

    plt.title(
        "Phase 6.4 - Robust Smoothed and Octave-Corrected Pitch"
    )

    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    graph_path = (
        "graphs/smoothed_pitch.png"
    )

    plt.savefig(
        graph_path,
        dpi=300
    )

    plt.close()

    print(
        "Smoothed pitch graph saved successfully!"
    )

    print(f"Saved graph: {graph_path}")

    print("\n" + "=" * 65)
    print("PHASE 6.4 COMPLETED SUCCESSFULLY")
    print("=" * 65)

    return corrected_f0, times