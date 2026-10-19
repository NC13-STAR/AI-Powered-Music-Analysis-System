import os
import numpy as np
import matplotlib.pyplot as plt


def create_final_melody_contour(
    f0,
    times,
    min_note_duration=0.08
):
    """
    Phase 6.5:
    Create the final melody contour from the
    smoothed and octave-corrected pitch.

    Processing:
    1. Keep valid pitch values.
    2. Preserve genuine gaps.
    3. Remove extremely short isolated pitch fragments.
    4. Keep the resulting melody contour unchanged.
    """

    print("\n" + "=" * 60)
    print("PHASE 6.5 - FINAL MELODY CONTOUR")
    print("=" * 60)

    print("\nCreating final melody contour...")

    final_f0 = np.array(
        f0,
        dtype=float
    ).copy()

    times = np.array(
        times,
        dtype=float
    )

    # ------------------------------------------------
    # Step 1: Identify valid pitch
    # ------------------------------------------------

    valid = (
        np.isfinite(final_f0)
        & (final_f0 > 0)
    )

    print(
        f"\nInitial valid pitch frames: "
        f"{np.sum(valid)}"
    )

    # ------------------------------------------------
    # Step 2: Find continuous pitch segments
    # ------------------------------------------------

    segments = []

    start = None

    for i in range(len(valid)):

        if valid[i] and start is None:

            start = i

        elif not valid[i] and start is not None:

            segments.append(
                (start, i)
            )

            start = None

    if start is not None:

        segments.append(
            (start, len(valid))
        )

    print(
        f"Continuous pitch segments found: "
        f"{len(segments)}"
    )

    # ------------------------------------------------
    # Step 3: Remove extremely short segments
    # ------------------------------------------------

    removed_segments = 0

    if len(times) > 1:

        frame_duration = np.median(
            np.diff(times)
        )

    else:

        frame_duration = 0.0

    minimum_frames = 1

    if frame_duration > 0:

        minimum_frames = max(
            1,
            int(
                np.ceil(
                    min_note_duration /
                    frame_duration
                )
            )
        )

    print(
        f"Minimum segment duration: "
        f"{min_note_duration:.2f} seconds"
    )

    print(
        f"Minimum frames required: "
        f"{minimum_frames}"
    )

    for start, end in segments:

        segment_length = end - start

        if segment_length < minimum_frames:

            final_f0[start:end] = np.nan

            removed_segments += 1

    # ------------------------------------------------
    # Step 4: Final statistics
    # ------------------------------------------------

    final_valid = (
        np.isfinite(final_f0)
        & (final_f0 > 0)
    )

    total_frames = len(final_f0)

    final_frames = np.sum(
        final_valid
    )

    coverage = (
        final_frames /
        total_frames *
        100
        if total_frames > 0
        else 0
    )

    print("\nFinal melody statistics:")

    print(
        f"Total frames            : "
        f"{total_frames}"
    )

    print(
        f"Final melody frames     : "
        f"{final_frames}"
    )

    print(
        f"Final melody coverage   : "
        f"{coverage:.2f}%"
    )

    print(
        f"Short segments removed  : "
        f"{removed_segments}"
    )

    if final_frames > 0:

        print(
            f"Minimum melody F0       : "
            f"{np.nanmin(final_f0):.2f} Hz"
        )

        print(
            f"Maximum melody F0       : "
            f"{np.nanmax(final_f0):.2f} Hz"
        )

        print(
            f"Mean melody F0          : "
            f"{np.nanmean(final_f0):.2f} Hz"
        )

    # ------------------------------------------------
    # Step 5: Save final melody data
    # ------------------------------------------------

    os.makedirs(
        "data/output",
        exist_ok=True
    )

    os.makedirs(
        "graphs",
        exist_ok=True
    )

    output_file = (
        "data/output/final_melody_data.npz"
    )

    np.savez(
        output_file,
        f0=final_f0,
        times=times
    )

    print(
        "\nFinal melody data saved successfully!"
    )

    print(
        f"Saved file: {output_file}"
    )

    # ------------------------------------------------
    # Step 6: Create final melody graph
    # ------------------------------------------------

    plt.figure(
        figsize=(14, 5)
    )

    plt.plot(
        times,
        final_f0,
        linewidth=1
    )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Frequency (Hz)"
    )

    plt.title(
        "Phase 6.5 - Final Melody Contour"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    graph_path = (
        "graphs/final_melody_contour.png"
    )

    plt.savefig(
        graph_path,
        dpi=300
    )

    plt.close()

    print(
        "Final melody graph saved successfully!"
    )

    print(
        f"Saved graph: {graph_path}"
    )

    print("\n" + "=" * 60)
    print(
        "PHASE 6.5 COMPLETED SUCCESSFULLY"
    )
    print("=" * 60)

    return final_f0, times