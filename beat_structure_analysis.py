import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt


def analyze_beat_structure(audio, sample_rate):
    print("\n" + "=" * 70)
    print("PHASE 8.2 - BEAT / DOWNBEAT AND RHYTHMIC STRUCTURE")
    print("=" * 70)

    # =========================================================
    # STEP 1 - LOAD PHASE 8.1 BEAT DATA
    # =========================================================

    input_file = "data/output/rhythm_analysis.csv"

    if not os.path.exists(input_file):
        print("\nERROR:")
        print("Phase 8.1 rhythm_analysis.csv was not found.")
        print("Please run Phase 8.1 first.")
        return pd.DataFrame()

    beat_data = pd.read_csv(input_file)

    if len(beat_data) == 0:
        print("\nNo beat data available.")
        return pd.DataFrame()

    print("\nStep 1 - Beat data loaded successfully.")
    print(f"Total beat events loaded: {len(beat_data)}")

    # =========================================================
    # STEP 2 - EXTRACT BEAT TIMES
    # =========================================================

    beat_times = beat_data[
        "beat_time_seconds"
    ].astype(float).to_numpy()

    beat_numbers = np.arange(
        1,
        len(beat_times) + 1
    )

    # =========================================================
    # STEP 3 - CALCULATE BEAT INTERVALS
    # =========================================================

    if len(beat_times) >= 2:

        beat_intervals = np.diff(
            beat_times
        )

    else:

        beat_intervals = np.array([])

    print("\nStep 2 - Beat interval analysis completed.")

    # =========================================================
    # STEP 4 - LOCAL BPM
    # =========================================================

    local_bpm = np.full(
        len(beat_times),
        np.nan
    )

    if len(beat_intervals) > 0:

        local_bpm[1:] = (
            60.0 / beat_intervals
        )

    print("Step 3 - Local BPM calculation completed.")

    # =========================================================
    # STEP 5 - ONSET STRENGTH
    # =========================================================

    onset_envelope = librosa.onset.onset_strength(
        y=audio,
        sr=sample_rate
    )

    onset_times = librosa.times_like(
        onset_envelope,
        sr=sample_rate
    )

    print("Step 4 - Onset strength calculation completed.")

    # =========================================================
    # STEP 6 - BEAT STRENGTH
    # =========================================================

    beat_strength = np.zeros(
        len(beat_times)
    )

    for i, beat_time in enumerate(beat_times):

        index = np.argmin(
            np.abs(
                onset_times - beat_time
            )
        )

        beat_strength[i] = (
            onset_envelope[index]
        )

    # Normalize beat strength
    maximum_strength = np.max(
        beat_strength
    )

    if maximum_strength > 0:

        normalized_strength = (
            beat_strength /
            maximum_strength
        )

    else:

        normalized_strength = np.zeros(
            len(beat_strength)
        )

    print("Step 5 - Beat strength estimation completed.")

    # =========================================================
    # STEP 7 - LOCAL BEAT STRENGTH
    # =========================================================

    # Calculate a local median strength.
    # This helps identify unusually strong beats.

    if len(normalized_strength) >= 5:

        local_reference = np.zeros(
            len(normalized_strength)
        )

        half_window = 2

        for i in range(
            len(normalized_strength)
        ):

            start = max(
                0,
                i - half_window
            )

            end = min(
                len(normalized_strength),
                i + half_window + 1
            )

            local_reference[i] = np.median(
                normalized_strength[start:end]
            )

    else:

        local_reference = np.full(
            len(normalized_strength),
            np.median(
                normalized_strength
            )
            if len(normalized_strength) > 0
            else 0.0
        )

    # =========================================================
    # STEP 8 - DOWNBEAT CANDIDATE SCORE
    # =========================================================

    # A downbeat candidate is based on:
    #
    # 1. Relative beat strength
    # 2. Local strength compared with neighboring beats
    #
    # This is only a candidate detector.
    # It does NOT yet identify taal.

    downbeat_score = np.zeros(
        len(normalized_strength)
    )

    for i in range(
        len(normalized_strength)
    ):

        strength_component = (
            normalized_strength[i]
        )

        if local_reference[i] > 0:

            local_component = (
                normalized_strength[i] /
                local_reference[i]
            )

            local_component = min(
                local_component,
                2.0
            ) / 2.0

        else:

            local_component = (
                strength_component
            )

        downbeat_score[i] = (
            0.6 * strength_component +
            0.4 * local_component
        )

    # =========================================================
    # STEP 9 - SELECT DOWNBEAT CANDIDATES
    # =========================================================

    # Use a relatively conservative threshold.
    # We don't want every strong beat to become a downbeat.

    candidate_threshold = 0.70

    downbeat_candidate = (
        downbeat_score >= candidate_threshold
    )

    # Avoid marking adjacent beats as downbeats.
    # Keep the stronger candidate.

    for i in range(
        1,
        len(downbeat_candidate)
    ):

        if (
            downbeat_candidate[i]
            and downbeat_candidate[i - 1]
        ):

            if (
                downbeat_score[i]
                >
                downbeat_score[i - 1]
            ):

                downbeat_candidate[i - 1] = False

            else:

                downbeat_candidate[i] = False

    print(
        "Step 6 - Downbeat candidate detection completed."
    )

    # =========================================================
    # STEP 10 - BEAT POSITION WITHIN CANDIDATE CYCLE
    # =========================================================

    candidate_indices = np.where(
        downbeat_candidate
    )[0]

    beat_position = np.full(
        len(beat_times),
        np.nan
    )

    if len(candidate_indices) > 0:

        for i in range(
            len(beat_times)
        ):

            previous_candidates = (
                candidate_indices[
                    candidate_indices <= i
                ]
            )

            if len(previous_candidates) == 0:

                beat_position[i] = 1

            else:

                last_downbeat_index = (
                    previous_candidates[-1]
                )

                beat_position[i] = (
                    i -
                    last_downbeat_index +
                    1
                )

    # =========================================================
    # STEP 11 - CREATE STRUCTURE DATAFRAME
    # =========================================================

    structure_data = pd.DataFrame({
        "beat_number": beat_numbers,
        "beat_time_seconds": beat_times,
        "beat_interval_seconds": np.nan,
        "local_bpm": local_bpm,
        "beat_strength": beat_strength,
        "normalized_beat_strength": normalized_strength,
        "local_strength_reference": local_reference,
        "downbeat_score": downbeat_score,
        "downbeat_candidate": downbeat_candidate,
        "beat_position_from_downbeat": beat_position
    })

    if len(beat_intervals) > 0:

        structure_data.loc[
            1:,
            "beat_interval_seconds"
        ] = beat_intervals

    # =========================================================
    # STEP 12 - RHYTHMIC STATISTICS
    # =========================================================

    print("\n" + "-" * 65)
    print("BEAT STRUCTURE STATISTICS")
    print("-" * 65)

    print(
        f"Total beat events          : {len(beat_times)}"
    )

    if len(beat_intervals) > 0:

        print(
            f"Mean beat interval         : "
            f"{np.mean(beat_intervals):.4f} sec"
        )

        print(
            f"Median beat interval       : "
            f"{np.median(beat_intervals):.4f} sec"
        )

        print(
            f"Minimum beat interval      : "
            f"{np.min(beat_intervals):.4f} sec"
        )

        print(
            f"Maximum beat interval      : "
            f"{np.max(beat_intervals):.4f} sec"
        )

    valid_bpm = local_bpm[
        np.isfinite(local_bpm)
    ]

    if len(valid_bpm) > 0:

        print(
            f"Mean local BPM             : "
            f"{np.mean(valid_bpm):.2f}"
        )

        print(
            f"Median local BPM           : "
            f"{np.median(valid_bpm):.2f}"
        )

    total_downbeats = np.sum(
        downbeat_candidate
    )

    print(
        f"Downbeat candidates        : "
        f"{total_downbeats}"
    )

    if len(candidate_indices) > 1:

        candidate_gaps = np.diff(
            candidate_indices
        )

        print(
            f"Mean beats between candidates: "
            f"{np.mean(candidate_gaps):.2f}"
        )

        print(
            f"Median beats between candidates: "
            f"{np.median(candidate_gaps):.2f}"
        )

    # =========================================================
    # STEP 13 - POSSIBLE BEAT GROUPINGS
    # =========================================================

    grouping_counts = {}

    if len(candidate_indices) > 1:

        candidate_gaps = np.diff(
            candidate_indices
        )

        for gap in candidate_gaps:

            gap = int(gap)

            grouping_counts[gap] = (
                grouping_counts.get(
                    gap,
                    0
                ) + 1
            )

    print("\nCandidate beat grouping frequencies:")

    if grouping_counts:

        sorted_groupings = sorted(
            grouping_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        for group_size, count in sorted_groupings:

            print(
                f"{group_size:2d} beats : "
                f"{count} occurrences"
            )

    else:

        print(
            "Not enough downbeat candidates "
            "to estimate grouping."
        )

    # =========================================================
    # STEP 14 - SAVE DATA
    # =========================================================

    os.makedirs(
        "data/output",
        exist_ok=True
    )

    output_file = (
        "data/output/beat_structure_analysis.csv"
    )

    structure_data.to_csv(
        output_file,
        index=False
    )

    print(
        "\nBeat structure data saved successfully!"
    )

    print(
        f"Saved file: {output_file}"
    )

    # =========================================================
    # STEP 15 - SAVE GROUPING SUMMARY
    # =========================================================

    grouping_rows = []

    for group_size, count in sorted(
        grouping_counts.items()
    ):

        grouping_rows.append({
            "candidate_group_size_beats": group_size,
            "occurrence_count": count
        })

    grouping_data = pd.DataFrame(
        grouping_rows
    )

    grouping_file = (
        "data/output/"
        "candidate_beat_groupings.csv"
    )

    grouping_data.to_csv(
        grouping_file,
        index=False
    )

    print(
        "Beat grouping summary saved successfully!"
    )

    print(
        f"Saved file: {grouping_file}"
    )

    # =========================================================
    # STEP 16 - BEAT STRENGTH GRAPH
    # =========================================================

    os.makedirs(
        "graphs",
        exist_ok=True
    )

    plt.figure(
        figsize=(16, 6)
    )

    plt.plot(
        beat_times,
        normalized_strength,
        linewidth=1,
        label="Beat strength"
    )

    downbeat_times = beat_times[
        downbeat_candidate
    ]

    downbeat_strengths = normalized_strength[
        downbeat_candidate
    ]

    if len(downbeat_times) > 0:

        plt.scatter(
            downbeat_times,
            downbeat_strengths,
            s=35,
            label="Downbeat candidates"
        )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Normalized Beat Strength"
    )

    plt.title(
        "Phase 8.2 - Beat Strength and Downbeat Candidates"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    graph_path = (
        "graphs/"
        "beat_strength_downbeat_analysis.png"
    )

    plt.savefig(
        graph_path,
        dpi=300
    )

    plt.close()

    print(
        "\nBeat strength graph saved successfully!"
    )

    print(
        f"Saved graph: {graph_path}"
    )

    # =========================================================
    # STEP 17 - LOCAL BPM GRAPH
    # =========================================================

    if len(valid_bpm) > 0:

        plt.figure(
            figsize=(16, 5)
        )

        plt.plot(
            beat_times[1:],
            local_bpm[1:],
            linewidth=1
        )

        plt.xlabel(
            "Time (seconds)"
        )

        plt.ylabel(
            "Local BPM"
        )

        plt.title(
            "Phase 8.2 - Local Tempo Across Beats"
        )

        plt.grid(
            True,
            alpha=0.3
        )

        plt.tight_layout()

        bpm_graph = (
            "graphs/"
            "local_bpm_analysis.png"
        )

        plt.savefig(
            bpm_graph,
            dpi=300
        )

        plt.close()

        print(
            "Local BPM graph saved successfully!"
        )

        print(
            f"Saved graph: {bpm_graph}"
        )

    # =========================================================
    # STEP 18 - BEAT POSITION GRAPH
    # =========================================================

    valid_positions = np.isfinite(
        beat_position
    )

    if np.any(valid_positions):

        plt.figure(
            figsize=(16, 5)
        )

        plt.scatter(
            beat_times[valid_positions],
            beat_position[valid_positions],
            s=12
        )

        plt.xlabel(
            "Time (seconds)"
        )

        plt.ylabel(
            "Beat Position From Candidate Downbeat"
        )

        plt.title(
            "Phase 8.2 - Candidate Beat Grouping Structure"
        )

        plt.grid(
            True,
            alpha=0.3
        )

        plt.tight_layout()

        position_graph = (
            "graphs/"
            "beat_grouping_structure.png"
        )

        plt.savefig(
            position_graph,
            dpi=300
        )

        plt.close()

        print(
            "Beat grouping graph saved successfully!"
        )

        print(
            f"Saved graph: {position_graph}"
        )

    # =========================================================
    # FINAL
    # =========================================================

    print("\n" + "=" * 70)
    print(
        "PHASE 8.2 COMPLETED SUCCESSFULLY"
    )
    print("=" * 70)

    return structure_data