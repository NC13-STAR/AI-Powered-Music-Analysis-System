import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def analyze_tempo_stability():

    print("\n" + "=" * 70)
    print("PHASE 8.3 - TEMPO STABILITY ANALYSIS")
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

    beat_data = pd.read_csv(
        input_file
    )

    if len(beat_data) < 2:

        print("\nNot enough beat data for tempo stability analysis.")

        return pd.DataFrame()

    print("\nStep 1 - Rhythm beat data loaded successfully.")

    print(
        f"Total beat events loaded: {len(beat_data)}"
    )

    # =========================================================
    # STEP 2 - EXTRACT BEAT TIMES
    # =========================================================

    beat_times = beat_data[
        "beat_time_seconds"
    ].astype(float).to_numpy()

    print(
        "Step 2 - Beat timing data extracted."
    )

    # =========================================================
    # STEP 3 - CALCULATE BEAT INTERVALS
    # =========================================================

    beat_intervals = np.diff(
        beat_times
    )

    valid_intervals = (
        np.isfinite(beat_intervals)
        &
        (beat_intervals > 0)
    )

    beat_intervals = beat_intervals[
        valid_intervals
    ]

    interval_times = beat_times[1:][
        valid_intervals
    ]

    print(
        "Step 3 - Beat intervals calculated."
    )

    # =========================================================
    # STEP 4 - CALCULATE LOCAL BPM
    # =========================================================

    local_bpm = (
        60.0 /
        beat_intervals
    )

    print(
        "Step 4 - Local BPM calculated."
    )

    # =========================================================
    # STEP 5 - REMOVE EXTREME OUTLIERS
    # =========================================================

    # Use percentile limits only for stability statistics.
    # Original beat timings remain unchanged.

    lower_limit = np.percentile(
        local_bpm,
        2
    )

    upper_limit = np.percentile(
        local_bpm,
        98
    )

    stable_mask = (
        (local_bpm >= lower_limit)
        &
        (local_bpm <= upper_limit)
    )

    stable_bpm = local_bpm[
        stable_mask
    ]

    stable_times = interval_times[
        stable_mask
    ]

    print(
        "Step 5 - Extreme tempo outliers identified."
    )

    print(
        f"2nd percentile BPM  : {lower_limit:.2f}"
    )

    print(
        f"98th percentile BPM : {upper_limit:.2f}"
    )

    # =========================================================
    # STEP 6 - MOVING TEMPO ANALYSIS
    # =========================================================

    window_size = 20

    if len(stable_bpm) >= window_size:

        moving_mean = np.convolve(
            stable_bpm,
            np.ones(window_size) /
            window_size,
            mode="valid"
        )

        moving_std = np.zeros(
            len(moving_mean)
        )

        moving_times = stable_times[
            window_size - 1:
        ]

        for i in range(
            len(moving_mean)
        ):

            window = stable_bpm[
                i:i + window_size
            ]

            moving_std[i] = np.std(
                window
            )

    else:

        moving_mean = np.array(
            [np.mean(stable_bpm)]
        )

        moving_std = np.array(
            [np.std(stable_bpm)]
        )

        moving_times = np.array(
            [np.mean(stable_times)]
        )

    print(
        "Step 6 - Moving tempo analysis completed."
    )

    # =========================================================
    # STEP 7 - TEMPO STATISTICS
    # =========================================================

    mean_bpm = np.mean(
        stable_bpm
    )

    median_bpm = np.median(
        stable_bpm
    )

    std_bpm = np.std(
        stable_bpm
    )

    minimum_bpm = np.min(
        stable_bpm
    )

    maximum_bpm = np.max(
        stable_bpm
    )

    coefficient_variation = (
        std_bpm /
        mean_bpm
        * 100
        if mean_bpm > 0
        else 0
    )

    # =========================================================
    # STEP 8 - TEMPO CHANGE
    # =========================================================

    first_part_count = max(
        1,
        len(stable_bpm) // 3
    )

    first_part = stable_bpm[
        :first_part_count
    ]

    last_part = stable_bpm[
        -first_part_count:
    ]

    first_mean = np.mean(
        first_part
    )

    last_mean = np.mean(
        last_part
    )

    overall_tempo_change = (
        (
            last_mean -
            first_mean
        )
        /
        first_mean
        * 100
        if first_mean > 0
        else 0
    )

    # =========================================================
    # STEP 9 - TEMPO STABILITY SCORE
    # =========================================================

    # Internal analytical score.
    # This is NOT an externally validated accuracy measure.

    stability_score = max(
        0.0,
        100.0 -
        (
            coefficient_variation
            * 5.0
        )
    )

    stability_score = min(
        stability_score,
        100.0
    )

    # =========================================================
    # STEP 10 - CLASSIFICATION
    # =========================================================

    if coefficient_variation < 2:

        stability_category = (
            "Highly Stable"
        )

    elif coefficient_variation < 5:

        stability_category = (
            "Stable"
        )

    elif coefficient_variation < 10:

        stability_category = (
            "Moderately Stable"
        )

    else:

        stability_category = (
            "Variable"
        )

    # =========================================================
    # STEP 11 - PRINT RESULTS
    # =========================================================

    print("\n" + "-" * 65)
    print("TEMPO STABILITY RESULTS")
    print("-" * 65)

    print(
        f"Mean stable BPM             : "
        f"{mean_bpm:.2f}"
    )

    print(
        f"Median stable BPM           : "
        f"{median_bpm:.2f}"
    )

    print(
        f"Minimum stable BPM          : "
        f"{minimum_bpm:.2f}"
    )

    print(
        f"Maximum stable BPM          : "
        f"{maximum_bpm:.2f}"
    )

    print(
        f"BPM standard deviation      : "
        f"{std_bpm:.2f}"
    )

    print(
        f"Tempo coefficient variation : "
        f"{coefficient_variation:.2f}%"
    )

    print(
        f"First-section mean BPM      : "
        f"{first_mean:.2f}"
    )

    print(
        f"Last-section mean BPM       : "
        f"{last_mean:.2f}"
    )

    print(
        f"Overall tempo change        : "
        f"{overall_tempo_change:.2f}%"
    )

    print(
        f"Tempo stability score       : "
        f"{stability_score:.2f}%"
    )

    print(
        f"Tempo stability category    : "
        f"{stability_category}"
    )

    # =========================================================
    # STEP 12 - CREATE DETAILED DATAFRAME
    # =========================================================

    stability_data = pd.DataFrame({
        "time_seconds": stable_times,
        "local_bpm": stable_bpm
    })

    # =========================================================
    # STEP 13 - SAVE DETAILED DATA
    # =========================================================

    os.makedirs(
        "data/output",
        exist_ok=True
    )

    detailed_file = (
        "data/output/"
        "tempo_stability_analysis.csv"
    )

    stability_data.to_csv(
        detailed_file,
        index=False
    )

    print(
        "\nTempo stability data saved successfully!"
    )

    print(
        f"Saved file: {detailed_file}"
    )

    # =========================================================
    # STEP 14 - SAVE SUMMARY
    # =========================================================

    summary_data = pd.DataFrame([{
        "mean_bpm": mean_bpm,
        "median_bpm": median_bpm,
        "minimum_bpm": minimum_bpm,
        "maximum_bpm": maximum_bpm,
        "bpm_standard_deviation": std_bpm,
        "tempo_coefficient_variation_percent":
            coefficient_variation,
        "first_section_mean_bpm":
            first_mean,
        "last_section_mean_bpm":
            last_mean,
        "overall_tempo_change_percent":
            overall_tempo_change,
        "tempo_stability_score":
            stability_score,
        "tempo_stability_category":
            stability_category
    }])

    summary_file = (
        "data/output/"
        "tempo_stability_summary.csv"
    )

    summary_data.to_csv(
        summary_file,
        index=False
    )

    print(
        "Tempo stability summary saved successfully!"
    )

    print(
        f"Saved file: {summary_file}"
    )

    # =========================================================
    # STEP 15 - LOCAL BPM GRAPH
    # =========================================================

    os.makedirs(
        "graphs",
        exist_ok=True
    )

    plt.figure(
        figsize=(16, 6)
    )

    plt.plot(
        stable_times,
        stable_bpm,
        linewidth=1,
        label="Local BPM"
    )

    plt.axhline(
        mean_bpm,
        linewidth=1.5,
        label="Mean BPM"
    )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "BPM"
    )

    plt.title(
        "Phase 8.3 - Tempo Stability Over Time"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    graph_path = (
        "graphs/"
        "tempo_stability.png"
    )

    plt.savefig(
        graph_path,
        dpi=300
    )

    plt.close()

    print(
        "\nTempo stability graph saved successfully!"
    )

    print(
        f"Saved graph: {graph_path}"
    )

    # =========================================================
    # STEP 16 - MOVING TEMPO GRAPH
    # =========================================================

    plt.figure(
        figsize=(16, 6)
    )

    plt.plot(
        moving_times,
        moving_mean,
        linewidth=1.5,
        label="20-beat moving mean"
    )

    plt.fill_between(
        moving_times,
        moving_mean - moving_std,
        moving_mean + moving_std,
        alpha=0.2,
        label="Local variation"
    )

    plt.axhline(
        mean_bpm,
        linewidth=1,
        label="Overall mean BPM"
    )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "BPM"
    )

    plt.title(
        "Phase 8.3 - Moving Tempo Stability"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    moving_graph = (
        "graphs/"
        "moving_tempo_stability.png"
    )

    plt.savefig(
        moving_graph,
        dpi=300
    )

    plt.close()

    print(
        "Moving tempo graph saved successfully!"
    )

    print(
        f"Saved graph: {moving_graph}"
    )

    # =========================================================
    # STEP 17 - TEMPO DISTRIBUTION
    # =========================================================

    plt.figure(
        figsize=(10, 6)
    )

    plt.hist(
        stable_bpm,
        bins=20
    )

    plt.axvline(
        mean_bpm,
        linewidth=1.5,
        label="Mean BPM"
    )

    plt.xlabel(
        "Local BPM"
    )

    plt.ylabel(
        "Number of Beat Intervals"
    )

    plt.title(
        "Phase 8.3 - Local Tempo Distribution"
    )

    plt.legend()

    plt.grid(
        True,
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    distribution_graph = (
        "graphs/"
        "tempo_distribution.png"
    )

    plt.savefig(
        distribution_graph,
        dpi=300
    )

    plt.close()

    print(
        "Tempo distribution graph saved successfully!"
    )

    print(
        f"Saved graph: {distribution_graph}"
    )

    # =========================================================
    # FINAL
    # =========================================================

    print("\n" + "=" * 70)
    print(
        "PHASE 8.3 COMPLETED SUCCESSFULLY"
    )
    print("=" * 70)

    return stability_data


if __name__ == "__main__":

    analyze_tempo_stability()