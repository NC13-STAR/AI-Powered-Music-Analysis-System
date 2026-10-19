import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def classify_duration(duration):
    """
    Descriptive classification of detected note duration.
    These categories are not yet musical rhythmic values.
    """

    if duration < 0.25:
        return "Very Short"

    elif duration < 0.50:
        return "Short"

    elif duration < 1.00:
        return "Medium"

    elif duration < 2.00:
        return "Long"

    else:
        return "Very Long"


def calculate_note_durations(note_segments):

    if note_segments is None or len(note_segments) == 0:

        print("\nNo note segments available.")

        return pd.DataFrame()


    data = note_segments.copy()

    print(
        f"\nNumber of note events received: {len(data)}"
    )


    # ========================================================
    # PHASE 7.1 COLUMN NAMES
    # ========================================================

    start_times = (
        data["start_time_seconds"]
        .astype(float)
        .to_numpy()
    )

    end_times = (
        data["end_time_seconds"]
        .astype(float)
        .to_numpy()
    )


    # ========================================================
    # NOTE DURATION
    # ========================================================

    durations = (
        end_times - start_times
    )

    durations = np.maximum(
        durations,
        0.0
    )

    data["note_duration_seconds"] = durations


    # ========================================================
    # NOTE CENTER TIME
    # ========================================================

    data["note_center_time"] = (
        start_times + end_times
    ) / 2.0


    # ========================================================
    # GAP BEFORE NOTE
    # ========================================================

    gap_before = np.zeros(
        len(data)
    )

    if len(data) > 1:

        gap_before[1:] = (
            start_times[1:]
            - end_times[:-1]
        )

    gap_before = np.maximum(
        gap_before,
        0.0
    )

    data["gap_before_seconds"] = gap_before


    # ========================================================
    # GAP AFTER NOTE
    # ========================================================

    gap_after = np.zeros(
        len(data)
    )

    if len(data) > 1:

        gap_after[:-1] = (
            start_times[1:]
            - end_times[:-1]
        )

    gap_after = np.maximum(
        gap_after,
        0.0
    )

    data["gap_after_seconds"] = gap_after


    # ========================================================
    # DURATION CATEGORY
    # ========================================================

    data["duration_category"] = [
        classify_duration(duration)
        for duration in durations
    ]


    return data


def print_duration_statistics(data):

    print("\n" + "-" * 60)
    print("NOTE DURATION STATISTICS")
    print("-" * 60)


    if len(data) == 0:

        print("No note events available.")

        return


    durations = (
        data["note_duration_seconds"]
        .to_numpy()
    )

    gaps = (
        data["gap_before_seconds"]
        .to_numpy()
    )


    print(
        f"Total note events       : {len(data)}"
    )

    print(
        f"Minimum duration        : "
        f"{np.min(durations):.3f} sec"
    )

    print(
        f"Maximum duration        : "
        f"{np.max(durations):.3f} sec"
    )

    print(
        f"Mean duration           : "
        f"{np.mean(durations):.3f} sec"
    )

    print(
        f"Median duration         : "
        f"{np.median(durations):.3f} sec"
    )

    print(
        f"Total detected melody   : "
        f"{np.sum(durations):.3f} sec"
    )

    print(
        f"Total detected gaps     : "
        f"{np.sum(gaps):.3f} sec"
    )


    print("\nDuration categories:")


    category_counts = (
        data["duration_category"]
        .value_counts()
    )


    category_order = [
        "Very Short",
        "Short",
        "Medium",
        "Long",
        "Very Long"
    ]


    for category in category_order:

        count = category_counts.get(
            category,
            0
        )

        print(
            f"{category:12s}: {count}"
        )


def save_duration_data(data):

    os.makedirs(
        "data/output",
        exist_ok=True
    )


    output_file = (
        "data/output/note_duration_analysis.csv"
    )


    data.to_csv(
        output_file,
        index=False
    )


    print(
        "\nNote duration analysis saved successfully!"
    )

    print(
        f"Saved file: {output_file}"
    )


    return output_file


def create_duration_graph(data):

    os.makedirs(
        "graphs",
        exist_ok=True
    )


    if len(data) == 0:
        return


    note_numbers = np.arange(
        1,
        len(data) + 1
    )


    durations = (
        data["note_duration_seconds"]
        .to_numpy()
    )


    plt.figure(
        figsize=(16, 6)
    )


    plt.bar(
        note_numbers,
        durations
    )


    plt.xlabel(
        "Note Event Number"
    )

    plt.ylabel(
        "Duration (seconds)"
    )


    plt.title(
        "Phase 7.3 - Musical Note Duration Analysis"
    )


    plt.grid(
        True,
        axis="y",
        alpha=0.3
    )


    plt.tight_layout()


    graph_path = (
        "graphs/note_duration_analysis.png"
    )


    plt.savefig(
        graph_path,
        dpi=300
    )


    plt.close()


    print(
        "\nNote duration graph saved successfully!"
    )

    print(
        f"Saved graph: {graph_path}"
    )


def create_timing_graph(data):

    os.makedirs(
        "graphs",
        exist_ok=True
    )


    if len(data) == 0:
        return


    start_times = (
        data["start_time_seconds"]
        .astype(float)
        .to_numpy()
    )

    end_times = (
        data["end_time_seconds"]
        .astype(float)
        .to_numpy()
    )


    note_numbers = np.arange(
        1,
        len(data) + 1
    )


    plt.figure(
        figsize=(16, 7)
    )


    plt.hlines(
        note_numbers,
        start_times,
        end_times,
        linewidth=5
    )


    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Note Event Number"
    )


    plt.title(
        "Phase 7.3 - Note Timing and Duration"
    )


    plt.grid(
        True,
        axis="x",
        alpha=0.3
    )


    plt.tight_layout()


    graph_path = (
        "graphs/note_timing_duration.png"
    )


    plt.savefig(
        graph_path,
        dpi=300
    )


    plt.close()


    print(
        "Note timing graph saved successfully!"
    )

    print(
        f"Saved graph: {graph_path}"
    )


def create_duration_distribution(data):

    os.makedirs(
        "graphs",
        exist_ok=True
    )


    if len(data) == 0:
        return


    durations = (
        data["note_duration_seconds"]
        .to_numpy()
    )


    plt.figure(
        figsize=(10, 6)
    )


    plt.hist(
        durations,
        bins=20
    )


    plt.xlabel(
        "Note Duration (seconds)"
    )

    plt.ylabel(
        "Number of Note Events"
    )


    plt.title(
        "Phase 7.3 - Distribution of Note Durations"
    )


    plt.grid(
        True,
        axis="y",
        alpha=0.3
    )


    plt.tight_layout()


    graph_path = (
        "graphs/note_duration_distribution.png"
    )


    plt.savefig(
        graph_path,
        dpi=300
    )


    plt.close()


    print(
        "Duration distribution graph saved successfully!"
    )

    print(
        f"Saved graph: {graph_path}"
    )


def analyze_note_durations(note_segments):

    print("\n" + "=" * 65)
    print("PHASE 7.3 - NOTE DURATION ANALYSIS")
    print("=" * 65)


    print(
        "\nAnalyzing timing and duration of detected notes..."
    )


    data = calculate_note_durations(
        note_segments
    )


    if len(data) == 0:

        print(
            "\nPhase 7.3 could not continue because "
            "no note events were available."
        )

        return data


    print_duration_statistics(
        data
    )


    save_duration_data(
        data
    )


    create_duration_graph(
        data
    )


    create_timing_graph(
        data
    )


    create_duration_distribution(
        data
    )


    print("\n" + "=" * 65)
    print("PHASE 7.3 COMPLETED SUCCESSFULLY")
    print("=" * 65)


    return data