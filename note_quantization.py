import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


NOTE_NAMES = [
    "C", "C#", "D", "D#", "E", "F",
    "F#", "G", "G#", "A", "A#", "B"
]


def midi_to_note(midi_number):
    """
    Convert an integer MIDI number to a Western note name.
    Example:
    60 -> C4
    61 -> C#4
    69 -> A4
    """

    if not np.isfinite(midi_number):
        return None

    midi_number = int(round(midi_number))

    note_index = midi_number % 12
    octave = (midi_number // 12) - 1

    return f"{NOTE_NAMES[note_index]}{octave}"


def calculate_quantization(note_segments):
    """
    Quantize each segmented note to the nearest equal-tempered semitone.

    The original detected pitch is preserved.
    Quantized MIDI and cents deviation are added.
    """

    if note_segments is None or len(note_segments) == 0:
        print("\nNo note segments available for quantization.")
        return pd.DataFrame()

    data = note_segments.copy()

    print("\nNumber of input note segments:", len(data))

    # ---------------------------------------------------------
    # Use median MIDI pitch from Phase 7.1
    # ---------------------------------------------------------

    if "median_midi" in data.columns:
        detected_midi = data["median_midi"].astype(float).to_numpy()

    elif "mean_midi" in data.columns:
        detected_midi = data["mean_midi"].astype(float).to_numpy()

    else:
        raise ValueError(
            "Phase 7.1 output does not contain median_midi or mean_midi."
        )

    # ---------------------------------------------------------
    # Quantize to nearest semitone
    # ---------------------------------------------------------

    quantized_midi = np.round(detected_midi)

    # ---------------------------------------------------------
    # Calculate cents deviation
    #
    # 100 cents = one semitone
    # ---------------------------------------------------------

    cents_deviation = (
        detected_midi - quantized_midi
    ) * 100.0

    # ---------------------------------------------------------
    # Convert quantized MIDI to Western note
    # ---------------------------------------------------------

    quantized_notes = [
        midi_to_note(midi)
        for midi in quantized_midi
    ]

    # ---------------------------------------------------------
    # Calculate absolute cents deviation
    # ---------------------------------------------------------

    absolute_cents = np.abs(cents_deviation)

    # ---------------------------------------------------------
    # Add results to dataframe
    # ---------------------------------------------------------

    data["detected_midi"] = detected_midi
    data["quantized_midi"] = quantized_midi
    data["quantized_note"] = quantized_notes
    data["cents_deviation"] = cents_deviation
    data["absolute_cents_deviation"] = absolute_cents

    return data


def print_quantization_statistics(data):
    """
    Print useful Phase 7.2 statistics.
    """

    print("\n" + "-" * 60)
    print("QUANTIZATION STATISTICS")
    print("-" * 60)

    total_notes = len(data)

    if total_notes == 0:
        print("No notes available.")
        return

    mean_abs_error = data["absolute_cents_deviation"].mean()
    median_abs_error = data["absolute_cents_deviation"].median()
    maximum_abs_error = data["absolute_cents_deviation"].max()

    within_25 = np.sum(
        data["absolute_cents_deviation"] <= 25
    )

    within_50 = np.sum(
        data["absolute_cents_deviation"] <= 50
    )

    print(f"Total note events          : {total_notes}")
    print(f"Mean absolute deviation    : {mean_abs_error:.2f} cents")
    print(f"Median absolute deviation  : {median_abs_error:.2f} cents")
    print(f"Maximum absolute deviation : {maximum_abs_error:.2f} cents")

    print(
        f"Notes within ±25 cents    : "
        f"{within_25} / {total_notes}"
    )

    print(
        f"Notes within ±50 cents    : "
        f"{within_50} / {total_notes}"
    )

    print("\nQuantized Western notes:")

    note_counts = data["quantized_note"].value_counts()

    for note, count in note_counts.items():
        print(f"{note:5s} : {count}")


def save_quantized_data(data):
    """
    Save Phase 7.2 results to CSV.
    """

    os.makedirs("data/output", exist_ok=True)

    output_file = "data/output/quantized_notes.csv"

    data.to_csv(
        output_file,
        index=False
    )

    print("\nQuantized note data saved successfully!")
    print(f"Saved file: {output_file}")

    return output_file


def create_quantization_graph(data):
    """
    Create a graph showing:
    - detected continuous MIDI pitch
    - quantized MIDI pitch
    """

    os.makedirs("graphs", exist_ok=True)

    if len(data) == 0:
        print("\nNo data available for quantization graph.")
        return

    # ---------------------------------------------------------
    # Calculate note center times
    # ---------------------------------------------------------

    if "start_time" in data.columns and "end_time" in data.columns:

        start_times = data["start_time"].astype(float).to_numpy()
        end_times = data["end_time"].astype(float).to_numpy()

        center_times = (
            start_times + end_times
        ) / 2.0

    elif "start_seconds" in data.columns and "end_seconds" in data.columns:

        start_times = data["start_seconds"].astype(float).to_numpy()
        end_times = data["end_seconds"].astype(float).to_numpy()

        center_times = (
            start_times + end_times
        ) / 2.0

    else:
        center_times = np.arange(len(data))

    detected_midi = data["detected_midi"].to_numpy()
    quantized_midi = data["quantized_midi"].to_numpy()

    # ---------------------------------------------------------
    # Plot detected pitch
    # ---------------------------------------------------------

    plt.figure(figsize=(16, 6))

    plt.scatter(
        center_times,
        detected_midi,
        s=18,
        label="Detected pitch"
    )

    # ---------------------------------------------------------
    # Plot quantized pitch
    # ---------------------------------------------------------

    plt.step(
        center_times,
        quantized_midi,
        where="mid",
        linewidth=1.5,
        label="Quantized pitch"
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("MIDI Pitch Number")

    plt.title(
        "Phase 7.2 - Note Stabilization / Quantization"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    graph_path = (
        "graphs/note_quantization.png"
    )

    plt.savefig(
        graph_path,
        dpi=300
    )

    plt.close()

    print("\nQuantization graph saved successfully!")
    print(f"Saved graph: {graph_path}")


def create_cents_graph(data):
    """
    Create a graph showing pitch deviation from
    the nearest equal-tempered semitone.
    """

    os.makedirs("graphs", exist_ok=True)

    if len(data) == 0:
        return

    if "start_time" in data.columns and "end_time" in data.columns:

        start_times = data["start_time"].astype(float).to_numpy()
        end_times = data["end_time"].astype(float).to_numpy()

        center_times = (
            start_times + end_times
        ) / 2.0

    elif "start_seconds" in data.columns and "end_seconds" in data.columns:

        start_times = data["start_seconds"].astype(float).to_numpy()
        end_times = data["end_seconds"].astype(float).to_numpy()

        center_times = (
            start_times + end_times
        ) / 2.0

    else:
        center_times = np.arange(len(data))

    cents = data["cents_deviation"].to_numpy()

    plt.figure(figsize=(16, 5))

    plt.axhline(
        0,
        linewidth=1
    )

    plt.scatter(
        center_times,
        cents,
        s=18
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Pitch Deviation (cents)")

    plt.title(
        "Phase 7.2 - Pitch Deviation from Nearest Semitone"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    graph_path = (
        "graphs/note_pitch_deviation.png"
    )

    plt.savefig(
        graph_path,
        dpi=300
    )

    plt.close()

    print(
        "Pitch deviation graph saved successfully!"
    )

    print(
        f"Saved graph: {graph_path}"
    )


def quantize_notes(note_segments):
    """
    Main Phase 7.2 function.
    """

    print("\n" + "=" * 65)
    print("PHASE 7.2 - NOTE STABILIZATION / QUANTIZATION")
    print("=" * 65)

    print(
        "\nStabilizing segmented musical notes..."
    )

    print(
        "Mapping each detected note to the nearest "
        "equal-tempered semitone..."
    )

    data = calculate_quantization(
        note_segments
    )

    if len(data) == 0:
        return data

    print_quantization_statistics(
        data
    )

    save_quantized_data(
        data
    )

    create_quantization_graph(
        data
    )

    create_cents_graph(
        data
    )

    print("\n" + "=" * 65)
    print("PHASE 7.2 COMPLETED SUCCESSFULLY")
    print("=" * 65)

    return data