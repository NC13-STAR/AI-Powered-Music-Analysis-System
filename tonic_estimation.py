import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


NOTE_NAMES = [
    "C", "C#", "D", "D#", "E", "F",
    "F#", "G", "G#", "A", "A#", "B"
]


# ------------------------------------------------------------
# BASIC NOTE UTILITIES
# ------------------------------------------------------------

def pitch_class_to_name(pitch_class):
    return NOTE_NAMES[int(pitch_class) % 12]


def midi_to_note_name(midi_number):
    midi_number = int(round(midi_number))
    note_index = midi_number % 12
    octave = (midi_number // 12) - 1
    return f"{NOTE_NAMES[note_index]}{octave}"


# ------------------------------------------------------------
# EXTRACT NOTE INFORMATION
# ------------------------------------------------------------

def prepare_note_data(note_data):

    if note_data is None or len(note_data) == 0:
        return pd.DataFrame()

    data = note_data.copy()

    # --------------------------------------------------------
    # Find MIDI column
    # --------------------------------------------------------

    if "quantized_midi" in data.columns:
        data["analysis_midi"] = pd.to_numeric(
            data["quantized_midi"],
            errors="coerce"
        )

    elif "rounded_midi" in data.columns:
        data["analysis_midi"] = pd.to_numeric(
            data["rounded_midi"],
            errors="coerce"
        )

    elif "mean_midi" in data.columns:
        data["analysis_midi"] = pd.to_numeric(
            data["mean_midi"],
            errors="coerce"
        )

    else:
        raise ValueError(
            "No MIDI information found in note data."
        )

    # --------------------------------------------------------
    # Find duration column
    # --------------------------------------------------------

    if "note_duration_seconds" in data.columns:
        data["analysis_duration"] = pd.to_numeric(
            data["note_duration_seconds"],
            errors="coerce"
        )

    elif "duration_seconds" in data.columns:
        data["analysis_duration"] = pd.to_numeric(
            data["duration_seconds"],
            errors="coerce"
        )

    else:
        data["analysis_duration"] = 0.1

    # --------------------------------------------------------
    # Find start time
    # --------------------------------------------------------

    if "start_time_seconds" in data.columns:
        data["analysis_start"] = pd.to_numeric(
            data["start_time_seconds"],
            errors="coerce"
        )

    elif "start_time" in data.columns:
        data["analysis_start"] = pd.to_numeric(
            data["start_time"],
            errors="coerce"
        )

    else:
        data["analysis_start"] = np.arange(len(data))

    # --------------------------------------------------------
    # Find end time
    # --------------------------------------------------------

    if "end_time_seconds" in data.columns:
        data["analysis_end"] = pd.to_numeric(
            data["end_time_seconds"],
            errors="coerce"
        )

    elif "end_time" in data.columns:
        data["analysis_end"] = pd.to_numeric(
            data["end_time"],
            errors="coerce"
        )

    else:
        data["analysis_end"] = (
            data["analysis_start"] +
            data["analysis_duration"]
        )

    # --------------------------------------------------------
    # Clean invalid rows
    # --------------------------------------------------------

    data = data[
        np.isfinite(data["analysis_midi"]) &
        np.isfinite(data["analysis_duration"]) &
        (data["analysis_duration"] > 0)
    ].copy()

    data["pitch_class"] = (
        data["analysis_midi"].round().astype(int) % 12
    )

    return data


# ------------------------------------------------------------
# PITCH CLASS HISTOGRAM
# ------------------------------------------------------------

def calculate_duration_weighted_histogram(data):

    histogram = np.zeros(12, dtype=float)

    for _, row in data.iterrows():

        pitch_class = int(row["pitch_class"])
        duration = float(row["analysis_duration"])

        histogram[pitch_class] += duration

    return histogram


# ------------------------------------------------------------
# LONG NOTE / SUSTAINED NOTE EVIDENCE
# ------------------------------------------------------------

def calculate_sustained_note_histogram(data):

    histogram = np.zeros(12, dtype=float)

    if len(data) == 0:
        return histogram

    durations = data["analysis_duration"].to_numpy()

    # 75th percentile identifies relatively long notes
    duration_threshold = np.percentile(
        durations,
        75
    )

    for _, row in data.iterrows():

        duration = float(row["analysis_duration"])

        if duration >= duration_threshold:

            pitch_class = int(row["pitch_class"])

            # Extra evidence for sustained notes
            histogram[pitch_class] += duration

    return histogram


# ------------------------------------------------------------
# ENDING EVIDENCE
# ------------------------------------------------------------

def calculate_ending_histogram(data):

    histogram = np.zeros(12, dtype=float)

    if len(data) == 0:
        return histogram

    # Sort chronologically
    data = data.sort_values("analysis_start")

    # Use last 15% of detected note events
    number_of_notes = len(data)

    ending_count = max(
        3,
        int(np.ceil(number_of_notes * 0.15))
    )

    ending_notes = data.tail(ending_count)

    for _, row in ending_notes.iterrows():

        pitch_class = int(row["pitch_class"])
        duration = float(row["analysis_duration"])

        histogram[pitch_class] += duration

    return histogram


# ------------------------------------------------------------
# NORMALIZATION
# ------------------------------------------------------------

def normalize(values):

    values = np.asarray(values, dtype=float)

    total = np.sum(values)

    if total <= 0:
        return np.zeros_like(values)

    return values / total


# ------------------------------------------------------------
# TONIC CANDIDATE SCORING
# ------------------------------------------------------------

def calculate_tonic_candidate_scores(
    histogram,
    sustained_histogram,
    ending_histogram
):

    histogram_norm = normalize(histogram)

    sustained_norm = normalize(
        sustained_histogram
    )

    ending_norm = normalize(
        ending_histogram
    )

    scores = np.zeros(12)

    # --------------------------------------------------------
    # Evaluate all 12 possible tonic candidates
    # --------------------------------------------------------

    for tonic in range(12):

        # Sa
        sa_strength = histogram_norm[tonic]

        # Pa = tonic + 7 semitones
        pa = (tonic + 7) % 12
        pa_strength = histogram_norm[pa]

        # Ma = tonic + 5 semitones
        ma = (tonic + 5) % 12
        ma_strength = histogram_norm[ma]

        # Sustained Sa
        sustained_sa = sustained_norm[tonic]

        # Ending Sa
        ending_sa = ending_norm[tonic]

        # ----------------------------------------------------
        # Candidate score
        # ----------------------------------------------------

        score = (
            0.40 * sa_strength
            + 0.20 * pa_strength
            + 0.10 * ma_strength
            + 0.20 * sustained_sa
            + 0.10 * ending_sa
        )

        scores[tonic] = score

    return scores


# ------------------------------------------------------------
# TONIC MIDI ESTIMATION
# ------------------------------------------------------------

def estimate_tonic_midi(data, tonic_pitch_class):

    candidates = data[
        data["pitch_class"] == tonic_pitch_class
    ]

    if len(candidates) == 0:

        # Fallback around middle C
        return int(
            tonic_pitch_class + 60
        )

    midi_values = candidates[
        "analysis_midi"
    ].to_numpy()

    durations = candidates[
        "analysis_duration"
    ].to_numpy()

    # Duration-weighted median
    order = np.argsort(midi_values)

    sorted_midi = midi_values[order]
    sorted_duration = durations[order]

    cumulative = np.cumsum(
        sorted_duration
    )

    midpoint = np.sum(
        sorted_duration
    ) / 2.0

    index = np.searchsorted(
        cumulative,
        midpoint
    )

    index = min(
        index,
        len(sorted_midi) - 1
    )

    return int(
        round(sorted_midi[index])
    )


# ------------------------------------------------------------
# MAIN TONIC ESTIMATION
# ------------------------------------------------------------

def estimate_tonic(note_data):

    print("\n" + "=" * 70)
    print("PHASE 7.4 - IMPROVED TONIC / SA ESTIMATION")
    print("=" * 70)

    print(
        "\nUsing multiple musical evidence sources:"
    )

    print("1. Duration-weighted pitch occurrence")
    print("2. Sustained-note evidence")
    print("3. Sa prominence")
    print("4. Pa relationship")
    print("5. Ma relationship")
    print("6. Ending-note evidence")
    print("7. Evaluation of all 12 tonic candidates")

    # --------------------------------------------------------
    # Prepare data
    # --------------------------------------------------------

    data = prepare_note_data(note_data)

    if len(data) == 0:

        raise ValueError(
            "Unable to estimate tonic because "
            "no valid note data was available."
        )

    print(
        f"\nValid note events used for tonic analysis: {len(data)}"
    )

    # --------------------------------------------------------
    # Calculate evidence
    # --------------------------------------------------------

    histogram = calculate_duration_weighted_histogram(
        data
    )

    sustained_histogram = calculate_sustained_note_histogram(
        data
    )

    ending_histogram = calculate_ending_histogram(
        data
    )

    # --------------------------------------------------------
    # Calculate candidate scores
    # --------------------------------------------------------

    scores = calculate_tonic_candidate_scores(
        histogram,
        sustained_histogram,
        ending_histogram
    )

    # --------------------------------------------------------
    # Rank candidates
    # --------------------------------------------------------

    ranking = np.argsort(
        scores
    )[::-1]

    best_pitch_class = int(
        ranking[0]
    )

    second_pitch_class = int(
        ranking[1]
    )

    best_score = float(
        scores[best_pitch_class]
    )

    second_score = float(
        scores[second_pitch_class]
    )

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    if best_score > 0:

        confidence = (
            best_score - second_score
        ) / best_score

    else:

        confidence = 0.0

    confidence = float(
        np.clip(
            confidence,
            0.0,
            1.0
        )
    )

    # --------------------------------------------------------
    # Determine tonic information
    # --------------------------------------------------------

    tonic_name = pitch_class_to_name(
        best_pitch_class
    )

    tonic_midi = estimate_tonic_midi(
        data,
        best_pitch_class
    )

    tonic_frequency = (
        440.0 *
        2.0 ** (
            (tonic_midi - 69) / 12.0
        )
    )

    # --------------------------------------------------------
    # Print main result
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("TONIC ESTIMATION RESULT")
    print("-" * 70)

    print(
        f"Estimated Sa / Tonic : {tonic_name}"
    )

    print(
        f"Tonic MIDI           : {tonic_midi}"
    )

    print(
        f"Tonic frequency      : {tonic_frequency:.2f} Hz"
    )

    print(
        f"Confidence            : {confidence * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Print ranking
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("TONIC CANDIDATE RANKING")
    print("-" * 70)

    ranking_rows = []

    for rank, pitch_class in enumerate(
        ranking,
        start=1
    ):

        name = pitch_class_to_name(
            pitch_class
        )

        score = float(
            scores[pitch_class]
        )

        duration_evidence = (
            normalize(histogram)[pitch_class]
        )

        sustained_evidence = (
            normalize(
                sustained_histogram
            )[pitch_class]
        )

        ending_evidence = (
            normalize(
                ending_histogram
            )[pitch_class]
        )

        pa = (
            pitch_class + 7
        ) % 12

        ma = (
            pitch_class + 5
        ) % 12

        pa_strength = (
            normalize(histogram)[pa]
        )

        ma_strength = (
            normalize(histogram)[ma]
        )

        ranking_rows.append({
            "rank": rank,
            "tonic": name,
            "pitch_class": pitch_class,
            "score": score,
            "sa_occurrence": duration_evidence,
            "pa_support": pa_strength,
            "ma_support": ma_strength,
            "sustained_sa": sustained_evidence,
            "ending_sa": ending_evidence
        })

        print(
            f"{rank:2d}. "
            f"{name:3s}  "
            f"score={score:.4f}  "
            f"Sa={duration_evidence:.4f}  "
            f"Pa={pa_strength:.4f}  "
            f"Ma={ma_strength:.4f}  "
            f"Sustain={sustained_evidence:.4f}  "
            f"Ending={ending_evidence:.4f}"
        )

    ranking_dataframe = pd.DataFrame(
        ranking_rows
    )

    # --------------------------------------------------------
    # Pitch-class evidence
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("DURATION-WEIGHTED PITCH-CLASS EVIDENCE")
    print("-" * 70)

    histogram_ranking = np.argsort(
        histogram
    )[::-1]

    for pitch_class in histogram_ranking:

        if histogram[pitch_class] <= 0:
            continue

        name = pitch_class_to_name(
            pitch_class
        )

        print(
            f"{name:3s} : "
            f"{histogram[pitch_class]:.3f}"
        )

    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    os.makedirs(
        "data/output",
        exist_ok=True
    )

    output_file = (
        "data/output/tonic_estimation.csv"
    )

    result_dataframe = pd.DataFrame({

        "estimated_tonic": [
            tonic_name
        ],

        "tonic_pitch_class": [
            best_pitch_class
        ],

        "tonic_midi": [
            tonic_midi
        ],

        "tonic_frequency_hz": [
            tonic_frequency
        ],

        "confidence": [
            confidence
        ],

        "second_candidate": [
            pitch_class_to_name(
                second_pitch_class
            )
        ],

        "second_candidate_score": [
            second_score
        ]
    })

    result_dataframe.to_csv(
        output_file,
        index=False
    )

    print(
        "\nTonic result saved successfully!"
    )

    print(
        f"Saved file: {output_file}"
    )

    # --------------------------------------------------------
    # Save complete candidate ranking
    # --------------------------------------------------------

    ranking_file = (
        "data/output/tonic_candidate_ranking.csv"
    )

    ranking_dataframe.to_csv(
        ranking_file,
        index=False
    )

    print(
        "Complete tonic candidate ranking saved!"
    )

    print(
        f"Saved file: {ranking_file}"
    )

    # --------------------------------------------------------
    # Create graph
    # --------------------------------------------------------

    os.makedirs(
        "graphs",
        exist_ok=True
    )

    pitch_names = [
        pitch_class_to_name(i)
        for i in range(12)
    ]

    plt.figure(
        figsize=(13, 6)
    )

    bars = plt.bar(
        pitch_names,
        scores
    )

    # Highlight the selected candidate
    bars[best_pitch_class].set_linewidth(3)
    bars[best_pitch_class].set_edgecolor(
        "black"
    )

    plt.xlabel(
        "Possible Tonic / Sa"
    )

    plt.ylabel(
        "Combined Evidence Score"
    )

    plt.title(
        "Phase 7.4 - Improved Tonic / Sa Candidate Analysis"
    )

    plt.grid(
        True,
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    graph_path = (
        "graphs/tonic_candidate_scores.png"
    )

    plt.savefig(
        graph_path,
        dpi=300
    )

    plt.close()

    print(
        "\nTonic candidate graph saved successfully!"
    )

    print(
        f"Saved graph: {graph_path}"
    )

    # --------------------------------------------------------
    # Final message
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(
        "PHASE 7.4 COMPLETED SUCCESSFULLY"
    )
    print("=" * 70)

    return (
        tonic_name,
        tonic_midi,
        tonic_frequency,
        confidence
    )