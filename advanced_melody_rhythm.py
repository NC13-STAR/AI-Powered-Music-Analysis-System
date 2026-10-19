import os
import json
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

OUTPUT_DIR = "data/output"
GRAPH_DIR = "graphs"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def safe_float(value, default=0.0):
    try:
        value = float(value)

        if np.isfinite(value):
            return value

        return default

    except (TypeError, ValueError):
        return default


def find_column(df, candidates):
    """
    Find the first available column from a list of candidates.
    """

    for column in candidates:
        if column in df.columns:
            return column

    return None


def load_csv_if_exists(filename):
    """
    Load a CSV from data/output.
    """

    path = os.path.join(OUTPUT_DIR, filename)

    if not os.path.exists(path):
        print(f"Warning: {path} not found.")
        return None

    try:
        return pd.read_csv(path)

    except Exception as error:
        print(f"Could not read {path}: {error}")
        return None


def save_dataframe(df, filename):
    """
    Save dataframe to data/output.
    """

    path = os.path.join(OUTPUT_DIR, filename)

    df.to_csv(path, index=False)

    return path


# ============================================================
# 11.2.1 MELODIC PITCH RANGE
# ============================================================

def analyze_pitch_range():
    print("\n" + "=" * 70)
    print("PHASE 11.2.1 - MELODIC PITCH RANGE")
    print("=" * 70)

    path = os.path.join(
        OUTPUT_DIR,
        "final_melody_data.npz"
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            "Required file not found:\n"
            f"{path}"
        )

    data = np.load(path)

    f0 = np.asarray(data["f0"], dtype=float)

    valid_f0 = f0[
        np.isfinite(f0) &
        (f0 > 0)
    ]

    if len(valid_f0) == 0:
        raise ValueError(
            "No valid pitch values were found."
        )

    midi_values = librosa.hz_to_midi(valid_f0)

    min_midi = float(np.min(midi_values))
    max_midi = float(np.max(midi_values))
    median_midi = float(np.median(midi_values))

    min_hz = float(np.min(valid_f0))
    max_hz = float(np.max(valid_f0))
    median_hz = float(np.median(valid_f0))

    pitch_range_semitones = max_midi - min_midi

    pitch_range_octaves = pitch_range_semitones / 12.0

    min_note = librosa.midi_to_note(
        int(round(min_midi))
    )

    max_note = librosa.midi_to_note(
        int(round(max_midi))
    )

    median_note = librosa.midi_to_note(
        int(round(median_midi))
    )

    result = pd.DataFrame(
        [{
            "minimum_frequency_hz": min_hz,
            "maximum_frequency_hz": max_hz,
            "median_frequency_hz": median_hz,
            "minimum_midi": min_midi,
            "maximum_midi": max_midi,
            "median_midi": median_midi,
            "minimum_note": min_note,
            "maximum_note": max_note,
            "median_note": median_note,
            "pitch_range_semitones": pitch_range_semitones,
            "pitch_range_octaves": pitch_range_octaves,
            "valid_pitch_frames": len(valid_f0)
        }]
    )

    output_path = save_dataframe(
        result,
        "advanced_pitch_range.csv"
    )

    print(f"Minimum pitch : {min_hz:.2f} Hz ({min_note})")
    print(f"Maximum pitch : {max_hz:.2f} Hz ({max_note})")
    print(f"Median pitch  : {median_hz:.2f} Hz ({median_note})")
    print(
        f"Pitch range   : "
        f"{pitch_range_semitones:.2f} semitones"
    )
    print(
        f"Pitch range   : "
        f"{pitch_range_octaves:.2f} octaves"
    )

    return result


# ============================================================
# 11.2.2 MELODIC MOVEMENT
# ============================================================

def analyze_melodic_movement():
    print("\n" + "=" * 70)
    print("PHASE 11.2.2 - MELODIC MOVEMENT")
    print("=" * 70)

    path = os.path.join(
        OUTPUT_DIR,
        "final_melody_data.npz"
    )

    data = np.load(path)

    f0 = np.asarray(data["f0"], dtype=float)

    valid = (
        np.isfinite(f0) &
        (f0 > 0)
    )

    midi = np.full(
        len(f0),
        np.nan,
        dtype=float
    )

    midi[valid] = librosa.hz_to_midi(
        f0[valid]
    )

    valid_midi = midi[np.isfinite(midi)]

    if len(valid_midi) < 2:
        raise ValueError(
            "Not enough valid pitch data."
        )

    intervals = np.diff(valid_midi)

    ascending = intervals[intervals > 0]
    descending = intervals[intervals < 0]
    stationary = intervals[
        np.isclose(intervals, 0)
    ]

    mean_absolute_interval = float(
        np.mean(np.abs(intervals))
    )

    median_absolute_interval = float(
        np.median(np.abs(intervals))
    )

    largest_upward = (
        float(np.max(ascending))
        if len(ascending) > 0
        else 0.0
    )

    largest_downward = (
        float(np.min(descending))
        if len(descending) > 0
        else 0.0
    )

    ascending_count = len(ascending)
    descending_count = len(descending)
    stationary_count = len(stationary)

    total = len(intervals)

    result = pd.DataFrame(
        [{
            "total_intervals": total,
            "ascending_movements": ascending_count,
            "descending_movements": descending_count,
            "stationary_movements": stationary_count,
            "ascending_percentage":
                ascending_count / total * 100,
            "descending_percentage":
                descending_count / total * 100,
            "stationary_percentage":
                stationary_count / total * 100,
            "mean_absolute_interval_semitones":
                mean_absolute_interval,
            "median_absolute_interval_semitones":
                median_absolute_interval,
            "largest_upward_interval_semitones":
                largest_upward,
            "largest_downward_interval_semitones":
                largest_downward
        }]
    )

    output_path = save_dataframe(
        result,
        "melodic_movement_analysis.csv"
    )

    print(
        f"Ascending movements : {ascending_count}"
    )
    print(
        f"Descending movements: {descending_count}"
    )
    print(
        f"Stationary movements: {stationary_count}"
    )
    print(
        f"Mean interval       : "
        f"{mean_absolute_interval:.3f} semitones"
    )

    return result


# ============================================================
# 11.2.3 INTERVAL ANALYSIS
# ============================================================

def analyze_intervals():
    print("\n" + "=" * 70)
    print("PHASE 11.2.3 - INTERVAL ANALYSIS")
    print("=" * 70)

    path = os.path.join(
        OUTPUT_DIR,
        "final_melody_data.npz"
    )

    data = np.load(path)

    f0 = np.asarray(data["f0"], dtype=float)

    valid_f0 = f0[
        np.isfinite(f0) &
        (f0 > 0)
    ]

    midi = librosa.hz_to_midi(valid_f0)

    if len(midi) < 2:
        raise ValueError(
            "Not enough pitch data for interval analysis."
        )

    intervals = np.diff(midi)

    absolute_intervals = np.abs(intervals)

    rounded_intervals = np.round(
        absolute_intervals
    ).astype(int)

    unique, counts = np.unique(
        rounded_intervals,
        return_counts=True
    )

    interval_rows = []

    total = len(rounded_intervals)

    for interval_value, count in zip(
        unique,
        counts
    ):
        interval_rows.append(
            {
                "interval_semitones":
                    int(interval_value),
                "count":
                    int(count),
                "percentage":
                    float(count / total * 100)
            }
        )

    interval_data = pd.DataFrame(
        interval_rows
    )

    save_dataframe(
        interval_data,
        "melodic_interval_distribution.csv"
    )

    print("\nMost frequent melodic intervals:")

    print(
        interval_data
        .sort_values(
            "count",
            ascending=False
        )
        .head(10)
        .to_string(index=False)
    )

    return interval_data


# ============================================================
# 11.2.4 MELODIC COMPLEXITY
# ============================================================

def calculate_melodic_complexity():
    print("\n" + "=" * 70)
    print("PHASE 11.2.4 - MELODIC COMPLEXITY")
    print("=" * 70)

    path = os.path.join(
        OUTPUT_DIR,
        "final_melody_data.npz"
    )

    data = np.load(path)

    f0 = np.asarray(data["f0"], dtype=float)

    valid_f0 = f0[
        np.isfinite(f0) &
        (f0 > 0)
    ]

    midi = librosa.hz_to_midi(valid_f0)

    if len(midi) < 2:
        raise ValueError(
            "Not enough pitch data."
        )

    intervals = np.diff(midi)

    absolute_intervals = np.abs(intervals)

    large_jumps = np.sum(
        absolute_intervals >= 5
    )

    very_large_jumps = np.sum(
        absolute_intervals >= 12
    )

    direction_changes = 0

    previous_direction = 0

    for interval in intervals:

        if interval > 0:
            direction = 1

        elif interval < 0:
            direction = -1

        else:
            direction = 0

        if (
            previous_direction != 0
            and direction != 0
            and direction != previous_direction
        ):
            direction_changes += 1

        if direction != 0:
            previous_direction = direction

    total_intervals = len(intervals)

    large_jump_ratio = (
        large_jumps / total_intervals
    )

    very_large_jump_ratio = (
        very_large_jumps / total_intervals
    )

    direction_change_ratio = (
        direction_changes / total_intervals
    )

    complexity_score = (
        large_jump_ratio * 40
        + very_large_jump_ratio * 30
        + direction_change_ratio * 30
    )

    complexity_score = float(
        min(100.0, complexity_score)
    )

    if complexity_score < 20:
        category = "Low"

    elif complexity_score < 40:
        category = "Moderate"

    elif complexity_score < 60:
        category = "Moderately High"

    else:
        category = "High"

    result = pd.DataFrame(
        [{
            "large_jumps_5_or_more_semitones":
                int(large_jumps),
            "very_large_jumps_12_or_more_semitones":
                int(very_large_jumps),
            "direction_changes":
                int(direction_changes),
            "large_jump_ratio":
                float(large_jump_ratio),
            "very_large_jump_ratio":
                float(very_large_jump_ratio),
            "direction_change_ratio":
                float(direction_change_ratio),
            "melodic_complexity_score":
                complexity_score,
            "complexity_category":
                category
        }]
    )

    save_dataframe(
        result,
        "melodic_complexity_analysis.csv"
    )

    print(
        f"Complexity score: "
        f"{complexity_score:.2f}/100"
    )

    print(
        f"Complexity category: {category}"
    )

    return result


# ============================================================
# 11.2.5 PITCH STABILITY
# ============================================================

def analyze_pitch_stability():
    print("\n" + "=" * 70)
    print("PHASE 11.2.5 - PITCH STABILITY")
    print("=" * 70)

    path = os.path.join(
        OUTPUT_DIR,
        "quantized_notes.csv"
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    df = pd.read_csv(path)

    deviation_column = find_column(
        df,
        [
            "absolute_cents_deviation",
            "cents_deviation"
        ]
    )

    if deviation_column is None:
        raise ValueError(
            "Pitch deviation column not found "
            "in quantized_notes.csv"
        )

    deviations = pd.to_numeric(
        df[deviation_column],
        errors="coerce"
    ).dropna()

    deviations = np.abs(
        deviations.to_numpy()
    )

    if len(deviations) == 0:
        raise ValueError(
            "No valid pitch deviation data."
        )

    mean_deviation = float(
        np.mean(deviations)
    )

    median_deviation = float(
        np.median(deviations)
    )

    std_deviation = float(
        np.std(deviations)
    )

    stable_percentage = float(
        np.mean(deviations <= 25) * 100
    )

    result = pd.DataFrame(
        [{
            "mean_absolute_cents_deviation":
                mean_deviation,
            "median_absolute_cents_deviation":
                median_deviation,
            "std_cents_deviation":
                std_deviation,
            "within_25_cents_percentage":
                stable_percentage
        }]
    )

    save_dataframe(
        result,
        "pitch_stability_analysis.csv"
    )

    print(
        f"Mean deviation   : "
        f"{mean_deviation:.2f} cents"
    )

    print(
        f"Median deviation : "
        f"{median_deviation:.2f} cents"
    )

    print(
        f"Within ±25 cents: "
        f"{stable_percentage:.2f}%"
    )

    return result


# ============================================================
# 11.2.6 PHRASE LEVEL MELODY
# ============================================================

def analyze_phrase_melody():
    print("\n" + "=" * 70)
    print("PHASE 11.2.6 - PHRASE LEVEL MELODY ANALYSIS")
    print("=" * 70)

    path = os.path.join(
        OUTPUT_DIR,
        "melody_phrase_note_data.csv"
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    df = pd.read_csv(path)

    phrase_column = find_column(
        df,
        [
            "phrase_id",
            "phrase"
        ]
    )

    start_column = find_column(
        df,
        [
            "start_time_seconds",
            "start_time"
        ]
    )

    end_column = find_column(
        df,
        [
            "end_time_seconds",
            "end_time"
        ]
    )

    midi_column = find_column(
        df,
        [
            "analysis_midi",
            "quantized_midi",
            "detected_midi"
        ]
    )

    if phrase_column is None:
        raise ValueError(
            "phrase_id column not found."
        )

    if midi_column is None:
        raise ValueError(
            "MIDI column not found."
        )

    phrase_rows = []

    for phrase_id, group in df.groupby(
        phrase_column
    ):

        midi_values = pd.to_numeric(
            group[midi_column],
            errors="coerce"
        ).dropna()

        if len(midi_values) == 0:
            continue

        phrase_range = (
            float(midi_values.max())
            - float(midi_values.min())
        )

        phrase_mean = float(
            midi_values.mean()
        )

        phrase_std = float(
            midi_values.std()
        ) if len(midi_values) > 1 else 0.0

        row = {
            "phrase_id": phrase_id,
            "note_count": len(midi_values),
            "mean_midi": phrase_mean,
            "pitch_std_semitones": phrase_std,
            "pitch_range_semitones": phrase_range
        }

        if (
            start_column is not None
            and end_column is not None
        ):
            starts = pd.to_numeric(
                group[start_column],
                errors="coerce"
            ).dropna()

            ends = pd.to_numeric(
                group[end_column],
                errors="coerce"
            ).dropna()

            if len(starts) > 0 and len(ends) > 0:

                row["start_time_seconds"] = float(
                    starts.min()
                )

                row["end_time_seconds"] = float(
                    ends.max()
                )

                row["duration_seconds"] = (
                    row["end_time_seconds"]
                    - row["start_time_seconds"]
                )

        phrase_rows.append(row)

    phrase_data = pd.DataFrame(
        phrase_rows
    )

    save_dataframe(
        phrase_data,
        "advanced_phrase_melody_analysis.csv"
    )

    print(
        f"Melodic phrases analyzed: "
        f"{len(phrase_data)}"
    )

    if len(phrase_data) > 0:

        print("\nFirst phrase results:")

        print(
            phrase_data.head(10)
            .to_string(index=False)
        )

    return phrase_data


# ============================================================
# 11.2.7 RHYTHMIC COMPLEXITY
# ============================================================

def analyze_rhythmic_complexity():
    print("\n" + "=" * 70)
    print("PHASE 11.2.7 - RHYTHMIC COMPLEXITY")
    print("=" * 70)

    path = os.path.join(
        OUTPUT_DIR,
        "beat_structure_analysis.csv"
    )

    if not os.path.exists(path):
        print(
            "beat_structure_analysis.csv not found."
        )

        print(
            "Using rhythm_analysis.csv instead."
        )

        path = os.path.join(
            OUTPUT_DIR,
            "rhythm_analysis.csv"
        )

    if not os.path.exists(path):
        raise FileNotFoundError(
            "No beat/rhythm analysis CSV was found."
        )

    df = pd.read_csv(path)

    interval_column = find_column(
        df,
        [
            "beat_interval_seconds",
            "interval_seconds",
            "beat_interval"
        ]
    )

    if interval_column is None:
        print(
            "No beat interval column available."
        )

        result = pd.DataFrame(
            [{
                "rhythmic_complexity_score": np.nan,
                "rhythmic_complexity_category":
                    "Insufficient data"
            }]
        )

        save_dataframe(
            result,
            "rhythmic_complexity_analysis.csv"
        )

        return result

    intervals = pd.to_numeric(
        df[interval_column],
        errors="coerce"
    ).dropna()

    intervals = intervals[
        intervals > 0
    ].to_numpy()

    if len(intervals) < 2:
        raise ValueError(
            "Not enough beat intervals."
        )

    mean_interval = float(
        np.mean(intervals)
    )

    std_interval = float(
        np.std(intervals)
    )

    coefficient_variation = (
        std_interval / mean_interval
        if mean_interval > 0
        else 0.0
    )

    interval_changes = np.abs(
        np.diff(intervals)
    )

    mean_interval_change = float(
        np.mean(interval_changes)
    )

    irregularity_score = min(
        100.0,
        coefficient_variation * 100
    )

    if irregularity_score < 5:
        category = "Very Low"

    elif irregularity_score < 10:
        category = "Low"

    elif irregularity_score < 20:
        category = "Moderate"

    elif irregularity_score < 35:
        category = "High"

    else:
        category = "Very High"

    result = pd.DataFrame(
        [{
            "mean_beat_interval_seconds":
                mean_interval,
            "std_beat_interval_seconds":
                std_interval,
            "beat_interval_coefficient_variation":
                coefficient_variation,
            "mean_interval_change_seconds":
                mean_interval_change,
            "rhythmic_complexity_score":
                irregularity_score,
            "rhythmic_complexity_category":
                category
        }]
    )

    save_dataframe(
        result,
        "rhythmic_complexity_analysis.csv"
    )

    print(
        f"Mean beat interval : "
        f"{mean_interval:.4f} sec"
    )

    print(
        f"Interval variation : "
        f"{coefficient_variation * 100:.2f}%"
    )

    print(
        f"Complexity score   : "
        f"{irregularity_score:.2f}/100"
    )

    print(
        f"Category           : {category}"
    )

    return result


# ============================================================
# 11.2.8 MELODY-RHYTHM RELATIONSHIP
# ============================================================

def analyze_melody_rhythm_relationship():
    print("\n" + "=" * 70)
    print("PHASE 11.2.8 - MELODY-RHYTHM RELATIONSHIP")
    print("=" * 70)

    melody_path = os.path.join(
        OUTPUT_DIR,
        "note_duration_analysis.csv"
    )

    if not os.path.exists(melody_path):
        raise FileNotFoundError(
            f"Required file not found:\n{melody_path}"
        )

    melody_df = pd.read_csv(
        melody_path
    )

    duration_column = find_column(
        melody_df,
        [
            "note_duration_seconds",
            "duration_seconds"
        ]
    )

    gap_column = find_column(
        melody_df,
        [
            "gap_before_seconds"
        ]
    )

    if duration_column is None:
        raise ValueError(
            "Note duration column not found."
        )

    durations = pd.to_numeric(
        melody_df[duration_column],
        errors="coerce"
    ).dropna()

    durations = durations[
        durations > 0
    ].to_numpy()

    if len(durations) == 0:
        raise ValueError(
            "No valid note durations."
        )

    mean_duration = float(
        np.mean(durations)
    )

    median_duration = float(
        np.median(durations)
    )

    std_duration = float(
        np.std(durations)
    )

    if gap_column is not None:

        gaps = pd.to_numeric(
            melody_df[gap_column],
            errors="coerce"
        ).dropna()

        gaps = gaps[
            gaps >= 0
        ].to_numpy()

    else:

        gaps = np.array([])

    if len(gaps) > 0:

        mean_gap = float(
            np.mean(gaps)
        )

        nonzero_gap_percentage = float(
            np.mean(gaps > 0.01) * 100
        )

    else:

        mean_gap = 0.0
        nonzero_gap_percentage = 0.0

    result = pd.DataFrame(
        [{
            "mean_note_duration_seconds":
                mean_duration,
            "median_note_duration_seconds":
                median_duration,
            "std_note_duration_seconds":
                std_duration,
            "mean_gap_before_note_seconds":
                mean_gap,
            "notes_with_nonzero_gap_percentage":
                nonzero_gap_percentage,
            "note_count":
                len(durations)
        }]
    )

    save_dataframe(
        result,
        "melody_rhythm_relationship.csv"
    )

    print(
        f"Mean note duration : "
        f"{mean_duration:.3f} sec"
    )

    print(
        f"Median duration    : "
        f"{median_duration:.3f} sec"
    )

    print(
        f"Mean gap           : "
        f"{mean_gap:.3f} sec"
    )

    return result


# ============================================================
# 11.2.9 VISUALIZATION
# ============================================================

def create_advanced_visualizations(
    interval_data,
    phrase_data
):

    print("\n" + "=" * 70)
    print("PHASE 11.2.9 - ADVANCED VISUALIZATION")
    print("=" * 70)

    # --------------------------------------------------------
    # INTERVAL DISTRIBUTION
    # --------------------------------------------------------

    if (
        interval_data is not None
        and len(interval_data) > 0
    ):

        plt.figure(
            figsize=(12, 6)
        )

        plt.bar(
            interval_data[
                "interval_semitones"
            ],
            interval_data["count"]
        )

        plt.xlabel(
            "Interval (semitones)"
        )

        plt.ylabel(
            "Frequency"
        )

        plt.title(
            "Melodic Interval Distribution"
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                GRAPH_DIR,
                "phase_11_2_interval_distribution.png"
            ),
            dpi=200
        )

        plt.close()

    # --------------------------------------------------------
    # PHRASE PITCH RANGE
    # --------------------------------------------------------

    if (
        phrase_data is not None
        and len(phrase_data) > 0
        and "pitch_range_semitones" in phrase_data.columns
    ):

        plot_data = phrase_data.copy()

        plot_data[
            "phrase_label"
        ] = plot_data[
            "phrase_id"
        ].astype(str)

        plt.figure(
            figsize=(12, 6)
        )

        plt.bar(
            plot_data["phrase_label"],
            plot_data[
                "pitch_range_semitones"
            ]
        )

        plt.xlabel(
            "Phrase"
        )

        plt.ylabel(
            "Pitch range (semitones)"
        )

        plt.title(
            "Pitch Range Across Melodic Phrases"
        )

        plt.xticks(
            rotation=90
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                GRAPH_DIR,
                "phase_11_2_phrase_pitch_range.png"
            ),
            dpi=200
        )

        plt.close()

    # --------------------------------------------------------
    # PITCH CONTOUR
    # --------------------------------------------------------

    melody_path = os.path.join(
        OUTPUT_DIR,
        "final_melody_data.npz"
    )

    if os.path.exists(melody_path):

        data = np.load(
            melody_path
        )

        f0 = np.asarray(
            data["f0"],
            dtype=float
        )

        times = np.asarray(
            data["times"],
            dtype=float
        )

        valid = (
            np.isfinite(f0)
            & (f0 > 0)
        )

        plt.figure(
            figsize=(14, 6)
        )

        plt.plot(
            times[valid],
            librosa.hz_to_midi(
                f0[valid]
            ),
            linewidth=1
        )

        plt.xlabel(
            "Time (seconds)"
        )

        plt.ylabel(
            "Pitch (MIDI)"
        )

        plt.title(
            "Advanced Melody Pitch Contour"
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                GRAPH_DIR,
                "phase_11_2_advanced_pitch_contour.png"
            ),
            dpi=200
        )

        plt.close()

    print(
        "Advanced graphs generated successfully."
    )


# ============================================================
# 11.2 SUMMARY
# ============================================================

def create_phase_11_2_summary(
    pitch_range,
    movement,
    intervals,
    complexity,
    stability,
    phrase_data,
    rhythm_complexity,
    melody_rhythm
):

    summary = {}

    if pitch_range is not None:
        summary["pitch_range"] = (
            pitch_range.iloc[0].to_dict()
        )

    if movement is not None:
        summary["melodic_movement"] = (
            movement.iloc[0].to_dict()
        )

    if complexity is not None:
        summary["melodic_complexity"] = (
            complexity.iloc[0].to_dict()
        )

    if stability is not None:
        summary["pitch_stability"] = (
            stability.iloc[0].to_dict()
        )

    if rhythm_complexity is not None:
        summary["rhythmic_complexity"] = (
            rhythm_complexity.iloc[0].to_dict()
        )

    if melody_rhythm is not None:
        summary["melody_rhythm_relationship"] = (
            melody_rhythm.iloc[0].to_dict()
        )

    if intervals is not None:
        summary["interval_count"] = int(
            len(intervals)
        )

    if phrase_data is not None:
        summary["phrase_count"] = int(
            len(phrase_data)
        )

    output_path = os.path.join(
        OUTPUT_DIR,
        "phase_11_2_summary.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
            default=str
        )

    print(
        f"\nSummary saved: {output_path}"
    )


# ============================================================
# MAIN PHASE 11.2 PIPELINE
# ============================================================

def run_phase_11_2():

    print("\n" + "=" * 80)
    print("PHASE 11.2 - ADVANCED MELODY & RHYTHM ANALYSIS")
    print("=" * 80)

    # 11.2.1
    pitch_range = analyze_pitch_range()

    # 11.2.2
    movement = analyze_melodic_movement()

    # 11.2.3
    intervals = analyze_intervals()

    # 11.2.4
    complexity = calculate_melodic_complexity()

    # 11.2.5
    stability = analyze_pitch_stability()

    # 11.2.6
    phrase_data = analyze_phrase_melody()

    # 11.2.7
    rhythm_complexity = analyze_rhythmic_complexity()

    # 11.2.8
    melody_rhythm = analyze_melody_rhythm_relationship()

    # 11.2.9
    create_advanced_visualizations(
        intervals,
        phrase_data
    )

    # Summary
    create_phase_11_2_summary(
        pitch_range,
        movement,
        intervals,
        complexity,
        stability,
        phrase_data,
        rhythm_complexity,
        melody_rhythm
    )

    print("\n" + "=" * 80)
    print("PHASE 11.2 COMPLETED SUCCESSFULLY")
    print("=" * 80)

    print("\nGenerated CSV files:")

    print(
        "  data/output/advanced_pitch_range.csv"
    )

    print(
        "  data/output/melodic_movement_analysis.csv"
    )

    print(
        "  data/output/melodic_interval_distribution.csv"
    )

    print(
        "  data/output/melodic_complexity_analysis.csv"
    )

    print(
        "  data/output/pitch_stability_analysis.csv"
    )

    print(
        "  data/output/advanced_phrase_melody_analysis.csv"
    )

    print(
        "  data/output/rhythmic_complexity_analysis.csv"
    )

    print(
        "  data/output/melody_rhythm_relationship.csv"
    )

    print("\nGenerated summary:")

    print(
        "  data/output/phase_11_2_summary.json"
    )

    print("\nGenerated graphs:")

    print(
        "  graphs/phase_11_2_interval_distribution.png"
    )

    print(
        "  graphs/phase_11_2_phrase_pitch_range.png"
    )

    print(
        "  graphs/phase_11_2_advanced_pitch_contour.png"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    run_phase_11_2()