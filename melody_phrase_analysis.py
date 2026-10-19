import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PHASE 7.6 - MELODY / MUSICAL PHRASE ANALYSIS
# ============================================================

SWARA_ORDER = [
    "Sa",
    "Komal Re",
    "Re",
    "Komal Ga",
    "Ga",
    "Ma",
    "Tivra Ma",
    "Pa",
    "Komal Dha",
    "Dha",
    "Komal Ni",
    "Ni"
]


def load_swara_data():

    input_file = "data/output/indian_swara_data.csv"

    if not os.path.exists(input_file):
        raise FileNotFoundError(
            f"\nIndian swara data not found:\n{input_file}\n\n"
            "Please run Phase 7.5 first."
        )

    data = pd.read_csv(input_file)

    if len(data) == 0:
        raise ValueError(
            "Indian swara data file is empty."
        )

    required_columns = [
        "start_time_seconds",
        "end_time_seconds",
        "indian_swara",
        "swara_short",
        "analysis_midi"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "\nRequired columns are missing from "
            "indian_swara_data.csv:\n"
            + ", ".join(missing_columns)
        )

    data = data.sort_values(
        "start_time_seconds"
    ).reset_index(drop=True)

    return data


# ============================================================
# PHRASE DETECTION
# ============================================================

def detect_melodic_phrases(
    data,
    phrase_gap_threshold=0.80
):

    print("\n" + "-" * 70)
    print("MELODIC PHRASE DETECTION")
    print("-" * 70)

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

    gaps = np.zeros(len(data))

    if len(data) > 1:

        gaps[1:] = (
            start_times[1:]
            - end_times[:-1]
        )

    phrase_ids = np.zeros(
        len(data),
        dtype=int
    )

    current_phrase = 1

    for i in range(len(data)):

        if i == 0:

            phrase_ids[i] = current_phrase

        else:

            if gaps[i] >= phrase_gap_threshold:

                current_phrase += 1

            phrase_ids[i] = current_phrase

    data = data.copy()

    data["gap_before_seconds"] = gaps
    data["phrase_id"] = phrase_ids

    number_of_phrases = (
        data["phrase_id"].nunique()
    )

    print(
        f"Phrase gap threshold : "
        f"{phrase_gap_threshold:.2f} seconds"
    )

    print(
        f"Melodic phrases found: "
        f"{number_of_phrases}"
    )

    return data


# ============================================================
# PHRASE FEATURE ANALYSIS
# ============================================================

def calculate_phrase_features(data):

    print("\n" + "-" * 70)
    print("CALCULATING PHRASE FEATURES")
    print("-" * 70)

    phrase_records = []

    for phrase_id, phrase in data.groupby(
        "phrase_id",
        sort=True
    ):

        phrase = phrase.reset_index(drop=True)

        start_time = float(
            phrase["start_time_seconds"].iloc[0]
        )

        end_time = float(
            phrase["end_time_seconds"].iloc[-1]
        )

        phrase_duration = (
            end_time - start_time
        )

        note_count = len(phrase)

        swaras = (
            phrase["indian_swara"]
            .astype(str)
            .tolist()
        )

        swara_sequence = " - ".join(
            swaras
        )

        unique_swaras = list(
            dict.fromkeys(swaras)
        )

        unique_swara_count = len(
            set(swaras)
        )

        midi_values = (
            pd.to_numeric(
                phrase["analysis_midi"],
                errors="coerce"
            )
            .to_numpy()
        )

        valid_midi = midi_values[
            np.isfinite(midi_values)
        ]

        ascending = 0
        descending = 0
        stable = 0

        for i in range(1, len(valid_midi)):

            difference = (
                valid_midi[i]
                - valid_midi[i - 1]
            )

            if difference > 0.01:

                ascending += 1

            elif difference < -0.01:

                descending += 1

            else:

                stable += 1

        if len(swaras) > 0:

            swara_counts = (
                pd.Series(swaras)
                .value_counts()
            )

            dominant_swara = (
                swara_counts.index[0]
            )

            dominant_swara_count = int(
                swara_counts.iloc[0]
            )

        else:

            dominant_swara = None
            dominant_swara_count = 0

        if len(valid_midi) > 0:

            minimum_midi = float(
                np.min(valid_midi)
            )

            maximum_midi = float(
                np.max(valid_midi)
            )

            pitch_range_semitones = (
                maximum_midi
                - minimum_midi
            )

        else:

            minimum_midi = np.nan
            maximum_midi = np.nan
            pitch_range_semitones = np.nan

        phrase_records.append({

            "phrase_id": int(
                phrase_id
            ),

            "start_time_seconds": start_time,

            "end_time_seconds": end_time,

            "phrase_duration_seconds":
                phrase_duration,

            "note_count":
                note_count,

            "unique_swara_count":
                unique_swara_count,

            "dominant_swara":
                dominant_swara,

            "dominant_swara_count":
                dominant_swara_count,

            "ascending_transitions":
                ascending,

            "descending_transitions":
                descending,

            "stable_transitions":
                stable,

            "minimum_midi":
                minimum_midi,

            "maximum_midi":
                maximum_midi,

            "pitch_range_semitones":
                pitch_range_semitones,

            "swara_sequence":
                swara_sequence,

            "unique_swaras":
                ", ".join(unique_swaras)
        })

    phrase_data = pd.DataFrame(
        phrase_records
    )

    return phrase_data


# ============================================================
# MELODIC MOVEMENT CLASSIFICATION
# ============================================================

def classify_phrase_movement(
    phrase_data
):

    movement_labels = []

    for _, row in phrase_data.iterrows():

        ascending = row[
            "ascending_transitions"
        ]

        descending = row[
            "descending_transitions"
        ]

        stable = row[
            "stable_transitions"
        ]

        if (
            ascending > descending
            and ascending > stable
        ):

            movement = "Ascending"

        elif (
            descending > ascending
            and descending > stable
        ):

            movement = "Descending"

        elif (
            stable >= ascending
            and stable >= descending
        ):

            movement = "Stable"

        else:

            movement = "Mixed"

        movement_labels.append(
            movement
        )

    phrase_data = phrase_data.copy()

    phrase_data[
        "melodic_movement"
    ] = movement_labels

    return phrase_data


# ============================================================
# REPEATED MOTIF DETECTION
# ============================================================

def detect_repeated_motifs(
    data,
    motif_length=3
):

    print("\n" + "-" * 70)
    print("REPEATED MOTIF ANALYSIS")
    print("-" * 70)

    swaras = (
        data["indian_swara"]
        .astype(str)
        .tolist()
    )

    motif_records = []

    if len(swaras) < motif_length:

        return pd.DataFrame(
            columns=[
                "motif",
                "occurrence_count",
                "first_occurrence_note"
            ]
        )

    motif_positions = {}

    for i in range(
        len(swaras) - motif_length + 1
    ):

        motif = tuple(
            swaras[
                i:i + motif_length
            ]
        )

        if motif not in motif_positions:

            motif_positions[motif] = []

        motif_positions[motif].append(i)

    sorted_motifs = sorted(
        motif_positions.items(),
        key=lambda item: len(
            item[1]
        ),
        reverse=True
    )

    for motif, positions in sorted_motifs:

        occurrence_count = len(
            positions
        )

        if occurrence_count < 2:

            continue

        motif_text = " - ".join(
            motif
        )

        motif_records.append({

            "motif":
                motif_text,

            "occurrence_count":
                occurrence_count,

            "first_occurrence_note":
                positions[0] + 1
        })

    motif_data = pd.DataFrame(
        motif_records
    )

    if len(motif_data) > 0:

        print(
            f"Repeated {motif_length}-note "
            f"motifs found: "
            f"{len(motif_data)}"
        )

        print("\nTop repeated motifs:")

        print(
            motif_data
            .head(15)
            .to_string(index=False)
        )

    else:

        print(
            "No repeated motifs found."
        )

    return motif_data


# ============================================================
# PRINT STATISTICS
# ============================================================

def print_phrase_statistics(
    phrase_data
):

    print("\n" + "=" * 70)
    print("PHASE 7.6 - MELODY / MUSICAL PHRASE ANALYSIS")
    print("=" * 70)

    print("\nPhrase analysis statistics:")

    print(
        f"Total melodic phrases : "
        f"{len(phrase_data)}"
    )

    if len(phrase_data) == 0:

        return

    print(
        f"Total note events     : "
        f"{phrase_data['note_count'].sum()}"
    )

    print(
        f"Mean notes / phrase   : "
        f"{phrase_data['note_count'].mean():.2f}"
    )

    print(
        f"Mean phrase duration  : "
        f"{phrase_data['phrase_duration_seconds'].mean():.2f} sec"
    )

    print(
        f"Longest phrase        : "
        f"{phrase_data['phrase_duration_seconds'].max():.2f} sec"
    )

    print(
        f"Shortest phrase       : "
        f"{phrase_data['phrase_duration_seconds'].min():.2f} sec"
    )

    print("\nMelodic movement:")

    movement_counts = (
        phrase_data[
            "melodic_movement"
        ]
        .value_counts()
    )

    for movement in [
        "Ascending",
        "Descending",
        "Stable",
        "Mixed"
    ]:

        count = movement_counts.get(
            movement,
            0
        )

        print(
            f"{movement:12s}: {count}"
        )

    print("\nDominant swaras across phrases:")

    dominant_counts = (
        phrase_data[
            "dominant_swara"
        ]
        .value_counts()
    )

    for swara, count in (
        dominant_counts.items()
    ):

        print(
            f"{swara:12s}: {count}"
        )


# ============================================================
# SAVE DATA
# ============================================================

def save_phrase_data(
    phrase_data,
    note_data
):

    os.makedirs(
        "data/output",
        exist_ok=True
    )

    phrase_file = (
        "data/output/"
        "melodic_phrase_analysis.csv"
    )

    note_file = (
        "data/output/"
        "melody_phrase_note_data.csv"
    )

    phrase_data.to_csv(
        phrase_file,
        index=False
    )

    note_data.to_csv(
        note_file,
        index=False
    )

    print(
        "\nMelodic phrase analysis saved!"
    )

    print(
        f"Saved file: {phrase_file}"
    )

    print(
        f"Saved file: {note_file}"
    )


# ============================================================
# PHRASE TIMELINE GRAPH
# ============================================================

def create_phrase_timeline_graph(
    phrase_data
):

    os.makedirs(
        "graphs",
        exist_ok=True
    )

    if len(phrase_data) == 0:

        return

    plt.figure(
        figsize=(16, 7)
    )

    for _, row in phrase_data.iterrows():

        start = row[
            "start_time_seconds"
        ]

        end = row[
            "end_time_seconds"
        ]

        phrase_id = row[
            "phrase_id"
        ]

        plt.hlines(
            phrase_id,
            start,
            end,
            linewidth=5
        )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Phrase Number"
    )

    plt.title(
        "Phase 7.6 - Melodic Phrase Timeline"
    )

    plt.grid(
        True,
        axis="x",
        alpha=0.3
    )

    plt.tight_layout()

    graph_path = (
        "graphs/"
        "melodic_phrase_timeline.png"
    )

    plt.savefig(
        graph_path,
        dpi=300
    )

    plt.close()

    print(
        "\nMelodic phrase timeline graph saved!"
    )

    print(
        f"Saved graph: {graph_path}"
    )


# ============================================================
# PHRASE NOTE COUNT GRAPH
# ============================================================

def create_phrase_note_count_graph(
    phrase_data
):

    os.makedirs(
        "graphs",
        exist_ok=True
    )

    if len(phrase_data) == 0:

        return

    phrase_numbers = (
        phrase_data[
            "phrase_id"
        ]
        .to_numpy()
    )

    note_counts = (
        phrase_data[
            "note_count"
        ]
        .to_numpy()
    )

    plt.figure(
        figsize=(16, 6)
    )

    plt.bar(
        phrase_numbers,
        note_counts
    )

    plt.xlabel(
        "Phrase Number"
    )

    plt.ylabel(
        "Number of Notes"
    )

    plt.title(
        "Phase 7.6 - Notes per Melodic Phrase"
    )

    plt.grid(
        True,
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    graph_path = (
        "graphs/"
        "melodic_phrase_note_count.png"
    )

    plt.savefig(
        graph_path,
        dpi=300
    )

    plt.close()

    print(
        "Phrase note-count graph saved!"
    )

    print(
        f"Saved graph: {graph_path}"
    )


# ============================================================
# PITCH MOVEMENT GRAPH
# ============================================================

def create_pitch_movement_graph(
    data
):

    os.makedirs(
        "graphs",
        exist_ok=True
    )

    if len(data) == 0:

        return

    center_times = (
        data[
            "start_time_seconds"
        ].astype(float).to_numpy()
        +
        data[
            "end_time_seconds"
        ].astype(float).to_numpy()
    ) / 2.0

    midi_values = (
        pd.to_numeric(
            data["analysis_midi"],
            errors="coerce"
        ).to_numpy()
    )

    valid = (
        np.isfinite(
            midi_values
        )
    )

    plt.figure(
        figsize=(16, 6)
    )

    plt.scatter(
        center_times[valid],
        midi_values[valid],
        s=18
    )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "MIDI Pitch"
    )

    plt.title(
        "Phase 7.6 - Melodic Pitch Movement"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    graph_path = (
        "graphs/"
        "melodic_pitch_movement.png"
    )

    plt.savefig(
        graph_path,
        dpi=300
    )

    plt.close()

    print(
        "Melodic pitch movement graph saved!"
    )

    print(
        f"Saved graph: {graph_path}"
    )


# ============================================================
# MAIN PHASE 7.6 FUNCTION
# ============================================================

def analyze_melodic_phrases():

    print("\n" + "=" * 70)
    print(
        "PHASE 7.6 - "
        "MELODY / MUSICAL PHRASE ANALYSIS"
    )
    print("=" * 70)

    print(
        "\nLoading Phase 7.5 Indian swara data..."
    )

    data = load_swara_data()

    print(
        f"Note events loaded: {len(data)}"
    )

    print(
        "\nDetecting melodic phrases..."
    )

    phrase_note_data = detect_melodic_phrases(
        data,
        phrase_gap_threshold=0.80
    )

    print(
        "\nCalculating phrase-level features..."
    )

    phrase_data = calculate_phrase_features(
        phrase_note_data
    )

    print(
        "\nClassifying melodic movement..."
    )

    phrase_data = classify_phrase_movement(
        phrase_data
    )

    motif_data = detect_repeated_motifs(
        phrase_note_data,
        motif_length=3
    )

    print_phrase_statistics(
        phrase_data
    )

    print(
        "\nFirst 15 detected melodic phrases:"
    )

    display_columns = [
        "phrase_id",
        "start_time_seconds",
        "end_time_seconds",
        "phrase_duration_seconds",
        "note_count",
        "dominant_swara",
        "melodic_movement"
    ]

    print(
        phrase_data[
            display_columns
        ]
        .head(15)
        .to_string(index=False)
    )

    save_phrase_data(
        phrase_data,
        phrase_note_data
    )

    if len(motif_data) > 0:

        motif_file = (
            "data/output/"
            "repeated_melodic_motifs.csv"
        )

        motif_data.to_csv(
            motif_file,
            index=False
        )

        print(
            "\nRepeated motif data saved!"
        )

        print(
            f"Saved file: {motif_file}"
        )

    create_phrase_timeline_graph(
        phrase_data
    )

    create_phrase_note_count_graph(
        phrase_data
    )

    create_pitch_movement_graph(
        phrase_note_data
    )

    print("\n" + "=" * 70)
    print(
        "PHASE 7.6 COMPLETED SUCCESSFULLY"
    )
    print("=" * 70)

    return (
        phrase_data,
        phrase_note_data,
        motif_data
    )