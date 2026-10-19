import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PHASE 7.7
# WESTERN + INDIAN NOTATION ALIGNMENT
# ============================================================


NOTE_NAMES = [
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B"
]


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


# ============================================================
# LOAD DATA
# ============================================================

def load_alignment_data():

    input_file = (
        "data/output/"
        "melody_phrase_note_data.csv"
    )

    if not os.path.exists(input_file):

        raise FileNotFoundError(
            f"\nRequired file not found:\n"
            f"{input_file}\n\n"
            "Please run Phase 7.6 first."
        )

    data = pd.read_csv(
        input_file
    )

    if len(data) == 0:

        raise ValueError(
            "Melody phrase note data is empty."
        )

    required_columns = [
        "start_time_seconds",
        "end_time_seconds",
        "indian_swara",
        "swara_short",
        "analysis_midi",
        "phrase_id"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:

        raise ValueError(
            "\nRequired columns are missing:\n"
            + ", ".join(missing_columns)
        )

    data = (
        data
        .sort_values(
            "start_time_seconds"
        )
        .reset_index(drop=True)
    )

    return data


# ============================================================
# LOAD TONIC
# ============================================================

def load_tonic():

    tonic_file = (
        "data/output/"
        "tonic_estimation.csv"
    )

    if not os.path.exists(tonic_file):

        raise FileNotFoundError(
            f"\nTonic estimation file not found:\n"
            f"{tonic_file}\n\n"
            "Please run Phase 7.4 first."
        )

    tonic_data = pd.read_csv(
        tonic_file
    )

    if len(tonic_data) == 0:

        raise ValueError(
            "Tonic estimation file is empty."
        )

    row = tonic_data.iloc[0]

    return {
        "tonic_name":
            str(row["estimated_tonic"]),

        "tonic_pitch_class":
            int(row["tonic_pitch_class"]),

        "tonic_midi":
            int(row["tonic_midi"]),

        "tonic_frequency":
            float(row["tonic_frequency_hz"]),

        "confidence":
            float(row["confidence"])
    }


# ============================================================
# MIDI TO WESTERN NOTE
# ============================================================

def midi_to_western_note(
    midi_number
):

    if not np.isfinite(
        midi_number
    ):

        return None

    midi_number = int(
        round(midi_number)
    )

    note_index = (
        midi_number % 12
    )

    octave = (
        midi_number // 12
    ) - 1

    return (
        f"{NOTE_NAMES[note_index]}"
        f"{octave}"
    )


# ============================================================
# CREATE NOTATION ALIGNMENT
# ============================================================

def create_notation_alignment(
    data
):

    aligned = data.copy()

    # --------------------------------------------------------
    # Convert important numerical columns
    # --------------------------------------------------------

    aligned[
        "start_time_seconds"
    ] = pd.to_numeric(
        aligned[
            "start_time_seconds"
        ],
        errors="coerce"
    )

    aligned[
        "end_time_seconds"
    ] = pd.to_numeric(
        aligned[
            "end_time_seconds"
        ],
        errors="coerce"
    )

    aligned[
        "analysis_midi"
    ] = pd.to_numeric(
        aligned[
            "analysis_midi"
        ],
        errors="coerce"
    )


    # --------------------------------------------------------
    # Note duration
    # --------------------------------------------------------

    aligned[
        "notation_duration_seconds"
    ] = (
        aligned[
            "end_time_seconds"
        ]
        -
        aligned[
            "start_time_seconds"
        ]
    )

    aligned[
        "notation_duration_seconds"
    ] = np.maximum(
        aligned[
            "notation_duration_seconds"
        ],
        0.0
    )


    # --------------------------------------------------------
    # Note center time
    # --------------------------------------------------------

    aligned[
        "note_center_time_seconds"
    ] = (
        aligned[
            "start_time_seconds"
        ]
        +
        aligned[
            "end_time_seconds"
        ]
    ) / 2.0


    # --------------------------------------------------------
    # Western note
    # --------------------------------------------------------

    aligned[
        "western_note"
    ] = aligned[
        "analysis_midi"
    ].apply(
        midi_to_western_note
    )


    # --------------------------------------------------------
    # Western pitch class
    # --------------------------------------------------------

    aligned[
        "western_pitch_class"
    ] = aligned[
        "analysis_midi"
    ].apply(
        lambda x:
        int(round(x)) % 12
        if np.isfinite(x)
        else np.nan
    )


    # --------------------------------------------------------
    # Musical sequence number
    # --------------------------------------------------------

    aligned[
        "notation_sequence"
    ] = np.arange(
        1,
        len(aligned) + 1
    )


    # --------------------------------------------------------
    # Combined notation
    # --------------------------------------------------------

    aligned[
        "western_indian_notation"
    ] = (
        aligned[
            "western_note"
        ].fillna("Unknown")
        +
        " = "
        +
        aligned[
            "indian_swara"
        ].fillna("Unknown")
    )


    # --------------------------------------------------------
    # Phrase label
    # --------------------------------------------------------

    aligned[
        "phrase_label"
    ] = (
        "Phrase "
        +
        aligned[
            "phrase_id"
        ].astype(str)
    )


    # --------------------------------------------------------
    # Final human-readable notation
    # --------------------------------------------------------

    aligned[
        "notation_label"
    ] = (
        aligned[
            "western_note"
        ].fillna("Unknown")
        +
        " / "
        +
        aligned[
            "indian_swara"
        ].fillna("Unknown")
    )


    return aligned


# ============================================================
# PRINT ALIGNMENT STATISTICS
# ============================================================

def print_alignment_statistics(
    data,
    tonic
):

    print("\n" + "=" * 70)
    print(
        "PHASE 7.7 - "
        "WESTERN + INDIAN NOTATION ALIGNMENT"
    )
    print("=" * 70)

    print("\nTonic information:")

    print(
        f"Estimated Sa / Tonic : "
        f"{tonic['tonic_name']}"
    )

    print(
        f"Tonic frequency      : "
        f"{tonic['tonic_frequency']:.2f} Hz"
    )

    print(
        f"Tonic confidence     : "
        f"{tonic['confidence'] * 100:.2f}%"
    )

    print("\nAlignment statistics:")

    print(
        f"Total notation events : "
        f"{len(data)}"
    )

    print(
        f"Total phrases         : "
        f"{data['phrase_id'].nunique()}"
    )

    print(
        f"Total melody duration : "
        f"{data['notation_duration_seconds'].sum():.2f} sec"
    )

    print(
        f"Unique Western notes  : "
        f"{data['western_note'].nunique()}"
    )

    print(
        f"Unique Indian swaras  : "
        f"{data['indian_swara'].nunique()}"
    )


# ============================================================
# DISPLAY FIRST NOTATION EVENTS
# ============================================================

def display_first_notation_events(
    data
):

    print("\n" + "-" * 70)
    print(
        "FIRST 25 WESTERN + INDIAN "
        "NOTATION EVENTS"
    )
    print("-" * 70)

    display_columns = [
        "notation_sequence",
        "phrase_id",
        "start_time_seconds",
        "end_time_seconds",
        "notation_duration_seconds",
        "western_note",
        "indian_swara",
        "swara_short"
    ]

    available_columns = [
        column
        for column in display_columns
        if column in data.columns
    ]

    print(
        data[
            available_columns
        ]
        .head(25)
        .to_string(
            index=False
        )
    )


# ============================================================
# SAVE COMPLETE ALIGNMENT DATA
# ============================================================

def save_alignment_data(
    data
):

    os.makedirs(
        "data/output",
        exist_ok=True
    )

    output_file = (
        "data/output/"
        "western_indian_notation_alignment.csv"
    )

    data.to_csv(
        output_file,
        index=False
    )

    print(
        "\nWestern + Indian notation data saved!"
    )

    print(
        f"Saved file: {output_file}"
    )

    return output_file


# ============================================================
# CREATE HUMAN-READABLE NOTATION TABLE
# ============================================================

def create_notation_table(
    data
):

    os.makedirs(
        "data/output",
        exist_ok=True
    )

    output_file = (
        "data/output/"
        "musical_notation_table.csv"
    )

    table = data[
        [
            "notation_sequence",
            "phrase_id",
            "start_time_seconds",
            "end_time_seconds",
            "notation_duration_seconds",
            "western_note",
            "indian_swara",
            "swara_short",
            "notation_label"
        ]
    ].copy()

    table.to_csv(
        output_file,
        index=False
    )

    print(
        "\nHuman-readable notation table saved!"
    )

    print(
        f"Saved file: {output_file}"
    )

    return table


# ============================================================
# NOTATION TIMELINE GRAPH
# ============================================================

def create_notation_timeline_graph(
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
        ].to_numpy()
        +
        data[
            "end_time_seconds"
        ].to_numpy()
    ) / 2.0


    midi_values = (
        data[
            "analysis_midi"
        ]
        .to_numpy()
    )


    valid = np.isfinite(
        midi_values
    )


    plt.figure(
        figsize=(16, 7)
    )

    plt.scatter(
        center_times[valid],
        midi_values[valid],
        s=20
    )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Western MIDI Pitch"
    )

    plt.title(
        "Phase 7.7 - Western + Indian "
        "Notation Alignment"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    graph_path = (
        "graphs/"
        "western_indian_notation_alignment.png"
    )

    plt.savefig(
        graph_path,
        dpi=300
    )

    plt.close()

    print(
        "\nNotation alignment graph saved!"
    )

    print(
        f"Saved graph: {graph_path}"
    )


# ============================================================
# SWARA + WESTERN DISTRIBUTION
# ============================================================

def create_notation_distribution_graph(
    data
):

    os.makedirs(
        "graphs",
        exist_ok=True
    )

    swara_counts = (
        data[
            "indian_swara"
        ]
        .value_counts()
    )

    values = [
        swara_counts.get(
            swara,
            0
        )
        for swara in SWARA_ORDER
    ]


    plt.figure(
        figsize=(14, 6)
    )

    plt.bar(
        SWARA_ORDER,
        values
    )

    plt.xlabel(
        "Indian Swara"
    )

    plt.ylabel(
        "Number of Aligned Notes"
    )

    plt.title(
        "Phase 7.7 - Aligned Indian Swara Distribution"
    )

    plt.xticks(
        rotation=45
    )

    plt.grid(
        True,
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    graph_path = (
        "graphs/"
        "aligned_swara_distribution.png"
    )

    plt.savefig(
        graph_path,
        dpi=300
    )

    plt.close()

    print(
        "Aligned swara distribution graph saved!"
    )

    print(
        f"Saved graph: {graph_path}"
    )


# ============================================================
# PHRASE NOTATION GRAPH
# ============================================================

def create_phrase_notation_graph(
    data
):

    os.makedirs(
        "graphs",
        exist_ok=True
    )

    if len(data) == 0:

        return


    phrase_numbers = (
        data[
            "phrase_id"
        ]
        .to_numpy()
    )

    midi_values = (
        data[
            "analysis_midi"
        ]
        .to_numpy()
    )


    valid = np.isfinite(
        midi_values
    )


    plt.figure(
        figsize=(16, 7)
    )

    plt.scatter(
        phrase_numbers[valid],
        midi_values[valid],
        s=18
    )

    plt.xlabel(
        "Melodic Phrase"
    )

    plt.ylabel(
        "Western MIDI Pitch"
    )

    plt.title(
        "Phase 7.7 - Musical Phrase "
        "and Notation Alignment"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    graph_path = (
        "graphs/"
        "phrase_notation_alignment.png"
    )

    plt.savefig(
        graph_path,
        dpi=300
    )

    plt.close()

    print(
        "Phrase notation graph saved!"
    )

    print(
        f"Saved graph: {graph_path}"
    )


# ============================================================
# MAIN PHASE 7.7 FUNCTION
# ============================================================

def align_western_indian_notation():

    print("\n" + "=" * 70)
    print(
        "PHASE 7.7 - "
        "WESTERN + INDIAN NOTATION ALIGNMENT"
    )
    print("=" * 70)


    print(
        "\nLoading Phase 7.6 melody phrase data..."
    )

    data = load_alignment_data()

    print(
        f"Note events loaded: {len(data)}"
    )


    print(
        "\nLoading tonic estimation..."
    )

    tonic = load_tonic()

    print(
        f"Current Sa: "
        f"{tonic['tonic_name']}"
    )


    print(
        "\nAligning Western notes "
        "with Indian swaras..."
    )

    aligned_data = create_notation_alignment(
        data
    )


    print(
        "\nNotation alignment completed."
    )


    print_alignment_statistics(
        aligned_data,
        tonic
    )


    display_first_notation_events(
        aligned_data
    )


    save_alignment_data(
        aligned_data
    )


    create_notation_table(
        aligned_data
    )


    create_notation_timeline_graph(
        aligned_data
    )


    create_notation_distribution_graph(
        aligned_data
    )


    create_phrase_notation_graph(
        aligned_data
    )


    print("\n" + "=" * 70)
    print(
        "PHASE 7.7 COMPLETED SUCCESSFULLY"
    )
    print("=" * 70)


    return aligned_data


if __name__ == "__main__":

    align_western_indian_notation()