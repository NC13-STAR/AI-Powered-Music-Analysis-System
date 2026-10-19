import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PHASE 7.5 - INDIAN SWARA MAPPING
# ============================================================

NOTE_NAMES = [
    "C", "C#", "D", "D#", "E", "F",
    "F#", "G", "G#", "A", "A#", "B"
]


# ------------------------------------------------------------
# CHROMATIC SWARA MAPPING
# ------------------------------------------------------------
#
# Semitone distance from Sa:
#
# 0  -> Sa
# 1  -> Komal Re
# 2  -> Re
# 3  -> Komal Ga
# 4  -> Ga
# 5  -> Ma
# 6  -> Tivra Ma
# 7  -> Pa
# 8  -> Komal Dha
# 9  -> Dha
# 10 -> Komal Ni
# 11 -> Ni
#
# This is a pitch-class mapping.
# Raga-specific interpretation is NOT performed here.
# ------------------------------------------------------------

SWARA_NAMES = {
    0: "Sa",
    1: "Komal Re",
    2: "Re",
    3: "Komal Ga",
    4: "Ga",
    5: "Ma",
    6: "Tivra Ma",
    7: "Pa",
    8: "Komal Dha",
    9: "Dha",
    10: "Komal Ni",
    11: "Ni"
}


# Short names useful for tables/graphs
SWARA_SHORT_NAMES = {
    0: "Sa",
    1: "Re♭",
    2: "Re",
    3: "Ga♭",
    4: "Ga",
    5: "Ma",
    6: "Ma#",
    7: "Pa",
    8: "Dha♭",
    9: "Dha",
    10: "Ni♭",
    11: "Ni"
}


# ------------------------------------------------------------
# UTILITY FUNCTIONS
# ------------------------------------------------------------

def get_pitch_class(midi_number):
    """
    Convert MIDI number to pitch class.
    """

    if not np.isfinite(midi_number):
        return None

    return int(round(midi_number)) % 12


def get_western_note_name(midi_number):
    """
    Convert MIDI number to Western note name.
    """

    if not np.isfinite(midi_number):
        return None

    midi_number = int(round(midi_number))

    note_index = midi_number % 12
    octave = (midi_number // 12) - 1

    return f"{NOTE_NAMES[note_index]}{octave}"


# ------------------------------------------------------------
# SWARA FROM SEMITONE DISTANCE
# ------------------------------------------------------------

def semitone_to_swara(semitone_distance):
    """
    Convert semitone distance from Sa to Indian swara.
    """

    semitone_distance = int(
        semitone_distance
    ) % 12

    return SWARA_NAMES[
        semitone_distance
    ]


def semitone_to_swara_short(semitone_distance):
    """
    Short swara representation for graphs.
    """

    semitone_distance = int(
        semitone_distance
    ) % 12

    return SWARA_SHORT_NAMES[
        semitone_distance
    ]


# ------------------------------------------------------------
# LOAD TONIC
# ------------------------------------------------------------

def load_tonic():

    tonic_file = (
        "data/output/tonic_estimation.csv"
    )

    if not os.path.exists(tonic_file):

        raise FileNotFoundError(
            "\nTonic estimation file not found:\n"
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

    tonic_name = str(
        tonic_data.iloc[0]["estimated_tonic"]
    )

    tonic_pitch_class = int(
        tonic_data.iloc[0]["tonic_pitch_class"]
    )

    tonic_midi = int(
        tonic_data.iloc[0]["tonic_midi"]
    )

    tonic_frequency = float(
        tonic_data.iloc[0]["tonic_frequency_hz"]
    )

    confidence = float(
        tonic_data.iloc[0]["confidence"]
    )

    return (
        tonic_name,
        tonic_pitch_class,
        tonic_midi,
        tonic_frequency,
        confidence
    )


# ------------------------------------------------------------
# LOAD QUANTIZED NOTE DATA
# ------------------------------------------------------------

def load_note_data():

    note_file = (
        "data/output/quantized_notes.csv"
    )

    if not os.path.exists(note_file):

        raise FileNotFoundError(
            "\nQuantized note file not found:\n"
            f"{note_file}\n\n"
            "Please run Phase 7.2 first."
        )

    data = pd.read_csv(
        note_file
    )

    if len(data) == 0:

        raise ValueError(
            "Quantized note data is empty."
        )

    return data


# ------------------------------------------------------------
# MAP NOTES TO INDIAN SWARAS
# ------------------------------------------------------------

def map_notes_to_swaras(
    note_data,
    tonic_pitch_class
):

    data = note_data.copy()

    # --------------------------------------------------------
    # Determine MIDI column
    # --------------------------------------------------------

    if "quantized_midi" in data.columns:

        midi_values = pd.to_numeric(
            data["quantized_midi"],
            errors="coerce"
        )

    elif "rounded_midi" in data.columns:

        midi_values = pd.to_numeric(
            data["rounded_midi"],
            errors="coerce"
        )

    elif "mean_midi" in data.columns:

        midi_values = pd.to_numeric(
            data["mean_midi"],
            errors="coerce"
        )

    else:

        raise ValueError(
            "No MIDI column found in quantized note data."
        )

    data["analysis_midi"] = midi_values

    # --------------------------------------------------------
    # Calculate pitch class
    # --------------------------------------------------------

    data["pitch_class"] = (
        data["analysis_midi"]
        .round()
        .astype("Int64")
        % 12
    )

    # --------------------------------------------------------
    # Calculate semitone distance from Sa
    # --------------------------------------------------------

    semitone_distances = []

    swaras = []
    swara_short = []
    western_notes = []

    for _, row in data.iterrows():

        midi = row["analysis_midi"]

        if not np.isfinite(midi):

            semitone_distances.append(
                np.nan
            )

            swaras.append(
                None
            )

            swara_short.append(
                None
            )

            western_notes.append(
                None
            )

            continue

        pitch_class = int(
            round(midi)
        ) % 12

        distance = (
            pitch_class -
            tonic_pitch_class
        ) % 12

        semitone_distances.append(
            distance
        )

        swaras.append(
            semitone_to_swara(
                distance
            )
        )

        swara_short.append(
            semitone_to_swara_short(
                distance
            )
        )

        western_notes.append(
            get_western_note_name(
                midi
            )
        )

    data["semitone_from_sa"] = (
        semitone_distances
    )

    data["indian_swara"] = (
        swaras
    )

    data["swara_short"] = (
        swara_short
    )

    data["western_note_calculated"] = (
        western_notes
    )

    return data


# ------------------------------------------------------------
# PRINT STATISTICS
# ------------------------------------------------------------

def print_swara_statistics(
    data,
    tonic_name,
    tonic_frequency,
    confidence
):

    print("\n" + "-" * 70)
    print("INDIAN SWARA MAPPING RESULT")
    print("-" * 70)

    print(
        f"Reference Sa          : {tonic_name}"
    )

    print(
        f"Sa frequency          : "
        f"{tonic_frequency:.2f} Hz"
    )

    print(
        f"Tonic confidence      : "
        f"{confidence * 100:.2f}%"
    )

    print(
        f"Total note events     : {len(data)}"
    )

    print("\nSwara occurrence:")

    counts = (
        data["indian_swara"]
        .value_counts()
    )

    # Fixed musical order
    swara_order = [
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

    for swara in swara_order:

        count = counts.get(
            swara,
            0
        )

        print(
            f"{swara:12s} : {count}"
        )


# ------------------------------------------------------------
# SAVE DATA
# ------------------------------------------------------------

def save_swara_data(data):

    os.makedirs(
        "data/output",
        exist_ok=True
    )

    output_file = (
        "data/output/indian_swara_data.csv"
    )

    data.to_csv(
        output_file,
        index=False
    )

    print(
        "\nIndian swara data saved successfully!"
    )

    print(
        f"Saved file: {output_file}"
    )

    return output_file


# ------------------------------------------------------------
# CREATE SWARA DISTRIBUTION GRAPH
# ------------------------------------------------------------

def create_swara_distribution_graph(
    data
):

    os.makedirs(
        "graphs",
        exist_ok=True
    )

    swara_order = [
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

    counts = (
        data["indian_swara"]
        .value_counts()
    )

    values = [
        counts.get(
            swara,
            0
        )
        for swara in swara_order
    ]

    plt.figure(
        figsize=(14, 6)
    )

    plt.bar(
        swara_order,
        values
    )

    plt.xlabel(
        "Indian Swara"
    )

    plt.ylabel(
        "Number of Note Events"
    )

    plt.title(
        "Phase 7.5 - Indian Swara Distribution"
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
        "graphs/indian_swara_distribution.png"
    )

    plt.savefig(
        graph_path,
        dpi=300
    )

    plt.close()

    print(
        "\nIndian swara distribution graph saved!"
    )

    print(
        f"Saved graph: {graph_path}"
    )


# ------------------------------------------------------------
# CREATE SWARA TIMELINE GRAPH
# ------------------------------------------------------------

def create_swara_timeline_graph(
    data
):

    os.makedirs(
        "graphs",
        exist_ok=True
    )

    if (
        "start_time_seconds" not in data.columns
        or
        "end_time_seconds" not in data.columns
    ):

        print(
            "\nTiming columns unavailable."
        )

        return

    start_times = (
        data[
            "start_time_seconds"
        ]
        .astype(float)
        .to_numpy()
    )

    end_times = (
        data[
            "end_time_seconds"
        ]
        .astype(float)
        .to_numpy()
    )

    center_times = (
        start_times +
        end_times
    ) / 2.0

    swara_order = [
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

    swara_to_number = {
        swara: index
        for index, swara
        in enumerate(
            swara_order
        )
    }

    y_values = []

    for swara in data[
        "indian_swara"
    ]:

        y_values.append(
            swara_to_number.get(
                swara,
                np.nan
            )
        )

    plt.figure(
        figsize=(16, 7)
    )

    plt.scatter(
        center_times,
        y_values,
        s=18
    )

    plt.yticks(
        range(
            len(swara_order)
        ),
        swara_order
    )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Indian Swara"
    )

    plt.title(
        "Phase 7.5 - Indian Swara Melody Timeline"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    graph_path = (
        "graphs/indian_swara_timeline.png"
    )

    plt.savefig(
        graph_path,
        dpi=300
    )

    plt.close()

    print(
        "Indian swara timeline graph saved!"
    )

    print(
        f"Saved graph: {graph_path}"
    )


# ------------------------------------------------------------
# MAIN PHASE 7.5 FUNCTION
# ------------------------------------------------------------

def perform_swara_mapping():

    print("\n" + "=" * 70)
    print(
        "PHASE 7.5 - INDIAN SWARA MAPPING"
    )
    print("=" * 70)

    print(
        "\nLoading tonic estimation..."
    )

    (
        tonic_name,
        tonic_pitch_class,
        tonic_midi,
        tonic_frequency,
        confidence
    ) = load_tonic()

    print(
        f"Detected Sa / Tonic : {tonic_name}"
    )

    print(
        f"Tonic pitch class   : "
        f"{tonic_pitch_class}"
    )

    print(
        f"Tonic MIDI          : "
        f"{tonic_midi}"
    )

    print(
        f"Tonic frequency     : "
        f"{tonic_frequency:.2f} Hz"
    )

    print(
        f"Tonic confidence    : "
        f"{confidence * 100:.2f}%"
    )

    print(
        "\nLoading quantized note data..."
    )

    note_data = load_note_data()

    print(
        f"Note events loaded  : "
        f"{len(note_data)}"
    )

    print(
        "\nMapping Western notes "
        "relative to Sa..."
    )

    mapped_data = map_notes_to_swaras(
        note_data,
        tonic_pitch_class
    )

    print(
        "\nSwara mapping completed."
    )

    # --------------------------------------------------------
    # Print first 20 mappings
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("FIRST 20 NOTE-TO-SWARA MAPPINGS")
    print("-" * 70)

    display_columns = [
        "start_time_seconds",
        "end_time_seconds",
        "western_note_calculated",
        "semitone_from_sa",
        "indian_swara"
    ]

    available_columns = [
        column
        for column in display_columns
        if column in mapped_data.columns
    ]

    print(
        mapped_data[
            available_columns
        ]
        .head(20)
        .to_string(
            index=False
        )
    )

    print_swara_statistics(
        mapped_data,
        tonic_name,
        tonic_frequency,
        confidence
    )

    save_swara_data(
        mapped_data
    )

    create_swara_distribution_graph(
        mapped_data
    )

    create_swara_timeline_graph(
        mapped_data
    )

    print("\n" + "=" * 70)
    print(
        "PHASE 7.5 COMPLETED SUCCESSFULLY"
    )
    print("=" * 70)

    return mapped_data