import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------------------------
# Western note conversion
# ------------------------------------------------

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


def frequency_to_midi(frequency):
    """
    Convert frequency in Hz to MIDI pitch number.
    """

    if frequency <= 0:
        return np.nan

    midi = (
        69
        + 12 * np.log2(frequency / 440.0)
    )

    return midi


def midi_to_note(midi_number):
    """
    Convert MIDI pitch number to
    Western musical note name.
    """

    if not np.isfinite(midi_number):
        return None

    midi_number = int(
        round(midi_number)
    )

    note_index = midi_number % 12

    octave = (
        midi_number // 12
    ) - 1

    note_name = NOTE_NAMES[note_index]

    return f"{note_name}{octave}"


# ------------------------------------------------
# Phase 6.6
# ------------------------------------------------

def convert_f0_to_western_notes(
    f0,
    times
):
    """
    Phase 6.6:

    Convert final F0 melody contour into
    Western musical note names.

    Processing:
    1. Read final F0 values.
    2. Convert Hz -> MIDI.
    3. Round MIDI to nearest semitone.
    4. Convert MIDI -> Western note name.
    5. Preserve genuine gaps.
    6. Save note data and graph.
    """

    print("\n" + "=" * 60)
    print("PHASE 6.6 - F0 TO WESTERN MUSICAL NOTES")
    print("=" * 60)

    print(
        "\nConverting final melody F0 "
        "into Western musical notes..."
    )

    f0 = np.array(
        f0,
        dtype=float
    )

    times = np.array(
        times,
        dtype=float
    )

    # ------------------------------------------------
    # Step 1 - Valid pitch detection
    # ------------------------------------------------

    valid = (
        np.isfinite(f0)
        & (f0 > 0)
    )

    valid_count = np.sum(valid)

    print("\nStep 1 - Valid F0 values")

    print(
        f"Total frames      : {len(f0)}"
    )

    print(
        f"Valid F0 frames   : {valid_count}"
    )

    print(
        f"Missing F0 frames : "
        f"{len(f0) - valid_count}"
    )

    # ------------------------------------------------
    # Step 2 - Frequency -> MIDI
    # ------------------------------------------------

    midi_values = np.full(
        len(f0),
        np.nan
    )

    for i in range(len(f0)):

        if valid[i]:

            midi_values[i] = (
                frequency_to_midi(
                    f0[i]
                )
            )

    print(
        "\nStep 2 - F0 to MIDI conversion completed."
    )

    # ------------------------------------------------
    # Step 3 - MIDI -> nearest semitone
    # ------------------------------------------------

    rounded_midi = np.full(
        len(midi_values),
        np.nan
    )

    for i in range(len(midi_values)):

        if np.isfinite(
            midi_values[i]
        ):

            rounded_midi[i] = round(
                midi_values[i]
            )

    print(
        "Step 3 - MIDI values rounded "
        "to nearest semitone."
    )

    # ------------------------------------------------
    # Step 4 - MIDI -> Western note
    # ------------------------------------------------

    note_names = []

    for midi in rounded_midi:

        if np.isfinite(midi):

            note_names.append(
                midi_to_note(midi)
            )

        else:

            note_names.append(
                None
            )

    print(
        "Step 4 - Western note conversion completed."
    )

    # ------------------------------------------------
    # Statistics
    # ------------------------------------------------

    detected_notes = [
        note
        for note in note_names
        if note is not None
    ]

    unique_notes = sorted(
        set(detected_notes)
    )

    print("\nWestern note statistics:")

    print(
        f"Detected note frames : "
        f"{len(detected_notes)}"
    )

    print(
        f"Unique notes detected: "
        f"{len(unique_notes)}"
    )

    if unique_notes:

        print(
            "\nNotes detected:"
        )

        print(
            ", ".join(
                unique_notes
            )
        )

    # ------------------------------------------------
    # Step 5 - Save data
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
        "data/output/"
        "western_note_data.csv"
    )

    dataframe = pd.DataFrame(
        {
            "time_seconds": times,
            "f0_hz": f0,
            "midi_number": rounded_midi,
            "western_note": note_names
        }
    )

    dataframe.to_csv(
        output_file,
        index=False
    )

    print(
        "\nWestern note data saved successfully!"
    )

    print(
        f"Saved file: {output_file}"
    )

    # ------------------------------------------------
    # Step 6 - Plot note sequence
    # ------------------------------------------------

    plt.figure(
        figsize=(16, 6)
    )

    plt.scatter(
        times[valid],
        rounded_midi[valid],
        s=8
    )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "MIDI Pitch Number"
    )

    plt.title(
        "Phase 6.6 - Western Musical Notes"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    graph_path = (
        "graphs/"
        "western_notes.png"
    )

    plt.savefig(
        graph_path,
        dpi=300
    )

    plt.close()

    print(
        "Western note graph saved successfully!"
    )

    print(
        f"Saved graph: {graph_path}"
    )

    # ------------------------------------------------
    # Final message
    # ------------------------------------------------

    print("\n" + "=" * 60)

    print(
        "PHASE 6.6 COMPLETED SUCCESSFULLY"
    )

    print("=" * 60)

    return (
        dataframe
    )