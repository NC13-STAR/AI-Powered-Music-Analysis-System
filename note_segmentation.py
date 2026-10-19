import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


NOTE_NAMES = [
    "C", "C#", "D", "D#", "E", "F",
    "F#", "G", "G#", "A", "A#", "B"
]


# ============================================================
# BASIC NOTE CONVERSION FUNCTIONS
# ============================================================

def frequency_to_midi(frequency):
    if frequency <= 0 or not np.isfinite(frequency):
        return np.nan

    return 69 + 12 * np.log2(frequency / 440.0)


def midi_to_note(midi_number):
    if not np.isfinite(midi_number):
        return None

    midi_number = int(round(midi_number))

    note_index = midi_number % 12
    octave = (midi_number // 12) - 1

    note_name = NOTE_NAMES[note_index]

    return f"{note_name}{octave}"


def note_class_from_name(note_name):
    """
    Converts D3 -> D
    Converts A#4 -> A#
    """

    if note_name is None:
        return None

    for note in NOTE_NAMES:
        if note_name.startswith(note):
            return note

    return None


# ============================================================
# FIND CONTINUOUS PITCH REGIONS
# ============================================================

def find_valid_segments(valid):

    segments = []

    start = None

    for i in range(len(valid)):

        if valid[i] and start is None:
            start = i

        elif not valid[i] and start is not None:

            segments.append((start, i))

            start = None

    if start is not None:
        segments.append((start, len(valid)))

    return segments


# ============================================================
# PHASE 7.1
# MUSICAL NOTE SEGMENTATION
# ============================================================

def segment_melody_notes(
    f0,
    times,
    min_note_duration=0.08,
    pitch_tolerance_semitones=0.75,
    minimum_stable_frames=3
):

    print("\n" + "=" * 70)
    print("PHASE 7.1 - MUSICAL NOTE SEGMENTATION")
    print("=" * 70)

    print("\nConverting final melody contour into musical note segments...")

    f0 = np.array(f0, dtype=float)
    times = np.array(times, dtype=float)

    if len(f0) != len(times):
        raise ValueError(
            "F0 and time arrays must have the same length."
        )

    # --------------------------------------------------------
    # VALID PITCH
    # --------------------------------------------------------

    valid = np.isfinite(f0) & (f0 > 0)

    total_frames = len(f0)
    valid_frames = np.sum(valid)

    print("\nInput statistics:")
    print(f"Total frames       : {total_frames}")
    print(f"Valid melody frames: {valid_frames}")
    print(f"Missing frames     : {total_frames - valid_frames}")

    # --------------------------------------------------------
    # F0 -> CONTINUOUS MIDI
    # --------------------------------------------------------

    midi_continuous = np.full(
        len(f0),
        np.nan,
        dtype=float
    )

    for i in range(len(f0)):

        if valid[i]:

            midi_continuous[i] = frequency_to_midi(
                f0[i]
            )

    print("\nContinuous MIDI conversion completed.")

    # --------------------------------------------------------
    # FIND CONTINUOUS REGIONS
    # --------------------------------------------------------

    continuous_regions = find_valid_segments(valid)

    print(
        f"\nContinuous pitch regions found: "
        f"{len(continuous_regions)}"
    )

    # --------------------------------------------------------
    # FRAME DURATION
    # --------------------------------------------------------

    if len(times) > 1:

        frame_duration = np.median(
            np.diff(times)
        )

    else:

        frame_duration = 0.0

    if frame_duration <= 0:

        raise ValueError(
            "Unable to determine frame duration."
        )

    minimum_frames = max(
        minimum_stable_frames,
        int(
            np.ceil(
                min_note_duration /
                frame_duration
            )
        )
    )

    print(
        f"Frame duration       : "
        f"{frame_duration:.4f} seconds"
    )

    print(
        f"Minimum note duration: "
        f"{min_note_duration:.2f} seconds"
    )

    print(
        f"Minimum frames       : "
        f"{minimum_frames}"
    )

    # --------------------------------------------------------
    # NOTE SEGMENTATION
    # --------------------------------------------------------

    note_events = []

    candidate_segments = 0
    rejected_segments = 0

    for region_start, region_end in continuous_regions:

        region_length = region_end - region_start

        if region_length < minimum_frames:

            rejected_segments += 1

            continue

        start = region_start

        while start < region_end:

            current_end = start + 1

            current_reference = midi_continuous[start]

            if not np.isfinite(current_reference):

                start += 1

                continue

            # ------------------------------------------------
            # FOLLOW SAME MUSICAL NOTE
            # ------------------------------------------------

            while current_end < region_end:

                current_pitch = midi_continuous[current_end]

                if not np.isfinite(current_pitch):

                    break

                pitch_difference = abs(
                    current_pitch -
                    current_reference
                )

                if pitch_difference > pitch_tolerance_semitones:

                    break

                reference_start = max(
                    start,
                    current_end - 5
                )

                reference_values = (
                    midi_continuous[
                        reference_start:
                        current_end + 1
                    ]
                )

                reference_values = (
                    reference_values[
                        np.isfinite(reference_values)
                    ]
                )

                if len(reference_values) > 0:

                    current_reference = np.median(
                        reference_values
                    )

                current_end += 1

            # ------------------------------------------------
            # SEGMENT LENGTH
            # ------------------------------------------------

            segment_length = (
                current_end - start
            )

            candidate_segments += 1

            if segment_length >= minimum_frames:

                segment_midi = (
                    midi_continuous[
                        start:current_end
                    ]
                )

                segment_f0 = (
                    f0[
                        start:current_end
                    ]
                )

                valid_midi = (
                    segment_midi[
                        np.isfinite(segment_midi)
                    ]
                )

                valid_f0 = (
                    segment_f0[
                        np.isfinite(segment_f0)
                    ]
                )

                if (
                    len(valid_midi) > 0
                    and len(valid_f0) > 0
                ):

                    median_midi = np.median(
                        valid_midi
                    )

                    mean_midi = np.mean(
                        valid_midi
                    )

                    median_f0 = np.median(
                        valid_f0
                    )

                    mean_f0 = np.mean(
                        valid_f0
                    )

                    start_time = times[start]

                    end_time = (
                        times[current_end - 1]
                        + frame_duration
                    )

                    duration = (
                        end_time -
                        start_time
                    )

                    rounded_midi = int(
                        round(median_midi)
                    )

                    western_note = midi_to_note(
                        rounded_midi
                    )

                    note_class = (
                        note_class_from_name(
                            western_note
                        )
                    )

                    note_events.append({

                        "start_time_seconds":
                            start_time,

                        "end_time_seconds":
                            end_time,

                        "duration_seconds":
                            duration,

                        "median_f0_hz":
                            median_f0,

                        "mean_f0_hz":
                            mean_f0,

                        "continuous_midi":
                            median_midi,

                        "mean_midi":
                            mean_midi,

                        "rounded_midi":
                            rounded_midi,

                        "western_note":
                            western_note,

                        "note_class":
                            note_class
                    })

            else:

                rejected_segments += 1

            start = current_end

    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    dataframe = pd.DataFrame(
        note_events
    )

    print("\nSegmentation statistics:")

    print(
        f"Candidate note segments: "
        f"{candidate_segments}"
    )

    print(
        f"Accepted note segments : "
        f"{len(dataframe)}"
    )

    print(
        f"Rejected short segments: "
        f"{rejected_segments}"
    )

    # ========================================================
    # SAVE CSV
    # ========================================================

    os.makedirs(
        "data/output",
        exist_ok=True
    )

    os.makedirs(
        "graphs",
        exist_ok=True
    )

    output_file = (
        "data/output/note_segments.csv"
    )

    dataframe.to_csv(
        output_file,
        index=False
    )

    print(
        "\nNote segmentation data saved successfully!"
    )

    print(
        f"Saved file: {output_file}"
    )

    # ========================================================
    # PRINT FIRST NOTES
    # ========================================================

    if len(dataframe) > 0:

        print(
            "\nFirst detected musical notes:"
        )

        display_columns = [

            "start_time_seconds",

            "end_time_seconds",

            "duration_seconds",

            "median_f0_hz",

            "rounded_midi",

            "western_note"
        ]

        print(
            dataframe[
                display_columns
            ].head(20).to_string(
                index=False
            )
        )

    else:

        print(
            "\nWARNING: "
            "No musical note segments were detected."
        )

    # ========================================================
    # CREATE IMPROVED GRAPHS
    # ========================================================

    if len(dataframe) > 0:

        create_note_graphs(
            dataframe
        )

    # ========================================================
    # COMPLETION MESSAGE
    # ========================================================

    print("\n" + "=" * 70)
    print("PHASE 7.1 COMPLETED SUCCESSFULLY")
    print("=" * 70)

    return dataframe


# ============================================================
# GRAPH GENERATION
# ============================================================

def create_note_graphs(dataframe):

    print(
        "\nCreating improved musical note graphs..."
    )

    # --------------------------------------------------------
    # FULL SONG OVERVIEW
    # --------------------------------------------------------

    create_overview_graph(
        dataframe
    )

    # --------------------------------------------------------
    # 30 SECOND DETAIL GRAPHS
    # --------------------------------------------------------

    create_detail_graphs(
        dataframe
    )


# ============================================================
# FULL SONG OVERVIEW
# ============================================================

def create_overview_graph(dataframe):

    max_time = dataframe[
        "end_time_seconds"
    ].max()

    min_midi = dataframe[
        "rounded_midi"
    ].min()

    max_midi = dataframe[
        "rounded_midi"
    ].max()

    # Add small vertical margin
    y_min = min_midi - 2
    y_max = max_midi + 2

    fig, ax = plt.subplots(
        figsize=(20, 9)
    )

    # --------------------------------------------------------
    # NOTE CLASS COLORS
    # --------------------------------------------------------

    cmap = plt.get_cmap(
        "tab20"
    )

    note_color_map = {}

    for i, note_class in enumerate(
        NOTE_NAMES
    ):

        note_color_map[note_class] = (
            cmap(i % 20)
        )

    # --------------------------------------------------------
    # DRAW NOTE BARS
    # --------------------------------------------------------

    for _, row in dataframe.iterrows():

        start = row[
            "start_time_seconds"
        ]

        end = row[
            "end_time_seconds"
        ]

        midi = row[
            "rounded_midi"
        ]

        note_class = row[
            "note_class"
        ]

        color = note_color_map[
            note_class
        ]

        ax.plot(
            [start, end],
            [midi, midi],
            linewidth=6,
            solid_capstyle="round",
            color=color
        )

    # --------------------------------------------------------
    # AXES
    # --------------------------------------------------------

    ax.set_xlim(
        0,
        max_time
    )

    ax.set_ylim(
        y_min,
        y_max
    )

    ax.set_xlabel(
        "Time (seconds)",
        fontsize=13
    )

    ax.set_ylabel(
        "Western Pitch",
        fontsize=13
    )

    ax.set_title(
        "Phase 7.1 - Complete Melody Note Overview",
        fontsize=17,
        fontweight="bold"
    )

    # --------------------------------------------------------
    # Y AXIS NOTE LABELS
    # --------------------------------------------------------

    midi_ticks = np.arange(
        int(y_min),
        int(y_max) + 1
    )

    midi_labels = [
        midi_to_note(midi)
        for midi in midi_ticks
    ]

    ax.set_yticks(
        midi_ticks
    )

    ax.set_yticklabels(
        midi_labels,
        fontsize=9
    )

    # --------------------------------------------------------
    # GRID
    # --------------------------------------------------------

    ax.grid(
        True,
        axis="both",
        alpha=0.25
    )

    # --------------------------------------------------------
    # LEGEND
    # --------------------------------------------------------

    legend_items = []

    for note_class in NOTE_NAMES:

        legend_items.append(
            Line2D(
                [0],
                [0],
                color=note_color_map[
                    note_class
                ],
                linewidth=5,
                label=note_class
            )
        )

    ax.legend(
        handles=legend_items,
        title="Note Class",
        bbox_to_anchor=(
            1.01,
            1
        ),
        loc="upper left",
        ncol=1
    )

    plt.tight_layout()

    graph_path = (
        "graphs/"
        "note_segmentation_overview.png"
    )

    plt.savefig(
        graph_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"\nFull-song overview saved:"
    )

    print(
        f"Saved graph: {graph_path}"
    )


# ============================================================
# 30 SECOND DETAIL GRAPHS
# ============================================================

def create_detail_graphs(dataframe):

    max_time = dataframe[
        "end_time_seconds"
    ].max()

    window_size = 30.0

    number_of_windows = int(
        np.ceil(
            max_time /
            window_size
        )
    )

    print(
        f"\nCreating {number_of_windows} "
        f"detailed 30-second graphs..."
    )

    # --------------------------------------------------------
    # NOTE CLASS COLORS
    # --------------------------------------------------------

    cmap = plt.get_cmap(
        "tab20"
    )

    note_color_map = {}

    for i, note_class in enumerate(
        NOTE_NAMES
    ):

        note_color_map[note_class] = (
            cmap(i % 20)
        )

    # --------------------------------------------------------
    # EACH TIME WINDOW
    # --------------------------------------------------------

    for window_index in range(
        number_of_windows
    ):

        window_start = (
            window_index *
            window_size
        )

        window_end = min(
            window_start +
            window_size,
            max_time
        )

        # ----------------------------------------------------
        # SELECT NOTES IN THIS WINDOW
        # ----------------------------------------------------

        window_notes = dataframe[
            (
                dataframe[
                    "end_time_seconds"
                ] >= window_start
            )
            &
            (
                dataframe[
                    "start_time_seconds"
                ] <= window_end
            )
        ]

        if len(window_notes) == 0:

            continue

        min_midi = window_notes[
            "rounded_midi"
        ].min()

        max_midi = window_notes[
            "rounded_midi"
        ].max()

        y_min = min_midi - 2
        y_max = max_midi + 2

        # ----------------------------------------------------
        # CREATE FIGURE
        # ----------------------------------------------------

        fig, ax = plt.subplots(
            figsize=(20, 10)
        )

        # ----------------------------------------------------
        # DRAW NOTE BARS
        # ----------------------------------------------------

        for _, row in window_notes.iterrows():

            start = max(
                row[
                    "start_time_seconds"
                ],
                window_start
            )

            end = min(
                row[
                    "end_time_seconds"
                ],
                window_end
            )

            midi = row[
                "rounded_midi"
            ]

            note_class = row[
                "note_class"
            ]

            western_note = row[
                "western_note"
            ]

            duration = row[
                "duration_seconds"
            ]

            color = note_color_map[
                note_class
            ]

            # ------------------------------------------------
            # NOTE BAR
            # ------------------------------------------------

            ax.plot(
                [start, end],
                [midi, midi],
                linewidth=8,
                solid_capstyle="round",
                color=color
            )

            # ------------------------------------------------
            # NOTE LABEL
            #
            # Only label reasonably long notes.
            # This prevents the graph from becoming
            # overcrowded.
            # ------------------------------------------------

            visible_duration = (
                end - start
            )

            if visible_duration >= 0.16:

                label_position = (
                    start +
                    visible_duration /
                    2
                )

                ax.text(
                    label_position,
                    midi + 0.22,
                    western_note,
                    fontsize=11,
                    fontweight="bold",
                    ha="center",
                    va="bottom"
                )

        # ----------------------------------------------------
        # AXIS RANGE
        # ----------------------------------------------------

        ax.set_xlim(
            window_start,
            window_end
        )

        ax.set_ylim(
            y_min,
            y_max
        )

        # ----------------------------------------------------
        # AXIS LABELS
        # ----------------------------------------------------

        ax.set_xlabel(
            "Time (seconds)",
            fontsize=14
        )

        ax.set_ylabel(
            "Western Pitch",
            fontsize=14
        )

        ax.set_title(
            "Phase 7.1 - Melody Notes "
            f"({window_start:.0f}-{window_end:.0f} seconds)",
            fontsize=18,
            fontweight="bold"
        )

        # ----------------------------------------------------
        # Y AXIS
        # ----------------------------------------------------

        midi_ticks = np.arange(
            int(y_min),
            int(y_max) + 1
        )

        midi_labels = [
            midi_to_note(midi)
            for midi in midi_ticks
        ]

        ax.set_yticks(
            midi_ticks
        )

        ax.set_yticklabels(
            midi_labels,
            fontsize=11
        )

        # ----------------------------------------------------
        # X AXIS
        # ----------------------------------------------------

        ax.tick_params(
            axis="x",
            labelsize=11
        )

        # ----------------------------------------------------
        # GRID
        # ----------------------------------------------------

        ax.grid(
            True,
            axis="both",
            alpha=0.25
        )

        # ----------------------------------------------------
        # LEGEND
        # ----------------------------------------------------

        legend_items = []

        used_note_classes = []

        for note_class in NOTE_NAMES:

            if note_class in (
                window_notes[
                    "note_class"
                ].values
            ):

                used_note_classes.append(
                    note_class
                )

        for note_class in used_note_classes:

            legend_items.append(
                Line2D(
                    [0],
                    [0],
                    color=note_color_map[
                        note_class
                    ],
                    linewidth=6,
                    label=note_class
                )
            )

        ax.legend(
            handles=legend_items,
            title="Note Class",
            loc="upper right",
            fontsize=10
        )

        # ----------------------------------------------------
        # EXPLANATION
        # ----------------------------------------------------

        ax.text(
            0.01,
            0.02,
            "Bar length = note duration   |   "
            "Vertical position = pitch   |   "
            "Color = note class   |   "
            "Blank area = no detected pitch",
            transform=ax.transAxes,
            fontsize=10,
            bbox=dict(
                alpha=0.8
            )
        )

        plt.tight_layout()

        # ----------------------------------------------------
        # FILE NAME
        # ----------------------------------------------------

        start_label = (
            f"{int(window_start):03d}"
        )

        end_label = (
            f"{int(window_end):03d}"
        )

        graph_path = (
            "graphs/"
            f"note_segmentation_"
            f"{start_label}_{end_label}s.png"
        )

        plt.savefig(
            graph_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(
            f"Saved detailed graph: "
            f"{graph_path}"
        )

    print(
        "\nAll improved note graphs "
        "created successfully!"
    )