from __future__ import annotations

import json
import math
import re
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
GRAPHS_DIR = PROJECT_ROOT / "graphs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# INPUT
# ============================================================

INPUT_FILE = OUTPUT_DIR / "repeated_melodic_motifs.csv"


# ============================================================
# OUTPUT FILES
# ============================================================

TRANSFORMATIONS_FILE = (
    OUTPUT_DIR / "phase_12_2_motif_transformations.csv"
)

SIMILARITY_FILE = (
    OUTPUT_DIR / "phase_12_2_motif_similarity.csv"
)

FREQUENCY_FILE = (
    OUTPUT_DIR / "phase_12_2_transformation_frequency.csv"
)

LOCATIONS_FILE = (
    OUTPUT_DIR / "phase_12_2_transformation_locations.csv"
)

COMPLEXITY_FILE = (
    OUTPUT_DIR / "phase_12_2_motif_complexity.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR / "phase_12_2_summary.json"
)


# ============================================================
# GRAPH FILES
# ============================================================

GRAPH_TRANSFORMATIONS = (
    GRAPHS_DIR / "phase_12_2_motif_transformations.png"
)

GRAPH_SIMILARITY = (
    GRAPHS_DIR / "phase_12_2_motif_similarity.png"
)

GRAPH_FREQUENCY = (
    GRAPHS_DIR / "phase_12_2_transformation_frequency.png"
)


# ============================================================
# SWARA DEFINITIONS
# ============================================================

SWARA_TO_VALUE = {
    "Sa": 0,
    "Re": 1,
    "Ga": 2,
    "Ma": 3,
    "Pa": 4,
    "Dha": 5,
    "Ni": 6,
}

VALUE_TO_SWARA = {
    value: swara
    for swara, value in SWARA_TO_VALUE.items()
}


# ============================================================
# DISPLAY
# ============================================================

def header(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def section(title):
    print()
    print("-" * 70)
    print(title)
    print("-" * 70)


# ============================================================
# SWARA NORMALIZATION
# ============================================================

def normalize_swara(value):
    """
    Normalize common swara spellings.

    Examples:
        sa -> Sa
        SA -> Sa
        re -> Re
        MA -> Ma
    """

    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    text = text.replace("'", "")
    text = text.replace('"', "")
    text = text.strip()

    lookup = {
        "sa": "Sa",
        "re": "Re",
        "ga": "Ga",
        "ma": "Ma",
        "pa": "Pa",
        "dha": "Dha",
        "ni": "Ni",
    }

    return lookup.get(text.lower())


# ============================================================
# PARSE SWARA MOTIF
# ============================================================

def parse_swara_motif(value):
    """
    Convert:

        Sa - Re - Ga

    into:

        ["Sa", "Re", "Ga"]
    """

    if value is None:
        return []

    text = str(value).strip()

    if not text:
        return []

    # Normalize separators
    text = text.replace("→", "-")
    text = text.replace(">", "-")
    text = text.replace("|", "-")
    text = text.replace(",", "-")
    text = text.replace(";", "-")
    text = text.replace("_", "-")

    parts = re.split(r"\s*-\s*|\s+", text)

    result = []

    for part in parts:
        part = part.strip()

        if not part:
            continue

        swara = normalize_swara(part)

        if swara is not None:
            result.append(swara)

    return result


# ============================================================
# SWARA TO NUMERIC VALUES
# ============================================================

def swara_values(sequence):
    return [
        SWARA_TO_VALUE[swara]
        for swara in sequence
        if swara in SWARA_TO_VALUE
    ]


# ============================================================
# INTERVALS
# ============================================================

def interval_sequence(values):
    if len(values) < 2:
        return []

    return [
        values[index + 1] - values[index]
        for index in range(len(values) - 1)
    ]


# ============================================================
# NORMALIZED INTERVAL PATTERN
# ============================================================

def normalized_intervals(values):
    intervals = interval_sequence(values)

    if not intervals:
        return []

    maximum = max(
        max(abs(value) for value in intervals),
        1
    )

    return [
        value / maximum
        for value in intervals
    ]


# ============================================================
# SEQUENCE COMPARISON
# ============================================================

def sequences_equal(a, b):
    if len(a) != len(b):
        return False

    if not a:
        return False

    return all(
        abs(float(x) - float(y)) < 0.0001
        for x, y in zip(a, b)
    )


# ============================================================
# TRANSPOSED RELATIONSHIP
# ============================================================

def is_transposition(a, b):
    """
    Same interval structure.

    Example:

        Sa Re Ga
        Re Ga Ma

    Both have:

        +1, +1
    """

    if len(a) != len(b):
        return False

    if len(a) < 2:
        return False

    intervals_a = interval_sequence(a)
    intervals_b = interval_sequence(b)

    return sequences_equal(
        intervals_a,
        intervals_b
    )


# ============================================================
# INVERSION
# ============================================================

def is_inversion(a, b):
    """
    Inversion reverses interval direction.

    Example:

        Sa Re Ga
        Ga Re Sa

    """

    if len(a) != len(b):
        return False

    if len(a) < 2:
        return False

    intervals_a = interval_sequence(a)
    intervals_b = interval_sequence(b)

    inverted = [
        -value
        for value in intervals_a
    ]

    return sequences_equal(
        inverted,
        intervals_b
    )


# ============================================================
# RETROGRADE
# ============================================================

def is_retrograde(a, b):
    """
    Retrograde reverses the motif.

    Example:

        Sa Re Ga

    becomes:

        Ga Re Sa
    """

    if len(a) != len(b):
        return False

    if len(a) < 2:
        return False

    reversed_a = list(reversed(a))

    offset = b[0] - reversed_a[0]

    transformed = [
        value + offset
        for value in reversed_a
    ]

    return sequences_equal(
        transformed,
        b
    )


# ============================================================
# RETROGRADE INVERSION
# ============================================================

def is_retrograde_inversion(a, b):
    """
    Reverse the motif and invert the intervals.
    """

    if len(a) != len(b):
        return False

    if len(a) < 2:
        return False

    reversed_a = list(reversed(a))

    base = reversed_a[0]

    inverted = [
        (2 * base) - value
        for value in reversed_a
    ]

    offset = b[0] - inverted[0]

    transformed = [
        value + offset
        for value in inverted
    ]

    return sequences_equal(
        transformed,
        b
    )


# ============================================================
# SIMILARITY
# ============================================================

def interval_similarity(a, b):
    """
    Compare interval structures.

    100 = identical
    0   = very different
    """

    intervals_a = interval_sequence(a)
    intervals_b = interval_sequence(b)

    if not intervals_a or not intervals_b:
        return 0.0

    if len(intervals_a) != len(intervals_b):
        return 0.0

    difference = np.mean(
        np.abs(
            np.asarray(intervals_a, dtype=float)
            -
            np.asarray(intervals_b, dtype=float)
        )
    )

    similarity = 100.0 * math.exp(
        -difference / 2.0
    )

    return max(
        0.0,
        min(100.0, float(similarity))
    )


# ============================================================
# SHAPE SIMILARITY
# ============================================================

def shape_similarity(a, b):
    """
    Compare directional movement.

    Example:

        up, up, down

    compared with:

        up, up, down
    """

    intervals_a = interval_sequence(a)
    intervals_b = interval_sequence(b)

    if len(intervals_a) != len(intervals_b):
        return 0.0

    if not intervals_a:
        return 0.0

    matches = 0

    for x, y in zip(intervals_a, intervals_b):

        direction_x = 0

        direction_y = 0

        if x > 0:
            direction_x = 1
        elif x < 0:
            direction_x = -1

        if y > 0:
            direction_y = 1
        elif y < 0:
            direction_y = -1

        if direction_x == direction_y:
            matches += 1

    return (
        100.0
        *
        matches
        /
        len(intervals_a)
    )


# ============================================================
# COMBINED SIMILARITY
# ============================================================

def combined_similarity(a, b):

    interval_score = interval_similarity(a, b)

    shape_score = shape_similarity(a, b)

    return (
        0.7 * interval_score
        +
        0.3 * shape_score
    )


# ============================================================
# LOAD PHASE 12.1
# ============================================================

def load_phase_12_1():

    section("LOADING PHASE 12.1 MOTIF DATA")

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(
        f"Loaded: {INPUT_FILE.name}"
    )

    print(
        f"Rows: {len(df)}"
    )

    print(
        f"Columns: {list(df.columns)}"
    )

    return df


# ============================================================
# BUILD MOTIF TABLE
# ============================================================

def build_motif_table(df):

    section("BUILDING SWARA MOTIF TABLE")

    required_columns = [
        "motif",
        "occurrence_count",
        "first_occurrence_note",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Phase 12.1 output is missing required "
            f"columns: {missing}"
        )

    records = []

    for index, row in df.iterrows():

        motif_text = str(
            row["motif"]
        ).strip()

        sequence = parse_swara_motif(
            motif_text
        )

        if not sequence:
            continue

        values = swara_values(
            sequence
        )

        occurrence_count = 1

        try:
            occurrence_count = int(
                float(
                    row["occurrence_count"]
                )
            )
        except (
            TypeError,
            ValueError
        ):
            occurrence_count = 1

        try:
            first_note = int(
                float(
                    row["first_occurrence_note"]
                )
            )
        except (
            TypeError,
            ValueError
        ):
            first_note = -1

        records.append(
            {
                "motif_id": f"M{index + 1}",
                "motif": motif_text,
                "swara_sequence": " - ".join(
                    sequence
                ),
                "sequence": sequence,
                "numeric_sequence": values,
                "note_count": len(sequence),
                "occurrence_count": max(
                    1,
                    occurrence_count
                ),
                "first_occurrence_note": first_note,
            }
        )

    result = pd.DataFrame(records)

    print(
        f"Usable swara motifs: {len(result)}"
    )

    if not result.empty:

        print()

        print(
            result[
                [
                    "motif_id",
                    "motif",
                    "occurrence_count",
                    "first_occurrence_note",
                ]
            ]
            .head(15)
            .to_string(index=False)
        )

    return result


# ============================================================
# TRANSFORMATION ANALYSIS
# ============================================================

def analyze_transformations(motifs):

    section(
        "PHASE 12.2.1 - TRANSFORMATION ANALYSIS"
    )

    records = []

    for i in range(len(motifs)):

        motif_a = motifs.iloc[i]

        sequence_a = motif_a[
            "numeric_sequence"
        ]

        for j in range(i + 1, len(motifs)):

            motif_b = motifs.iloc[j]

            sequence_b = motif_b[
                "numeric_sequence"
            ]

            if len(sequence_a) != len(sequence_b):
                continue

            transformation = None

            if is_transposition(
                sequence_a,
                sequence_b
            ):

                transformation = (
                    "Transposition"
                )

            elif is_inversion(
                sequence_a,
                sequence_b
            ):

                transformation = (
                    "Inversion"
                )

            elif is_retrograde(
                sequence_a,
                sequence_b
            ):

                transformation = (
                    "Retrograde"
                )

            elif is_retrograde_inversion(
                sequence_a,
                sequence_b
            ):

                transformation = (
                    "Retrograde-Inversion"
                )

            if transformation is None:
                continue

            similarity = combined_similarity(
                sequence_a,
                sequence_b
            )

            records.append(
                {
                    "motif_a": motif_a[
                        "motif_id"
                    ],
                    "motif_b": motif_b[
                        "motif_id"
                    ],
                    "motif_a_swara": motif_a[
                        "motif"
                    ],
                    "motif_b_swara": motif_b[
                        "motif"
                    ],
                    "transformation": transformation,
                    "note_count": len(sequence_a),
                    "occurrence_a": motif_a[
                        "occurrence_count"
                    ],
                    "occurrence_b": motif_b[
                        "occurrence_count"
                    ],
                    "similarity_score": round(
                        similarity,
                        4
                    ),
                    "motif_a_note": motif_a[
                        "first_occurrence_note"
                    ],
                    "motif_b_note": motif_b[
                        "first_occurrence_note"
                    ],
                }
            )

    result = pd.DataFrame(records)

    if result.empty:

        print(
            "No exact transformations detected."
        )

    else:

        print(
            f"Transformation relationships: "
            f"{len(result)}"
        )

        print()

        print(
            result[
                [
                    "motif_a_swara",
                    "motif_b_swara",
                    "transformation",
                    "similarity_score",
                ]
            ]
            .head(20)
            .to_string(index=False)
        )

    return result


# ============================================================
# ALL-PAIR SIMILARITY
# ============================================================

def analyze_similarity(motifs):

    section(
        "PHASE 12.2.2 - MOTIF SIMILARITY"
    )

    records = []

    for i in range(len(motifs)):

        motif_a = motifs.iloc[i]

        for j in range(i + 1, len(motifs)):

            motif_b = motifs.iloc[j]

            sequence_a = motif_a[
                "numeric_sequence"
            ]

            sequence_b = motif_b[
                "numeric_sequence"
            ]

            if len(sequence_a) != len(sequence_b):
                continue

            interval_score = interval_similarity(
                sequence_a,
                sequence_b
            )

            shape_score = shape_similarity(
                sequence_a,
                sequence_b
            )

            combined = (
                0.7 * interval_score
                +
                0.3 * shape_score
            )

            records.append(
                {
                    "motif_a": motif_a[
                        "motif_id"
                    ],
                    "motif_b": motif_b[
                        "motif_id"
                    ],
                    "motif_a_swara": motif_a[
                        "motif"
                    ],
                    "motif_b_swara": motif_b[
                        "motif"
                    ],
                    "note_count": len(
                        sequence_a
                    ),
                    "interval_similarity": round(
                        interval_score,
                        4
                    ),
                    "shape_similarity": round(
                        shape_score,
                        4
                    ),
                    "combined_similarity": round(
                        combined,
                        4
                    ),
                    "occurrence_a": motif_a[
                        "occurrence_count"
                    ],
                    "occurrence_b": motif_b[
                        "occurrence_count"
                    ],
                }
            )

    result = pd.DataFrame(records)

    if result.empty:

        print(
            "No same-length motif pairs available."
        )

    else:

        result = result.sort_values(
            "combined_similarity",
            ascending=False
        ).reset_index(drop=True)

        print(
            f"Similarity pairs: {len(result)}"
        )

        print()

        print(
            result[
                [
                    "motif_a_swara",
                    "motif_b_swara",
                    "combined_similarity",
                ]
            ]
            .head(15)
            .to_string(index=False)
        )

    return result


# ============================================================
# TRANSFORMATION FREQUENCY
# ============================================================

def analyze_frequency(transformations):

    section(
        "PHASE 12.2.3 - TRANSFORMATION FREQUENCY"
    )

    if transformations.empty:

        result = pd.DataFrame(
            {
                "transformation": [
                    "No exact transformation"
                ],
                "count": [0],
                "percentage": [0.0],
            }
        )

        return result

    counts = (
        transformations[
            "transformation"
        ]
        .value_counts()
        .reset_index()
    )

    counts.columns = [
        "transformation",
        "count",
    ]

    total = counts[
        "count"
    ].sum()

    counts["percentage"] = (
        counts["count"]
        /
        total
        *
        100.0
    ).round(4)

    print(
        counts.to_string(
            index=False
        )
    )

    return counts


# ============================================================
# TRANSFORMATION LOCATIONS
# ============================================================

def analyze_locations(transformations):

    section(
        "PHASE 12.2.4 - TRANSFORMATION LOCATIONS"
    )

    if transformations.empty:

        return pd.DataFrame(
            columns=[
                "motif_a",
                "motif_b",
                "transformation",
                "motif_a_note",
                "motif_b_note",
                "note_distance",
            ]
        )

    result = transformations.copy()

    result["note_distance"] = (
        result[
            "motif_b_note"
        ]
        -
        result[
            "motif_a_note"
        ]
    ).abs()

    result = result.sort_values(
        [
            "transformation",
            "motif_a_note",
        ]
    )

    print(
        f"Locations recorded: {len(result)}"
    )

    return result


# ============================================================
# MOTIF COMPLEXITY
# ============================================================

def calculate_complexity(motifs):

    section(
        "PHASE 12.2.5 - MOTIF COMPLEXITY"
    )

    records = []

    for _, motif in motifs.iterrows():

        values = motif[
            "numeric_sequence"
        ]

        intervals = interval_sequence(
            values
        )

        if not intervals:

            unique_intervals = 0
            direction_changes = 0
            average_interval = 0.0

        else:

            unique_intervals = len(
                set(intervals)
            )

            directions = []

            for interval in intervals:

                if interval > 0:
                    directions.append(1)

                elif interval < 0:
                    directions.append(-1)

                else:
                    directions.append(0)

            direction_changes = 0

            for index in range(
                1,
                len(directions)
            ):

                if (
                    directions[index]
                    !=
                    directions[index - 1]
                ):

                    direction_changes += 1

            average_interval = float(
                np.mean(
                    np.abs(
                        intervals
                    )
                )
            )

        length_score = min(
            40.0,
            len(values) * 5.0
        )

        interval_score = min(
            30.0,
            unique_intervals * 5.0
        )

        direction_score = min(
            30.0,
            direction_changes * 5.0
        )

        complexity = (
            length_score
            +
            interval_score
            +
            direction_score
        )

        records.append(
            {
                "motif_id": motif[
                    "motif_id"
                ],
                "motif": motif[
                    "motif"
                ],
                "note_count": len(
                    values
                ),
                "unique_interval_count":
                    unique_intervals,
                "direction_changes":
                    direction_changes,
                "average_interval_size":
                    round(
                        average_interval,
                        4
                    ),
                "complexity_score":
                    round(
                        complexity,
                        4
                    ),
            }
        )

    result = pd.DataFrame(records)

    print(
        f"Motifs analyzed: {len(result)}"
    )

    return result


# ============================================================
# GRAPH 1
# ============================================================

def create_transformation_graph(
    transformations
):

    section(
        "GRAPH 1 - MOTIF TRANSFORMATION TYPES"
    )

    plt.figure(
        figsize=(11, 7)
    )

    if transformations.empty:

        plt.text(
            0.5,
            0.5,
            "No exact motif transformations detected",
            ha="center",
            va="center",
            fontsize=16,
        )

        plt.axis("off")

    else:

        counts = (
            transformations[
                "transformation"
            ]
            .value_counts()
        )

        labels = counts.index.tolist()

        values = counts.values.tolist()

        x = np.arange(
            len(labels)
        )

        plt.bar(
            x,
            values
        )

        plt.xticks(
            x,
            labels,
            rotation=25,
            ha="right"
        )

        plt.ylabel(
            "Number of relationships"
        )

        plt.xlabel(
            "Transformation type"
        )

        plt.title(
            "Phase 12.2 - Motif Transformation Relationships"
        )

        for index, value in enumerate(
            values
        ):

            plt.text(
                index,
                value,
                str(value),
                ha="center",
                va="bottom"
            )

        plt.grid(
            axis="y",
            alpha=0.3
        )

    plt.tight_layout()

    plt.savefig(
        GRAPH_TRANSFORMATIONS,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {GRAPH_TRANSFORMATIONS}"
    )


# ============================================================
# GRAPH 2
# ============================================================

def create_similarity_graph(
    similarity
):

    section(
        "GRAPH 2 - MOTIF SIMILARITY"
    )

    plt.figure(
        figsize=(12, 7)
    )

    if similarity.empty:

        plt.text(
            0.5,
            0.5,
            "No comparable motif pairs found",
            ha="center",
            va="center",
            fontsize=16,
        )

        plt.axis("off")

    else:

        top = similarity.head(20)

        labels = [
            f"{row.motif_a_swara}\nvs\n{row.motif_b_swara}"
            for _, row in top.iterrows()
        ]

        values = (
            top[
                "combined_similarity"
            ]
            .to_numpy()
        )

        x = np.arange(
            len(labels)
        )

        plt.bar(
            x,
            values
        )

        plt.xticks(
            x,
            labels,
            rotation=70,
            ha="right"
        )

        plt.ylabel(
            "Similarity score"
        )

        plt.xlabel(
            "Motif pair"
        )

        plt.title(
            "Phase 12.2 - Highest Motif Similarity Relationships"
        )

        plt.ylim(
            0,
            100
        )

        plt.grid(
            axis="y",
            alpha=0.3
        )

    plt.tight_layout()

    plt.savefig(
        GRAPH_SIMILARITY,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {GRAPH_SIMILARITY}"
    )


# ============================================================
# GRAPH 3
# ============================================================

def create_frequency_graph(
    frequency
):

    section(
        "GRAPH 3 - TRANSFORMATION FREQUENCY"
    )

    plt.figure(
        figsize=(10, 7)
    )

    if frequency.empty:

        plt.text(
            0.5,
            0.5,
            "No transformation frequency data",
            ha="center",
            va="center",
            fontsize=16,
        )

        plt.axis("off")

    else:

        real_frequency = frequency[
            frequency["count"] > 0
        ]

        if real_frequency.empty:

            plt.text(
                0.5,
                0.5,
                "No exact transformation frequency detected",
                ha="center",
                va="center",
                fontsize=16,
            )

            plt.axis("off")

        else:

            labels = (
                real_frequency[
                    "transformation"
                ]
                .astype(str)
                .tolist()
            )

            values = (
                real_frequency[
                    "percentage"
                ]
                .to_numpy(
                    dtype=float
                )
            )

            x = np.arange(
                len(labels)
            )

            plt.bar(
                x,
                values
            )

            plt.xticks(
                x,
                labels,
                rotation=25,
                ha="right"
            )

            plt.ylabel(
                "Percentage of transformations"
            )

            plt.xlabel(
                "Transformation type"
            )

            plt.title(
                "Phase 12.2 - Transformation Frequency"
            )

            for index, value in enumerate(
                values
            ):

                plt.text(
                    index,
                    value,
                    f"{value:.1f}%",
                    ha="center",
                    va="bottom"
                )

            plt.grid(
                axis="y",
                alpha=0.3
            )

    plt.tight_layout()

    plt.savefig(
        GRAPH_FREQUENCY,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {GRAPH_FREQUENCY}"
    )


# ============================================================
# SUMMARY
# ============================================================

def create_summary(
    motifs,
    transformations,
    similarity,
    frequency,
    complexity
):

    if similarity.empty:

        mean_similarity = 0.0
        maximum_similarity = 0.0

    else:

        mean_similarity = float(
            similarity[
                "combined_similarity"
            ].mean()
        )

        maximum_similarity = float(
            similarity[
                "combined_similarity"
            ].max()
        )

    if complexity.empty:

        mean_complexity = 0.0

    else:

        mean_complexity = float(
            complexity[
                "complexity_score"
            ].mean()
        )

    transformation_counts = {}

    if not transformations.empty:

        counts = (
            transformations[
                "transformation"
            ]
            .value_counts()
        )

        transformation_counts = {
            str(key): int(value)
            for key, value
            in counts.items()
        }

    summary = {
        "phase": "12.2",
        "title":
            "Motif Transformation & Variation Analysis",
        "input_file":
            str(INPUT_FILE),
        "total_motifs":
            int(len(motifs)),
        "total_transformation_relationships":
            int(len(transformations)),
        "total_similarity_pairs":
            int(len(similarity)),
        "mean_similarity":
            round(
                mean_similarity,
                4
            ),
        "maximum_similarity":
            round(
                maximum_similarity,
                4
            ),
        "mean_complexity":
            round(
                mean_complexity,
                4
            ),
        "transformation_counts":
            transformation_counts,
        "analysis_type":
            "Indian swara-based motif analysis",
        "swara_system":
            [
                "Sa",
                "Re",
                "Ga",
                "Ma",
                "Pa",
                "Dha",
                "Ni",
            ],
        "notes": [
            "Motifs are read from Phase 12.1 swara strings.",
            "Transposition compares identical interval structures.",
            "Inversion compares opposite interval directions.",
            "Retrograde compares reversed motif sequences.",
            "Retrograde-inversion combines reversal and inversion.",
            "Similarity scores describe structural similarity and do not represent artistic quality.",
        ],
    }

    return summary


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_outputs(
    transformations,
    similarity,
    frequency,
    locations,
    complexity,
    summary
):

    section(
        "SAVING PHASE 12.2 OUTPUTS"
    )

    transformations.to_csv(
        TRANSFORMATIONS_FILE,
        index=False
    )

    similarity.to_csv(
        SIMILARITY_FILE,
        index=False
    )

    frequency.to_csv(
        FREQUENCY_FILE,
        index=False
    )

    locations.to_csv(
        LOCATIONS_FILE,
        index=False
    )

    complexity.to_csv(
        COMPLEXITY_FILE,
        index=False
    )

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    print(
        f"Saved: {TRANSFORMATIONS_FILE}"
    )

    print(
        f"Saved: {SIMILARITY_FILE}"
    )

    print(
        f"Saved: {FREQUENCY_FILE}"
    )

    print(
        f"Saved: {LOCATIONS_FILE}"
    )

    print(
        f"Saved: {COMPLEXITY_FILE}"
    )

    print(
        f"Saved: {SUMMARY_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    header(
        "PHASE 12.2 STANDALONE TEST"
    )

    print(
        "Swara-based Motif Transformation & Variation Analysis"
    )

    print(
        "No audio path is required."
    )

    print(
        f"Input: {INPUT_FILE}"
    )

    try:

        # ------------------------------------------------------
        # LOAD
        # ------------------------------------------------------

        raw_data = load_phase_12_1()

        # ------------------------------------------------------
        # BUILD MOTIFS
        # ------------------------------------------------------

        motifs = build_motif_table(
            raw_data
        )

        if motifs.empty:

            raise ValueError(
                "No valid swara motifs could be extracted "
                "from repeated_melodic_motifs.csv."
            )

        # ------------------------------------------------------
        # TRANSFORMATIONS
        # ------------------------------------------------------

        transformations = (
            analyze_transformations(
                motifs
            )
        )

        # ------------------------------------------------------
        # SIMILARITY
        # ------------------------------------------------------

        similarity = (
            analyze_similarity(
                motifs
            )
        )

        # ------------------------------------------------------
        # FREQUENCY
        # ------------------------------------------------------

        frequency = (
            analyze_frequency(
                transformations
            )
        )

        # ------------------------------------------------------
        # LOCATIONS
        # ------------------------------------------------------

        locations = (
            analyze_locations(
                transformations
            )
        )

        # ------------------------------------------------------
        # COMPLEXITY
        # ------------------------------------------------------

        complexity = (
            calculate_complexity(
                motifs
            )
        )

        # ------------------------------------------------------
        # SUMMARY
        # ------------------------------------------------------

        summary = create_summary(
            motifs,
            transformations,
            similarity,
            frequency,
            complexity
        )

        # ------------------------------------------------------
        # GRAPHS
        # ------------------------------------------------------

        create_transformation_graph(
            transformations
        )

        create_similarity_graph(
            similarity
        )

        create_frequency_graph(
            frequency
        )

        # ------------------------------------------------------
        # SAVE
        # ------------------------------------------------------

        save_outputs(
            transformations,
            similarity,
            frequency,
            locations,
            complexity,
            summary
        )

        # ------------------------------------------------------
        # FINAL SUMMARY
        # ------------------------------------------------------

        header(
            "PHASE 12.2 COMPLETED SUCCESSFULLY"
        )

        print(
            f"Motifs analyzed              : {len(motifs)}"
        )

        print(
            f"Transformations detected     : "
            f"{len(transformations)}"
        )

        print(
            f"Similarity pairs              : "
            f"{len(similarity)}"
        )

        print(
            f"Mean similarity               : "
            f"{summary['mean_similarity']:.2f}"
        )

        print(
            f"Mean complexity               : "
            f"{summary['mean_complexity']:.2f}"
        )

        print()

        print(
            "CSV OUTPUTS:"
        )

        print(
            f"  {TRANSFORMATIONS_FILE}"
        )

        print(
            f"  {SIMILARITY_FILE}"
        )

        print(
            f"  {FREQUENCY_FILE}"
        )

        print(
            f"  {LOCATIONS_FILE}"
        )

        print(
            f"  {COMPLEXITY_FILE}"
        )

        print()

        print(
            "SUMMARY:"
        )

        print(
            f"  {SUMMARY_FILE}"
        )

        print()

        print(
            "GRAPHS:"
        )

        print(
            f"  {GRAPH_TRANSFORMATIONS}"
        )

        print(
            f"  {GRAPH_SIMILARITY}"
        )

        print(
            f"  {GRAPH_FREQUENCY}"
        )

        print()

        print(
            "Phase 12.2 standalone test finished."
        )

    except FileNotFoundError as error:

        header(
            "PHASE 12.2 INPUT ERROR"
        )

        print(
            str(error)
        )

    except Exception as error:

        header(
            "PHASE 12.2 FAILED"
        )

        print(
            f"{type(error).__name__}: {error}"
        )

        raise


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()