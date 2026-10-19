from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# =============================================================================
# PHASE 12.3
# MOTIF DEVELOPMENT & STRUCTURAL ROLE ANALYSIS
#
# Standalone phase.
#
# Run from project root:
#
#     python src/phase_12_3.py
#
# Main input:
#
#     data/output/repeated_melodic_motifs.csv
#
# Optional inputs:
#
#     data/output/phase_12_2_motif_transformations.csv
#     data/output/phase_12_2_motif_similarity.csv
#     data/output/melodic_phrase_analysis.csv
#     data/output/advanced_phrase_melody_analysis.csv
#
# The phase is designed for SWARA-BASED motif data.
# =============================================================================


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
GRAPHS_DIR = PROJECT_ROOT / "graphs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# INPUT FILES
# =============================================================================

MOTIF_FILE = OUTPUT_DIR / "repeated_melodic_motifs.csv"

TRANSFORMATION_FILE = (
    OUTPUT_DIR / "phase_12_2_motif_transformations.csv"
)

SIMILARITY_FILE = (
    OUTPUT_DIR / "phase_12_2_motif_similarity.csv"
)

PHRASE_FILE = OUTPUT_DIR / "melodic_phrase_analysis.csv"

ADVANCED_PHRASE_FILE = (
    OUTPUT_DIR / "advanced_phrase_melody_analysis.csv"
)


# =============================================================================
# OUTPUT FILES
# =============================================================================

DEVELOPMENT_FILE = (
    OUTPUT_DIR / "phase_12_3_motif_development.csv"
)

DENSITY_FILE = (
    OUTPUT_DIR / "phase_12_3_motif_density.csv"
)

PROMINENCE_FILE = (
    OUTPUT_DIR / "phase_12_3_motif_prominence.csv"
)

STRUCTURE_FILE = (
    OUTPUT_DIR / "phase_12_3_structural_roles.csv"
)

TIMELINE_FILE = (
    OUTPUT_DIR / "phase_12_3_motif_timeline.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR / "phase_12_3_summary.json"
)


# =============================================================================
# GRAPH FILES
# =============================================================================

GRAPH_TIMELINE = (
    GRAPHS_DIR / "phase_12_3_motif_timeline.png"
)

GRAPH_FREQUENCY = (
    GRAPHS_DIR / "phase_12_3_motif_frequency.png"
)

GRAPH_DEVELOPMENT = (
    GRAPHS_DIR / "phase_12_3_motif_development.png"
)

GRAPH_PROMINENCE = (
    GRAPHS_DIR / "phase_12_3_motif_prominence.png"
)


# =============================================================================
# DISPLAY HELPERS
# =============================================================================

def print_header(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def print_section(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# =============================================================================
# TEXT HELPERS
# =============================================================================

def clean_text(value) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value).strip()


def normalize_swara(value) -> str:
    """
    Normalize a single swara name.

    Examples:
        Sa
        SA
        sa
        Sa#
    """

    text = clean_text(value)

    if not text:
        return ""

    return text.strip()


def parse_swara_sequence(value) -> list[str]:
    """
    Parse swara motifs such as:

        Pa - Pa - Pa
        Ni - Sa - Sa
        Sa, Re, Ga
        Sa|Re|Ga
        Sa Re Ga

    Returns:

        ["Pa", "Pa", "Pa"]
    """

    text = clean_text(value)

    if not text:
        return []

    text = text.replace("[", "")
    text = text.replace("]", "")
    text = text.replace("(", "")
    text = text.replace(")", "")

    text = text.replace("|", "-")
    text = text.replace(",", "-")
    text = text.replace(";", "-")
    text = text.replace(">", "-")

    parts = text.split("-")

    result = []

    for part in parts:
        part = normalize_swara(part)

        if not part:
            continue

        result.append(part)

    if len(result) == 1:
        parts = result[0].split()

        if len(parts) > 1:
            result = [
                normalize_swara(part)
                for part in parts
                if normalize_swara(part)
            ]

    return result


def swara_sequence_to_text(sequence: list[str]) -> str:
    return " - ".join(sequence)


# =============================================================================
# COLUMN HELPERS
# =============================================================================

def find_column(
    dataframe: pd.DataFrame,
    candidates: list[str]
):
    """
    Find a column using exact, normalized, and partial matching.
    """

    if dataframe.empty:
        return None

    columns = list(dataframe.columns)

    # Exact match.
    for candidate in candidates:
        if candidate in columns:
            return candidate

    # Normalized match.
    normalized_columns = {}

    for column in columns:
        key = (
            str(column)
            .lower()
            .replace("_", "")
            .replace("-", "")
            .replace(" ", "")
        )

        normalized_columns[key] = column

    for candidate in candidates:
        key = (
            str(candidate)
            .lower()
            .replace("_", "")
            .replace("-", "")
            .replace(" ", "")
        )

        if key in normalized_columns:
            return normalized_columns[key]

    # Partial match.
    for column in columns:
        column_key = (
            str(column)
            .lower()
            .replace("_", "")
            .replace("-", "")
            .replace(" ", "")
        )

        for candidate in candidates:
            candidate_key = (
                str(candidate)
                .lower()
                .replace("_", "")
                .replace("-", "")
                .replace(" ", "")
            )

            if (
                candidate_key
                and candidate_key in column_key
            ):
                return column

    return None


# =============================================================================
# SAFE NUMERIC CONVERSION
# =============================================================================

def safe_int(value, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default

        return int(float(value))

    except (TypeError, ValueError):
        return default


def safe_float(value, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default

        return float(value)

    except (TypeError, ValueError):
        return default


# =============================================================================
# LOAD MOTIF DATA
# =============================================================================

def load_motif_data() -> pd.DataFrame:
    print_section("LOADING PHASE 12.1 MOTIF DATA")

    if not MOTIF_FILE.exists():
        raise FileNotFoundError(
            "Phase 12.1 motif file was not found:\n"
            f"{MOTIF_FILE}\n\n"
            "Run Phase 12.1 successfully before Phase 12.3."
        )

    dataframe = pd.read_csv(MOTIF_FILE)

    print(
        f"[loaded] {MOTIF_FILE.name} "
        f"({len(dataframe)} rows, "
        f"{len(dataframe.columns)} columns)"
    )

    print()
    print("Available columns:")

    for column in dataframe.columns:
        print(f"    - {column}")

    return dataframe


# =============================================================================
# BUILD SWARA MOTIF TABLE
# =============================================================================

def build_motif_table(
    dataframe: pd.DataFrame
) -> pd.DataFrame:

    print_section("BUILDING SWARA MOTIF TABLE")

    if dataframe.empty:
        return pd.DataFrame(
            columns=[
                "motif_id",
                "motif",
                "swara_sequence",
                "note_count",
                "occurrence_count",
                "first_occurrence_note",
            ]
        )

    motif_column = find_column(
        dataframe,
        [
            "motif",
            "motif_sequence",
            "sequence",
            "swara_sequence",
            "pattern",
        ],
    )

    occurrence_column = find_column(
        dataframe,
        [
            "occurrence_count",
            "occurrences",
            "frequency",
            "count",
            "motif_frequency",
        ],
    )

    first_occurrence_column = find_column(
        dataframe,
        [
            "first_occurrence_note",
            "first_occurrence",
            "first_note",
            "first_position",
            "start_note",
        ],
    )

    if motif_column is None:
        raise ValueError(
            "Could not find a swara motif column in "
            "repeated_melodic_motifs.csv."
        )

    records = []

    for index, row in dataframe.iterrows():

        motif_text = clean_text(
            row[motif_column]
        )

        sequence = parse_swara_sequence(
            motif_text
        )

        if not sequence:
            continue

        occurrence_count = 1

        if occurrence_column is not None:
            occurrence_count = max(
                1,
                safe_int(
                    row[occurrence_column],
                    1,
                ),
            )

        first_occurrence_note = 0

        if first_occurrence_column is not None:
            first_occurrence_note = safe_int(
                row[first_occurrence_column],
                0,
            )

        records.append(
            {
                "motif_id": f"M{index + 1:03d}",
                "motif": swara_sequence_to_text(
                    sequence
                ),
                "swara_sequence": sequence,
                "note_count": len(sequence),
                "occurrence_count": occurrence_count,
                "first_occurrence_note": (
                    first_occurrence_note
                ),
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
                    "note_count",
                    "occurrence_count",
                    "first_occurrence_note",
                ]
            ]
            .head(15)
            .to_string(index=False)
        )

    return result


# =============================================================================
# LOAD PHASE 12.2 DATA
# =============================================================================

def load_optional_phase_12_2_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame
]:

    print_section("LOADING OPTIONAL PHASE 12.2 DATA")

    transformations = pd.DataFrame()
    similarity = pd.DataFrame()

    if TRANSFORMATION_FILE.exists():

        transformations = pd.read_csv(
            TRANSFORMATION_FILE
        )

        print(
            f"[loaded] {TRANSFORMATION_FILE.name} "
            f"({len(transformations)} rows)"
        )

    else:
        print(
            "[info] Phase 12.2 transformation file "
            "not found. Continuing without it."
        )

    if SIMILARITY_FILE.exists():

        similarity = pd.read_csv(
            SIMILARITY_FILE
        )

        print(
            f"[loaded] {SIMILARITY_FILE.name} "
            f"({len(similarity)} rows)"
        )

    else:
        print(
            "[info] Phase 12.2 similarity file "
            "not found. Continuing without it."
        )

    return transformations, similarity


# =============================================================================
# MOTIF DEVELOPMENT
# =============================================================================

def analyze_motif_development(
    motifs: pd.DataFrame
) -> pd.DataFrame:

    print_section(
        "PHASE 12.3.1 - MOTIF DEVELOPMENT ANALYSIS"
    )

    if motifs.empty:
        return pd.DataFrame()

    records = []

    max_occurrence = max(
        motifs["occurrence_count"].max(),
        1,
    )

    max_note_position = max(
        motifs["first_occurrence_note"].max(),
        1,
    )

    for _, motif in motifs.iterrows():

        occurrence_count = safe_int(
            motif["occurrence_count"],
            1,
        )

        first_note = safe_int(
            motif["first_occurrence_note"],
            0,
        )

        note_count = safe_int(
            motif["note_count"],
            0,
        )

        recurrence_strength = (
            occurrence_count
            / max_occurrence
            * 100.0
        )

        structural_position = (
            first_note
            / max_note_position
            * 100.0
        )

        persistence_score = min(
            100.0,
            occurrence_count * 10.0
        )

        development_score = (
            recurrence_strength * 0.45
            + persistence_score * 0.35
            + min(note_count * 5.0, 100.0) * 0.20
        )

        records.append(
            {
                "motif_id": motif["motif_id"],
                "motif": motif["motif"],
                "note_count": note_count,
                "occurrence_count": occurrence_count,
                "first_occurrence_note": first_note,
                "recurrence_strength": round(
                    recurrence_strength,
                    4,
                ),
                "structural_position_percent": round(
                    structural_position,
                    4,
                ),
                "persistence_score": round(
                    persistence_score,
                    4,
                ),
                "development_score": round(
                    development_score,
                    4,
                ),
            }
        )

    result = pd.DataFrame(records)

    result = result.sort_values(
        [
            "development_score",
            "occurrence_count",
        ],
        ascending=False,
    ).reset_index(drop=True)

    print(
        f"Motifs analyzed: {len(result)}"
    )

    print()
    print(
        result.head(15).to_string(
            index=False
        )
    )

    return result


# =============================================================================
# MOTIF DENSITY
# =============================================================================

def analyze_motif_density(
    motifs: pd.DataFrame
) -> pd.DataFrame:

    print_section(
        "PHASE 12.3.2 - MOTIF DENSITY ANALYSIS"
    )

    if motifs.empty:
        return pd.DataFrame()

    total_occurrences = int(
        motifs["occurrence_count"].sum()
    )

    if total_occurrences <= 0:
        total_occurrences = 1

    records = []

    for _, motif in motifs.iterrows():

        occurrence_count = safe_int(
            motif["occurrence_count"],
            1,
        )

        density_percent = (
            occurrence_count
            / total_occurrences
            * 100.0
        )

        records.append(
            {
                "motif_id": motif["motif_id"],
                "motif": motif["motif"],
                "occurrence_count": occurrence_count,
                "density_percent": round(
                    density_percent,
                    4,
                ),
            }
        )

    result = pd.DataFrame(records)

    result = result.sort_values(
        "density_percent",
        ascending=False,
    ).reset_index(drop=True)

    print(
        f"Total motif occurrences: "
        f"{total_occurrences}"
    )

    print()
    print(
        result.head(15).to_string(
            index=False
        )
    )

    return result


# =============================================================================
# MOTIF PROMINENCE
# =============================================================================

def analyze_motif_prominence(
    motifs: pd.DataFrame
) -> pd.DataFrame:

    print_section(
        "PHASE 12.3.3 - MOTIF PROMINENCE ANALYSIS"
    )

    if motifs.empty:
        return pd.DataFrame()

    max_frequency = max(
        motifs["occurrence_count"].max(),
        1,
    )

    max_length = max(
        motifs["note_count"].max(),
        1,
    )

    records = []

    for _, motif in motifs.iterrows():

        frequency_score = (
            safe_int(
                motif["occurrence_count"],
                1,
            )
            / max_frequency
            * 100.0
        )

        length_score = (
            safe_int(
                motif["note_count"],
                1,
            )
            / max_length
            * 100.0
        )

        prominence_score = (
            frequency_score * 0.70
            + length_score * 0.30
        )

        if prominence_score >= 75.0:
            category = "High"

        elif prominence_score >= 45.0:
            category = "Moderate"

        else:
            category = "Low"

        records.append(
            {
                "motif_id": motif["motif_id"],
                "motif": motif["motif"],
                "occurrence_count": safe_int(
                    motif["occurrence_count"],
                    1,
                ),
                "note_count": safe_int(
                    motif["note_count"],
                    0,
                ),
                "frequency_score": round(
                    frequency_score,
                    4,
                ),
                "length_score": round(
                    length_score,
                    4,
                ),
                "prominence_score": round(
                    prominence_score,
                    4,
                ),
                "prominence_category": category,
            }
        )

    result = pd.DataFrame(records)

    result = result.sort_values(
        "prominence_score",
        ascending=False,
    ).reset_index(drop=True)

    print(
        f"Motifs evaluated: {len(result)}"
    )

    print()
    print(
        result.head(15).to_string(
            index=False
        )
    )

    return result


# =============================================================================
# STRUCTURAL ROLE CLASSIFICATION
# =============================================================================

def classify_structural_role(
    occurrence_count: int,
    first_occurrence_note: int,
    prominence_score: float,
    total_notes: int,
) -> str:

    if total_notes <= 0:
        total_notes = 1

    position = (
        first_occurrence_note
        / total_notes
    )

    if occurrence_count >= 10:
        if position < 0.33:
            return "Early recurring motif"

        if position < 0.66:
            return "Middle recurring motif"

        return "Late recurring motif"

    if occurrence_count >= 5:
        if prominence_score >= 60:
            return "Developing motif"

        return "Moderately recurring motif"

    if position < 0.33:
        return "Early motif"

    if position < 0.66:
        return "Middle motif"

    return "Late motif"


def analyze_structural_roles(
    motifs: pd.DataFrame,
    prominence: pd.DataFrame,
) -> pd.DataFrame:

    print_section(
        "PHASE 12.3.4 - STRUCTURAL ROLE ANALYSIS"
    )

    if motifs.empty:
        return pd.DataFrame()

    prominence_lookup = {}

    if not prominence.empty:

        for _, row in prominence.iterrows():

            prominence_lookup[
                clean_text(row["motif_id"])
            ] = safe_float(
                row["prominence_score"],
                0.0,
            )

    total_notes = int(
        motifs["first_occurrence_note"].max()
    )

    records = []

    for _, motif in motifs.iterrows():

        motif_id = clean_text(
            motif["motif_id"]
        )

        occurrence_count = safe_int(
            motif["occurrence_count"],
            1,
        )

        first_occurrence_note = safe_int(
            motif["first_occurrence_note"],
            0,
        )

        prominence_score = prominence_lookup.get(
            motif_id,
            0.0,
        )

        role = classify_structural_role(
            occurrence_count,
            first_occurrence_note,
            prominence_score,
            total_notes,
        )

        records.append(
            {
                "motif_id": motif_id,
                "motif": motif["motif"],
                "occurrence_count": occurrence_count,
                "first_occurrence_note": first_occurrence_note,
                "prominence_score": round(
                    prominence_score,
                    4,
                ),
                "structural_role": role,
            }
        )

    result = pd.DataFrame(records)

    print()
    print(
        "Structural role distribution:"
    )

    role_counts = (
        result["structural_role"]
        .value_counts()
    )

    for role, count in role_counts.items():

        print(
            f"    {role:<32} {count}"
        )

    return result


# =============================================================================
# MOTIF TIMELINE
# =============================================================================

def build_motif_timeline(
    motifs: pd.DataFrame
) -> pd.DataFrame:

    print_section(
        "PHASE 12.3.5 - MOTIF TIMELINE"
    )

    if motifs.empty:
        return pd.DataFrame()

    records = []

    for _, motif in motifs.iterrows():

        motif_id = motif["motif_id"]

        motif_text = motif["motif"]

        first_note = safe_int(
            motif["first_occurrence_note"],
            0,
        )

        occurrence_count = safe_int(
            motif["occurrence_count"],
            1,
        )

        # Phase 12.1 gives the first known
        # occurrence position. We do not invent
        # timestamps that are not present.
        records.append(
            {
                "motif_id": motif_id,
                "motif": motif_text,
                "first_occurrence_note": first_note,
                "occurrence_count": occurrence_count,
                "timeline_position": first_note,
            }
        )

    result = pd.DataFrame(records)

    result = result.sort_values(
        "timeline_position"
    ).reset_index(drop=True)

    print(
        f"Timeline entries: {len(result)}"
    )

    return result


# =============================================================================
# PHASE 12.2 RELATIONSHIP SUMMARY
# =============================================================================

def summarize_phase_12_2(
    transformations: pd.DataFrame,
    similarity: pd.DataFrame,
) -> dict:

    print_section(
        "PHASE 12.3.6 - PHASE 12.2 RELATIONSHIP SUMMARY"
    )

    transformation_count = 0
    similarity_count = 0

    transformation_types = {}

    if (
        transformations is not None
        and not transformations.empty
        and "transformation" in transformations.columns
    ):

        transformation_count = len(
            transformations
        )

        counts = (
            transformations["transformation"]
            .value_counts()
        )

        transformation_types = {
            clean_text(key): int(value)
            for key, value in counts.items()
        }

    if similarity is not None:
        similarity_count = len(
            similarity
        )

    print(
        f"Transformation relationships: "
        f"{transformation_count}"
    )

    print(
        f"Similarity relationships: "
        f"{similarity_count}"
    )

    return {
        "transformation_relationships": (
            transformation_count
        ),
        "similarity_relationships": (
            similarity_count
        ),
        "transformation_types": (
            transformation_types
        ),
    }


# =============================================================================
# SUMMARY
# =============================================================================

def build_summary(
    motifs: pd.DataFrame,
    development: pd.DataFrame,
    density: pd.DataFrame,
    prominence: pd.DataFrame,
    structural_roles: pd.DataFrame,
    timeline: pd.DataFrame,
    phase_12_2_summary: dict,
) -> dict:

    print_section(
        "PHASE 12.3.7 - FINAL SUMMARY"
    )

    total_motifs = len(motifs)

    total_occurrences = 0

    if (
        not motifs.empty
        and "occurrence_count" in motifs.columns
    ):

        total_occurrences = int(
            motifs["occurrence_count"].sum()
        )

    average_occurrences = 0.0

    if (
        not motifs.empty
        and "occurrence_count" in motifs.columns
    ):

        average_occurrences = float(
            motifs["occurrence_count"].mean()
        )

    maximum_occurrences = 0

    if (
        not motifs.empty
        and "occurrence_count" in motifs.columns
    ):

        maximum_occurrences = int(
            motifs["occurrence_count"].max()
        )

    mean_development_score = 0.0

    if (
        not development.empty
        and "development_score" in development.columns
    ):

        mean_development_score = float(
            development["development_score"].mean()
        )

    mean_prominence = 0.0

    if (
        not prominence.empty
        and "prominence_score" in prominence.columns
    ):

        mean_prominence = float(
            prominence["prominence_score"].mean()
        )

    structural_distribution = {}

    if (
        not structural_roles.empty
        and "structural_role" in structural_roles.columns
    ):

        counts = (
            structural_roles[
                "structural_role"
            ]
            .value_counts()
        )

        structural_distribution = {
            clean_text(key): int(value)
            for key, value in counts.items()
        }

    if total_occurrences >= 50:
        recurrence_level = "High"

    elif total_occurrences >= 20:
        recurrence_level = "Moderate"

    elif total_occurrences > 0:
        recurrence_level = "Low"

    else:
        recurrence_level = "None"

    summary = {
        "phase": "12.3",
        "title": (
            "Motif Development & "
            "Structural Role Analysis"
        ),
        "input_file": str(MOTIF_FILE),
        "total_motifs": int(total_motifs),
        "total_motif_occurrences": int(
            total_occurrences
        ),
        "average_occurrences_per_motif": round(
            average_occurrences,
            4,
        ),
        "maximum_motif_occurrences": int(
            maximum_occurrences
        ),
        "mean_development_score": round(
            mean_development_score,
            4,
        ),
        "mean_prominence_score": round(
            mean_prominence,
            4,
        ),
        "recurrence_level": recurrence_level,
        "structural_role_distribution": (
            structural_distribution
        ),
        "phase_12_2_relationships": (
            phase_12_2_summary
        ),
        "timeline_entries": int(
            len(timeline)
        ),
        "analysis_notes": [
            (
                "Motif development is evaluated "
                "from recurrence, motif length, "
                "and first observed position."
            ),
            (
                "Motif prominence is a feature-based "
                "measurement and is not an artistic judgment."
            ),
            (
                "Structural roles are inferred from "
                "available motif recurrence and position data."
            ),
            (
                "Phase 12.3 does not claim composer intent."
            ),
            (
                "Only positions explicitly available "
                "in the Phase 12.1 data are used."
            ),
        ],
    }

    print(
        f"Total motifs              : {total_motifs}"
    )

    print(
        f"Total motif occurrences   : "
        f"{total_occurrences}"
    )

    print(
        f"Average occurrences       : "
        f"{average_occurrences:.2f}"
    )

    print(
        f"Maximum motif frequency   : "
        f"{maximum_occurrences}"
    )

    print(
        f"Mean development score    : "
        f"{mean_development_score:.2f}"
    )

    print(
        f"Mean prominence score     : "
        f"{mean_prominence:.2f}"
    )

    print(
        f"Recurrence level          : "
        f"{recurrence_level}"
    )

    return summary


# =============================================================================
# GRAPH 1 - MOTIF TIMELINE
# =============================================================================

def create_timeline_graph(
    timeline: pd.DataFrame
) -> None:

    print_section(
        "PHASE 12.3.8 - MOTIF TIMELINE GRAPH"
    )

    plt.figure(
        figsize=(14, 8)
    )

    if timeline.empty:

        plt.text(
            0.5,
            0.5,
            "No motif timeline data available",
            ha="center",
            va="center",
            fontsize=14,
        )

        plt.axis("off")

    else:

        plot_data = timeline.copy()

        plot_data = plot_data.sort_values(
            "timeline_position"
        )

        # Display the most frequently recurring
        # motifs first if there are many motifs.
        plot_data = plot_data.head(30)

        x = plot_data[
            "timeline_position"
        ].to_numpy()

        y = list(
            range(
                len(plot_data)
            )
        )

        sizes = (
            plot_data[
                "occurrence_count"
            ]
            .astype(float)
            * 35.0
            + 40.0
        )

        plt.scatter(
            x,
            y,
            s=sizes,
        )

        labels = plot_data[
            "motif"
        ].astype(str).tolist()

        plt.yticks(
            y,
            labels,
        )

        plt.xlabel(
            "First occurrence note position"
        )

        plt.ylabel(
            "Swara motif"
        )

        plt.title(
            "Phase 12.3 - Recurring Motif Timeline"
        )

        plt.grid(
            axis="x",
            alpha=0.3,
        )

    plt.tight_layout()

    plt.savefig(
        GRAPH_TIMELINE,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {GRAPH_TIMELINE}"
    )


# =============================================================================
# GRAPH 2 - MOTIF FREQUENCY
# =============================================================================

def create_frequency_graph(
    motifs: pd.DataFrame
) -> None:

    print_section(
        "PHASE 12.3.9 - MOTIF FREQUENCY GRAPH"
    )

    plt.figure(
        figsize=(14, 8)
    )

    if motifs.empty:

        plt.text(
            0.5,
            0.5,
            "No motif frequency data available",
            ha="center",
            va="center",
            fontsize=14,
        )

        plt.axis("off")

    else:

        plot_data = motifs.copy()

        plot_data = plot_data.sort_values(
            "occurrence_count",
            ascending=False,
        )

        plot_data = plot_data.head(20)

        labels = plot_data[
            "motif"
        ].astype(str).tolist()

        values = plot_data[
            "occurrence_count"
        ].astype(float).tolist()

        x = list(
            range(
                len(labels)
            )
        )

        plt.bar(
            x,
            values,
        )

        plt.xticks(
            x,
            labels,
            rotation=45,
            ha="right",
        )

        plt.xlabel(
            "Swara motif"
        )

        plt.ylabel(
            "Occurrence count"
        )

        plt.title(
            "Phase 12.3 - Most Frequently Recurring Motifs"
        )

        plt.grid(
            axis="y",
            alpha=0.3,
        )

    plt.tight_layout()

    plt.savefig(
        GRAPH_FREQUENCY,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {GRAPH_FREQUENCY}"
    )


# =============================================================================
# GRAPH 3 - MOTIF DEVELOPMENT
# =============================================================================

def create_development_graph(
    development: pd.DataFrame
) -> None:

    print_section(
        "PHASE 12.3.10 - MOTIF DEVELOPMENT GRAPH"
    )

    plt.figure(
        figsize=(14, 8)
    )

    if development.empty:

        plt.text(
            0.5,
            0.5,
            "No motif development data available",
            ha="center",
            va="center",
            fontsize=14,
        )

        plt.axis("off")

    else:

        plot_data = development.copy()

        plot_data = plot_data.sort_values(
            "development_score",
            ascending=False,
        )

        plot_data = plot_data.head(20)

        labels = plot_data[
            "motif"
        ].astype(str).tolist()

        values = plot_data[
            "development_score"
        ].astype(float).tolist()

        x = list(
            range(
                len(labels)
            )
        )

        plt.bar(
            x,
            values,
        )

        plt.xticks(
            x,
            labels,
            rotation=45,
            ha="right",
        )

        plt.xlabel(
            "Swara motif"
        )

        plt.ylabel(
            "Development score"
        )

        plt.title(
            "Phase 12.3 - Motif Development Scores"
        )

        plt.ylim(
            0,
            100,
        )

        plt.grid(
            axis="y",
            alpha=0.3,
        )

    plt.tight_layout()

    plt.savefig(
        GRAPH_DEVELOPMENT,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {GRAPH_DEVELOPMENT}"
    )


# =============================================================================
# GRAPH 4 - MOTIF PROMINENCE
# =============================================================================

def create_prominence_graph(
    prominence: pd.DataFrame
) -> None:

    print_section(
        "PHASE 12.3.11 - MOTIF PROMINENCE GRAPH"
    )

    plt.figure(
        figsize=(14, 8)
    )

    if prominence.empty:

        plt.text(
            0.5,
            0.5,
            "No motif prominence data available",
            ha="center",
            va="center",
            fontsize=14,
        )

        plt.axis("off")

    else:

        plot_data = prominence.copy()

        plot_data = plot_data.sort_values(
            "prominence_score",
            ascending=False,
        )

        plot_data = plot_data.head(20)

        labels = plot_data[
            "motif"
        ].astype(str).tolist()

        values = plot_data[
            "prominence_score"
        ].astype(float).tolist()

        x = list(
            range(
                len(labels)
            )
        )

        plt.bar(
            x,
            values,
        )

        plt.xticks(
            x,
            labels,
            rotation=45,
            ha="right",
        )

        plt.xlabel(
            "Swara motif"
        )

        plt.ylabel(
            "Prominence score"
        )

        plt.title(
            "Phase 12.3 - Motif Prominence"
        )

        plt.ylim(
            0,
            100,
        )

        plt.grid(
            axis="y",
            alpha=0.3,
        )

    plt.tight_layout()

    plt.savefig(
        GRAPH_PROMINENCE,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {GRAPH_PROMINENCE}"
    )


# =============================================================================
# SAVE OUTPUTS
# =============================================================================

def save_outputs(
    development: pd.DataFrame,
    density: pd.DataFrame,
    prominence: pd.DataFrame,
    structural_roles: pd.DataFrame,
    timeline: pd.DataFrame,
    summary: dict,
) -> None:

    print_section(
        "SAVING PHASE 12.3 OUTPUTS"
    )

    development.to_csv(
        DEVELOPMENT_FILE,
        index=False,
    )

    density.to_csv(
        DENSITY_FILE,
        index=False,
    )

    prominence.to_csv(
        PROMINENCE_FILE,
        index=False,
    )

    structural_roles.to_csv(
        STRUCTURE_FILE,
        index=False,
    )

    timeline.to_csv(
        TIMELINE_FILE,
        index=False,
    )

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    print(
        f"Saved: {DEVELOPMENT_FILE}"
    )

    print(
        f"Saved: {DENSITY_FILE}"
    )

    print(
        f"Saved: {PROMINENCE_FILE}"
    )

    print(
        f"Saved: {STRUCTURE_FILE}"
    )

    print(
        f"Saved: {TIMELINE_FILE}"
    )

    print(
        f"Saved: {SUMMARY_FILE}"
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    print()
    print("=" * 80)
    print(
        "PHASE 12.3 STANDALONE TEST"
    )
    print("=" * 80)

    print()
    print(
        "PHASE 12.3 - MOTIF DEVELOPMENT "
        "& STRUCTURAL ROLE ANALYSIS"
    )

    print()
    print(
        "This phase uses the swara-based "
        "Phase 12.1 motif output."
    )

    print(
        "No audio path is required."
    )

    try:

        # -------------------------------------------------------------
        # LOAD PHASE 12.1
        # -------------------------------------------------------------

        raw_motifs = load_motif_data()

        # -------------------------------------------------------------
        # BUILD SWARA MOTIF REPRESENTATION
        # -------------------------------------------------------------

        motifs = build_motif_table(
            raw_motifs
        )

        if motifs.empty:

            print()
            print(
                "=" * 80
            )

            print(
                "PHASE 12.3 WARNING"
            )

            print(
                "=" * 80
            )

            print()
            print(
                "No usable swara motifs were found."
            )

            print(
                "Check repeated_melodic_motifs.csv."
            )

            print()

            return

        # -------------------------------------------------------------
        # LOAD OPTIONAL PHASE 12.2 RESULTS
        # -------------------------------------------------------------

        transformations, similarity = (
            load_optional_phase_12_2_data()
        )

        # -------------------------------------------------------------
        # PHASE 12.3 ANALYSIS
        # -------------------------------------------------------------

        development = analyze_motif_development(
            motifs
        )

        density = analyze_motif_density(
            motifs
        )

        prominence = analyze_motif_prominence(
            motifs
        )

        structural_roles = analyze_structural_roles(
            motifs,
            prominence,
        )

        timeline = build_motif_timeline(
            motifs
        )

        phase_12_2_summary = summarize_phase_12_2(
            transformations,
            similarity,
        )

        # -------------------------------------------------------------
        # SUMMARY
        # -------------------------------------------------------------

        summary = build_summary(
            motifs,
            development,
            density,
            prominence,
            structural_roles,
            timeline,
            phase_12_2_summary,
        )

        # -------------------------------------------------------------
        # GRAPHS
        # -------------------------------------------------------------

        create_timeline_graph(
            timeline
        )

        create_frequency_graph(
            motifs
        )

        create_development_graph(
            development
        )

        create_prominence_graph(
            prominence
        )

        # -------------------------------------------------------------
        # SAVE
        # -------------------------------------------------------------

        save_outputs(
            development,
            density,
            prominence,
            structural_roles,
            timeline,
            summary,
        )

        # -------------------------------------------------------------
        # SUCCESS
        # -------------------------------------------------------------

        print()
        print("=" * 80)
        print(
            "PHASE 12.3 COMPLETED SUCCESSFULLY"
        )
        print("=" * 80)

        print()
        print(
            "Generated CSV files:"
        )

        print(
            "  data/output/"
            "phase_12_3_motif_development.csv"
        )

        print(
            "  data/output/"
            "phase_12_3_motif_density.csv"
        )

        print(
            "  data/output/"
            "phase_12_3_motif_prominence.csv"
        )

        print(
            "  data/output/"
            "phase_12_3_structural_roles.csv"
        )

        print(
            "  data/output/"
            "phase_12_3_motif_timeline.csv"
        )

        print()
        print(
            "Generated summary:"
        )

        print(
            "  data/output/"
            "phase_12_3_summary.json"
        )

        print()
        print(
            "Generated graphs:"
        )

        print(
            "  graphs/"
            "phase_12_3_motif_timeline.png"
        )

        print(
            "  graphs/"
            "phase_12_3_motif_frequency.png"
        )

        print(
            "  graphs/"
            "phase_12_3_motif_development.png"
        )

        print(
            "  graphs/"
            "phase_12_3_motif_prominence.png"
        )

        print()
        print(
            "Phase 12.3 standalone test finished."
        )

    except FileNotFoundError as exc:

        print()
        print("=" * 80)
        print(
            "PHASE 12.3 INPUT FILE ERROR"
        )
        print("=" * 80)

        print()
        print(
            str(exc)
        )

    except ImportError as exc:

        print()
        print("=" * 80)
        print(
            "PHASE 12.3 DEPENDENCY ERROR"
        )
        print("=" * 80)

        print()
        print(
            f"{type(exc).__name__}: {exc}"
        )

        print()
        print(
            "Make sure the active virtual environment "
            "contains pandas and matplotlib."
        )

    except Exception as exc:

        print()
        print("=" * 80)
        print(
            "PHASE 12.3 FAILED"
        )
        print("=" * 80)

        print()
        print(
            f"{type(exc).__name__}: {exc}"
        )

        print()
        print(
            "The Phase 12.3 standalone test stopped."
        )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()