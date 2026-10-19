"""
PHASE 12.1 - RECURRING MELODIC MOTIF ANALYSIS

Analyzes existing Phase 11 melodic outputs for:

1. Recurring melodic motifs
2. Repeated note sequences
3. Repeated interval patterns
4. Phrase-to-phrase similarity
5. Motif frequency
6. Motif locations / timestamps
7. Longest recurring patterns
8. Melodic repetition density
9. Motif complexity
10. Motif visualizations

Phase 12.1 is standalone.
It does NOT modify main.py.
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
GRAPH_DIR = PROJECT_ROOT / "graphs"

GRAPH_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# CONSOLE HELPERS
# ============================================================================

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


# ============================================================================
# CSV HELPERS
# ============================================================================

def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def write_csv(
    path: Path,
    rows: List[Dict[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fieldnames),
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    field: row.get(field, "")
                    for field in fieldnames
                }
            )


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default

        text = str(value).strip()

        if not text:
            return default

        return float(text)

    except (ValueError, TypeError):
        return default


def to_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default

        text = str(value).strip()

        if not text:
            return default

        return int(round(float(text)))

    except (ValueError, TypeError):
        return default


def first_value(
    row: Dict[str, Any],
    names: Sequence[str],
    default: Any = None,
) -> Any:
    for name in names:
        if name in row:
            value = row[name]

            if value is not None and str(value).strip() != "":
                return value

    return default


# ============================================================================
# NOTE COLUMN DETECTION
# ============================================================================

def find_note_value(row: Dict[str, Any]) -> Optional[float]:
    candidates = [
        "midi",
        "midi_note",
        "midi_pitch",
        "pitch_midi",
        "quantized_midi",
        "quantized_pitch",
        "note_midi",
        "note",
        "pitch",
    ]

    value = first_value(row, candidates)

    if value is None:
        return None

    result = to_float(value, float("nan"))

    if not math.isfinite(result):
        return None

    return result


def find_start_time(row: Dict[str, Any]) -> float:
    value = first_value(
        row,
        [
            "start_time_seconds",
            "start_time",
            "start",
            "time_seconds",
            "time",
            "onset_seconds",
            "onset",
        ],
        0.0,
    )

    return to_float(value, 0.0)


def find_end_time(row: Dict[str, Any]) -> float:
    value = first_value(
        row,
        [
            "end_time_seconds",
            "end_time",
            "end",
            "offset_seconds",
            "offset",
        ],
        None,
    )

    if value is None:
        return find_start_time(row)

    return to_float(value, find_start_time(row))


def find_phrase_id(row: Dict[str, Any]) -> str:
    value = first_value(
        row,
        [
            "phrase_id",
            "phrase",
            "phrase_number",
            "phrase_index",
            "section_id",
        ],
        "1",
    )

    return str(value)


# ============================================================================
# NOTE DATA LOADING
# ============================================================================

def load_best_note_file() -> Tuple[Path, List[Dict[str, str]]]:
    candidates = [
        OUTPUT_DIR / "melody_phrase_note_data.csv",
        OUTPUT_DIR / "quantized_notes.csv",
        OUTPUT_DIR / "note_segments.csv",
        OUTPUT_DIR / "western_note_data.csv",
        OUTPUT_DIR / "musical_notation_table.csv",
    ]

    for path in candidates:
        rows = read_csv(path)

        if not rows:
            continue

        valid_notes = 0

        for row in rows[:1000]:
            if find_note_value(row) is not None:
                valid_notes += 1

        if valid_notes > 0:
            return path, rows

    raise FileNotFoundError(
        "No usable melodic note CSV was found in data/output."
    )


# ============================================================================
# NOTE SEQUENCE CONSTRUCTION
# ============================================================================

def build_note_events(
    rows: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []

    for index, row in enumerate(rows):
        pitch = find_note_value(row)

        if pitch is None:
            continue

        start = find_start_time(row)
        end = find_end_time(row)

        if end < start:
            end = start

        events.append(
            {
                "original_index": index,
                "phrase_id": find_phrase_id(row),
                "pitch": pitch,
                "start": start,
                "end": end,
            }
        )

    events.sort(
        key=lambda item: (
            item["start"],
            item["original_index"],
        )
    )

    return events


# ============================================================================
# PHRASE GROUPING
# ============================================================================

def group_events_by_phrase(
    events: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for event in events:
        groups[str(event["phrase_id"])].append(event)

    for phrase_events in groups.values():
        phrase_events.sort(
            key=lambda item: item["start"]
        )

    return dict(groups)


# ============================================================================
# MOTIF REPRESENTATIONS
# ============================================================================

def pitch_token(value: float) -> int:
    return int(round(value))


def make_pitch_sequence(
    events: Sequence[Dict[str, Any]],
) -> Tuple[int, ...]:
    return tuple(
        pitch_token(event["pitch"])
        for event in events
    )


def make_interval_sequence(
    events: Sequence[Dict[str, Any]],
) -> Tuple[int, ...]:
    pitches = [
        pitch_token(event["pitch"])
        for event in events
    ]

    if len(pitches) < 2:
        return ()

    return tuple(
        pitches[index + 1] - pitches[index]
        for index in range(len(pitches) - 1)
    )


# ============================================================================
# SUBSEQUENCE GENERATION
# ============================================================================

def all_subsequences(
    sequence: Sequence[int],
    minimum_length: int = 3,
    maximum_length: int = 12,
) -> Dict[Tuple[int, ...], List[int]]:
    result: Dict[Tuple[int, ...], List[int]] = defaultdict(list)

    length = len(sequence)

    maximum_length = min(
        maximum_length,
        length,
    )

    for size in range(
        minimum_length,
        maximum_length + 1,
    ):
        for start in range(
            0,
            length - size + 1,
        ):
            motif = tuple(
                sequence[start:start + size]
            )

            result[motif].append(start)

    return dict(result)


# ============================================================================
# GLOBAL MOTIF SEARCH
# ============================================================================

def find_recurring_pitch_motifs(
    events: List[Dict[str, Any]],
    minimum_length: int = 3,
    maximum_length: int = 12,
) -> List[Dict[str, Any]]:
    pitches = [
        pitch_token(event["pitch"])
        for event in events
    ]

    starts_by_motif = all_subsequences(
        pitches,
        minimum_length,
        maximum_length,
    )

    results: List[Dict[str, Any]] = []

    for motif, positions in starts_by_motif.items():
        if len(positions) < 2:
            continue

        occurrences = []

        for position in positions:
            if position >= len(events):
                continue

            end_index = min(
                position + len(motif) - 1,
                len(events) - 1,
            )

            occurrences.append(
                {
                    "start_time": events[position]["start"],
                    "end_time": events[end_index]["end"],
                }
            )

        if len(occurrences) < 2:
            continue

        results.append(
            {
                "motif_type": "pitch",
                "length_notes": len(motif),
                "sequence": ",".join(
                    str(value)
                    for value in motif
                ),
                "frequency": len(occurrences),
                "first_start_time": occurrences[0]["start_time"],
                "last_start_time": occurrences[-1]["start_time"],
                "locations": ";".join(
                    f"{item['start_time']:.3f}"
                    for item in occurrences
                ),
            }
        )

    results.sort(
        key=lambda item: (
            -item["length_notes"],
            -item["frequency"],
        )
    )

    return results


# ============================================================================
# INTERVAL MOTIFS
# ============================================================================

def find_recurring_interval_motifs(
    events: List[Dict[str, Any]],
    minimum_length: int = 3,
    maximum_length: int = 12,
) -> List[Dict[str, Any]]:
    intervals = list(
        make_interval_sequence(events)
    )

    starts_by_motif = all_subsequences(
        intervals,
        minimum_length,
        maximum_length,
    )

    results: List[Dict[str, Any]] = []

    for motif, positions in starts_by_motif.items():
        if len(positions) < 2:
            continue

        occurrences = []

        for position in positions:
            if position >= len(events):
                continue

            end_index = min(
                position + len(motif),
                len(events) - 1,
            )

            occurrences.append(
                {
                    "start_time": events[position]["start"],
                    "end_time": events[end_index]["end"],
                }
            )

        if len(occurrences) < 2:
            continue

        results.append(
            {
                "motif_type": "interval",
                "length_intervals": len(motif),
                "sequence": ",".join(
                    f"{value:+d}"
                    for value in motif
                ),
                "frequency": len(occurrences),
                "first_start_time": occurrences[0]["start_time"],
                "last_start_time": occurrences[-1]["start_time"],
                "locations": ";".join(
                    f"{item['start_time']:.3f}"
                    for item in occurrences
                ),
            }
        )

    results.sort(
        key=lambda item: (
            -item["length_intervals"],
            -item["frequency"],
        )
    )

    return results


# ============================================================================
# PHRASE SIMILARITY
# ============================================================================

def normalized_pitch_sequence(
    events: Sequence[Dict[str, Any]],
) -> List[int]:
    pitches = [
        pitch_token(event["pitch"])
        for event in events
    ]

    if not pitches:
        return []

    first = pitches[0]

    return [
        pitch - first
        for pitch in pitches
    ]


def sequence_similarity(
    first: Sequence[int],
    second: Sequence[int],
) -> float:
    if not first or not second:
        return 0.0

    length = max(
        len(first),
        len(second),
    )

    matches = 0

    for index in range(length):
        if index >= len(first):
            continue

        if index >= len(second):
            continue

        if first[index] == second[index]:
            matches += 1

    return matches / length


def analyze_phrase_similarity(
    phrase_groups: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    phrase_ids = list(phrase_groups.keys())

    results: List[Dict[str, Any]] = []

    for first_index in range(len(phrase_ids)):
        for second_index in range(
            first_index + 1,
            len(phrase_ids),
        ):
            first_id = phrase_ids[first_index]
            second_id = phrase_ids[second_index]

            first_sequence = normalized_pitch_sequence(
                phrase_groups[first_id]
            )

            second_sequence = normalized_pitch_sequence(
                phrase_groups[second_id]
            )

            similarity = sequence_similarity(
                first_sequence,
                second_sequence,
            )

            if similarity <= 0.0:
                continue

            results.append(
                {
                    "phrase_1": first_id,
                    "phrase_2": second_id,
                    "similarity_percent": round(
                        similarity * 100.0,
                        3,
                    ),
                    "phrase_1_note_count": len(first_sequence),
                    "phrase_2_note_count": len(second_sequence),
                }
            )

    results.sort(
        key=lambda item: -item["similarity_percent"]
    )

    return results


# ============================================================================
# MOTIF COMPLEXITY
# ============================================================================

def motif_complexity(
    sequence: Sequence[int],
) -> float:
    if not sequence:
        return 0.0

    unique_values = len(set(sequence))
    length = len(sequence)

    if length == 1:
        return 0.0

    diversity = unique_values / length

    intervals = [
        abs(sequence[index + 1] - sequence[index])
        for index in range(length - 1)
    ]

    if intervals:
        interval_diversity = (
            len(set(intervals))
            / len(intervals)
        )
    else:
        interval_diversity = 0.0

    score = (
        0.6 * diversity
        + 0.4 * interval_diversity
    ) * 100.0

    return round(
        min(100.0, score),
        3,
    )


# ============================================================================
# REPETITION DENSITY
# ============================================================================

def calculate_repetition_density(
    events: List[Dict[str, Any]],
    motifs: List[Dict[str, Any]],
) -> float:
    if not events:
        return 0.0

    repeated_note_counts: Counter = Counter()

    for motif in motifs:
        frequency = to_int(
            motif.get("frequency"),
            0,
        )

        length = to_int(
            motif.get("length_notes"),
            0,
        )

        if frequency < 2 or length <= 0:
            continue

        sequence_text = motif.get(
            "sequence",
            "",
        )

        try:
            sequence = [
                int(value.strip())
                for value in sequence_text.split(",")
                if value.strip()
            ]
        except ValueError:
            continue

        for value in sequence:
            repeated_note_counts[value] += (
                frequency
            )

    if not repeated_note_counts:
        return 0.0

    repeated_weight = sum(
        repeated_note_counts.values()
    )

    possible_weight = (
        len(events)
        * max(1, len(motifs))
    )

    density = (
        repeated_weight
        / possible_weight
    ) * 100.0

    return round(
        min(100.0, density),
        3,
    )


# ============================================================================
# MOTIF SUMMARY
# ============================================================================

def build_motif_summary(
    pitch_motifs: List[Dict[str, Any]],
    interval_motifs: List[Dict[str, Any]],
    phrase_similarity: List[Dict[str, Any]],
    events: List[Dict[str, Any]],
) -> Dict[str, Any]:
    longest_pitch = (
        max(
            pitch_motifs,
            key=lambda item: (
                item["length_notes"],
                item["frequency"],
            ),
        )
        if pitch_motifs
        else None
    )

    longest_interval = (
        max(
            interval_motifs,
            key=lambda item: (
                item["length_intervals"],
                item["frequency"],
            ),
        )
        if interval_motifs
        else None
    )

    top_similarity = (
        phrase_similarity[0]
        if phrase_similarity
        else None
    )

    complexity_values = []

    for motif in pitch_motifs:
        sequence_text = motif.get(
            "sequence",
            "",
        )

        try:
            sequence = [
                int(value.strip())
                for value in sequence_text.split(",")
                if value.strip()
            ]
        except ValueError:
            continue

        complexity_values.append(
            motif_complexity(sequence)
        )

    mean_complexity = (
        sum(complexity_values)
        / len(complexity_values)
        if complexity_values
        else 0.0
    )

    repetition_density = calculate_repetition_density(
        events,
        pitch_motifs,
    )

    return {
        "note_count": len(events),
        "recurring_pitch_motifs": len(pitch_motifs),
        "recurring_interval_motifs": len(interval_motifs),
        "phrase_comparisons": len(phrase_similarity),
        "highest_phrase_similarity_percent": (
            top_similarity["similarity_percent"]
            if top_similarity
            else 0.0
        ),
        "longest_pitch_motif_notes": (
            longest_pitch["length_notes"]
            if longest_pitch
            else 0
        ),
        "longest_interval_motif_intervals": (
            longest_interval["length_intervals"]
            if longest_interval
            else 0
        ),
        "mean_motif_complexity": round(
            mean_complexity,
            3,
        ),
        "melodic_repetition_density_percent": (
            repetition_density
        ),
    }


# ============================================================================
# VISUALIZATIONS
# ============================================================================

def plot_motif_frequency(
    motifs: List[Dict[str, Any]],
) -> None:
    if not motifs:
        return

    selected = motifs[:10]

    labels = [
        f"M{i + 1}"
        for i in range(len(selected))
    ]

    frequencies = [
        to_int(item["frequency"])
        for item in selected
    ]

    plt.figure(figsize=(10, 6))

    plt.bar(
        labels,
        frequencies,
    )

    plt.title(
        "Phase 12.1 - Recurring Motif Frequency"
    )

    plt.xlabel("Motif")
    plt.ylabel("Occurrences")

    plt.tight_layout()

    path = GRAPH_DIR / (
        "phase_12_1_motif_frequency.png"
    )

    plt.savefig(
        path,
        dpi=150,
    )

    plt.close()

    print(f"  Saved: {path}")


def plot_motif_locations(
    motifs: List[Dict[str, Any]],
) -> None:
    if not motifs:
        return

    selected = motifs[:10]

    plt.figure(figsize=(12, 6))

    for index, motif in enumerate(selected):
        location_text = motif.get(
            "locations",
            "",
        )

        if not location_text:
            continue

        try:
            locations = [
                float(value)
                for value in location_text.split(";")
                if value.strip()
            ]
        except ValueError:
            continue

        y_values = [
            index + 1
            for _ in locations
        ]

        plt.scatter(
            locations,
            y_values,
            s=35,
        )

    plt.title(
        "Phase 12.1 - Recurring Motif Locations"
    )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Motif"
    )

    plt.tight_layout()

    path = GRAPH_DIR / (
        "phase_12_1_motif_locations.png"
    )

    plt.savefig(
        path,
        dpi=150,
    )

    plt.close()

    print(f"  Saved: {path}")


def plot_phrase_similarity(
    similarities: List[Dict[str, Any]],
) -> None:
    if not similarities:
        return

    selected = similarities[:15]

    labels = [
        f"{item['phrase_1']}-{item['phrase_2']}"
        for item in selected
    ]

    values = [
        item["similarity_percent"]
        for item in selected
    ]

    plt.figure(figsize=(12, 6))

    plt.bar(
        labels,
        values,
    )

    plt.title(
        "Phase 12.1 - Phrase-to-Phrase Similarity"
    )

    plt.xlabel(
        "Phrase pair"
    )

    plt.ylabel(
        "Similarity (%)"
    )

    plt.xticks(
        rotation=45,
        ha="right",
    )

    plt.tight_layout()

    path = GRAPH_DIR / (
        "phase_12_1_phrase_similarity.png"
    )

    plt.savefig(
        path,
        dpi=150,
    )

    plt.close()

    print(f"  Saved: {path}")


def plot_motif_complexity(
    motifs: List[Dict[str, Any]],
) -> None:
    if not motifs:
        return

    selected = motifs[:10]

    labels = [
        f"M{i + 1}"
        for i in range(len(selected))
    ]

    values = []

    for motif in selected:
        sequence_text = motif.get(
            "sequence",
            "",
        )

        try:
            sequence = [
                int(value.strip())
                for value in sequence_text.split(",")
                if value.strip()
            ]
        except ValueError:
            sequence = []

        values.append(
            motif_complexity(sequence)
        )

    plt.figure(figsize=(10, 6))

    plt.bar(
        labels,
        values,
    )

    plt.title(
        "Phase 12.1 - Motif Complexity"
    )

    plt.xlabel("Motif")
    plt.ylabel("Complexity score")

    plt.ylim(
        0,
        100,
    )

    plt.tight_layout()

    path = GRAPH_DIR / (
        "phase_12_1_motif_complexity.png"
    )

    plt.savefig(
        path,
        dpi=150,
    )

    plt.close()

    print(f"  Saved: {path}")


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def run_phase_12_1() -> Dict[str, Any]:
    print_header(
        "PHASE 12.1 - RECURRING MELODIC MOTIF ANALYSIS"
    )

    print()
    print(
        "Using completed Phase 11 melodic outputs."
    )

    print_section(
        "LOADING MELODIC NOTE DATA"
    )

    note_path, raw_rows = load_best_note_file()

    print(
        f"Source file : {note_path}"
    )

    print(
        f"Rows loaded : {len(raw_rows)}"
    )

    events = build_note_events(
        raw_rows
    )

    if len(events) < 3:
        raise RuntimeError(
            "Not enough melodic note events "
            "were found for motif analysis."
        )

    print(
        f"Usable notes: {len(events)}"
    )

    phrase_groups = group_events_by_phrase(
        events
    )

    print(
        f"Melodic phrases: {len(phrase_groups)}"
    )

    print_section(
        "12.1.1 - RECURRING MELODIC MOTIFS"
    )

    pitch_motifs = find_recurring_pitch_motifs(
        events
    )

    print(
        f"Recurring pitch motifs: "
        f"{len(pitch_motifs)}"
    )

    for index, motif in enumerate(
        pitch_motifs[:10],
        start=1,
    ):
        print(
            f"  Motif {index}: "
            f"{motif['length_notes']} notes, "
            f"{motif['frequency']} occurrences"
        )

    print_section(
        "12.1.2 - REPEATED INTERVAL PATTERNS"
    )

    interval_motifs = find_recurring_interval_motifs(
        events
    )

    print(
        f"Recurring interval motifs: "
        f"{len(interval_motifs)}"
    )

    for index, motif in enumerate(
        interval_motifs[:10],
        start=1,
    ):
        print(
            f"  Pattern {index}: "
            f"{motif['sequence']} "
            f"({motif['frequency']} occurrences)"
        )

    print_section(
        "12.1.3 - PHRASE-TO-PHRASE SIMILARITY"
    )

    similarities = analyze_phrase_similarity(
        phrase_groups
    )

    print(
        f"Phrase comparisons: "
        f"{len(similarities)}"
    )

    for item in similarities[:10]:
        print(
            f"  Phrase {item['phrase_1']} "
            f"<-> Phrase {item['phrase_2']}: "
            f"{item['similarity_percent']:.2f}%"
        )

    print_section(
        "12.1.4 - MOTIF COMPLEXITY"
    )

    complexity_values = []

    for motif in pitch_motifs:
        sequence_text = motif["sequence"]

        try:
            sequence = [
                int(value.strip())
                for value in sequence_text.split(",")
                if value.strip()
            ]
        except ValueError:
            sequence = []

        complexity = motif_complexity(
            sequence
        )

        complexity_values.append(
            complexity
        )

    mean_complexity = (
        sum(complexity_values)
        / len(complexity_values)
        if complexity_values
        else 0.0
    )

    print(
        f"Mean motif complexity: "
        f"{mean_complexity:.2f}/100"
    )

    print_section(
        "12.1.5 - MELODIC REPETITION DENSITY"
    )

    repetition_density = (
        calculate_repetition_density(
            events,
            pitch_motifs,
        )
    )

    print(
        f"Repetition density: "
        f"{repetition_density:.2f}%"
    )

    print_section(
        "12.1.6 - LONGEST RECURRING PATTERNS"
    )

    if pitch_motifs:
        longest_pitch = max(
            pitch_motifs,
            key=lambda item: (
                item["length_notes"],
                item["frequency"],
            ),
        )

        print(
            f"Longest pitch motif: "
            f"{longest_pitch['length_notes']} notes"
        )

        print(
            f"Frequency: "
            f"{longest_pitch['frequency']}"
        )

        print(
            f"Locations: "
            f"{longest_pitch['locations']}"
        )

    if interval_motifs:
        longest_interval = max(
            interval_motifs,
            key=lambda item: (
                item["length_intervals"],
                item["frequency"],
            ),
        )

        print(
            f"Longest interval pattern: "
            f"{longest_interval['length_intervals']} intervals"
        )

        print(
            f"Pattern: "
            f"{longest_interval['sequence']}"
        )

    print_section(
        "SAVING PHASE 12.1 RESULTS"
    )

    pitch_fields = [
        "motif_type",
        "length_notes",
        "sequence",
        "frequency",
        "first_start_time",
        "last_start_time",
        "locations",
    ]

    write_csv(
        OUTPUT_DIR / (
            "phase_12_1_recurring_pitch_motifs.csv"
        ),
        pitch_motifs,
        pitch_fields,
    )

    interval_fields = [
        "motif_type",
        "length_intervals",
        "sequence",
        "frequency",
        "first_start_time",
        "last_start_time",
        "locations",
    ]

    write_csv(
        OUTPUT_DIR / (
            "phase_12_1_interval_motifs.csv"
        ),
        interval_motifs,
        interval_fields,
    )

    similarity_fields = [
        "phrase_1",
        "phrase_2",
        "similarity_percent",
        "phrase_1_note_count",
        "phrase_2_note_count",
    ]

    write_csv(
        OUTPUT_DIR / (
            "phase_12_1_phrase_similarity.csv"
        ),
        similarities,
        similarity_fields,
    )

    summary = build_motif_summary(
        pitch_motifs,
        interval_motifs,
        similarities,
        events,
    )

    summary["source_file"] = str(
        note_path
    )

    summary["mean_motif_complexity"] = round(
        mean_complexity,
        3,
    )

    summary_path = OUTPUT_DIR / (
        "phase_12_1_summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            indent=4,
        )

    print(
        f"Summary saved: {summary_path}"
    )

    print_section(
        "GENERATING VISUALIZATIONS"
    )

    plot_motif_frequency(
        pitch_motifs
    )

    plot_motif_locations(
        pitch_motifs
    )

    plot_phrase_similarity(
        similarities
    )

    plot_motif_complexity(
        pitch_motifs
    )

    print_header(
        "PHASE 12.1 COMPLETED SUCCESSFULLY"
    )

    print()
    print("Generated files:")

    print(
        "  data/output/"
        "phase_12_1_recurring_pitch_motifs.csv"
    )

    print(
        "  data/output/"
        "phase_12_1_interval_motifs.csv"
    )

    print(
        "  data/output/"
        "phase_12_1_phrase_similarity.csv"
    )

    print(
        "  data/output/"
        "phase_12_1_summary.json"
    )

    print()
    print("Generated graphs:")

    print(
        "  graphs/"
        "phase_12_1_motif_frequency.png"
    )

    print(
        "  graphs/"
        "phase_12_1_motif_locations.png"
    )

    print(
        "  graphs/"
        "phase_12_1_phrase_similarity.png"
    )

    print(
        "  graphs/"
        "phase_12_1_motif_complexity.png"
    )

    return summary


# ============================================================================
# STANDALONE ENTRY POINT
# ============================================================================

def main() -> None:
    print_header(
        "PHASE 12.1 STANDALONE TEST"
    )

    print()
    print(
        "Phase 12.1 does not require an audio path."
    )

    print(
        "It reads the completed melodic outputs "
        "from Phase 11."
    )

    try:
        run_phase_12_1()

    except Exception as exc:
        print()
        print("=" * 80)
        print("PHASE 12.1 FAILED")
        print("=" * 80)
        print()
        print(
            f"{type(exc).__name__}: {exc}"
        )
        print()

        raise


if __name__ == "__main__":
    main()