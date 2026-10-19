"""
PHASE 11.4 - MUSICAL INTERPRETATION & FINAL FEATURE SYNTHESIS

Combines:
    Phase 11.1 - Musical Element Analysis
    Phase 11.2 - Advanced Melody & Rhythm Analysis
    Phase 11.3 - Musical Expression & Performance Analysis

Phase 10 remains ON HOLD.

No PyTorch / Whisper dependency is used.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
GRAPH_DIR = PROJECT_ROOT / "graphs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# GENERAL HELPERS
# ============================================================================

def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert common numeric/text values to float safely."""

    if value is None:
        return default

    try:
        if isinstance(value, bool):
            return float(value)

        if isinstance(value, (int, float, np.integer, np.floating)):
            value = float(value)

            if math.isfinite(value):
                return value

            return default

        text = str(value).strip()

        if not text:
            return default

        # Handle strings such as:
        # "5.24/100"
        # "154.67 BPM"
        # "50.32%"
        # "29.91 semitones"
        for token in [
            "semitones",
            "semitone",
            "octaves",
            "octave",
            "cents",
            "BPM",
            "Hz",
            "dB",
            "%",
        ]:
            text = text.replace(token, "")

        text = text.strip()

        if "/" in text:
            text = text.split("/")[0].strip()

        value = float(text)

        if math.isfinite(value):
            return value

        return default

    except (ValueError, TypeError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(safe_float(value, default)))
    except Exception:
        return default


def clean_text(value: Any, default: str = "Unknown") -> str:
    if value is None:
        return default

    text = str(value).strip()

    if not text:
        return default

    if text.lower() in {"nan", "none", "null"}:
        return default

    return text


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


# ============================================================================
# FILE HELPERS
# ============================================================================

def read_csv(filename: str) -> Optional[pd.DataFrame]:
    path = OUTPUT_DIR / filename

    if not path.exists():
        print(f"  [missing] {filename}")
        return None

    try:
        df = pd.read_csv(path)

        print(
            f"  [loaded] {filename} "
            f"({len(df)} rows, {len(df.columns)} columns)"
        )

        return df

    except Exception as exc:
        print(f"  [error] {filename}: {exc}")
        return None


def read_json(filename: str) -> Optional[dict]:
    path = OUTPUT_DIR / filename

    if not path.exists():
        print(f"  [missing] {filename}")
        return None

    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

        if isinstance(data, dict):
            return data

        return {"data": data}

    except Exception as exc:
        print(f"  [error] {filename}: {exc}")
        return None


# ============================================================================
# NORMALIZED COLUMN / VALUE SEARCH
# ============================================================================

def normalize_name(value: Any) -> str:
    """
    Normalize column/key names.

    This allows:
        mean_local_tempo
        Mean Local Tempo
        mean-local-tempo
        MeanLocalTempo

    to be compared more reliably.
    """

    text = str(value).strip().lower()

    for char in [
        "_",
        "-",
        " ",
        "(",
        ")",
        "[",
        "]",
        "%",
        "/",
        ".",
    ]:
        text = text.replace(char, "")

    return text


def find_column(
    dataframe: Optional[pd.DataFrame],
    names: list[str],
) -> Optional[str]:

    if dataframe is None or dataframe.empty:
        return None

    normalized_columns = {
        normalize_name(column): column
        for column in dataframe.columns
    }

    for name in names:
        normalized = normalize_name(name)

        if normalized in normalized_columns:
            return normalized_columns[normalized]

    return None


def find_dataframe_value(
    dataframe: Optional[pd.DataFrame],
    names: list[str],
    default: Any = None,
) -> Any:

    if dataframe is None or dataframe.empty:
        return default

    column = find_column(dataframe, names)

    if column is not None:
        series = dataframe[column].dropna()

        if not series.empty:
            return series.iloc[0]

    # ------------------------------------------------------------------------
    # IMPORTANT FALLBACK:
    #
    # Many of our phase CSVs may be in:
    #
    # feature,value
    #
    # format rather than:
    #
    # melodic_complexity,...
    #
    # Search those rows too.
    # ------------------------------------------------------------------------

    feature_column = find_column(
        dataframe,
        [
            "feature",
            "metric",
            "parameter",
            "name",
            "measure",
        ],
    )

    value_column = find_column(
        dataframe,
        [
            "value",
            "metric_value",
            "measurement",
            "score",
        ],
    )

    if feature_column is not None and value_column is not None:

        wanted = {
            normalize_name(name)
            for name in names
        }

        for _, row in dataframe.iterrows():

            feature_name = normalize_name(
                row.get(feature_column, "")
            )

            if feature_name in wanted:
                return row.get(value_column)

    return default


def find_json_recursive(
    data: Any,
    names: list[str],
) -> Any:
    """
    Recursively search JSON dictionaries/lists.

    This prevents Phase 11.4 from becoming dependent on one exact
    JSON nesting structure.
    """

    wanted = {
        normalize_name(name)
        for name in names
    }

    if isinstance(data, dict):

        for key, value in data.items():

            if normalize_name(key) in wanted:
                return value

        for value in data.values():

            result = find_json_recursive(value, names)

            if result is not None:
                return result

    elif isinstance(data, list):

        for item in data:

            result = find_json_recursive(item, names)

            if result is not None:
                return result

    return None


def find_value(
    dataframe: Optional[pd.DataFrame],
    json_data: Optional[dict],
    names: list[str],
    default: Any = None,
) -> Any:

    value = find_dataframe_value(
        dataframe,
        names,
        None,
    )

    if value is not None:
        return value

    value = find_json_recursive(
        json_data,
        names,
    )

    if value is not None:
        return value

    return default


# ============================================================================
# PHASE 11.1
# ============================================================================

def load_phase_11_1() -> dict:

    print()
    print("Loading Phase 11.1 results...")

    instrument_df = read_csv("instrument_analysis.csv")
    chord_df = read_csv("chord_analysis.csv")
    key_df = read_csv("key_scale_analysis.csv")
    structure_df = read_csv("musical_structure.csv")
    summary_json = read_json("phase_11_1_summary.json")

    result = {
        "brightness": clean_text(
            find_value(
                instrument_df,
                summary_json,
                ["brightness", "brightness_category"],
            )
        ),
        "noisiness": clean_text(
            find_value(
                instrument_df,
                summary_json,
                ["noisiness", "noisiness_category"],
            )
        ),
        "mean_rms": safe_float(
            find_value(
                instrument_df,
                summary_json,
                [
                    "mean_rms",
                    "mean_rms_energy",
                    "rms_energy",
                ],
            )
        ),
        "spectral_centroid": safe_float(
            find_value(
                instrument_df,
                summary_json,
                [
                    "spectral_centroid",
                    "mean_spectral_centroid",
                ],
            )
        ),
        "spectral_bandwidth": safe_float(
            find_value(
                instrument_df,
                summary_json,
                [
                    "spectral_bandwidth",
                    "mean_spectral_bandwidth",
                ],
            )
        ),
        "dominant_chord": "Unknown",
        "chord_segments": 0,
        "key": clean_text(
            find_value(
                key_df,
                summary_json,
                ["key", "estimated_key"],
            )
        ),
        "scale": clean_text(
            find_value(
                key_df,
                summary_json,
                ["scale", "estimated_scale"],
            )
        ),
        "key_confidence": safe_float(
            find_value(
                key_df,
                summary_json,
                [
                    "confidence",
                    "key_confidence",
                    "similarity",
                ],
            )
        ),
        "sections": 0,
    }

    # ------------------------------------------------------------------------
    # Chords
    # ------------------------------------------------------------------------

    if chord_df is not None and not chord_df.empty:

        result["chord_segments"] = len(chord_df)

        chord_column = find_column(
            chord_df,
            ["chord"],
        )

        if chord_column is not None:

            counts = (
                chord_df[chord_column]
                .dropna()
                .astype(str)
                .value_counts()
            )

            if not counts.empty:
                result["dominant_chord"] = counts.index[0]

    # ------------------------------------------------------------------------
    # Structure
    # ------------------------------------------------------------------------

    if structure_df is not None and not structure_df.empty:
        result["sections"] = len(structure_df)

    print(
        f"  Key              : {result['key']}"
    )
    print(
        f"  Scale            : {result['scale']}"
    )
    print(
        f"  Key confidence   : {result['key_confidence']:.2f}%"
    )
    print(
        f"  Chord segments   : {result['chord_segments']}"
    )
    print(
        f"  Sections         : {result['sections']}"
    )

    return result


# ============================================================================
# PHASE 11.2
# ============================================================================

def load_phase_11_2() -> dict:

    print()
    print("Loading Phase 11.2 results...")

    pitch_df = read_csv(
        "advanced_pitch_range.csv"
    )

    movement_df = read_csv(
        "melodic_movement_analysis.csv"
    )

    complexity_df = read_csv(
        "melodic_complexity_analysis.csv"
    )

    stability_df = read_csv(
        "pitch_stability_analysis.csv"
    )

    phrase_df = read_csv(
        "advanced_phrase_melody_analysis.csv"
    )

    rhythm_df = read_csv(
        "rhythmic_complexity_analysis.csv"
    )

    relationship_df = read_csv(
        "melody_rhythm_relationship.csv"
    )

    summary_json = read_json(
        "phase_11_2_summary.json"
    )

    result = {
        "min_pitch_hz": safe_float(
            find_value(
                pitch_df,
                summary_json,
                [
                    "minimum_pitch_hz",
                    "min_pitch_hz",
                    "minimum_pitch",
                ],
            )
        ),
        "max_pitch_hz": safe_float(
            find_value(
                pitch_df,
                summary_json,
                [
                    "maximum_pitch_hz",
                    "max_pitch_hz",
                    "maximum_pitch",
                ],
            )
        ),
        "median_pitch_hz": safe_float(
            find_value(
                pitch_df,
                summary_json,
                [
                    "median_pitch_hz",
                    "median_pitch",
                ],
            )
        ),
        "pitch_range_semitones": safe_float(
            find_value(
                pitch_df,
                summary_json,
                [
                    "pitch_range_semitones",
                    "pitch_range",
                ],
            )
        ),
        "pitch_range_octaves": safe_float(
            find_value(
                pitch_df,
                summary_json,
                [
                    "pitch_range_octaves",
                    "range_octaves",
                ],
            )
        ),
        "ascending": safe_int(
            find_value(
                movement_df,
                summary_json,
                [
                    "ascending_movements",
                    "ascending",
                    "ascending_count",
                ],
            )
        ),
        "descending": safe_int(
            find_value(
                movement_df,
                summary_json,
                [
                    "descending_movements",
                    "descending",
                    "descending_count",
                ],
            )
        ),
        "stationary": safe_int(
            find_value(
                movement_df,
                summary_json,
                [
                    "stationary_movements",
                    "stationary",
                    "stationary_count",
                ],
            )
        ),
        "mean_interval": safe_float(
            find_value(
                movement_df,
                summary_json,
                [
                    "mean_interval",
                    "mean_interval_semitones",
                ],
            )
        ),
        "melodic_complexity": safe_float(
            find_value(
                complexity_df,
                summary_json,
                [
                    "complexity_score",
                    "melodic_complexity",
                    "melodic_complexity_score",
                ],
            )
        ),
        "melodic_complexity_category": clean_text(
            find_value(
                complexity_df,
                summary_json,
                [
                    "complexity_category",
                    "melodic_complexity_category",
                    "category",
                ],
            )
        ),
        "pitch_stability_cents": safe_float(
            find_value(
                stability_df,
                summary_json,
                [
                    "mean_deviation_cents",
                    "mean_deviation",
                    "pitch_stability_cents",
                ],
            )
        ),
        "pitch_within_25": safe_float(
            find_value(
                stability_df,
                summary_json,
                [
                    "within_25_cents",
                    "within_25",
                    "percent_within_25",
                ],
            )
        ),
        "phrase_count": (
            len(phrase_df)
            if phrase_df is not None
            else 0
        ),
        "rhythmic_complexity": safe_float(
            find_value(
                rhythm_df,
                summary_json,
                [
                    "complexity_score",
                    "rhythmic_complexity",
                    "rhythmic_complexity_score",
                ],
            )
        ),
        "rhythmic_complexity_category": clean_text(
            find_value(
                rhythm_df,
                summary_json,
                [
                    "category",
                    "complexity_category",
                    "rhythmic_complexity_category",
                ],
            )
        ),
        "mean_beat_interval": safe_float(
            find_value(
                rhythm_df,
                summary_json,
                [
                    "mean_beat_interval",
                    "beat_interval",
                ],
            )
        ),
        "interval_variation": safe_float(
            find_value(
                rhythm_df,
                summary_json,
                [
                    "interval_variation",
                    "variation_percent",
                ],
            )
        ),
        "mean_note_duration": safe_float(
            find_value(
                relationship_df,
                summary_json,
                [
                    "mean_note_duration",
                    "mean_duration",
                ],
            )
        ),
        "median_note_duration": safe_float(
            find_value(
                relationship_df,
                summary_json,
                [
                    "median_duration",
                    "median_note_duration",
                ],
            )
        ),
        "mean_gap": safe_float(
            find_value(
                relationship_df,
                summary_json,
                [
                    "mean_gap",
                    "mean_note_gap",
                    "gap",
                ],
            )
        ),
    }

    print()
    print("  Phase 11.2 extracted values:")
    print(
        f"    Melodic complexity : "
        f"{result['melodic_complexity']:.2f}"
    )
    print(
        f"    Rhythmic complexity: "
        f"{result['rhythmic_complexity']:.2f}"
    )
    print(
        f"    Pitch deviation    : "
        f"{result['pitch_stability_cents']:.2f} cents"
    )
    print(
        f"    Pitch range        : "
        f"{result['pitch_range_semitones']:.2f} semitones"
    )
    print(
        f"    Mean beat interval : "
        f"{result['mean_beat_interval']:.4f} sec"
    )
    print(
        f"    Mean note duration : "
        f"{result['mean_note_duration']:.4f} sec"
    )
    print(
        f"    Mean gap           : "
        f"{result['mean_gap']:.4f} sec"
    )

    return result


# ============================================================================
# PHASE 11.3
# ============================================================================

def load_phase_11_3() -> dict:

    print()
    print("Loading Phase 11.3 results...")

    dynamics_df = read_csv(
        "phase_11_3_dynamics.csv"
    )

    tempo_df = read_csv(
        "phase_11_3_tempo_expression.csv"
    )

    pitch_df = read_csv(
        "phase_11_3_pitch_expression.csv"
    )

    summary_json = read_json(
        "phase_11_3_summary.json"
    )

    result = {
        "mean_rms": safe_float(
            find_value(
                dynamics_df,
                summary_json,
                [
                    "mean_rms_energy",
                    "mean_rms",
                ],
            )
        ),
        "median_rms": safe_float(
            find_value(
                dynamics_df,
                summary_json,
                [
                    "median_rms_energy",
                    "median_rms",
                ],
            )
        ),
        "mean_db": safe_float(
            find_value(
                dynamics_df,
                summary_json,
                [
                    "mean_energy_db",
                    "mean_db",
                ],
            )
        ),
        "median_db": safe_float(
            find_value(
                dynamics_df,
                summary_json,
                [
                    "median_energy_db",
                    "median_db",
                ],
            )
        ),
        "dynamic_range_db": safe_float(
            find_value(
                dynamics_df,
                summary_json,
                [
                    "dynamic_range_db",
                    "dynamic_range",
                ],
            )
        ),
        "dynamics_category": clean_text(
            find_value(
                dynamics_df,
                summary_json,
                [
                    "dynamics_category",
                    "category",
                ],
            )
        ),
        "onsets": safe_int(
            find_value(
                tempo_df,
                summary_json,
                [
                    "detected_onsets",
                    "onsets",
                ],
            )
        ),
        "mean_local_tempo": safe_float(
            find_value(
                tempo_df,
                summary_json,
                [
                    "mean_local_tempo",
                    "mean_tempo",
                    "local_tempo",
                ],
            )
        ),
        "median_local_tempo": safe_float(
            find_value(
                tempo_df,
                summary_json,
                [
                    "median_local_tempo",
                    "median_tempo",
                ],
            )
        ),
        "tempo_std": safe_float(
            find_value(
                tempo_df,
                summary_json,
                [
                    "tempo_standard_deviation",
                    "tempo_std",
                ],
            )
        ),
        "tempo_variation": safe_float(
            find_value(
                tempo_df,
                summary_json,
                [
                    "tempo_variation",
                    "tempo_variation_percent",
                    "variation_percent",
                ],
            )
        ),
        "timing_category": clean_text(
            find_value(
                tempo_df,
                summary_json,
                [
                    "timing_category",
                    "category",
                ],
            )
        ),
        "voiced_frames": safe_int(
            find_value(
                pitch_df,
                summary_json,
                [
                    "voiced_frames",
                ],
            )
        ),
        "pitch_modulation_depth": safe_float(
            find_value(
                pitch_df,
                summary_json,
                [
                    "pitch_modulation_depth",
                    "modulation_depth_cents",
                    "modulation_depth",
                ],
            )
        ),
        "modulation_rate": safe_float(
            find_value(
                pitch_df,
                summary_json,
                [
                    "estimated_modulation_rate",
                    "modulation_rate",
                ],
            )
        ),
        "pitch_expression": clean_text(
            find_value(
                pitch_df,
                summary_json,
                [
                    "pitch_expression",
                    "expression",
                ],
            )
        ),
    }

    print()
    print("  Phase 11.3 extracted values:")
    print(
        f"    Dynamic range      : "
        f"{result['dynamic_range_db']:.2f} dB"
    )
    print(
        f"    Mean local tempo   : "
        f"{result['mean_local_tempo']:.2f} BPM"
    )
    print(
        f"    Tempo variation    : "
        f"{result['tempo_variation']:.2f}%"
    )
    print(
        f"    Modulation depth   : "
        f"{result['pitch_modulation_depth']:.2f} cents"
    )

    return result


# ============================================================================
# CROSS FEATURE SYNTHESIS
# ============================================================================

def calculate_cross_feature_analysis(
    p11_1: dict,
    p11_2: dict,
    p11_3: dict,
) -> dict:

    ascending = safe_float(
        p11_2["ascending"]
    )

    descending = safe_float(
        p11_2["descending"]
    )

    stationary = safe_float(
        p11_2["stationary"]
    )

    total_movement = (
        ascending
        + descending
        + stationary
    )

    if total_movement > 0:

        ascending_pct = (
            ascending / total_movement * 100
        )

        descending_pct = (
            descending / total_movement * 100
        )

        stationary_pct = (
            stationary / total_movement * 100
        )

    else:

        ascending_pct = 0.0
        descending_pct = 0.0
        stationary_pct = 0.0

    # ------------------------------------------------------------------------
    # Note activity
    # ------------------------------------------------------------------------

    mean_note_duration = safe_float(
        p11_2["mean_note_duration"]
    )

    mean_gap = safe_float(
        p11_2["mean_gap"]
    )

    total_note_time = (
        mean_note_duration + mean_gap
    )

    if total_note_time > 0:
        note_activity_ratio = (
            mean_note_duration
            / total_note_time
        )
    else:
        note_activity_ratio = 0.0

    # ------------------------------------------------------------------------
    # Theoretical tempo
    # ------------------------------------------------------------------------

    mean_beat_interval = safe_float(
        p11_2["mean_beat_interval"]
    )

    if mean_beat_interval > 0:
        theoretical_tempo = (
            60.0 / mean_beat_interval
        )
    else:
        theoretical_tempo = 0.0

    measured_tempo = safe_float(
        p11_3["mean_local_tempo"]
    )

    if measured_tempo > 0 and theoretical_tempo > 0:

        tempo_consistency_ratio = clamp(
            1.0
            - (
                abs(
                    measured_tempo
                    - theoretical_tempo
                )
                / max(
                    measured_tempo,
                    theoretical_tempo,
                )
            ),
            0.0,
            1.0,
        )

    else:
        tempo_consistency_ratio = 0.0

    # ------------------------------------------------------------------------
    # Pitch stability
    # ------------------------------------------------------------------------

    pitch_deviation = safe_float(
        p11_2["pitch_stability_cents"]
    )

    if pitch_deviation <= 10:
        pitch_stability_category = "Very stable"

    elif pitch_deviation <= 20:
        pitch_stability_category = "Stable"

    elif pitch_deviation <= 35:
        pitch_stability_category = (
            "Moderately variable"
        )

    else:
        pitch_stability_category = (
            "Highly variable"
        )

    # ------------------------------------------------------------------------
    # Complexity
    # ------------------------------------------------------------------------

    melodic_complexity = clamp(
        safe_float(
            p11_2["melodic_complexity"]
        ),
        0.0,
        100.0,
    )

    rhythmic_complexity = clamp(
        safe_float(
            p11_2["rhythmic_complexity"]
        ),
        0.0,
        100.0,
    )

    tempo_variation = clamp(
        safe_float(
            p11_3["tempo_variation"]
        ),
        0.0,
        100.0,
    )

    overall_complexity = (
        melodic_complexity * 0.45
        + rhythmic_complexity * 0.35
        + tempo_variation * 0.20
    )

    if overall_complexity < 20:
        complexity_category = "Low"

    elif overall_complexity < 40:
        complexity_category = "Moderate"

    elif overall_complexity < 65:
        complexity_category = "High"

    else:
        complexity_category = "Very high"

    # ------------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------------

    dynamic_range = safe_float(
        p11_3["dynamic_range_db"]
    )

    if dynamic_range < 6:
        dynamic_category = "Narrow"

    elif dynamic_range < 12:
        dynamic_category = "Moderate"

    elif dynamic_range < 20:
        dynamic_category = "Wide"

    else:
        dynamic_category = "Very wide"

    # ------------------------------------------------------------------------
    # Melodic range
    # ------------------------------------------------------------------------

    pitch_range = safe_float(
        p11_2["pitch_range_semitones"]
    )

    if pitch_range < 12:
        melodic_range_category = "Compact"

    elif pitch_range < 24:
        melodic_range_category = "Moderate"

    elif pitch_range < 36:
        melodic_range_category = "Wide"

    else:
        melodic_range_category = "Very wide"

    return {
        "ascending_percentage": ascending_pct,
        "descending_percentage": descending_pct,
        "stationary_percentage": stationary_pct,
        "note_activity_ratio": note_activity_ratio,
        "theoretical_tempo_bpm": theoretical_tempo,
        "tempo_consistency_ratio": tempo_consistency_ratio,
        "pitch_stability_category": pitch_stability_category,
        "overall_complexity_score": overall_complexity,
        "overall_complexity_category": complexity_category,
        "dynamic_range_category": dynamic_category,
        "melodic_range_category": melodic_range_category,
    }


# ============================================================================
# MUSICAL PROFILE CSV
# ============================================================================

def create_musical_profile(
    p11_1: dict,
    p11_2: dict,
    p11_3: dict,
    cross: dict,
) -> pd.DataFrame:

    rows = [

        ("Estimated key", p11_1["key"]),
        ("Estimated scale", p11_1["scale"]),
        ("Key confidence (%)", p11_1["key_confidence"]),

        ("Dominant chord", p11_1["dominant_chord"]),
        ("Chord segments", p11_1["chord_segments"]),
        ("Structural sections", p11_1["sections"]),

        ("Spectral centroid (Hz)", p11_1["spectral_centroid"]),
        ("Spectral bandwidth (Hz)", p11_1["spectral_bandwidth"]),

        ("Minimum pitch (Hz)", p11_2["min_pitch_hz"]),
        ("Maximum pitch (Hz)", p11_2["max_pitch_hz"]),
        ("Median pitch (Hz)", p11_2["median_pitch_hz"]),
        ("Pitch range (semitones)", p11_2["pitch_range_semitones"]),
        ("Pitch range (octaves)", p11_2["pitch_range_octaves"]),

        ("Melodic complexity", p11_2["melodic_complexity"]),
        (
            "Pitch stability deviation (cents)",
            p11_2["pitch_stability_cents"],
        ),
        (
            "Pitch within ±25 cents (%)",
            p11_2["pitch_within_25"],
        ),

        ("Melodic phrases", p11_2["phrase_count"]),

        ("Rhythmic complexity", p11_2["rhythmic_complexity"]),
        (
            "Mean beat interval (sec)",
            p11_2["mean_beat_interval"],
        ),
        (
            "Mean note duration (sec)",
            p11_2["mean_note_duration"],
        ),
        (
            "Mean note gap (sec)",
            p11_2["mean_gap"],
        ),

        ("Mean local tempo (BPM)", p11_3["mean_local_tempo"]),
        ("Tempo variation (%)", p11_3["tempo_variation"]),
        ("Dynamic range (dB)", p11_3["dynamic_range_db"]),
        (
            "Pitch modulation depth (cents)",
            p11_3["pitch_modulation_depth"],
        ),

        (
            "Overall complexity score",
            cross["overall_complexity_score"],
        ),
    ]

    return pd.DataFrame(
        rows,
        columns=["feature", "value"],
    )


# ============================================================================
# INTERPRETATION
# ============================================================================

def generate_interpretation(
    p11_1: dict,
    p11_2: dict,
    p11_3: dict,
    cross: dict,
) -> list[dict[str, str]]:

    return [

        {
            "area": "Tonal profile",
            "interpretation": (
                f"The estimated tonal center is "
                f"{p11_1['key']} with a "
                f"{p11_1['scale']} scale classification. "
                f"Estimated key confidence is "
                f"{p11_1['key_confidence']:.2f}%."
            ),
        },

        {
            "area": "Melodic profile",
            "interpretation": (
                f"The detected melodic range spans "
                f"{p11_2['pitch_range_semitones']:.2f} "
                f"semitones "
                f"({p11_2['pitch_range_octaves']:.2f} octaves). "
                f"The melodic complexity score is "
                f"{p11_2['melodic_complexity']:.2f}/100."
            ),
        },

        {
            "area": "Melodic movement",
            "interpretation": (
                f"Melodic movement consists of approximately "
                f"{cross['ascending_percentage']:.1f}% ascending, "
                f"{cross['descending_percentage']:.1f}% descending, "
                f"and "
                f"{cross['stationary_percentage']:.1f}% stationary "
                f"movement."
            ),
        },

        {
            "area": "Rhythmic profile",
            "interpretation": (
                f"Measured rhythmic complexity is "
                f"{p11_2['rhythmic_complexity']:.2f}/100. "
                f"The Phase 11.3 local tempo estimate is "
                f"{p11_3['mean_local_tempo']:.2f} BPM."
            ),
        },

        {
            "area": "Timing expression",
            "interpretation": (
                f"Tempo variation is approximately "
                f"{p11_3['tempo_variation']:.2f}%, "
                f"classified as "
                f"{p11_3['timing_category']}."
            ),
        },

        {
            "area": "Pitch stability",
            "interpretation": (
                f"Mean pitch deviation is approximately "
                f"{p11_2['pitch_stability_cents']:.2f} cents. "
                f"This corresponds to the descriptive category "
                f"{cross['pitch_stability_category']}."
            ),
        },

        {
            "area": "Dynamics",
            "interpretation": (
                f"The measured dynamic range is approximately "
                f"{p11_3['dynamic_range_db']:.2f} dB. "
                f"The resulting descriptive profile is "
                f"{cross['dynamic_range_category']}."
            ),
        },

        {
            "area": "Pitch expression",
            "interpretation": (
                f"Phase 11.3 reports: "
                f"{p11_3['pitch_expression']}. "
                f"Modulation depth was approximately "
                f"{p11_3['pitch_modulation_depth']:.2f} cents."
            ),
        },

        {
            "area": "Overall musical profile",
            "interpretation": (
                f"Combining the Phase 11 melody, rhythm, "
                f"and performance features gives an overall "
                f"descriptive complexity score of "
                f"{cross['overall_complexity_score']:.2f}/100, "
                f"classified as "
                f"{cross['overall_complexity_category']}. "
                f"This is a feature-based characterization."
            ),
        },
    ]


# ============================================================================
# CROSS FEATURE CSV
# ============================================================================

def save_cross_feature_analysis(
    cross: dict,
) -> pd.DataFrame:

    rows = [

        {
            "feature": "Ascending movement (%)",
            "value": cross["ascending_percentage"],
        },

        {
            "feature": "Descending movement (%)",
            "value": cross["descending_percentage"],
        },

        {
            "feature": "Stationary movement (%)",
            "value": cross["stationary_percentage"],
        },

        {
            "feature": "Note activity ratio (%)",
            "value": cross["note_activity_ratio"] * 100,
        },

        {
            "feature": "Tempo consistency (%)",
            "value": cross["tempo_consistency_ratio"] * 100,
        },

        {
            "feature": "Overall complexity score",
            "value": cross["overall_complexity_score"],
        },
    ]

    dataframe = pd.DataFrame(rows)

    dataframe.to_csv(
        OUTPUT_DIR
        / "phase_11_4_cross_feature_analysis.csv",
        index=False,
    )

    return dataframe


# ============================================================================
# GRAPH 1 - MUSICAL PROFILE
# ============================================================================

def generate_musical_profile_graph(
    p11_2: dict,
    p11_3: dict,
) -> None:

    labels = [
        "Melodic\ncomplexity",
        "Rhythmic\ncomplexity",
        "Tempo\nvariation",
        "Pitch\nvariation",
        "Dynamic\nrange",
    ]

    melodic = clamp(
        safe_float(
            p11_2["melodic_complexity"]
        ),
        0,
        100,
    )

    rhythmic = clamp(
        safe_float(
            p11_2["rhythmic_complexity"]
        ),
        0,
        100,
    )

    tempo = clamp(
        safe_float(
            p11_3["tempo_variation"]
        ),
        0,
        100,
    )

    pitch_deviation = safe_float(
        p11_2["pitch_stability_cents"]
    )

    pitch_variation = clamp(
        (pitch_deviation / 50.0) * 100.0,
        0,
        100,
    )

    dynamic_range = safe_float(
        p11_3["dynamic_range_db"]
    )

    dynamic_variation = clamp(
        (dynamic_range / 20.0) * 100.0,
        0,
        100,
    )

    values = [
        melodic,
        rhythmic,
        tempo,
        pitch_variation,
        dynamic_variation,
    ]

    # ------------------------------------------------------------------------
    # DEBUG INFORMATION
    # ------------------------------------------------------------------------

    print()
    print("Musical profile graph values:")

    for label, value in zip(labels, values):
        print(
            f"  {label.replace(chr(10), ' '):25s}: "
            f"{value:.2f}"
        )

    # ------------------------------------------------------------------------
    # GRAPH
    # ------------------------------------------------------------------------

    plt.figure(
        figsize=(11, 6)
    )

    bars = plt.bar(
        labels,
        values,
    )

    plt.ylim(
        0,
        100,
    )

    plt.ylabel(
        "Normalized feature value (0–100)"
    )

    plt.title(
        "Phase 11.4 - Musical Profile"
    )

    plt.grid(
        axis="y",
        alpha=0.25,
    )

    # Put actual values above bars.
    for bar, value in zip(bars, values):

        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            value + 2,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()

    output_path = (
        GRAPH_DIR
        / "phase_11_4_musical_profile.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"  Saved: {output_path}"
    )


# ============================================================================
# GRAPH 2 - MOVEMENT DISTRIBUTION
# ============================================================================

def generate_cross_feature_graph(
    cross: dict,
) -> None:

    labels = [
        "Ascending",
        "Descending",
        "Stationary",
    ]

    values = [
        cross["ascending_percentage"],
        cross["descending_percentage"],
        cross["stationary_percentage"],
    ]

    print()
    print("Movement graph values:")

    for label, value in zip(labels, values):
        print(
            f"  {label:15s}: {value:.2f}%"
        )

    plt.figure(
        figsize=(9, 6)
    )

    bars = plt.bar(
        labels,
        values,
    )

    plt.ylabel(
        "Percentage (%)"
    )

    plt.title(
        "Phase 11.4 - Melodic Movement Distribution"
    )

    plt.ylim(
        0,
        max(
            100,
            max(values) * 1.2
            if values
            else 100,
        ),
    )

    plt.grid(
        axis="y",
        alpha=0.25,
    )

    for bar, value in zip(bars, values):

        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            value + 1,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()

    output_path = (
        GRAPH_DIR
        / "phase_11_4_cross_feature_analysis.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"  Saved: {output_path}"
    )


# ============================================================================
# SUMMARY
# ============================================================================

def save_summary(
    p11_1: dict,
    p11_2: dict,
    p11_3: dict,
    cross: dict,
    interpretation: list[dict[str, str]],
) -> None:

    summary = {

        "phase": "11.4",

        "title":
            "Musical Interpretation & Final Feature Synthesis",

        "source_phases": [
            "11.1 Musical Element Analysis",
            "11.2 Advanced Melody & Rhythm Analysis",
            "11.3 Musical Expression & Performance Analysis",
        ],

        "phase_11_1": p11_1,
        "phase_11_2": p11_2,
        "phase_11_3": p11_3,

        "cross_feature_analysis": cross,

        "interpretation": interpretation,

        "notes": [
            "Phase 11.4 is descriptive and feature-based.",
            "No new audio DSP processing is performed.",
            "No PyTorch or Whisper dependency is used.",
            "Phase 10 lyrics/word-to-notation remains on hold.",
        ],
    }

    output_path = (
        OUTPUT_DIR
        / "phase_11_4_summary.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print(
        f"  Saved: {output_path}"
    )


# ============================================================================
# MAIN PHASE 11.4
# ============================================================================

def run_phase_11_4() -> dict:

    print()
    print("=" * 80)
    print(
        "PHASE 11.4 - "
        "MUSICAL INTERPRETATION & FINAL FEATURE SYNTHESIS"
    )
    print("=" * 80)

    # ------------------------------------------------------------------------
    # Load previous phases
    # ------------------------------------------------------------------------

    p11_1 = load_phase_11_1()
    p11_2 = load_phase_11_2()
    p11_3 = load_phase_11_3()

    # ------------------------------------------------------------------------
    # Cross synthesis
    # ------------------------------------------------------------------------

    print()
    print("=" * 70)
    print("CROSS-FEATURE SYNTHESIS")
    print("=" * 70)

    cross = calculate_cross_feature_analysis(
        p11_1,
        p11_2,
        p11_3,
    )

    print(
        f"Overall complexity : "
        f"{cross['overall_complexity_score']:.2f}/100"
    )

    print(
        f"Complexity category: "
        f"{cross['overall_complexity_category']}"
    )

    print(
        f"Melodic range      : "
        f"{cross['melodic_range_category']}"
    )

    print(
        f"Pitch stability    : "
        f"{cross['pitch_stability_category']}"
    )

    print(
        f"Dynamic range      : "
        f"{cross['dynamic_range_category']}"
    )

    # ------------------------------------------------------------------------
    # Musical profile
    # ------------------------------------------------------------------------

    profile_df = create_musical_profile(
        p11_1,
        p11_2,
        p11_3,
        cross,
    )

    profile_path = (
        OUTPUT_DIR
        / "phase_11_4_musical_profile.csv"
    )

    profile_df.to_csv(
        profile_path,
        index=False,
    )

    # ------------------------------------------------------------------------
    # Cross feature CSV
    # ------------------------------------------------------------------------

    save_cross_feature_analysis(
        cross
    )

    # ------------------------------------------------------------------------
    # Interpretation
    # ------------------------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL MUSICAL INTERPRETATION")
    print("=" * 70)

    interpretation = generate_interpretation(
        p11_1,
        p11_2,
        p11_3,
        cross,
    )

    interpretation_df = pd.DataFrame(
        interpretation
    )

    interpretation_df.to_csv(
        OUTPUT_DIR
        / "phase_11_4_final_interpretation.csv",
        index=False,
    )

    for item in interpretation:

        print()
        print(
            f"{item['area']}:"
        )

        print(
            f"  {item['interpretation']}"
        )

    # ------------------------------------------------------------------------
    # Graphs
    # ------------------------------------------------------------------------

    print()
    print("=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    generate_musical_profile_graph(
        p11_2,
        p11_3,
    )

    generate_cross_feature_graph(
        cross
    )

    # ------------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------------

    save_summary(
        p11_1,
        p11_2,
        p11_3,
        cross,
        interpretation,
    )

    # ------------------------------------------------------------------------
    # Completion
    # ------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("PHASE 11.4 COMPLETED SUCCESSFULLY")
    print("=" * 80)

    print()
    print("Generated files:")

    print(
        "  data/output/"
        "phase_11_4_musical_profile.csv"
    )

    print(
        "  data/output/"
        "phase_11_4_cross_feature_analysis.csv"
    )

    print(
        "  data/output/"
        "phase_11_4_final_interpretation.csv"
    )

    print(
        "  data/output/"
        "phase_11_4_summary.json"
    )

    print()
    print("Generated graphs:")

    print(
        "  graphs/"
        "phase_11_4_musical_profile.png"
    )

    print(
        "  graphs/"
        "phase_11_4_cross_feature_analysis.png"
    )

    print()
    print("Phase 10 remains ON HOLD.")
    print("No PyTorch/Whisper dependency is used.")

    return {
        "phase": "11.4",
        "status": "completed",
    }


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":

    print()
    print("=" * 80)
    print("PHASE 11.4 STANDALONE TEST")
    print("=" * 80)

    print()
    print(
        "Phase 11.4 does not require an audio path."
    )

    print(
        "It reads the completed Phase 11.1, "
        "11.2 and 11.3 outputs."
    )

    print()

    try:

        run_phase_11_4()

        print()
        print("=" * 80)
        print(
            "PHASE 11.4 STANDALONE TEST FINISHED"
        )
        print("=" * 80)

    except Exception as exc:

        print()
        print("=" * 80)
        print("PHASE 11.4 FAILED")
        print("=" * 80)

        print()
        print(
            f"Error type    : {type(exc).__name__}"
        )

        print(
            f"Error message : {exc}"
        )

        raise