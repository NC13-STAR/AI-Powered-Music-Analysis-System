import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PHASE 9.2 - GENERAL RAGA RECOGNITION & VALIDATION
# ============================================================

OUTPUT_DIR = "data/output"
GRAPH_DIR = "graphs"

FEATURE_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "raga_feature_analysis.csv"
)

VALIDATION_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "raga_validation_results.csv"
)

SUMMARY_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "raga_validation_summary.csv"
)

GRAPH_OUTPUT = os.path.join(
    GRAPH_DIR,
    "raga_validation_analysis.png"
)


# ============================================================
# RAGA MUSICAL PROFILES
# ============================================================

# Relative semitone positions from Sa.
#
# These are simplified computational profiles.
# They are NOT intended to replace expert raga grammar.
#
# 0  = Sa
# 1  = Komal Re
# 2  = Re
# 3  = Komal Ga
# 4  = Ga
# 5  = Ma
# 6  = Tivra Ma
# 7  = Pa
# 8  = Komal Dha
# 9  = Dha
# 10 = Komal Ni
# 11 = Ni


RAGA_PROFILES = {

    "Desh": {
        "allowed": [0, 2, 4, 5, 7, 9, 11],
        "preferred": [0, 2, 4, 5, 7, 11],
        "characteristic": [2, 4, 5, 7, 11],
        "forbidden": [1, 3, 6, 8, 10],
        "ascending": [0, 2, 4, 5, 7, 11],
        "descending": [11, 9, 7, 5, 4, 2, 0]
    },

    "Yaman": {
        "allowed": [0, 2, 4, 6, 7, 9, 11],
        "preferred": [0, 2, 4, 6, 7, 9, 11],
        "characteristic": [4, 6, 7, 11],
        "forbidden": [1, 3, 5, 8, 10],
        "ascending": [0, 2, 4, 6, 7, 9, 11],
        "descending": [11, 9, 7, 6, 4, 2, 0]
    },

    "Khamaj": {
        "allowed": [0, 2, 4, 5, 7, 9, 10, 11],
        "preferred": [0, 2, 4, 5, 7, 9, 10],
        "characteristic": [9, 10, 7, 5],
        "forbidden": [1, 3, 6, 8],
        "ascending": [0, 2, 4, 5, 7, 9, 11],
        "descending": [11, 10, 9, 7, 5, 4, 2, 0]
    },

    "Kafi": {
        "allowed": [0, 2, 3, 5, 7, 9, 10],
        "preferred": [0, 2, 3, 5, 7, 9, 10],
        "characteristic": [3, 5, 7, 10],
        "forbidden": [1, 4, 6, 8, 11],
        "ascending": [0, 2, 3, 5, 7, 9, 10],
        "descending": [10, 9, 7, 5, 3, 2, 0]
    },

    "Bhupali": {
        "allowed": [0, 2, 4, 7, 9],
        "preferred": [0, 2, 4, 7, 9],
        "characteristic": [0, 2, 4, 7, 9],
        "forbidden": [1, 3, 5, 6, 8, 10, 11],
        "ascending": [0, 2, 4, 7, 9],
        "descending": [9, 7, 4, 2, 0]
    },

    "Bilawal": {
        "allowed": [0, 2, 4, 5, 7, 9, 11],
        "preferred": [0, 2, 4, 5, 7, 9, 11],
        "characteristic": [0, 4, 7, 11],
        "forbidden": [1, 3, 6, 8, 10],
        "ascending": [0, 2, 4, 5, 7, 9, 11],
        "descending": [11, 9, 7, 5, 4, 2, 0]
    },

    "Bhairav": {
        "allowed": [0, 1, 4, 5, 7, 8, 11],
        "preferred": [0, 1, 4, 5, 7, 8, 11],
        "characteristic": [1, 4, 8, 11],
        "forbidden": [2, 3, 6, 9, 10],
        "ascending": [0, 1, 4, 5, 7, 8, 11],
        "descending": [11, 8, 7, 5, 4, 1, 0]
    },

    "Todi": {
        "allowed": [0, 1, 3, 5, 6, 7, 8, 10],
        "preferred": [0, 1, 3, 5, 6, 7, 8, 10],
        "characteristic": [1, 3, 6, 8],
        "forbidden": [2, 4, 9, 11],
        "ascending": [0, 1, 3, 6, 7, 8, 10],
        "descending": [10, 8, 7, 6, 3, 1, 0]
    }
}


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def safe_float(value, default=np.nan):

    try:

        value = float(value)

        if np.isfinite(value):
            return value

    except (TypeError, ValueError):

        pass

    return default


def normalize_score(value):

    value = safe_float(value, 0.0)

    return max(
        0.0,
        min(
            1.0,
            value
        )
    )


def parse_list(value):

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if pd.isna(value):
        return []

    try:

        parsed = ast.literal_eval(
            str(value)
        )

        if isinstance(
            parsed,
            (list, tuple)
        ):

            return list(parsed)

    except Exception:

        pass

    return []


# ============================================================
# LOAD TONIC
# ============================================================

def load_tonic():

    path = os.path.join(
        OUTPUT_DIR,
        "tonic_estimation.csv"
    )

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    data = pd.read_csv(path)

    if len(data) == 0:

        raise ValueError(
            "tonic_estimation.csv is empty."
        )

    row = data.iloc[0]

    tonic_name = str(
        row.get(
            "estimated_tonic",
            "Unknown"
        )
    )

    tonic_pitch_class = safe_float(
        row.get(
            "tonic_pitch_class",
            np.nan
        )
    )

    tonic_frequency = safe_float(
        row.get(
            "tonic_frequency_hz",
            np.nan
        )
    )

    tonic_confidence = safe_float(
        row.get(
            "confidence",
            np.nan
        )
    )

    if not np.isfinite(
        tonic_pitch_class
    ):

        raise ValueError(
            "Tonic pitch class is unavailable."
        )

    return (
        tonic_name,
        int(round(tonic_pitch_class)) % 12,
        tonic_frequency,
        tonic_confidence
    )


# ============================================================
# LOAD SWARA DATA
# ============================================================

def load_swara_data():

    path = os.path.join(
        OUTPUT_DIR,
        "indian_swara_data.csv"
    )

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    data = pd.read_csv(path)

    if len(data) == 0:

        raise ValueError(
            "indian_swara_data.csv is empty."
        )

    return data


# ============================================================
# DETERMINE RELATIVE PITCH COLUMN
# ============================================================

def find_pitch_column(data):

    candidates = [

        "semitone_from_sa",

        "relative_pitch_class",

        "relative_semitone",

        "pitch_class_relative",

        "analysis_midi"

    ]

    for column in candidates:

        if column in data.columns:

            return column

    return None


# ============================================================
# CREATE RELATIVE PITCH DATA
# ============================================================

def prepare_pitch_data(
    data,
    tonic_pitch_class
):

    working = data.copy()

    # --------------------------------------------------------
    # Preferred case:
    # Phase 7.5 already calculated semitone_from_sa.
    # --------------------------------------------------------

    if "semitone_from_sa" in working.columns:

        relative = pd.to_numeric(
            working["semitone_from_sa"],
            errors="coerce"
        )

    elif "relative_pitch_class" in working.columns:

        relative = pd.to_numeric(
            working["relative_pitch_class"],
            errors="coerce"
        )

    elif "analysis_midi" in working.columns:

        midi = pd.to_numeric(
            working["analysis_midi"],
            errors="coerce"
        )

        relative = (
            midi.round()
            - tonic_pitch_class
        ) % 12

    elif "quantized_midi" in working.columns:

        midi = pd.to_numeric(
            working["quantized_midi"],
            errors="coerce"
        )

        relative = (
            midi.round()
            - tonic_pitch_class
        ) % 12

    else:

        raise ValueError(
            "No usable pitch information found "
            "in indian_swara_data.csv."
        )

    working[
        "relative_pitch_class_analysis"
    ] = (
        relative
        .round()
        % 12
    )

    return working


# ============================================================
# FEATURE 1 - SWARA DISTRIBUTION
# ============================================================

def calculate_swara_distribution(
    data
):

    counts = np.zeros(
        12,
        dtype=float
    )

    duration_weights = np.zeros(
        12,
        dtype=float
    )

    for _, row in data.iterrows():

        relative = safe_float(
            row.get(
                "relative_pitch_class_analysis",
                np.nan
            )
        )

        if not np.isfinite(relative):
            continue

        relative = int(
            round(relative)
        ) % 12

        counts[relative] += 1

        duration = safe_float(
            row.get(
                "note_duration_seconds",
                row.get(
                    "duration_seconds",
                    row.get(
                        "duration",
                        1.0
                    )
                )
            ),
            1.0
        )

        if duration <= 0:
            duration = 1.0

        duration_weights[
            relative
        ] += duration

    count_total = counts.sum()
    duration_total = duration_weights.sum()

    if count_total > 0:

        count_distribution = (
            counts / count_total
        )

    else:

        count_distribution = (
            np.zeros(12)
        )

    if duration_total > 0:

        duration_distribution = (
            duration_weights
            / duration_total
        )

    else:

        duration_distribution = (
            np.zeros(12)
        )

    return (
        count_distribution,
        duration_distribution
    )


# ============================================================
# FEATURE 2 - NOTE TRANSITIONS
# ============================================================

def calculate_transitions(
    data
):

    transitions = []

    ordered = data.copy()

    if "start_time_seconds" in ordered.columns:

        ordered = ordered.sort_values(
            "start_time_seconds"
        )

    previous = None

    for value in ordered[
        "relative_pitch_class_analysis"
    ]:

        if not np.isfinite(value):
            continue

        current = int(
            round(value)
        ) % 12

        if previous is not None:

            transitions.append(
                (
                    previous,
                    current
                )
            )

        previous = current

    return transitions


# ============================================================
# FEATURE 3 - CHARACTERISTIC MOVEMENT
# ============================================================

def characteristic_movement_score(
    transitions,
    characteristic_notes
):

    if len(transitions) == 0:

        return 0.0

    matches = 0

    for first, second in transitions:

        if (
            first in characteristic_notes
            and
            second in characteristic_notes
        ):

            matches += 1

    return matches / len(transitions)


# ============================================================
# FEATURE 4 - ASCENDING / DESCENDING BEHAVIOR
# ============================================================

def movement_direction_score(
    transitions,
    ascending_pattern,
    descending_pattern
):

    if len(transitions) == 0:

        return 0.0

    score = 0.0

    for first, second in transitions:

        if second > first:

            if (
                first in ascending_pattern
                and
                second in ascending_pattern
            ):

                score += 1.0

        elif second < first:

            if (
                first in descending_pattern
                and
                second in descending_pattern
            ):

                score += 1.0

    return score / len(transitions)


# ============================================================
# FEATURE 5 - ALLOWED NOTE SCORE
# ============================================================

def allowed_note_score(
    distribution,
    allowed_notes
):

    total = 0.0

    for pitch in allowed_notes:

        total += distribution[pitch]

    return total


# ============================================================
# FEATURE 6 - PREFERRED NOTE SCORE
# ============================================================

def preferred_note_score(
    distribution,
    preferred_notes
):

    if len(preferred_notes) == 0:

        return 0.0

    values = [

        distribution[pitch]

        for pitch in preferred_notes
    ]

    return float(
        np.mean(values)
    )


# ============================================================
# FEATURE 7 - FORBIDDEN NOTE PENALTY
# ============================================================

def forbidden_note_penalty(
    distribution,
    forbidden_notes
):

    if len(forbidden_notes) == 0:

        return 0.0

    values = [

        distribution[pitch]

        for pitch in forbidden_notes
    ]

    return float(
        np.sum(values)
    )


# ============================================================
# FEATURE 8 - SCALE COMPLETENESS
# ============================================================

def scale_completeness(
    distribution,
    allowed_notes
):

    if len(allowed_notes) == 0:

        return 0.0

    present = 0

    for pitch in allowed_notes:

        if distribution[pitch] > 0.01:

            present += 1

    return (
        present
        / len(allowed_notes)
    )


# ============================================================
# FEATURE ANALYSIS
# ============================================================

def analyze_raga_features():

    print("\n" + "=" * 70)
    print(
        "PHASE 9.2.1 - RAGA FEATURE ANALYSIS"
    )
    print("=" * 70)

    print(
        "\nLoading current song analysis..."
    )

    (
        tonic_name,
        tonic_pitch_class,
        tonic_frequency,
        tonic_confidence
    ) = load_tonic()

    data = load_swara_data()

    print(
        f"Estimated tonic : {tonic_name}"
    )

    print(
        f"Tonic pitch class : "
        f"{tonic_pitch_class}"
    )

    if np.isfinite(tonic_frequency):

        print(
            f"Tonic frequency : "
            f"{tonic_frequency:.2f} Hz"
        )

    print(
        f"Melodic note events : "
        f"{len(data)}"
    )

    data = prepare_pitch_data(
        data,
        tonic_pitch_class
    )

    (
        count_distribution,
        duration_distribution
    ) = calculate_swara_distribution(
        data
    )

    transitions = calculate_transitions(
        data
    )

    feature_rows = []

    for pitch in range(12):

        feature_rows.append({

            "relative_pitch_class":
                pitch,

            "note_event_percentage":
                count_distribution[pitch]
                * 100.0,

            "duration_percentage":
                duration_distribution[pitch]
                * 100.0

        })

    feature_data = pd.DataFrame(
        feature_rows
    )

    # --------------------------------------------------------
    # Add overall analysis information
    # --------------------------------------------------------

    feature_data[
        "estimated_tonic"
    ] = tonic_name

    feature_data[
        "tonic_pitch_class"
    ] = tonic_pitch_class

    feature_data[
        "tonic_confidence"
    ] = tonic_confidence

    feature_data[
        "transition_count"
    ] = len(transitions)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    feature_data.to_csv(
        FEATURE_OUTPUT,
        index=False
    )

    # --------------------------------------------------------
    # Print feature summary
    # --------------------------------------------------------

    print(
        "\nRelative swara / pitch-class evidence:"
    )

    for pitch in range(12):

        print(
            f"  Relative pitch {pitch:2d} : "
            f"{count_distribution[pitch] * 100:6.2f}% "
            f"events | "
            f"{duration_distribution[pitch] * 100:6.2f}% "
            f"duration"
        )

    print(
        f"\nMelodic transitions analyzed : "
        f"{len(transitions)}"
    )

    print(
        "\nFeature analysis saved:"
    )

    print(
        FEATURE_OUTPUT
    )

    print("\n" + "=" * 70)
    print(
        "PHASE 9.2.1 COMPLETED SUCCESSFULLY"
    )
    print("=" * 70)

    return {
        "tonic_name": tonic_name,
        "tonic_pitch_class": tonic_pitch_class,
        "tonic_frequency": tonic_frequency,
        "tonic_confidence": tonic_confidence,
        "data": data,
        "count_distribution": count_distribution,
        "duration_distribution": duration_distribution,
        "transitions": transitions,
        "feature_data": feature_data
    }


# ============================================================
# RAGA CANDIDATE SCORING
# ============================================================

def score_raga_candidate(
    distribution,
    transitions,
    profile
):

    allowed_score = (
        allowed_note_score(
            distribution,
            profile["allowed"]
        )
    )

    preferred_score = (
        preferred_note_score(
            distribution,
            profile["preferred"]
        )
    )

    forbidden_penalty = (
        forbidden_note_penalty(
            distribution,
            profile["forbidden"]
        )
    )

    completeness = (
        scale_completeness(
            distribution,
            profile["allowed"]
        )
    )

    characteristic_score = (
        characteristic_movement_score(
            transitions,
            profile["characteristic"]
        )
    )

    movement_score = (
        movement_direction_score(
            transitions,
            profile["ascending"],
            profile["descending"]
        )
    )

    # --------------------------------------------------------
    # Combined evidence score
    # --------------------------------------------------------

    score = (

        0.30 * allowed_score

        + 0.20 * preferred_score

        + 0.15 * completeness

        + 0.15 * characteristic_score

        + 0.10 * movement_score

        + 0.10 * (
            1.0
            - min(
                1.0,
                forbidden_penalty
            )
        )

    )

    score = max(
        0.0,
        min(
            1.0,
            score
        )
    )

    return {

        "allowed_note_score":
            allowed_score,

        "preferred_note_score":
            preferred_score,

        "forbidden_note_penalty":
            forbidden_penalty,

        "scale_completeness":
            completeness,

        "characteristic_movement_score":
            characteristic_score,

        "movement_score":
            movement_score,

        "evidence_score":
            score

    }


# ============================================================
# RECOGNIZE RAGA CANDIDATES
# ============================================================

def recognize_raga_candidates(
    feature_result
):

    distribution = (
        feature_result[
            "duration_distribution"
        ]
    )

    transitions = (
        feature_result[
            "transitions"
        ]
    )

    results = []

    for raga_name, profile in RAGA_PROFILES.items():

        scores = score_raga_candidate(
            distribution,
            transitions,
            profile
        )

        scores["raga"] = raga_name

        results.append(
            scores
        )

    results = sorted(
        results,
        key=lambda row:
            row["evidence_score"],
        reverse=True
    )

    return results


# ============================================================
# VALIDATE FINAL CANDIDATE
# ============================================================

def validate_raga_candidates(
    results
):

    if len(results) == 0:

        return (
            "Unknown",
            "Insufficient Evidence",
            0.0,
            0.0
        )

    top = results[0]

    if len(results) > 1:

        second = results[1]

        margin = (
            top["evidence_score"]
            -
            second["evidence_score"]
        )

    else:

        margin = top[
            "evidence_score"
        ]

    top_score = top[
        "evidence_score"
    ]

    # --------------------------------------------------------
    # Conservative validation
    # --------------------------------------------------------

    if (
        top_score >= 0.75
        and
        margin >= 0.10
    ):

        status = "Strong Candidate"

    elif (
        top_score >= 0.60
        and
        margin >= 0.06
    ):

        status = "Probable"

    elif top_score >= 0.45:

        status = "Uncertain"

    else:

        status = "Insufficient Evidence"

    if status == "Insufficient Evidence":

        estimated_raga = "Unknown"

    elif status == "Uncertain":

        estimated_raga = "Uncertain"

    else:

        estimated_raga = top[
            "raga"
        ]

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    confidence = (
        0.70 * top_score
        +
        0.30 * max(
            0.0,
            min(
                1.0,
                margin * 3.0
            )
        )
    )

    confidence = max(
        0.0,
        min(
            1.0,
            confidence
        )
    )

    return (
        estimated_raga,
        status,
        confidence,
        margin
    )


# ============================================================
# SAVE VALIDATION RESULTS
# ============================================================

def save_validation_results(
    results,
    feature_result
):

    rows = []

    for rank, result in enumerate(
        results,
        start=1
    ):

        rows.append({

            "rank":
                rank,

            "raga":
                result["raga"],

            "evidence_score":
                result["evidence_score"],

            "allowed_note_score":
                result["allowed_note_score"],

            "preferred_note_score":
                result["preferred_note_score"],

            "forbidden_note_penalty":
                result["forbidden_note_penalty"],

            "scale_completeness":
                result["scale_completeness"],

            "characteristic_movement_score":
                result[
                    "characteristic_movement_score"
                ],

            "movement_score":
                result["movement_score"]

        })

    validation_data = pd.DataFrame(
        rows
    )

    validation_data.to_csv(
        VALIDATION_OUTPUT,
        index=False
    )

    (
        estimated_raga,
        status,
        confidence,
        margin
    ) = validate_raga_candidates(
        results
    )

    top_candidate = (
        results[0]["raga"]
        if results
        else "Unknown"
    )

    top_score = (
        results[0]["evidence_score"]
        if results
        else 0.0
    )

    summary = pd.DataFrame([{

        "estimated_tonic":
            feature_result[
                "tonic_name"
            ],

        "tonic_pitch_class":
            feature_result[
                "tonic_pitch_class"
            ],

        "tonic_frequency_hz":
            feature_result[
                "tonic_frequency"
            ],

        "tonic_confidence":
            feature_result[
                "tonic_confidence"
            ],

        "top_raga_candidate":
            top_candidate,

        "top_evidence_score":
            top_score,

        "candidate_margin":
            margin,

        "raga_confidence":
            confidence,

        "estimated_raga":
            estimated_raga,

        "validation_status":
            status,

        "reference_song_dependency":
            "None"

    }])

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False
    )

    return (
        validation_data,
        summary
    )


# ============================================================
# CREATE GRAPH
# ============================================================

def create_validation_graph(
    results
):

    os.makedirs(
        GRAPH_DIR,
        exist_ok=True
    )

    names = [

        result["raga"]

        for result in results
    ]

    scores = [

        result["evidence_score"]

        for result in results
    ]

    plt.figure(
        figsize=(12, 6)
    )

    plt.bar(
        names,
        scores
    )

    plt.xlabel(
        "Raga Candidate"
    )

    plt.ylabel(
        "Validation Evidence Score"
    )

    plt.title(
        "Phase 9.2 - General Raga Recognition"
    )

    plt.ylim(
        0,
        1
    )

    plt.xticks(
        rotation=25
    )

    plt.grid(
        axis="y",
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        GRAPH_OUTPUT,
        dpi=300
    )

    plt.close()

    return GRAPH_OUTPUT


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(
    results,
    summary
):

    row = summary.iloc[0]

    print("\n")
    print("=" * 70)
    print(
        "PHASE 9.2.2 - RAGA CANDIDATE "
        "RECOGNITION & VALIDATION"
    )
    print("=" * 70)

    print(
        f"\nEstimated tonic : "
        f"{row['estimated_tonic']}"
    )

    print(
        f"Tonic confidence : "
        f"{row['tonic_confidence'] * 100:.2f}%"
    )

    print(
        "\nRaga candidates:"
    )

    print(
        "-" * 70
    )

    for index, result in enumerate(
        results,
        start=1
    ):

        print(

            f"{index}. "
            f"{result['raga']:<12} "
            f"Evidence: "
            f"{result['evidence_score'] * 100:6.2f}%"

        )

    print(
        "\n" + "-" * 70
    )

    print(
        f"Top candidate     : "
        f"{row['top_raga_candidate']}"
    )

    print(
        f"Estimated raga    : "
        f"{row['estimated_raga']}"
    )

    print(
        f"Confidence        : "
        f"{row['raga_confidence'] * 100:.2f}%"
    )

    print(
        f"Candidate margin  : "
        f"{row['candidate_margin'] * 100:.2f}%"
    )

    print(
        f"Validation status  : "
        f"{row['validation_status']}"
    )

    print(
        "\nImportant:"
    )

    print(
        "This result is generated from "
        "the musical evidence extracted from "
        "the current input song."
    )

    print(
        "No specific song or raga is hard-coded."
    )

    print(
        "\nFiles saved:"
    )

    print(
        f"  {FEATURE_OUTPUT}"
    )

    print(
        f"  {VALIDATION_OUTPUT}"
    )

    print(
        f"  {SUMMARY_OUTPUT}"
    )

    print(
        f"  {GRAPH_OUTPUT}"
    )

    print("\n" + "=" * 70)


# ============================================================
# MAIN PHASE 9.2 FUNCTION
# ============================================================

def recognize_and_validate_raga():

    print("\n")
    print("=" * 80)
    print(
        "PHASE 9.2 - GENERAL RAGA "
        "RECOGNITION & VALIDATION"
    )
    print("=" * 80)

    # --------------------------------------------------------
    # Phase 9.2.1
    # --------------------------------------------------------

    feature_result = (
        analyze_raga_features()
    )

    # --------------------------------------------------------
    # Phase 9.2.2
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print(
        "PHASE 9.2.2 - RAGA CANDIDATE "
        "RECOGNITION & VALIDATION"
    )
    print("=" * 70)

    print(
        "\nComparing melodic evidence "
        "against raga profiles..."
    )

    results = recognize_raga_candidates(
        feature_result
    )

    (
        validation_data,
        summary
    ) = save_validation_results(
        results,
        feature_result
    )

    graph_path = create_validation_graph(
        results
    )

    display_results(
        results,
        summary
    )

    print("\n")
    print("=" * 80)
    print(
        "PHASE 9.2 COMPLETED SUCCESSFULLY"
    )
    print("=" * 80)

    return {
        "features": feature_result,
        "candidates": validation_data,
        "summary": summary,
        "graph": graph_path
    }


# ============================================================
# STANDALONE EXECUTION
# ============================================================

if __name__ == "__main__":

    try:

        recognize_and_validate_raga()

    except KeyboardInterrupt:

        print(
            "\nPhase 9.2 interrupted by user."
        )

    except Exception as error:

        print("\n")
        print("=" * 70)
        print(
            "PHASE 9.2 STOPPED BECAUSE OF AN ERROR"
        )
        print("=" * 70)

        print(
            f"\nError type    : "
            f"{type(error).__name__}"
        )

        print(
            f"Error message : "
            f"{error}"
        )

        print("\n")
        print(
            "Check that Phases 7.4, 7.5 and 9.1 "
            "have already completed successfully."
        )

        print("=" * 70)

        raise