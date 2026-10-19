import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PHASE 8.5 - FINAL TAAL ESTIMATION
# ============================================================
#
# This phase estimates possible Indian rhythmic cycles using:
#   1. Actual beat timing
#   2. Beat strength
#   3. Cycle periodicity
#   4. Repetition of complete cycles
#   5. Position-wise beat strength
#   6. Vibhag structure
#   7. Sam / cycle-start evidence
#
# IMPORTANT:
# beat_number is NEVER used as time.
# beat_time_seconds is the authoritative timing column.
#
# The system is conservative:
# weak evidence -> Uncertain
#
# ============================================================


# ------------------------------------------------------------
# TAAL CANDIDATES
# ------------------------------------------------------------

TAAL_CANDIDATES = {
    "Dadra": {
        "matras": 6,
        "vibhag": [3, 3]
    },

    "Rupak": {
        "matras": 7,
        "vibhag": [3, 2, 2]
    },

    "Keherwa": {
        "matras": 8,
        "vibhag": [4, 4]
    },

    "Jhaptaal": {
        "matras": 10,
        "vibhag": [2, 3, 2, 3]
    },

    "Ektaal": {
        "matras": 12,
        "vibhag": [2, 2, 2, 2, 2, 2]
    },

    "Teentaal": {
        "matras": 16,
        "vibhag": [4, 4, 4, 4]
    }
}


# ------------------------------------------------------------
# UTILITY
# ------------------------------------------------------------

def normalize_array(values):
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return values

    minimum = np.nanmin(values)
    maximum = np.nanmax(values)

    if maximum - minimum < 1e-12:
        return np.ones_like(values) * 0.5

    return (values - minimum) / (maximum - minimum)


# ------------------------------------------------------------
# LOAD ACTUAL BEAT TIMING
# ------------------------------------------------------------

def load_beat_timing():
    path = os.path.join(
        "data",
        "output",
        "rhythm_analysis.csv"
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    df = pd.read_csv(path)

    print("\nInspecting rhythm-analysis columns...")

    print("\nAvailable columns:")

    for column in df.columns:
        print(f"  - {column}")

    # --------------------------------------------------------
    # IMPORTANT:
    # Directly use the real beat-time column.
    # --------------------------------------------------------

    if "beat_time_seconds" not in df.columns:
        raise ValueError(
            "\nThe required column 'beat_time_seconds' "
            "was not found in rhythm_analysis.csv.\n\n"
            "Available columns are:\n"
            + "\n".join(
                f"  - {column}"
                for column in df.columns
            )
        )

    beat_times = pd.to_numeric(
        df["beat_time_seconds"],
        errors="coerce"
    ).to_numpy()

    valid = np.isfinite(beat_times)

    beat_times = beat_times[valid]

    if len(beat_times) < 20:
        raise ValueError(
            "Not enough valid beat-time events for taal analysis."
        )

    # Remove duplicate timestamps
    beat_times = np.unique(beat_times)

    # Make sure timestamps are sorted
    beat_times = np.sort(beat_times)

    intervals = np.diff(beat_times)

    valid_intervals = (
        np.isfinite(intervals)
        & (intervals > 0)
        & (intervals < 5.0)
    )

    intervals = intervals[valid_intervals]

    print("\nBeat timing column selected:")
    print("beat_time_seconds")

    print(f"Valid beat events: {len(beat_times)}")

    print(
        f"Mean beat interval: "
        f"{np.mean(intervals):.4f} sec"
    )

    print(
        f"Median beat interval: "
        f"{np.median(intervals):.4f} sec"
    )

    print(
        f"Minimum beat interval: "
        f"{np.min(intervals):.4f} sec"
    )

    print(
        f"Maximum beat interval: "
        f"{np.max(intervals):.4f} sec"
    )

    return beat_times, intervals


# ------------------------------------------------------------
# LOAD BEAT STRENGTH
# ------------------------------------------------------------

def load_beat_strength():
    path = os.path.join(
        "data",
        "output",
        "rhythm_pattern_analysis.csv"
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    df = pd.read_csv(path)

    possible_columns = [
        "normalized_beat_strength",
        "beat_strength",
        "strength"
    ]

    strength_column = None

    for column in possible_columns:
        if column in df.columns:
            strength_column = column
            break

    if strength_column is None:
        raise ValueError(
            "\nNo beat-strength column was found in "
            "rhythm_pattern_analysis.csv.\n\n"
            "Available columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in df.columns
            )
        )

    strength = pd.to_numeric(
        df[strength_column],
        errors="coerce"
    ).to_numpy()

    strength = strength[np.isfinite(strength)]

    strength = normalize_array(strength)

    print("\nBeat-strength column selected:")
    print(strength_column)

    print(
        f"Beat-strength events: {len(strength)}"
    )

    return strength


# ------------------------------------------------------------
# SYNCHRONIZE BEAT TIMES AND STRENGTH
# ------------------------------------------------------------

def synchronize_data(beat_times, strength):
    n = min(
        len(beat_times),
        len(strength)
    )

    beat_times = beat_times[:n]
    strength = strength[:n]

    print("\nSynchronized rhythm events:")
    print(n)

    if n < 20:
        raise ValueError(
            "Insufficient synchronized beat events."
        )

    return beat_times, strength


# ------------------------------------------------------------
# ESTIMATE TEMPO
# ------------------------------------------------------------

def estimate_tempo(intervals):
    median_interval = np.median(intervals)

    if median_interval <= 0:
        return np.nan

    return 60.0 / median_interval


# ------------------------------------------------------------
# BUILD CYCLE PROFILES
# ------------------------------------------------------------

def build_cycle_profiles(strength, matras):
    """
    Divide beat-strength sequence into complete cycles.

    Example:
        Teentaal -> 16 beats per cycle

    Incomplete final cycles are discarded.
    """

    complete_cycles = len(strength) // matras

    if complete_cycles < 2:
        return np.empty((0, matras))

    usable = strength[
        :complete_cycles * matras
    ]

    cycles = usable.reshape(
        complete_cycles,
        matras
    )

    return cycles


# ------------------------------------------------------------
# PROFILE CONSISTENCY
# ------------------------------------------------------------

def profile_consistency(cycles):
    if len(cycles) < 2:
        return 0.0

    normalized_cycles = np.array([
        normalize_array(cycle)
        for cycle in cycles
    ])

    mean_profile = np.mean(
        normalized_cycles,
        axis=0
    )

    similarities = []

    for cycle in normalized_cycles:

        a = cycle - np.mean(cycle)
        b = mean_profile - np.mean(mean_profile)

        denominator = (
            np.linalg.norm(a)
            * np.linalg.norm(b)
        )

        if denominator <= 1e-12:
            similarity = 0.0
        else:
            similarity = (
                np.dot(a, b)
                / denominator
            )

        similarities.append(
            (similarity + 1.0) / 2.0
        )

    return float(
        np.mean(similarities)
    )


# ------------------------------------------------------------
# CYCLE-TO-CYCLE STABILITY
# ------------------------------------------------------------

def cycle_stability(beat_times, matras):
    """
    Measures whether cycle duration remains stable.
    """

    intervals = np.diff(beat_times)

    if len(intervals) < matras * 2:
        return 0.0

    usable = (
        len(intervals) // matras
    ) * matras

    interval_cycles = intervals[
        :usable
    ].reshape(
        -1,
        matras
    )

    cycle_durations = np.sum(
        interval_cycles,
        axis=1
    )

    if len(cycle_durations) < 2:
        return 0.0

    mean_duration = np.mean(
        cycle_durations
    )

    std_duration = np.std(
        cycle_durations
    )

    if mean_duration <= 0:
        return 0.0

    cv = (
        std_duration
        / mean_duration
    )

    stability = 1.0 / (
        1.0 + 4.0 * cv
    )

    return float(
        np.clip(stability, 0.0, 1.0)
    )


# ------------------------------------------------------------
# POSITION STABILITY
# ------------------------------------------------------------

def position_stability(cycles):
    """
    Checks whether strong/weak positions remain
    reasonably consistent across cycles.
    """

    if len(cycles) < 2:
        return 0.0

    cycles = np.array([
        normalize_array(cycle)
        for cycle in cycles
    ])

    std_by_position = np.std(
        cycles,
        axis=0
    )

    mean_std = np.mean(
        std_by_position
    )

    stability = 1.0 / (
        1.0 + 3.0 * mean_std
    )

    return float(
        np.clip(stability, 0.0, 1.0)
    )


# ------------------------------------------------------------
# CYCLE PERIODICITY
# ------------------------------------------------------------

def cycle_periodicity(strength, matras):
    """
    Compare consecutive complete cycles.

    High similarity between repeated cycles indicates
    stronger rhythmic periodicity.
    """

    cycles = build_cycle_profiles(
        strength,
        matras
    )

    if len(cycles) < 3:
        return 0.0

    similarities = []

    for i in range(
        len(cycles) - 1
    ):

        a = normalize_array(
            cycles[i]
        )

        b = normalize_array(
            cycles[i + 1]
        )

        denominator = (
            np.linalg.norm(
                a - np.mean(a)
            )
            *
            np.linalg.norm(
                b - np.mean(b)
            )
        )

        if denominator <= 1e-12:
            similarity = 0.0
        else:
            similarity = (
                np.dot(
                    a - np.mean(a),
                    b - np.mean(b)
                )
                / denominator
            )

        similarity = (
            similarity + 1.0
        ) / 2.0

        similarities.append(
            similarity
        )

    if len(similarities) == 0:
        return 0.0

    return float(
        np.mean(similarities)
    )


# ------------------------------------------------------------
# REPEATED-CYCLE SUPPORT
# ------------------------------------------------------------

def repeated_cycle_support(
    strength,
    matras
):
    """
    Measures how consistently complete cycles resemble
    the average cycle.

    This is deliberately strict.
    """

    cycles = build_cycle_profiles(
        strength,
        matras
    )

    if len(cycles) < 3:
        return 0.0

    normalized = np.array([
        normalize_array(cycle)
        for cycle in cycles
    ])

    template = np.mean(
        normalized,
        axis=0
    )

    similarities = []

    for cycle in normalized:

        a = cycle - np.mean(cycle)
        b = template - np.mean(template)

        denominator = (
            np.linalg.norm(a)
            * np.linalg.norm(b)
        )

        if denominator <= 1e-12:
            continue

        correlation = (
            np.dot(a, b)
            / denominator
        )

        similarities.append(
            (correlation + 1.0) / 2.0
        )

    if len(similarities) == 0:
        return 0.0

    strong_cycles = [
        value
        for value in similarities
        if value >= 0.65
    ]

    support = (
        len(strong_cycles)
        / len(similarities)
    )

    return float(
        np.clip(support, 0.0, 1.0)
    )


# ------------------------------------------------------------
# SAM EVIDENCE
# ------------------------------------------------------------

def sam_score(cycles):
    """
    Sam = first matra of each cycle.

    We compare first-position strength against the
    remaining positions.
    """

    if len(cycles) < 2:
        return 0.0

    cycles = np.array([
        normalize_array(cycle)
        for cycle in cycles
    ])

    first_positions = cycles[:, 0]

    other_positions = cycles[:, 1:]

    mean_first = np.mean(
        first_positions
    )

    mean_other = np.mean(
        other_positions
    )

    difference = (
        mean_first
        - mean_other
    )

    # Convert difference into 0..1 evidence.
    score = (
        0.5
        + difference
    )

    return float(
        np.clip(score, 0.0, 1.0)
    )


# ------------------------------------------------------------
# VIBHAG STRUCTURE
# ------------------------------------------------------------

def vibhag_score(cycles, vibhag):
    """
    Evaluate whether boundaries between vibhags show
    consistent changes in beat strength.

    This is evidence only, not a probability.
    """

    if len(cycles) < 2:
        return 0.0

    cycles = np.array([
        normalize_array(cycle)
        for cycle in cycles
    ])

    boundaries = []

    position = 0

    for size in vibhag[:-1]:
        position += size
        boundaries.append(position)

    if not boundaries:
        return 0.0

    scores = []

    for boundary in boundaries:

        left = cycles[:, boundary - 1]
        right = cycles[:, boundary]

        difference = np.abs(
            left - right
        )

        scores.append(
            np.mean(difference)
        )

    if not scores:
        return 0.0

    raw = np.mean(scores)

    return float(
        np.clip(raw, 0.0, 1.0)
    )


# ------------------------------------------------------------
# EVIDENCE SCORE
# ------------------------------------------------------------

def calculate_evidence(
    periodicity,
    repeated_support,
    profile,
    stability,
    position,
    sam,
    vibhag
):

    score = (
        0.30 * periodicity
        +
        0.25 * repeated_support
        +
        0.18 * profile
        +
        0.10 * stability
        +
        0.07 * position
        +
        0.06 * sam
        +
        0.04 * vibhag
    )

    return float(
        np.clip(score, 0.0, 1.0)
    )


# ------------------------------------------------------------
# CONFIDENCE
# ------------------------------------------------------------

def calculate_confidence(
    top_score,
    second_score,
    periodicity,
    repeated_support,
    profile
):

    margin = (
        top_score
        - second_score
    )

    # Evidence strength
    evidence_strength = np.mean([
        periodicity,
        repeated_support,
        profile
    ])

    # Margin contribution
    margin_strength = np.clip(
        margin / 0.15,
        0.0,
        1.0
    )

    confidence = (
        0.55 * evidence_strength
        +
        0.25 * top_score
        +
        0.20 * margin_strength
    )

    confidence *= 100.0

    # Conservative caps
    if periodicity < 0.20:
        confidence = min(
            confidence,
            35.0
        )

    if repeated_support < 0.20:
        confidence = min(
            confidence,
            40.0
        )

    if profile < 0.30:
        confidence = min(
            confidence,
            45.0
        )

    return float(
        np.clip(confidence, 0.0, 100.0)
    )


# ------------------------------------------------------------
# CONFIDENCE CATEGORY
# ------------------------------------------------------------

def confidence_category(confidence):

    if confidence >= 80:
        return "High"

    if confidence >= 60:
        return "Moderate"

    if confidence >= 40:
        return "Low"

    return "Very Low"


# ------------------------------------------------------------
# TAAL DECISION
# ------------------------------------------------------------

def decide_taal(
    candidates
):

    if len(candidates) == 0:
        return (
            "Uncertain",
            0.0,
            0.0
        )

    candidates = sorted(
        candidates,
        key=lambda x: x["evidence_score"],
        reverse=True
    )

    top = candidates[0]

    if len(candidates) > 1:
        second = candidates[1]
    else:
        second = {
            "evidence_score": 0.0
        }

    top_score = top[
        "evidence_score"
    ]

    second_score = second[
        "evidence_score"
    ]

    margin = (
        top_score
        - second_score
    )

    confidence = calculate_confidence(
        top_score,
        second_score,
        top["periodicity"],
        top["repeated_support"],
        top["profile_consistency"]
    )

    # --------------------------------------------------------
    # STRICT ASSIGNMENT
    # --------------------------------------------------------

    reliable = (
        confidence >= 60.0
        and top_score >= 0.45
        and top["periodicity"] >= 0.25
        and top["repeated_support"] >= 0.20
        and top["profile_consistency"] >= 0.30
        and margin >= 0.06
    )

    if reliable:
        taal = top["taal"]
    else:
        taal = "Uncertain"

    return (
        taal,
        confidence,
        margin
    )


# ------------------------------------------------------------
# GRAPH
# ------------------------------------------------------------

def create_taal_graph(results):

    os.makedirs(
        "graphs",
        exist_ok=True
    )

    results = sorted(
        results,
        key=lambda x: x["evidence_score"],
        reverse=True
    )

    names = [
        result["taal"]
        for result in results
    ]

    scores = [
        result["evidence_score"]
        for result in results
    ]

    plt.figure(
        figsize=(11, 6)
    )

    plt.bar(
        names,
        scores
    )

    plt.xlabel(
        "Taal Candidate"
    )

    plt.ylabel(
        "Rhythmic Evidence Score"
    )

    plt.title(
        "Phase 8.5 - Taal Candidate Evidence"
    )

    plt.xticks(
        rotation=20
    )

    plt.grid(
        axis="y",
        alpha=0.25
    )

    plt.tight_layout()

    output_path = os.path.join(
        "graphs",
        "taal_candidate_analysis.png"
    )

    plt.savefig(
        output_path,
        dpi=150
    )

    plt.close()

    return output_path


# ------------------------------------------------------------
# MAIN FUNCTION
# ------------------------------------------------------------

def estimate_taal():

    print("\n" + "=" * 60)
    print(
        "PHASE 8.5 - FINAL TAAL ESTIMATION"
    )
    print(
        "Cycle-Level Rhythmic Structure Analysis"
    )
    print("=" * 60)

    # --------------------------------------------------------
    # STEP 1
    # --------------------------------------------------------

    print(
        "\nStep 1 - Loading actual beat timing..."
    )

    beat_times, intervals = load_beat_timing()

    tempo = estimate_tempo(
        intervals
    )

    print(
        f"\nMedian beat-derived tempo: "
        f"{tempo:.2f} BPM"
    )

    # --------------------------------------------------------
    # STEP 2
    # --------------------------------------------------------

    print(
        "\nStep 2 - Loading beat-strength information..."
    )

    strength = load_beat_strength()

    # --------------------------------------------------------
    # STEP 3
    # --------------------------------------------------------

    beat_times, strength = synchronize_data(
        beat_times,
        strength
    )

    # --------------------------------------------------------
    # STEP 4
    # --------------------------------------------------------

    print(
        "\nStep 3 - Testing complete rhythmic cycles..."
    )

    results = []

    for taal_name, specification in TAAL_CANDIDATES.items():

        matras = specification[
            "matras"
        ]

        vibhag = specification[
            "vibhag"
        ]

        cycles = build_cycle_profiles(
            strength,
            matras
        )

        periodicity = cycle_periodicity(
            strength,
            matras
        )

        repeated_support = repeated_cycle_support(
            strength,
            matras
        )

        profile = profile_consistency(
            cycles
        )

        stability = cycle_stability(
            beat_times,
            matras
        )

        position = position_stability(
            cycles
        )

        sam = sam_score(
            cycles
        )

        vibhag_evidence = vibhag_score(
            cycles,
            vibhag
        )

        evidence = calculate_evidence(
            periodicity,
            repeated_support,
            profile,
            stability,
            position,
            sam,
            vibhag_evidence
        )

        results.append({
            "taal": taal_name,
            "matras": matras,
            "vibhag": str(vibhag),
            "periodicity": periodicity,
            "repeated_support": repeated_support,
            "profile_consistency": profile,
            "cycle_stability": stability,
            "position_stability": position,
            "sam_score": sam,
            "vibhag_score": vibhag_evidence,
            "evidence_score": evidence
        })

    # --------------------------------------------------------
    # STEP 5
    # --------------------------------------------------------

    results = sorted(
        results,
        key=lambda x: x["evidence_score"],
        reverse=True
    )

    print(
        "\nStep 4 - Ranking taal candidates..."
    )

    for index, result in enumerate(
        results,
        start=1
    ):

        print(
            f"{index}. "
            f"{result['taal']} | "
            f"Matras {result['matras']} | "
            f"Score {result['evidence_score']:.3f} | "
            f"Periodicity {result['periodicity']:.3f} | "
            f"Repeated {result['repeated_support']:.3f} | "
            f"Profile {result['profile_consistency']:.3f} | "
            f"Stability {result['cycle_stability']:.3f} | "
            f"Sam {result['sam_score']:.3f} | "
            f"Vibhag {result['vibhag_score']:.3f}"
        )

    # --------------------------------------------------------
    # STEP 6
    # --------------------------------------------------------

    taal, confidence, margin = decide_taal(
        results
    )

    top = results[0]

    category = confidence_category(
        confidence
    )

    print("\n" + "=" * 60)
    print(
        "FINAL TAAL ESTIMATION RESULT"
    )
    print("=" * 60)

    print(
        f"\nEstimated Taal : {taal}"
    )

    print(
        f"Confidence     : {confidence:.2f}%"
    )

    print(
        f"Confidence category : {category}"
    )

    print(
        "\nTop rhythmic candidate:"
    )

    print(
        f"Taal              : {top['taal']}"
    )

    print(
        f"Matras            : {top['matras']}"
    )

    print(
        f"Vibhag pattern    : {top['vibhag']}"
    )

    print(
        f"Evidence score    : "
        f"{top['evidence_score']:.3f}"
    )

    print(
        f"Cycle periodicity : "
        f"{top['periodicity']:.3f}"
    )

    print(
        f"Repeated support  : "
        f"{top['repeated_support']:.3f}"
    )

    print(
        f"Profile consistency: "
        f"{top['profile_consistency']:.3f}"
    )

    print(
        f"Candidate margin  : "
        f"{margin:.3f}"
    )

    print("\nDecision:")

    if taal == "Uncertain":

        print(
            "Taal remains UNCERTAIN."
        )

        print(
            "The system did not find sufficiently "
            "repeated cycle-level evidence to make "
            "a reliable taal assignment."
        )

        print(
            "The result is intentionally conservative "
            "and does not force a taal from weak evidence."
        )

    else:

        print(
            f"Estimated taal: {taal}"
        )

        print(
            "The candidate passed the minimum "
            "cycle-level evidence requirements."
        )

    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    os.makedirs(
        "data/output",
        exist_ok=True
    )

    candidate_df = pd.DataFrame(
        results
    )

    candidate_path = os.path.join(
        "data",
        "output",
        "taal_candidate_analysis.csv"
    )

    candidate_df.to_csv(
        candidate_path,
        index=False
    )

    summary = pd.DataFrame([{
        "estimated_taal": taal,
        "confidence_percent": confidence,
        "confidence_category": category,
        "top_candidate": top["taal"],
        "top_candidate_matras": top["matras"],
        "top_candidate_vibhag": top["vibhag"],
        "top_evidence_score": top["evidence_score"],
        "top_periodicity": top["periodicity"],
        "top_repeated_support": top["repeated_support"],
        "top_profile_consistency": top["profile_consistency"],
        "candidate_margin": margin,
        "median_bpm": tempo,
        "decision": (
            "Reliable taal assignment"
            if taal != "Uncertain"
            else "Insufficient cycle-level evidence"
        )
    }])

    summary_path = os.path.join(
        "data",
        "output",
        "taal_estimation_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    # --------------------------------------------------------
    # GRAPH
    # --------------------------------------------------------

    graph_path = create_taal_graph(
        results
    )

    print(
        "\nTaal candidate analysis saved:"
    )

    print(
        candidate_path
    )

    print(
        "\nTaal estimation summary saved:"
    )

    print(
        summary_path
    )

    print(
        "\nTaal candidate graph saved:"
    )

    print(
        graph_path
    )

    print(
        "\nPHASE 8.5 FINAL REFINEMENT COMPLETED SUCCESSFULLY"
    )

    print("=" * 60)

    return {
        "estimated_taal": taal,
        "confidence": confidence,
        "confidence_category": category,
        "top_candidate": top["taal"],
        "top_matras": top["matras"],
        "top_vibhag": top["vibhag"],
        "evidence_score": top["evidence_score"],
        "periodicity": top["periodicity"],
        "repeated_support": top["repeated_support"],
        "profile_consistency": top["profile_consistency"],
        "candidate_margin": margin
    }