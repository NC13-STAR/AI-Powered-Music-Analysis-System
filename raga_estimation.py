import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PHASE 9.1 - RAGA / MELODIC-MODE ESTIMATION
# ============================================================


# ------------------------------------------------------------
# Raga profiles
# ------------------------------------------------------------

RAGA_PROFILES = {
    "Desh": {
        "allowed": {
            "Sa", "Re", "Ga", "Ma", "Pa",
            "Dha", "Ni", "Komal Ni"
        },
        "preferred": {
            "Re", "Ma", "Pa", "Ni", "Komal Ni"
        },
        "characteristic": {
            "Ni", "Komal Ni", "Re", "Ma"
        },
        "forbidden": set()
    },

    "Yaman": {
        "allowed": {
            "Sa", "Re", "Ga", "Tivra Ma",
            "Pa", "Dha", "Ni"
        },
        "preferred": {
            "Re", "Ga", "Tivra Ma",
            "Dha", "Ni"
        },
        "characteristic": {
            "Tivra Ma", "Ga", "Ni"
        },
        "forbidden": {
            "Ma"
        }
    },

    "Khamaj": {
        "allowed": {
            "Sa", "Re", "Ga", "Ma", "Pa",
            "Dha", "Ni", "Komal Ni"
        },
        "preferred": {
            "Ga", "Ma", "Pa", "Dha",
            "Komal Ni"
        },
        "characteristic": {
            "Ga", "Dha", "Komal Ni"
        },
        "forbidden": set()
    },

    "Kafi": {
        "allowed": {
            "Sa", "Re", "Komal Ga", "Ga",
            "Ma", "Pa", "Dha",
            "Komal Ni", "Ni"
        },
        "preferred": {
            "Re", "Komal Ga", "Ma",
            "Pa", "Komal Ni"
        },
        "characteristic": {
            "Komal Ga", "Komal Ni"
        },
        "forbidden": set()
    },

    "Bhupali": {
        "allowed": {
            "Sa", "Re", "Ga", "Pa", "Dha"
        },
        "preferred": {
            "Re", "Ga", "Pa", "Dha"
        },
        "characteristic": {
            "Ga", "Pa", "Dha"
        },
        "forbidden": {
            "Ma",
            "Tivra Ma",
            "Ni",
            "Komal Ni",
            "Komal Ga",
            "Komal Re",
            "Komal Dha"
        }
    },

    "Bilawal": {
        "allowed": {
            "Sa", "Re", "Ga", "Ma",
            "Pa", "Dha", "Ni"
        },
        "preferred": {
            "Re", "Ga", "Ma",
            "Pa", "Dha", "Ni"
        },
        "characteristic": {
            "Ga", "Ma", "Dha", "Ni"
        },
        "forbidden": {
            "Komal Re",
            "Komal Ga",
            "Tivra Ma",
            "Komal Dha",
            "Komal Ni"
        }
    },

    "Bhairav": {
        "allowed": {
            "Sa", "Komal Re", "Ga", "Ma",
            "Pa", "Komal Dha", "Ni"
        },
        "preferred": {
            "Komal Re", "Ga", "Ma",
            "Pa", "Komal Dha", "Ni"
        },
        "characteristic": {
            "Komal Re", "Komal Dha"
        },
        "forbidden": {
            "Re", "Dha"
        }
    },

    "Todi": {
        "allowed": {
            "Sa",
            "Komal Re",
            "Komal Ga",
            "Tivra Ma",
            "Pa",
            "Komal Dha",
            "Ni"
        },
        "preferred": {
            "Komal Re",
            "Komal Ga",
            "Tivra Ma",
            "Komal Dha",
            "Ni"
        },
        "characteristic": {
            "Komal Re",
            "Komal Ga",
            "Tivra Ma",
            "Komal Dha"
        },
        "forbidden": {
            "Re",
            "Ga",
            "Ma",
            "Dha"
        }
    }
}


# ------------------------------------------------------------
# Load tonic information
# ------------------------------------------------------------

def load_tonic_information():
    """
    Load tonic information from Phase 7.4.

    Actual file structure:

        estimated_tonic
        tonic_pitch_class
        tonic_midi
        tonic_frequency_hz
        confidence
        second_candidate
        second_candidate_score

    Confidence is stored as a fraction.
    Example:
        0.4559 = 45.59%
    """

    tonic_file = "data/output/tonic_estimation.csv"

    if not os.path.exists(tonic_file):
        raise FileNotFoundError(
            f"Required file not found: {tonic_file}"
        )

    tonic_data = pd.read_csv(
        tonic_file
    )

    if tonic_data.empty:
        raise ValueError(
            "Tonic estimation file is empty."
        )

    print(
        f"Tonic information loaded successfully: "
        f"{tonic_file}"
    )

    return tonic_data


# ------------------------------------------------------------
# Extract tonic information
# ------------------------------------------------------------

def get_tonic_information(tonic_data):
    """
    Extract tonic information from the actual
    Phase 7.4 CSV format.
    """

    required_columns = [
        "estimated_tonic",
        "tonic_frequency_hz",
        "confidence"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in tonic_data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required tonic columns: "
            + ", ".join(missing_columns)
        )

    if tonic_data.empty:
        raise ValueError(
            "Tonic estimation data is empty."
        )

    row = tonic_data.iloc[0]

    # --------------------------------------------------------
    # Tonic name
    # --------------------------------------------------------

    tonic_name = str(
        row["estimated_tonic"]
    )

    # --------------------------------------------------------
    # Tonic frequency
    # --------------------------------------------------------

    tonic_frequency = float(
        row["tonic_frequency_hz"]
    )

    # --------------------------------------------------------
    # Tonic confidence
    # --------------------------------------------------------

    raw_confidence = float(
        row["confidence"]
    )

    # The CSV stores confidence as 0-1.
    tonic_confidence = (
        raw_confidence * 100
    )

    # --------------------------------------------------------
    # Second candidate
    # --------------------------------------------------------

    second_candidate = None

    if "second_candidate" in tonic_data.columns:
        second_candidate = str(
            row["second_candidate"]
        )

    # --------------------------------------------------------
    # Second candidate score
    # --------------------------------------------------------

    second_candidate_score = None

    if "second_candidate_score" in tonic_data.columns:

        raw_second_score = float(
            row["second_candidate_score"]
        )

        second_candidate_score = (
            raw_second_score * 100
        )

    print("\nTonic information extracted:")

    print(
        f"Estimated tonic       : "
        f"{tonic_name}"
    )

    print(
        f"Tonic frequency       : "
        f"{tonic_frequency:.2f} Hz"
    )

    print(
        f"Tonic confidence      : "
        f"{tonic_confidence:.2f}%"
    )

    if second_candidate is not None:

        print(
            f"Second candidate      : "
            f"{second_candidate}"
        )

    if second_candidate_score is not None:

        print(
            f"Second candidate score: "
            f"{second_candidate_score:.2f}%"
        )

    return (
        tonic_name,
        tonic_frequency,
        tonic_confidence,
        second_candidate,
        second_candidate_score
    )


# ------------------------------------------------------------
# Load Indian swara information
# ------------------------------------------------------------

def load_note_information():
    """
    Find the Phase 7 output containing Indian swaras.

    The actual current project contains:

        data/output/indian_swara_data.csv

    with the column:

        indian_swara
    """

    possible_files = [
        "data/output/indian_swara_data.csv",
        "data/output/western_indian_notation_alignment.csv",
        "data/output/musical_notation_table.csv",
        "data/output/melody_phrase_note_data.csv"
    ]

    possible_swara_columns = [
        "indian_swara",
        "swara",
        "Swara",
        "Indian_Swara",
        "Indian Swara",
        "Indian_Swara_Name",
        "Indian Swara Name"
    ]

    checked_files = []

    for file_path in possible_files:

        if not os.path.exists(file_path):
            continue

        checked_files.append(
            file_path
        )

        try:
            note_data = pd.read_csv(
                file_path
            )
        except Exception:
            continue

        for column in possible_swara_columns:

            if column in note_data.columns:

                if note_data.empty:
                    continue

                print(
                    f"Note information loaded successfully: "
                    f"{file_path}"
                )

                print(
                    f"Indian swara column found: "
                    f"{column}"
                )

                return (
                    note_data,
                    file_path,
                    column
                )

    if checked_files:

        raise ValueError(
            "Phase 9.1 could not find an Indian "
            "swara column.\n\n"
            "Files checked:\n"
            + "\n".join(
                checked_files
            )
        )

    raise FileNotFoundError(
        "No suitable Phase 7 output file containing "
        "Indian swara information was found."
    )


# ------------------------------------------------------------
# Calculate swara distribution
# ------------------------------------------------------------

def calculate_swara_distribution(
    note_data,
    swara_column
):
    """
    Calculate swara occurrence counts
    and percentages.
    """

    swara_series = (
        note_data[swara_column]
        .dropna()
        .astype(str)
        .str.strip()
    )

    swara_series = swara_series[
        swara_series != ""
    ]

    if len(swara_series) == 0:

        return pd.DataFrame(
            columns=[
                "swara",
                "count",
                "percentage"
            ]
        )

    counts = (
        swara_series
        .value_counts()
    )

    total_notes = len(
        swara_series
    )

    rows = []

    for swara, count in counts.items():

        percentage = (
            count /
            total_notes *
            100
        )

        rows.append(
            {
                "swara": swara,
                "count": int(count),
                "percentage": float(
                    percentage
                )
            }
        )

    distribution = pd.DataFrame(
        rows
    )

    distribution = (
        distribution
        .sort_values(
            by="count",
            ascending=False
        )
        .reset_index(drop=True)
    )

    return distribution


# ------------------------------------------------------------
# Calculate raga score
# ------------------------------------------------------------

def calculate_raga_score(
    distribution,
    raga_name,
    profile
):
    """
    Calculate an algorithmic compatibility score.

    Components:

        30% allowed-note compatibility
        20% preferred-note presence
        15% characteristic-note presence
        15% characteristic-note frequency
        10% preferred-note frequency
        10% forbidden-note penalty

    This is a heuristic score and not a classical
    raga-recognition model.
    """

    count_map = dict(
        zip(
            distribution["swara"],
            distribution["count"]
        )
    )

    total_notes = int(
        distribution["count"].sum()
    )

    if total_notes == 0:

        return {
            "raga": raga_name,
            "allowed_score": 0.0,
            "preferred_presence": 0.0,
            "characteristic_score": 0.0,
            "movement_score": 0.0,
            "preferred_frequency": 0.0,
            "forbidden_penalty": 0.0,
            "final_score": 0.0
        }

    observed_swaras = set(
        distribution["swara"]
    )

    # --------------------------------------------------------
    # Allowed-note compatibility
    # --------------------------------------------------------

    allowed_count = sum(
        count
        for swara, count in count_map.items()
        if swara in profile["allowed"]
    )

    allowed_score = (
        allowed_count /
        total_notes
    )

    # --------------------------------------------------------
    # Preferred-note presence
    # --------------------------------------------------------

    preferred_observed = (
        observed_swaras &
        profile["preferred"]
    )

    if profile["preferred"]:

        preferred_presence = (
            len(preferred_observed) /
            len(profile["preferred"])
        )

    else:

        preferred_presence = 0.0

    # --------------------------------------------------------
    # Characteristic-note presence
    # --------------------------------------------------------

    characteristic_observed = (
        observed_swaras &
        profile["characteristic"]
    )

    if profile["characteristic"]:

        characteristic_score = (
            len(characteristic_observed) /
            len(profile["characteristic"])
        )

    else:

        characteristic_score = 0.0

    # --------------------------------------------------------
    # Characteristic-note frequency
    # --------------------------------------------------------

    characteristic_count = sum(
        count_map.get(
            swara,
            0
        )
        for swara in profile["characteristic"]
    )

    movement_score = (
        characteristic_count /
        total_notes
    )

    # --------------------------------------------------------
    # Preferred-note frequency
    # --------------------------------------------------------

    preferred_count = sum(
        count_map.get(
            swara,
            0
        )
        for swara in profile["preferred"]
    )

    preferred_frequency = (
        preferred_count /
        total_notes
    )

    # --------------------------------------------------------
    # Forbidden-note penalty
    # --------------------------------------------------------

    forbidden_count = sum(
        count_map.get(
            swara,
            0
        )
        for swara in profile["forbidden"]
    )

    forbidden_penalty = (
        forbidden_count /
        total_notes
    )

    # --------------------------------------------------------
    # Final score
    # --------------------------------------------------------

    final_score = (
        0.30 * allowed_score
        + 0.20 * preferred_presence
        + 0.15 * characteristic_score
        + 0.15 * movement_score
        + 0.10 * preferred_frequency
        - 0.10 * forbidden_penalty
    )

    final_score = max(
        0.0,
        min(
            1.0,
            final_score
        )
    )

    return {
        "raga": raga_name,
        "allowed_score": allowed_score,
        "preferred_presence": preferred_presence,
        "characteristic_score": characteristic_score,
        "movement_score": movement_score,
        "preferred_frequency": preferred_frequency,
        "forbidden_penalty": forbidden_penalty,
        "final_score": final_score
    }


# ------------------------------------------------------------
# Estimate raga
# ------------------------------------------------------------

def estimate_raga():
    """
    Phase 9.1 main function.

    Estimates the probable raga using the detected
    Indian swara distribution.

    The result is explicitly treated as an algorithmic
    estimate rather than definitive classical identification.
    """

    print("\n" + "=" * 70)
    print(
        "PHASE 9.1 - RAGA / MELODIC-MODE ESTIMATION"
    )
    print("=" * 70)

    print(
        "\nStarting algorithmic raga estimation..."
    )

    # --------------------------------------------------------
    # Load tonic
    # --------------------------------------------------------

    tonic_data = (
        load_tonic_information()
    )

    (
        tonic_name,
        tonic_frequency,
        tonic_confidence,
        second_candidate,
        second_candidate_score
    ) = get_tonic_information(
        tonic_data
    )

    # --------------------------------------------------------
    # Load swara information
    # --------------------------------------------------------

    (
        note_data,
        note_file,
        swara_column
    ) = load_note_information()

    # --------------------------------------------------------
    # Musical context
    # --------------------------------------------------------

    print("\nMusical context:")

    print(
        f"Tonic              : "
        f"{tonic_name}"
    )

    print(
        f"Tonic frequency    : "
        f"{tonic_frequency:.2f} Hz"
    )

    print(
        f"Tonic confidence   : "
        f"{tonic_confidence:.2f}%"
    )

    print(
        f"Swara source       : "
        f"{note_file}"
    )

    print(
        f"Swara column       : "
        f"{swara_column}"
    )

    # --------------------------------------------------------
    # Swara distribution
    # --------------------------------------------------------

    distribution = (
        calculate_swara_distribution(
            note_data,
            swara_column
        )
    )

    if distribution.empty:

        raise ValueError(
            "No valid Indian swara information "
            "was found."
        )

    total_notes = int(
        distribution["count"].sum()
    )

    print(
        "\nDetected swara distribution:"
    )

    print(
        distribution.to_string(
            index=False
        )
    )

    print(
        f"\nTotal swara events analysed: "
        f"{total_notes}"
    )

    # --------------------------------------------------------
    # Calculate raga candidates
    # --------------------------------------------------------

    print(
        "\nComparing detected swaras "
        "with raga profiles..."
    )

    results = []

    for (
        raga_name,
        profile
    ) in RAGA_PROFILES.items():

        result = (
            calculate_raga_score(
                distribution,
                raga_name,
                profile
            )
        )

        results.append(
            result
        )

    results_df = pd.DataFrame(
        results
    )

    results_df = (
        results_df
        .sort_values(
            by="final_score",
            ascending=False
        )
        .reset_index(drop=True)
    )

    results_df.insert(
        0,
        "rank",
        np.arange(
            1,
            len(results_df) + 1
        )
    )

    # --------------------------------------------------------
    # Top candidate
    # --------------------------------------------------------

    top_candidate = str(
        results_df.iloc[0]["raga"]
    )

    top_score = float(
        results_df.iloc[0]["final_score"]
    )

    if len(results_df) > 1:

        second_raga = str(
            results_df.iloc[1]["raga"]
        )

        second_score = float(
            results_df.iloc[1]["final_score"]
        )

    else:

        second_raga = ""

        second_score = 0.0

    score_margin = (
        top_score -
        second_score
    )

    # --------------------------------------------------------
    # Conservative confidence decision
    # --------------------------------------------------------

    minimum_score = 0.60
    minimum_margin = 0.08

    if (
        top_score >= minimum_score
        and
        score_margin >= minimum_margin
    ):

        estimated_raga = (
            top_candidate
        )

        confidence_level = (
            "Moderate / High"
        )

        decision = (
            "Probable raga identified"
        )

    else:

        estimated_raga = (
            "Uncertain"
        )

        confidence_level = (
            "Low"
        )

        decision = (
            "Raga identification remains uncertain"
        )

    # --------------------------------------------------------
    # Candidate ranking
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print(
        "RAGA CANDIDATE RANKING"
    )
    print("-" * 70)

    ranking_display = (
        results_df[
            [
                "rank",
                "raga",
                "final_score"
            ]
        ]
        .copy()
    )

    ranking_display[
        "final_score"
    ] = (
        ranking_display[
            "final_score"
        ] *
        100
    ).round(2)

    ranking_display = (
        ranking_display
        .rename(
            columns={
                "final_score":
                    "score_percent"
            }
        )
    )

    print(
        ranking_display.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print(
        "RAGA ESTIMATION RESULT"
    )
    print("-" * 70)

    print(
        f"Top candidate      : "
        f"{top_candidate}"
    )

    print(
        f"Top score          : "
        f"{top_score * 100:.2f}%"
    )

    print(
        f"Second candidate   : "
        f"{second_raga}"
    )

    print(
        f"Second score       : "
        f"{second_score * 100:.2f}%"
    )

    print(
        f"Score margin       : "
        f"{score_margin * 100:.2f}%"
    )

    print(
        f"Estimated raga     : "
        f"{estimated_raga}"
    )

    print(
        f"Confidence level   : "
        f"{confidence_level}"
    )

    print(
        f"Decision           : "
        f"{decision}"
    )

    print(
        "\nNote: This is an algorithmic "
        "raga estimate based on detected "
        "swara information."
    )

    print(
        "It is not a definitive classical "
        "raga identification."
    )

    # --------------------------------------------------------
    # Output directories
    # --------------------------------------------------------

    os.makedirs(
        "data/output",
        exist_ok=True
    )

    os.makedirs(
        "graphs",
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save candidate analysis
    # --------------------------------------------------------

    candidate_file = (
        "data/output/"
        "raga_candidate_analysis.csv"
    )

    results_df.to_csv(
        candidate_file,
        index=False
    )

    print(
        "\nRaga candidate analysis saved:"
    )

    print(
        candidate_file
    )

    # --------------------------------------------------------
    # Save swara distribution
    # --------------------------------------------------------

    distribution_file = (
        "data/output/"
        "raga_swara_distribution.csv"
    )

    distribution.to_csv(
        distribution_file,
        index=False
    )

    print(
        "\nRaga swara distribution saved:"
    )

    print(
        distribution_file
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary = pd.DataFrame(
        [
            {
                "tonic": tonic_name,

                "tonic_frequency_hz":
                    tonic_frequency,

                "tonic_confidence_percent":
                    tonic_confidence,

                "tonic_second_candidate":
                    second_candidate,

                "tonic_second_candidate_score_percent":
                    second_candidate_score,

                "top_candidate":
                    top_candidate,

                "top_score_percent":
                    top_score * 100,

                "second_raga_candidate":
                    second_raga,

                "second_raga_score_percent":
                    second_score * 100,

                "raga_score_margin_percent":
                    score_margin * 100,

                "estimated_raga":
                    estimated_raga,

                "confidence_level":
                    confidence_level,

                "decision":
                    decision,

                "total_swara_events":
                    total_notes
            }
        ]
    )

    summary_file = (
        "data/output/"
        "raga_estimation_summary.csv"
    )

    summary.to_csv(
        summary_file,
        index=False
    )

    print(
        "\nRaga estimation summary saved:"
    )

    print(
        summary_file
    )

    # --------------------------------------------------------
    # Raga candidate graph
    # --------------------------------------------------------

    plt.figure(
        figsize=(12, 7)
    )

    plot_data = (
        results_df.copy()
    )

    plt.bar(
        plot_data["raga"],
        plot_data["final_score"] * 100
    )

    plt.xlabel(
        "Raga Candidate"
    )

    plt.ylabel(
        "Compatibility Score (%)"
    )

    plt.title(
        "Phase 9.1 - Raga Candidate Analysis"
    )

    plt.xticks(
        rotation=35,
        ha="right"
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    graph_path = (
        "graphs/"
        "raga_candidate_analysis.png"
    )

    plt.savefig(
        graph_path,
        dpi=300
    )

    plt.close()

    print(
        "\nRaga candidate graph saved:"
    )

    print(
        graph_path
    )

    # --------------------------------------------------------
    # Completion
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(
        "PHASE 9.1 COMPLETED SUCCESSFULLY"
    )
    print("=" * 70)

    return {
        "estimated_raga":
            estimated_raga,

        "top_candidate":
            top_candidate,

        "top_score":
            top_score,

        "second_candidate":
            second_raga,

        "second_score":
            second_score,

        "score_margin":
            score_margin,

        "confidence_level":
            confidence_level,

        "tonic":
            tonic_name,

        "tonic_frequency":
            tonic_frequency,

        "tonic_confidence":
            tonic_confidence,

        "candidate_analysis":
            results_df,

        "swara_distribution":
            distribution
    }