import os
import json
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt


# ============================================================
# PHASE 11.1 - MUSICAL ELEMENT ANALYSIS
# ============================================================

OUTPUT_DIR = "data/output"
GRAPH_DIR = "graphs"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)

NOTE_NAMES = [
    "C", "C#", "D", "D#", "E", "F",
    "F#", "G", "G#", "A", "A#", "B"
]


# ============================================================
# UTILITY
# ============================================================

def normalize(values):
    values = np.asarray(values, dtype=float)

    total = np.sum(values)

    if total <= 0:
        return np.zeros_like(values)

    return values / total


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    denominator = (
        np.linalg.norm(a) *
        np.linalg.norm(b)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(a, b) / denominator
    )


# ============================================================
# 11.1.1 SOUND CHARACTERISTICS
# ============================================================

def analyze_sound(audio, sample_rate):

    print("\n" + "=" * 70)
    print("PHASE 11.1.1 - SOUND CHARACTERISTICS")
    print("=" * 70)

    rms = librosa.feature.rms(
        y=audio,
        frame_length=2048,
        hop_length=512
    )[0]

    centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sample_rate,
        n_fft=2048,
        hop_length=512
    )[0]

    bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=sample_rate,
        n_fft=2048,
        hop_length=512
    )[0]

    zcr = librosa.feature.zero_crossing_rate(
        audio,
        frame_length=2048,
        hop_length=512
    )[0]

    result = {
        "mean_rms_energy": float(np.mean(rms)),
        "mean_spectral_centroid_hz": float(
            np.mean(centroid)
        ),
        "mean_spectral_bandwidth_hz": float(
            np.mean(bandwidth)
        ),
        "mean_zero_crossing_rate": float(
            np.mean(zcr)
        )
    }

    centroid_value = result[
        "mean_spectral_centroid_hz"
    ]

    if centroid_value < 1800:
        result["brightness"] = "Low"
    elif centroid_value < 3500:
        result["brightness"] = "Moderate"
    else:
        result["brightness"] = "High"

    zcr_value = result[
        "mean_zero_crossing_rate"
    ]

    if zcr_value < 0.08:
        result["noisiness"] = "Low"
    elif zcr_value < 0.15:
        result["noisiness"] = "Moderate"
    else:
        result["noisiness"] = "High"

    output = pd.DataFrame([result])

    output_file = os.path.join(
        OUTPUT_DIR,
        "instrument_analysis.csv"
    )

    output.to_csv(
        output_file,
        index=False
    )

    print(
        f"Mean RMS energy    : "
        f"{result['mean_rms_energy']:.5f}"
    )

    print(
        f"Spectral centroid  : "
        f"{result['mean_spectral_centroid_hz']:.2f} Hz"
    )

    print(
        f"Spectral bandwidth : "
        f"{result['mean_spectral_bandwidth_hz']:.2f} Hz"
    )

    print(
        f"Zero crossing rate : "
        f"{result['mean_zero_crossing_rate']:.5f}"
    )

    print(
        f"Brightness         : "
        f"{result['brightness']}"
    )

    print(
        f"Noisiness          : "
        f"{result['noisiness']}"
    )

    print(
        "\nNote: exact instrument names are not "
        "claimed without a trained classifier."
    )

    # Graph
    times = librosa.times_like(
        rms,
        sr=sample_rate,
        hop_length=512
    )

    plt.figure(figsize=(12, 5))
    plt.plot(times, rms)

    plt.title(
        "Phase 11.1 - Sound Energy Profile"
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("RMS Energy")

    plt.tight_layout()

    graph_file = os.path.join(
        GRAPH_DIR,
        "phase_11_1_sound_energy.png"
    )

    plt.savefig(
        graph_file,
        dpi=150
    )

    plt.close()

    return result


# ============================================================
# 11.1.2 CHORD ANALYSIS
# ============================================================

def create_chord_templates():

    templates = {}

    for root in range(12):

        major = np.zeros(12)

        major[root] = 1
        major[(root + 4) % 12] = 1
        major[(root + 7) % 12] = 1

        templates[
            f"{NOTE_NAMES[root]} major"
        ] = major

        minor = np.zeros(12)

        minor[root] = 1
        minor[(root + 3) % 12] = 1
        minor[(root + 7) % 12] = 1

        templates[
            f"{NOTE_NAMES[root]} minor"
        ] = minor

    return templates


def analyze_chords(audio, sample_rate):

    print("\n" + "=" * 70)
    print("PHASE 11.1.2 - CHORD ANALYSIS")
    print("=" * 70)

    chroma = librosa.feature.chroma_cqt(
        y=audio,
        sr=sample_rate,
        hop_length=512
    )

    templates = create_chord_templates()

    names = list(templates.keys())

    template_matrix = np.array(
        [
            templates[name]
            for name in names
        ]
    )

    rows = []

    for frame in range(
        chroma.shape[1]
    ):

        current = chroma[:, frame]

        norm = np.linalg.norm(current)

        if norm == 0:
            continue

        current = current / norm

        scores = []

        for template in template_matrix:

            score = cosine_similarity(
                current,
                template
            )

            scores.append(score)

        best_index = int(
            np.argmax(scores)
        )

        time = librosa.frames_to_time(
            frame,
            sr=sample_rate,
            hop_length=512
        )

        rows.append({
            "time_seconds": float(time),
            "chord": names[best_index],
            "confidence": float(
                scores[best_index]
            )
        })

    data = pd.DataFrame(rows)

    if data.empty:

        print("No chord data generated.")

        return data

    # --------------------------------------------------------
    # Smooth labels
    # --------------------------------------------------------

    smoothed = []

    labels = data["chord"].tolist()

    window = 5

    for i in range(len(labels)):

        start = max(
            0,
            i - window
        )

        end = min(
            len(labels),
            i + window + 1
        )

        section = labels[start:end]

        counts = {}

        for label in section:
            counts[label] = (
                counts.get(label, 0) + 1
            )

        best = max(
            counts,
            key=counts.get
        )

        smoothed.append(best)

    data["smoothed_chord"] = smoothed

    # --------------------------------------------------------
    # Convert frames into chord segments
    # --------------------------------------------------------

    segments = []

    current_chord = data.iloc[0][
        "smoothed_chord"
    ]

    start_time = float(
        data.iloc[0]["time_seconds"]
    )

    for i in range(
        1,
        len(data)
    ):

        chord = data.iloc[i][
            "smoothed_chord"
        ]

        time = float(
            data.iloc[i]["time_seconds"]
        )

        if chord != current_chord:

            segments.append({
                "start_time_seconds":
                    start_time,

                "end_time_seconds":
                    time,

                "chord":
                    current_chord
            })

            current_chord = chord
            start_time = time

    final_time = float(
        data.iloc[-1]["time_seconds"]
    )

    segments.append({
        "start_time_seconds":
            start_time,

        "end_time_seconds":
            final_time,

        "chord":
            current_chord
    })

    result = pd.DataFrame(
        segments
    )

    output_file = os.path.join(
        OUTPUT_DIR,
        "chord_analysis.csv"
    )

    result.to_csv(
        output_file,
        index=False
    )

    print(
        f"Chord segments detected: "
        f"{len(result)}"
    )

    print("\nFirst chord segments:")

    print(
        result.head(15).to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Graph
    # --------------------------------------------------------

    categories = (
        data["smoothed_chord"]
        .astype("category")
        .cat.codes
    )

    plt.figure(figsize=(14, 5))

    plt.step(
        data["time_seconds"],
        categories,
        where="post"
    )

    plt.title(
        "Phase 11.1 - Estimated Chord Progression"
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Chord index")

    plt.tight_layout()

    graph_file = os.path.join(
        GRAPH_DIR,
        "phase_11_1_chord_progression.png"
    )

    plt.savefig(
        graph_file,
        dpi=150
    )

    plt.close()

    return result


# ============================================================
# 11.1.3 KEY AND SCALE
# ============================================================

MAJOR_PROFILE = np.array([
    6.35, 2.23, 3.48, 2.33,
    4.38, 4.09, 2.52, 5.19,
    2.39, 3.66, 2.29, 2.88
])

MINOR_PROFILE = np.array([
    6.33, 2.68, 3.52, 5.38,
    2.60, 3.53, 2.54, 4.75,
    3.98, 2.69, 3.34, 3.17
])


def analyze_key_scale(
    audio,
    sample_rate,
    tonic_name=None
):

    print("\n" + "=" * 70)
    print("PHASE 11.1.3 - KEY & SCALE ANALYSIS")
    print("=" * 70)

    chroma = librosa.feature.chroma_cqt(
        y=audio,
        sr=sample_rate,
        hop_length=512
    )

    chroma_mean = np.mean(
        chroma,
        axis=1
    )

    chroma_mean = normalize(
        chroma_mean
    )

    candidates = []

    for root in range(12):

        major = np.roll(
            MAJOR_PROFILE,
            root
        )

        minor = np.roll(
            MINOR_PROFILE,
            root
        )

        major_score = cosine_similarity(
            chroma_mean,
            normalize(major)
        )

        minor_score = cosine_similarity(
            chroma_mean,
            normalize(minor)
        )

        candidates.append({
            "key": NOTE_NAMES[root],
            "scale": "Major",
            "score": major_score
        })

        candidates.append({
            "key": NOTE_NAMES[root],
            "scale": "Minor",
            "score": minor_score
        })

    candidates = pd.DataFrame(
        candidates
    )

    candidates = (
        candidates
        .sort_values(
            "score",
            ascending=False
        )
        .reset_index(drop=True)
    )

    best = candidates.iloc[0]
    second = candidates.iloc[1]

    best_score = float(
        best["score"]
    )

    second_score = float(
        second["score"]
    )

    margin = (
        best_score -
        second_score
    )

    confidence = max(
        0.0,
        min(
            1.0,
            margin / 0.10
        )
    )

    result = {
        "estimated_key":
            str(best["key"]),

        "estimated_scale":
            str(best["scale"]),

        "similarity_score":
            best_score,

        "second_candidate":
            str(second["key"]),

        "second_candidate_scale":
            str(second["scale"]),

        "candidate_margin":
            margin,

        "confidence":
            confidence,

        "phase_7_tonic":
            tonic_name
    }

    pd.DataFrame(
        [result]
    ).to_csv(
        os.path.join(
            OUTPUT_DIR,
            "key_scale_analysis.csv"
        ),
        index=False
    )

    candidates.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "key_scale_candidates.csv"
        ),
        index=False
    )

    print(
        f"Estimated key    : "
        f"{result['estimated_key']}"
    )

    print(
        f"Estimated scale  : "
        f"{result['estimated_scale']}"
    )

    print(
        f"Similarity       : "
        f"{best_score:.4f}"
    )

    print(
        f"Confidence       : "
        f"{confidence * 100:.2f}%"
    )

    if tonic_name:
        print(
            f"Phase 7 tonic    : "
            f"{tonic_name}"
        )

    # Graph
    top = candidates.head(10)

    labels = (
        top["key"] +
        " " +
        top["scale"]
    )

    plt.figure(figsize=(12, 6))

    plt.bar(
        labels,
        top["score"]
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.ylabel(
        "Similarity score"
    )

    plt.title(
        "Phase 11.1 - Key / Scale Candidates"
    )

    plt.tight_layout()

    graph_file = os.path.join(
        GRAPH_DIR,
        "phase_11_1_key_scale_candidates.png"
    )

    plt.savefig(
        graph_file,
        dpi=150
    )

    plt.close()

    return result


# ============================================================
# 11.1.4 MUSICAL STRUCTURE
# ============================================================

def analyze_structure(
    audio,
    sample_rate
):

    print("\n" + "=" * 70)
    print("PHASE 11.1.4 - MUSICAL STRUCTURE ANALYSIS")
    print("=" * 70)

    chroma = librosa.feature.chroma_cqt(
        y=audio,
        sr=sample_rate,
        hop_length=1024
    )

    onset = librosa.onset.onset_strength(
        y=audio,
        sr=sample_rate,
        hop_length=1024
    )

    if chroma.shape[1] < 2:

        return pd.DataFrame()

    # Harmonic change
    harmonic_change = np.mean(
        np.abs(
            np.diff(
                chroma,
                axis=1
            )
        ),
        axis=0
    )

    harmonic_change = np.concatenate(
        [[0], harmonic_change]
    )

    # Normalize both signals
    harmonic_change = normalize(
        harmonic_change
    )

    onset = normalize(
        onset
    )

    length = min(
        len(harmonic_change),
        len(onset)
    )

    structure_signal = (
        harmonic_change[:length] +
        onset[:length]
    ) / 2.0

    # Strong changes
    threshold = np.percentile(
        structure_signal,
        90
    )

    indices = np.where(
        structure_signal >= threshold
    )[0]

    times = librosa.frames_to_time(
        indices,
        sr=sample_rate,
        hop_length=1024
    )

    # Remove nearby changes
    selected = []

    minimum_gap = 15.0

    for time in times:

        time = float(time)

        if not selected:

            selected.append(time)

        elif (
            time - selected[-1]
            >= minimum_gap
        ):

            selected.append(time)

    total_duration = (
        len(audio) /
        sample_rate
    )

    boundaries = [0.0]

    boundaries.extend(selected)

    boundaries.append(
        total_duration
    )

    boundaries = sorted(
        set(
            round(
                value,
                3
            )
            for value in boundaries
            if 0 <= value <= total_duration
        )
    )

    rows = []

    number_of_sections = (
        len(boundaries) - 1
    )

    for i in range(
        number_of_sections
    ):

        start = boundaries[i]
        end = boundaries[i + 1]

        if end <= start:
            continue

        if i == 0:

            name = "Intro / Opening"

        elif i == number_of_sections - 1:

            name = "Outro / Final Section"

        else:

            name = f"Section {i}"

        rows.append({
            "section_id": i + 1,
            "section_name": name,
            "start_time_seconds": start,
            "end_time_seconds": end,
            "duration_seconds":
                end - start
        })

    result = pd.DataFrame(
        rows
    )

    output_file = os.path.join(
        OUTPUT_DIR,
        "musical_structure.csv"
    )

    result.to_csv(
        output_file,
        index=False
    )

    print(
        f"Sections detected: "
        f"{len(result)}"
    )

    print("\nDetected structure:")

    if not result.empty:

        print(
            result.to_string(
                index=False
            )
        )

    # Graph
    graph_times = librosa.frames_to_time(
        np.arange(
            len(structure_signal)
        ),
        sr=sample_rate,
        hop_length=1024
    )

    plt.figure(
        figsize=(14, 5)
    )

    plt.plot(
        graph_times,
        structure_signal
    )

    for boundary in boundaries:

        plt.axvline(
            boundary,
            linestyle="--",
            alpha=0.5
        )

    plt.title(
        "Phase 11.1 - Musical Structure"
    )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Structure-change strength"
    )

    plt.tight_layout()

    graph_file = os.path.join(
        GRAPH_DIR,
        "phase_11_1_musical_structure.png"
    )

    plt.savefig(
        graph_file,
        dpi=150
    )

    plt.close()

    return result


# ============================================================
# COMPLETE PHASE 11.1
# ============================================================

def analyze_musical_elements(
    audio,
    sample_rate,
    tonic_name=None
):

    print("\n")
    print("=" * 80)
    print("PHASE 11.1 - MUSICAL ELEMENT ANALYSIS")
    print("=" * 80)

    sound_result = analyze_sound(
        audio,
        sample_rate
    )

    chord_result = analyze_chords(
        audio,
        sample_rate
    )

    key_result = analyze_key_scale(
        audio,
        sample_rate,
        tonic_name
    )

    structure_result = analyze_structure(
        audio,
        sample_rate
    )

    summary = {
        "phase": "11.1",
        "module": "Musical Element Analysis",

        "sound_characteristics":
            sound_result,

        "chord_segments":
            int(len(chord_result)),

        "key_scale":
            key_result,

        "structure_sections":
            int(len(structure_result)),

        "limitations": [
            "Instrument names are not claimed.",
            "Chord detection is template-based.",
            "Key and scale are estimated from chroma.",
            "Musical structure is algorithmically estimated."
        ]
    }

    summary_file = os.path.join(
        OUTPUT_DIR,
        "phase_11_1_summary.json"
    )

    with open(
        summary_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    print("\n" + "=" * 80)
    print("PHASE 11.1 COMPLETED SUCCESSFULLY")
    print("=" * 80)

    print("\nGenerated files:")

    print(
        "  data/output/instrument_analysis.csv"
    )

    print(
        "  data/output/chord_analysis.csv"
    )

    print(
        "  data/output/key_scale_analysis.csv"
    )

    print(
        "  data/output/key_scale_candidates.csv"
    )

    print(
        "  data/output/musical_structure.csv"
    )

    print(
        "  data/output/phase_11_1_summary.json"
    )

    return summary


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("PHASE 11.1 - MUSICAL ELEMENT ANALYSIS")
    print("=" * 70)

    audio_path = r"C:\Users\vikra\Downloads\Vande_Mataram_by_Pt._Chandrashekhar_Vaze.ogg"

    if not os.path.isfile(audio_path):
        raise FileNotFoundError(
            f"Audio file not found:\n{audio_path}"
        )

    print("\nAudio file:")
    print(audio_path)

    print("\nLoading audio...")

    audio, sample_rate = librosa.load(
        audio_path,
        sr=None,
        mono=True
    )

    print(f"Sample rate : {sample_rate} Hz")
    print(f"Duration    : {len(audio) / sample_rate:.2f} seconds")

    print("\nRunning Phase 11.1 analysis...")

    results = analyze_musical_elements(
        audio,
        sample_rate
    )

    print("\n" + "=" * 70)
    print("PHASE 11.1 COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print("\nGenerated files:")

    print("\nCSV files:")
    print("  data/output/instrument_analysis.csv")
    print("  data/output/chord_analysis.csv")
    print("  data/output/key_scale_analysis.csv")
    print("  data/output/key_scale_candidates.csv")
    print("  data/output/musical_structure.csv")

    print("\nSummary:")
    print("  data/output/phase_11_1_summary.json")

    print("\nGraphs:")
    print("  graphs/phase_11_1_sound_energy.png")
    print("  graphs/phase_11_1_chord_progression.png")
    print("  graphs/phase_11_1_key_scale_candidates.png")
    print("  graphs/phase_11_1_musical_structure.png")