# ============================================================
# PHASE 11.3 - MUSICAL EXPRESSION & PERFORMANCE ANALYSIS
# ============================================================
#
# Purpose:
#   Analyze expressive/performance characteristics from audio.
#
# Analyses:
#   11.3.1 - Dynamics / Energy Analysis
#   11.3.2 - Tempo & Timing Expression
#   11.3.3 - Pitch Expression / Vibrato Indicators
#   11.3.4 - Performance Summary
#
# Dependencies:
#   librosa
#   numpy
#   pandas
#   matplotlib
#
# No PyTorch.
# No Whisper.
# No machine-learning classifier.
#
# ============================================================

import os
import json
import math
import traceback

import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

OUTPUT_DIR = os.path.join(
    "data",
    "output"
)

GRAPH_DIR = os.path.join(
    "graphs"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    GRAPH_DIR,
    exist_ok=True
)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def safe_float(value, default=0.0):
    """
    Convert a value safely to float.
    """

    try:

        value = float(value)

        if np.isfinite(value):
            return value

    except Exception:
        pass

    return default


def percentile_safe(values, percentile, default=0.0):
    """
    Safe percentile calculation.
    """

    values = np.asarray(
        values,
        dtype=float
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return default

    return safe_float(
        np.percentile(
            values,
            percentile
        ),
        default
    )


def classify_dynamics(dynamic_range_db):
    """
    Classify overall dynamic variation.
    """

    if dynamic_range_db < 6:
        return "Very low"

    if dynamic_range_db < 12:
        return "Low"

    if dynamic_range_db < 20:
        return "Moderate"

    if dynamic_range_db < 30:
        return "High"

    return "Very high"


def classify_tempo_variation(cv_percent):
    """
    Classify local tempo variation.
    """

    if cv_percent < 3:
        return "Very stable"

    if cv_percent < 7:
        return "Stable"

    if cv_percent < 12:
        return "Moderately variable"

    if cv_percent < 20:
        return "Highly variable"

    return "Very highly variable"


def classify_vibrato_rate(rate):
    """
    Very conservative vibrato-rate classification.
    """

    if rate <= 0:
        return "Not detected"

    if rate < 4:
        return "Slow modulation"

    if rate <= 8:
        return "Typical vibrato-range modulation"

    return "Fast modulation"


# ============================================================
# 11.3.1 - DYNAMICS / ENERGY ANALYSIS
# ============================================================

def analyze_dynamics(
    audio,
    sample_rate
):
    """
    Analyze loudness and energy variation.
    """

    print("\n" + "=" * 70)
    print("PHASE 11.3.1 - DYNAMICS / ENERGY ANALYSIS")
    print("=" * 70)

    audio = np.asarray(
        audio,
        dtype=float
    )

    if len(audio) == 0:
        raise ValueError(
            "Audio is empty."
        )

    # --------------------------------------------------------
    # RMS ENERGY
    # --------------------------------------------------------

    rms = librosa.feature.rms(
        y=audio,
        frame_length=2048,
        hop_length=512
    )[0]

    rms = np.asarray(
        rms,
        dtype=float
    )

    rms = np.nan_to_num(
        rms,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    rms_db = librosa.amplitude_to_db(
        rms + 1e-10,
        ref=np.max
    )

    times = librosa.frames_to_time(
        np.arange(len(rms)),
        sr=sample_rate,
        hop_length=512
    )

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    mean_rms = safe_float(
        np.mean(rms)
    )

    median_rms = safe_float(
        np.median(rms)
    )

    min_rms = safe_float(
        np.min(rms)
    )

    max_rms = safe_float(
        np.max(rms)
    )

    mean_db = safe_float(
        np.mean(rms_db)
    )

    median_db = safe_float(
        np.median(rms_db)
    )

    low_db = percentile_safe(
        rms_db,
        10
    )

    high_db = percentile_safe(
        rms_db,
        90
    )

    dynamic_range_db = (
        high_db -
        low_db
    )

    dynamics_category = classify_dynamics(
        dynamic_range_db
    )

    print(
        f"Mean RMS energy       : {mean_rms:.5f}"
    )

    print(
        f"Median RMS energy     : {median_rms:.5f}"
    )

    print(
        f"Mean energy (dB)      : {mean_db:.2f} dB"
    )

    print(
        f"Median energy (dB)    : {median_db:.2f} dB"
    )

    print(
        f"10th percentile       : {low_db:.2f} dB"
    )

    print(
        f"90th percentile       : {high_db:.2f} dB"
    )

    print(
        f"Dynamic range         : {dynamic_range_db:.2f} dB"
    )

    print(
        f"Dynamics category     : {dynamics_category}"
    )

    # --------------------------------------------------------
    # ENERGY PROFILE
    # --------------------------------------------------------

    if len(rms) > 0:

        energy_normalized = (
            rms /
            (np.max(rms) + 1e-10)
        )

    else:

        energy_normalized = np.array([])

    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    dynamics_df = pd.DataFrame({

        "time_seconds": times,

        "rms_energy": rms,

        "rms_db": rms_db,

        "normalized_energy":
            energy_normalized

    })

    dynamics_csv = os.path.join(
        OUTPUT_DIR,
        "phase_11_3_dynamics.csv"
    )

    dynamics_df.to_csv(
        dynamics_csv,
        index=False
    )

    # --------------------------------------------------------
    # GRAPH
    # --------------------------------------------------------

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        times,
        rms_db
    )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "RMS Energy (dB)"
    )

    plt.title(
        "Phase 11.3 - Dynamics / Energy Profile"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    dynamics_graph = os.path.join(
        GRAPH_DIR,
        "phase_11_3_dynamics.png"
    )

    plt.savefig(
        dynamics_graph,
        dpi=150
    )

    plt.close()

    return {

        "mean_rms": mean_rms,

        "median_rms": median_rms,

        "mean_db": mean_db,

        "median_db": median_db,

        "low_db": low_db,

        "high_db": high_db,

        "dynamic_range_db":
            dynamic_range_db,

        "dynamics_category":
            dynamics_category,

        "csv":
            dynamics_csv,

        "graph":
            dynamics_graph

    }


# ============================================================
# 11.3.2 - TEMPO & TIMING EXPRESSION
# ============================================================

def analyze_tempo_expression(
    audio,
    sample_rate
):
    """
    Analyze local tempo/timing variation.

    This is not intended to replace the existing Phase 8
    tempo analysis. It focuses specifically on expressive
    timing variation.
    """

    print("\n" + "=" * 70)
    print("PHASE 11.3.2 - TEMPO & TIMING EXPRESSION")
    print("=" * 70)

    # --------------------------------------------------------
    # ONSET DETECTION
    # --------------------------------------------------------

    onset_envelope = librosa.onset.onset_strength(
        y=audio,
        sr=sample_rate
    )

    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_envelope,
        sr=sample_rate,
        units="frames"
    )

    onset_times = librosa.frames_to_time(
        onset_frames,
        sr=sample_rate
    )

    # --------------------------------------------------------
    # INTER-ONSET INTERVALS
    # --------------------------------------------------------

    if len(onset_times) >= 2:

        intervals = np.diff(
            onset_times
        )

        intervals = intervals[
            np.isfinite(intervals)
        ]

        intervals = intervals[
            intervals > 0
        ]

    else:

        intervals = np.array([])

    # --------------------------------------------------------
    # LOCAL TEMPO
    # --------------------------------------------------------

    if len(intervals) > 0:

        local_tempo = (
            60.0 /
            intervals
        )

        # Remove extreme accidental detections.
        local_tempo = local_tempo[
            (local_tempo >= 30) &
            (local_tempo <= 300)
        ]

    else:

        local_tempo = np.array([])

    if len(local_tempo) > 0:

        mean_tempo = safe_float(
            np.mean(local_tempo)
        )

        median_tempo = safe_float(
            np.median(local_tempo)
        )

        std_tempo = safe_float(
            np.std(local_tempo)
        )

        tempo_cv = (
            std_tempo /
            (mean_tempo + 1e-10)
        ) * 100

    else:

        mean_tempo = 0.0
        median_tempo = 0.0
        std_tempo = 0.0
        tempo_cv = 0.0

    timing_category = classify_tempo_variation(
        tempo_cv
    )

    print(
        f"Detected onsets       : {len(onset_times)}"
    )

    print(
        f"Mean local tempo      : {mean_tempo:.2f} BPM"
    )

    print(
        f"Median local tempo    : {median_tempo:.2f} BPM"
    )

    print(
        f"Tempo standard dev.   : {std_tempo:.2f} BPM"
    )

    print(
        f"Tempo variation       : {tempo_cv:.2f}%"
    )

    print(
        f"Timing category       : {timing_category}"
    )

    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    if len(local_tempo) > 0:

        tempo_times = onset_times[
            1:
        ]

        min_len = min(
            len(tempo_times),
            len(local_tempo)
        )

        tempo_df = pd.DataFrame({

            "time_seconds":
                tempo_times[:min_len],

            "local_tempo_bpm":
                local_tempo[:min_len]

        })

    else:

        tempo_df = pd.DataFrame(
            columns=[
                "time_seconds",
                "local_tempo_bpm"
            ]
        )

    tempo_csv = os.path.join(
        OUTPUT_DIR,
        "phase_11_3_tempo_expression.csv"
    )

    tempo_df.to_csv(
        tempo_csv,
        index=False
    )

    # --------------------------------------------------------
    # GRAPH
    # --------------------------------------------------------

    plt.figure(
        figsize=(12, 5)
    )

    if len(tempo_df) > 0:

        plt.plot(
            tempo_df["time_seconds"],
            tempo_df["local_tempo_bpm"]
        )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Local Tempo (BPM)"
    )

    plt.title(
        "Phase 11.3 - Local Tempo Expression"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    tempo_graph = os.path.join(
        GRAPH_DIR,
        "phase_11_3_tempo_expression.png"
    )

    plt.savefig(
        tempo_graph,
        dpi=150
    )

    plt.close()

    return {

        "onset_count":
            int(len(onset_times)),

        "mean_tempo_bpm":
            mean_tempo,

        "median_tempo_bpm":
            median_tempo,

        "tempo_std_bpm":
            std_tempo,

        "tempo_variation_percent":
            tempo_cv,

        "timing_category":
            timing_category,

        "csv":
            tempo_csv,

        "graph":
            tempo_graph

    }


# ============================================================
# 11.3.3 - PITCH EXPRESSION / VIBRATO INDICATORS
# ============================================================

def analyze_pitch_expression(
    audio,
    sample_rate
):
    """
    Estimate pitch modulation characteristics.

    This is deliberately conservative.

    It does NOT claim that every detected modulation is
    definitely musical vibrato.
    """

    print("\n" + "=" * 70)
    print("PHASE 11.3.3 - PITCH EXPRESSION / VIBRATO INDICATORS")
    print("=" * 70)

    # --------------------------------------------------------
    # PITCH TRACKING
    # --------------------------------------------------------

    f0, voiced_flag, voiced_prob = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz(
            "C2"
        ),
        fmax=librosa.note_to_hz(
            "C6"
        ),
        sr=sample_rate,
        frame_length=2048,
        hop_length=512
    )

    times = librosa.times_like(
        f0,
        sr=sample_rate,
        hop_length=512
    )

    valid = (
        np.isfinite(f0)
        &
        (f0 > 0)
    )

    valid_f0 = f0[
        valid
    ]

    valid_times = times[
        valid
    ]

    if len(valid_f0) < 20:

        print(
            "Not enough voiced pitch data "
            "for reliable expression analysis."
        )

        return {

            "voiced_frames":
                int(len(valid_f0)),

            "pitch_modulation_detected":
                False,

            "estimated_modulation_rate_hz":
                0.0,

            "estimated_modulation_depth_cents":
                0.0,

            "classification":
                "Insufficient data"

        }

    # --------------------------------------------------------
    # CONVERT TO CENTS / SEMITONES
    # --------------------------------------------------------

    pitch_semitones = (
        12 *
        np.log2(
            valid_f0 /
            440.0
        ) +
        69
    )

    # --------------------------------------------------------
    # REMOVE SLOW TREND
    # --------------------------------------------------------

    if len(pitch_semitones) >= 9:

        window = 9

        kernel = np.ones(
            window
        ) / window

        smooth_pitch = np.convolve(
            pitch_semitones,
            kernel,
            mode="same"
        )

        modulation = (
            pitch_semitones -
            smooth_pitch
        )

    else:

        modulation = (
            pitch_semitones -
            np.mean(pitch_semitones)
        )

    modulation_cents = (
        modulation *
        100
    )

    # --------------------------------------------------------
    # MODULATION DEPTH
    # --------------------------------------------------------

    modulation_depth = safe_float(
        np.std(
            modulation_cents
        )
    )

    # --------------------------------------------------------
    # MODULATION RATE
    # --------------------------------------------------------

    if len(valid_times) >= 2:

        dt = np.median(
            np.diff(
                valid_times
            )
        )

    else:

        dt = 0.0

    estimated_rate = 0.0

    if dt > 0:

        # FFT-based rough modulation estimate.
        signal = (
            modulation -
            np.mean(modulation)
        )

        n = len(signal)

        if n >= 32:

            window = np.hanning(
                n
            )

            spectrum = np.abs(
                np.fft.rfft(
                    signal *
                    window
                )
            )

            frequencies = np.fft.rfftfreq(
                n,
                d=dt
            )

            valid_freq = (
                (frequencies >= 2.0)
                &
                (frequencies <= 12.0)
            )

            if np.any(valid_freq):

                candidate_spectrum = (
                    spectrum[
                        valid_freq
                    ]
                )

                candidate_freqs = (
                    frequencies[
                        valid_freq
                    ]
                )

                if len(
                    candidate_spectrum
                ) > 0:

                    peak_index = int(
                        np.argmax(
                            candidate_spectrum
                        )
                    )

                    peak_strength = (
                        candidate_spectrum[
                            peak_index
                        ]
                    )

                    total_strength = (
                        np.sum(
                            candidate_spectrum
                        ) +
                        1e-10
                    )

                    concentration = (
                        peak_strength /
                        total_strength
                    )

                    if concentration >= 0.08:

                        estimated_rate = (
                            float(
                                candidate_freqs[
                                    peak_index
                                ]
                            )
                        )

    # --------------------------------------------------------
    # CONSERVATIVE DETECTION
    # --------------------------------------------------------

    modulation_detected = (
        modulation_depth >= 8.0
        and
        estimated_rate > 0
    )

    if modulation_detected:

        classification = classify_vibrato_rate(
            estimated_rate
        )

    else:

        classification = (
            "No clear sustained pitch modulation detected"
        )

    print(
        f"Voiced frames             : {len(valid_f0)}"
    )

    print(
        f"Pitch modulation depth    : "
        f"{modulation_depth:.2f} cents"
    )

    print(
        f"Estimated modulation rate : "
        f"{estimated_rate:.2f} Hz"
    )

    print(
        f"Pitch expression           : "
        f"{classification}"
    )

    print(
        "Note: modulation is an indicator, "
        "not a definitive vibrato classification."
    )

    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    pitch_df = pd.DataFrame({

        "time_seconds":
            valid_times,

        "f0_hz":
            valid_f0,

        "pitch_semitones":
            pitch_semitones,

        "pitch_modulation_cents":
            modulation_cents

    })

    pitch_csv = os.path.join(
        OUTPUT_DIR,
        "phase_11_3_pitch_expression.csv"
    )

    pitch_df.to_csv(
        pitch_csv,
        index=False
    )

    # --------------------------------------------------------
    # GRAPH
    # --------------------------------------------------------

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        valid_times,
        modulation_cents
    )

    plt.axhline(
        0,
        linestyle="--",
        linewidth=1
    )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Pitch deviation (cents)"
    )

    plt.title(
        "Phase 11.3 - Pitch Expression"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    pitch_graph = os.path.join(
        GRAPH_DIR,
        "phase_11_3_pitch_expression.png"
    )

    plt.savefig(
        pitch_graph,
        dpi=150
    )

    plt.close()

    return {

        "voiced_frames":
            int(len(valid_f0)),

        "pitch_modulation_detected":
            bool(modulation_detected),

        "estimated_modulation_rate_hz":
            estimated_rate,

        "estimated_modulation_depth_cents":
            modulation_depth,

        "classification":
            classification,

        "csv":
            pitch_csv,

        "graph":
            pitch_graph

    }


# ============================================================
# 11.3.4 - PERFORMANCE SUMMARY
# ============================================================

def create_performance_summary(
    dynamics_data,
    tempo_data,
    pitch_data
):
    """
    Combine Phase 11.3 measurements into a structured summary.
    """

    print("\n" + "=" * 70)
    print("PHASE 11.3.4 - PERFORMANCE SUMMARY")
    print("=" * 70)

    dynamic_range = safe_float(
        dynamics_data.get(
            "dynamic_range_db",
            0
        )
    )

    tempo_variation = safe_float(
        tempo_data.get(
            "tempo_variation_percent",
            0
        )
    )

    modulation_depth = safe_float(
        pitch_data.get(
            "estimated_modulation_depth_cents",
            0
        )
    )

    modulation_rate = safe_float(
        pitch_data.get(
            "estimated_modulation_rate_hz",
            0
        )
    )

    # --------------------------------------------------------
    # OBJECTIVE INTERPRETATION
    # --------------------------------------------------------

    if dynamic_range < 12:

        dynamics_interpretation = (
            "The recording shows relatively limited "
            "frame-level energy variation."
        )

    elif dynamic_range < 25:

        dynamics_interpretation = (
            "The recording shows moderate "
            "frame-level energy variation."
        )

    else:

        dynamics_interpretation = (
            "The recording shows substantial "
            "frame-level energy variation."
        )

    if tempo_variation < 7:

        timing_interpretation = (
            "Detected onset timing is relatively stable."
        )

    elif tempo_variation < 15:

        timing_interpretation = (
            "Detected onset timing shows moderate variation."
        )

    else:

        timing_interpretation = (
            "Detected onset timing shows substantial variation."
        )

    if pitch_data.get(
        "pitch_modulation_detected",
        False
    ):

        pitch_interpretation = (
            "Sustained pitch modulation was detected; "
            "this is consistent with expressive pitch movement, "
            "but it is not by itself proof of vibrato."
        )

    else:

        pitch_interpretation = (
            "No clear sustained pitch modulation pattern "
            "was detected by this analysis."
        )

    # --------------------------------------------------------
    # SUMMARY OBJECT
    # --------------------------------------------------------

    summary = {

        "phase": "11.3",

        "title":
            "Musical Expression & Performance Analysis",

        "dynamics": {

            "dynamic_range_db":
                dynamic_range,

            "category":
                dynamics_data.get(
                    "dynamics_category",
                    "Unknown"
                ),

            "interpretation":
                dynamics_interpretation

        },

        "timing": {

            "mean_tempo_bpm":
                safe_float(
                    tempo_data.get(
                        "mean_tempo_bpm",
                        0
                    )
                ),

            "median_tempo_bpm":
                safe_float(
                    tempo_data.get(
                        "median_tempo_bpm",
                        0
                    )
                ),

            "tempo_variation_percent":
                tempo_variation,

            "category":
                tempo_data.get(
                    "timing_category",
                    "Unknown"
                ),

            "interpretation":
                timing_interpretation

        },

        "pitch_expression": {

            "modulation_detected":
                bool(
                    pitch_data.get(
                        "pitch_modulation_detected",
                        False
                    )
                ),

            "modulation_rate_hz":
                modulation_rate,

            "modulation_depth_cents":
                modulation_depth,

            "classification":
                pitch_data.get(
                    "classification",
                    "Unknown"
                ),

            "interpretation":
                pitch_interpretation

        },

        "interpretation_notes": [

            "Results are signal-analysis measurements.",

            "They do not identify a specific performer technique "
            "without a dedicated trained classifier.",

            "Pitch modulation is reported as an indicator rather "
            "than a definitive vibrato label.",

            "This phase does not replace the existing "
            "raga-recognition pipeline."

        ]

    }

    # --------------------------------------------------------
    # SAVE JSON
    # --------------------------------------------------------

    summary_json = os.path.join(
        OUTPUT_DIR,
        "phase_11_3_summary.json"
    )

    with open(
        summary_json,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # SAVE SUMMARY CSV
    # --------------------------------------------------------

    summary_rows = [

        {
            "metric":
                "dynamic_range_db",

            "value":
                dynamic_range
        },

        {
            "metric":
                "tempo_variation_percent",

            "value":
                tempo_variation
        },

        {
            "metric":
                "pitch_modulation_depth_cents",

            "value":
                modulation_depth
        },

        {
            "metric":
                "pitch_modulation_rate_hz",

            "value":
                modulation_rate
        }

    ]

    summary_csv = os.path.join(
        OUTPUT_DIR,
        "phase_11_3_musical_interpretation.csv"
    )

    pd.DataFrame(
        summary_rows
    ).to_csv(
        summary_csv,
        index=False
    )

    print(
        "\nDynamics:"
    )

    print(
        f"  {dynamics_interpretation}"
    )

    print(
        "\nTiming:"
    )

    print(
        f"  {timing_interpretation}"
    )

    print(
        "\nPitch expression:"
    )

    print(
        f"  {pitch_interpretation}"
    )

    print(
        f"\nSummary saved:"
        f"\n  {summary_json}"
    )

    print(
        f"\nCSV saved:"
        f"\n  {summary_csv}"
    )

    return summary


# ============================================================
# MAIN PHASE 11.3 FUNCTION
# ============================================================

def analyze_musical_expression(
    audio,
    sample_rate
):
    """
    Complete Phase 11.3 execution.
    """

    print("\n")
    print("=" * 80)
    print("PHASE 11.3 - MUSICAL EXPRESSION & PERFORMANCE ANALYSIS")
    print("=" * 80)

    # --------------------------------------------------------
    # 11.3.1
    # --------------------------------------------------------

    dynamics_data = analyze_dynamics(
        audio,
        sample_rate
    )

    # --------------------------------------------------------
    # 11.3.2
    # --------------------------------------------------------

    tempo_data = analyze_tempo_expression(
        audio,
        sample_rate
    )

    # --------------------------------------------------------
    # 11.3.3
    # --------------------------------------------------------

    pitch_data = analyze_pitch_expression(
        audio,
        sample_rate
    )

    # --------------------------------------------------------
    # 11.3.4
    # --------------------------------------------------------

    summary = create_performance_summary(
        dynamics_data,
        tempo_data,
        pitch_data
    )

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    result = {

        "phase":
            "11.3",

        "dynamics":
            dynamics_data,

        "tempo_expression":
            tempo_data,

        "pitch_expression":
            pitch_data,

        "summary":
            summary

    }

    print("\n")
    print("=" * 80)
    print("PHASE 11.3 COMPLETED SUCCESSFULLY")
    print("=" * 80)

    print("\nGenerated files:")

    print(
        "  data/output/phase_11_3_dynamics.csv"
    )

    print(
        "  data/output/phase_11_3_tempo_expression.csv"
    )

    print(
        "  data/output/phase_11_3_pitch_expression.csv"
    )

    print(
        "  data/output/phase_11_3_musical_interpretation.csv"
    )

    print(
        "  data/output/phase_11_3_summary.json"
    )

    print("\nGenerated graphs:")

    print(
        "  graphs/phase_11_3_dynamics.png"
    )

    print(
        "  graphs/phase_11_3_tempo_expression.png"
    )

    print(
        "  graphs/phase_11_3_pitch_expression.png"
    )

    return result


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 80)
    print("PHASE 11.3 STANDALONE TEST")
    print("=" * 80)

    audio_path = input(
        "\nEnter audio file path: "
    ).strip()

    # Remove accidental surrounding quotes.
    audio_path = audio_path.strip(
        '"'
    ).strip("'")

    if not os.path.isfile(
        audio_path
    ):

        raise FileNotFoundError(
            f"\nAudio file not found:\n"
            f"{audio_path}"
        )

    print(
        "\nLoading audio..."
    )

    audio, sample_rate = librosa.load(
        audio_path,
        sr=None,
        mono=True
    )

    duration = (
        len(audio) /
        sample_rate
    )

    print(
        f"\nSample rate : "
        f"{sample_rate} Hz"
    )

    print(
        f"Duration    : "
        f"{duration:.2f} seconds"
    )

    print(
        "\nRunning Phase 11.3..."
    )

    try:

        analyze_musical_expression(
            audio,
            sample_rate
        )

        print("\n")
        print("=" * 80)
        print("PHASE 11.3 STANDALONE TEST FINISHED")
        print("=" * 80)

    except KeyboardInterrupt:

        print(
            "\nPhase 11.3 interrupted by user."
        )

    except Exception as error:

        print("\n")
        print("=" * 80)
        print("PHASE 11.3 FAILED")
        print("=" * 80)

        print(
            f"\nError type:"
            f"\n{type(error).__name__}"
        )

        print(
            f"\nError:"
            f"\n{error}"
        )

        print(
            "\nTraceback:"
        )

        traceback.print_exc()