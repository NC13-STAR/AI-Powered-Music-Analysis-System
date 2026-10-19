import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt


def analyze_rhythm(audio, sample_rate):
    print("\n" + "=" * 70)
    print("PHASE 8.1 - RHYTHM AND BPM ANALYSIS")
    print("=" * 70)

    print("\nAnalyzing rhythmic structure...")
    print("Detecting tempo and beat positions using onset-based beat tracking...")

    # ---------------------------------------------------------
    # STEP 1 - ONSET STRENGTH
    # ---------------------------------------------------------
    onset_envelope = librosa.onset.onset_strength(
        y=audio,
        sr=sample_rate
    )

    print("\nStep 1 - Onset strength analysis completed.")

    # ---------------------------------------------------------
    # STEP 2 - TEMPO AND BEAT TRACKING
    # ---------------------------------------------------------
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_envelope,
        sr=sample_rate
    )

    # librosa may return a NumPy array for tempo
    if np.ndim(tempo) > 0:
        tempo_value = float(np.asarray(tempo).flatten()[0])
    else:
        tempo_value = float(tempo)

    beat_frames = np.asarray(beat_frames, dtype=int)

    # Convert beat frame positions to time
    beat_times = librosa.frames_to_time(
        beat_frames,
        sr=sample_rate
    )

    print("\nStep 2 - Beat tracking completed.")

    # ---------------------------------------------------------
    # STEP 3 - BEAT INTERVAL ANALYSIS
    # ---------------------------------------------------------
    if len(beat_times) >= 2:
        beat_intervals = np.diff(beat_times)

        mean_beat_interval = float(np.mean(beat_intervals))
        median_beat_interval = float(np.median(beat_intervals))

        estimated_bpm_from_median = (
            60.0 / median_beat_interval
            if median_beat_interval > 0
            else 0.0
        )

        beat_interval_std = float(np.std(beat_intervals))

    else:
        beat_intervals = np.array([])
        mean_beat_interval = 0.0
        median_beat_interval = 0.0
        estimated_bpm_from_median = 0.0
        beat_interval_std = 0.0

    # ---------------------------------------------------------
    # STEP 4 - BPM CONSISTENCY
    # ---------------------------------------------------------
    if len(beat_intervals) >= 2 and median_beat_interval > 0:
        bpm_values = 60.0 / beat_intervals

        bpm_median = float(np.median(bpm_values))
        bpm_mean = float(np.mean(bpm_values))
        bpm_std = float(np.std(bpm_values))

        # Relative consistency measure
        if bpm_mean > 0:
            tempo_variation_percent = (
                bpm_std / bpm_mean
            ) * 100.0
        else:
            tempo_variation_percent = 0.0
    else:
        bpm_values = np.array([])
        bpm_median = tempo_value
        bpm_mean = tempo_value
        bpm_std = 0.0
        tempo_variation_percent = 0.0

    # ---------------------------------------------------------
    # STEP 5 - BASIC CONFIDENCE ESTIMATE
    # ---------------------------------------------------------
    # This is NOT a machine-learning confidence score.
    # It measures beat-interval consistency.

    if len(beat_intervals) >= 4 and bpm_mean > 0:

        consistency = 1.0 - (
            min(tempo_variation_percent, 100.0) / 100.0
        )

        beat_count_factor = min(len(beat_times) / 100.0, 1.0)

        confidence = (
            0.7 * consistency +
            0.3 * beat_count_factor
        )

    elif len(beat_times) >= 2:
        confidence = 0.4

    else:
        confidence = 0.0

    confidence_percent = confidence * 100.0

    # ---------------------------------------------------------
    # PRINT RESULTS
    # ---------------------------------------------------------
    print("\n" + "-" * 65)
    print("RHYTHM ANALYSIS RESULTS")
    print("-" * 65)

    print(f"Estimated BPM              : {tempo_value:.2f}")
    print(f"Median beat-derived BPM    : {bpm_median:.2f}")
    print(f"Mean beat-derived BPM      : {bpm_mean:.2f}")
    print(f"BPM standard deviation     : {bpm_std:.2f}")

    print(f"Total beats detected       : {len(beat_times)}")

    if len(beat_times) >= 2:
        print(f"Mean beat interval         : {mean_beat_interval:.4f} sec")
        print(f"Median beat interval       : {median_beat_interval:.4f} sec")
        print(f"Beat interval deviation    : {beat_interval_std:.4f} sec")

    print(f"Tempo variation            : {tempo_variation_percent:.2f}%")
    print(f"Beat consistency confidence: {confidence_percent:.2f}%")

    # ---------------------------------------------------------
    # STEP 6 - CREATE BEAT DATAFRAME
    # ---------------------------------------------------------
    beat_numbers = np.arange(1, len(beat_times) + 1)

    beat_data = pd.DataFrame({
        "beat_number": beat_numbers,
        "beat_time_seconds": beat_times
    })

    if len(beat_times) > 0:
        beat_data["beat_interval_seconds"] = np.nan

        if len(beat_times) >= 2:
            beat_data.loc[
                1:,
                "beat_interval_seconds"
            ] = np.diff(beat_times)

    # ---------------------------------------------------------
    # STEP 7 - SAVE RESULTS
    # ---------------------------------------------------------
    os.makedirs("data/output", exist_ok=True)

    output_file = "data/output/rhythm_analysis.csv"

    beat_data.to_csv(
        output_file,
        index=False
    )

    print("\nRhythm beat data saved successfully!")
    print(f"Saved file: {output_file}")

    # ---------------------------------------------------------
    # STEP 8 - SAVE SUMMARY
    # ---------------------------------------------------------
    summary_data = pd.DataFrame({
        "parameter": [
            "estimated_bpm",
            "median_beat_bpm",
            "mean_beat_bpm",
            "bpm_standard_deviation",
            "total_beats",
            "mean_beat_interval_seconds",
            "median_beat_interval_seconds",
            "beat_interval_standard_deviation",
            "tempo_variation_percent",
            "beat_consistency_confidence_percent"
        ],
        "value": [
            tempo_value,
            bpm_median,
            bpm_mean,
            bpm_std,
            len(beat_times),
            mean_beat_interval,
            median_beat_interval,
            beat_interval_std,
            tempo_variation_percent,
            confidence_percent
        ]
    })

    summary_file = "data/output/rhythm_summary.csv"

    summary_data.to_csv(
        summary_file,
        index=False
    )

    print("Rhythm summary saved successfully!")
    print(f"Saved file: {summary_file}")

    # ---------------------------------------------------------
    # STEP 9 - BEAT TIMELINE GRAPH
    # ---------------------------------------------------------
    os.makedirs("graphs", exist_ok=True)

    duration = len(audio) / sample_rate

    plt.figure(figsize=(16, 6))

    # Audio amplitude envelope for visual reference
    hop_length = 512

    audio_times = librosa.times_like(
        onset_envelope,
        sr=sample_rate,
        hop_length=hop_length
    )

    normalized_onset = onset_envelope.copy()

    if np.max(normalized_onset) > 0:
        normalized_onset = (
            normalized_onset /
            np.max(normalized_onset)
        )

    plt.plot(
        audio_times,
        normalized_onset,
        linewidth=1,
        label="Onset strength"
    )

    if len(beat_times) > 0:
        beat_y = np.ones(len(beat_times))

        plt.scatter(
            beat_times,
            beat_y,
            s=18,
            label="Detected beats"
        )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Normalized Onset Strength")
    plt.title("Phase 8.1 - Rhythm and Beat Detection")

    plt.xlim(0, duration)

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    graph_path = "graphs/rhythm_beat_detection.png"

    plt.savefig(
        graph_path,
        dpi=300
    )

    plt.close()

    print("\nBeat detection graph saved successfully!")
    print(f"Saved graph: {graph_path}")

    # ---------------------------------------------------------
    # STEP 10 - BEAT INTERVAL GRAPH
    # ---------------------------------------------------------
    if len(beat_times) >= 2:

        plt.figure(figsize=(14, 5))

        interval_numbers = np.arange(
            1,
            len(beat_intervals) + 1
        )

        plt.plot(
            interval_numbers,
            beat_intervals,
            linewidth=1
        )

        plt.xlabel("Beat Number")
        plt.ylabel("Beat Interval (seconds)")
        plt.title("Phase 8.1 - Beat Interval Stability")

        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        interval_graph = (
            "graphs/beat_interval_stability.png"
        )

        plt.savefig(
            interval_graph,
            dpi=300
        )

        plt.close()

        print(
            "Beat interval graph saved successfully!"
        )
        print(
            f"Saved graph: {interval_graph}"
        )

    # ---------------------------------------------------------
    # FINAL
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 8.1 COMPLETED SUCCESSFULLY")
    print("=" * 70)

    return {
        "tempo": tempo_value,
        "beat_times": beat_times,
        "beat_data": beat_data,
        "summary_data": summary_data,
        "confidence": confidence_percent
    }