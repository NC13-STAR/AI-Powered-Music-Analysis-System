import os
import runpy
import traceback


# ============================================================
# PROJECT PATH
# ============================================================
BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ============================================================
# INPUT
# ============================================================

from input_handler import (
    select_local_audio,
    download_audio_from_url
)


# ============================================================
# AUDIO LOADING
# ============================================================
from audio_loader import load_audio


# ============================================================
# PHASE 4 - PREPROCESSING
# ============================================================

from preprocessing import (
    preprocess_audio
)


# ============================================================
# PHASE 5 - VISUALIZATION
# ============================================================

from visualization import (
    plot_waveform,
    plot_fft,
    plot_spectrogram
)


# ============================================================
# PHASE 6 - MELODY / PITCH ANALYSIS
# ============================================================

from melody_extraction import (
    extract_melodic_component
)

from pitch_detection import (
    detect_pitch
)

from pitch_cleaning import (
    clean_pitch
)

from pitch_smoothing import (
    smooth_and_correct_pitch
)

from final_melody import (
    create_final_melody_contour
)

from western_notes import (
    convert_f0_to_western_notes
)


# ============================================================
# PHASE 7 - NOTE / TONIC / SWARA / NOTATION
# ============================================================

from note_segmentation import (
    segment_melody_notes
)

from note_quantization import (
    quantize_notes
)

from note_duration import (
    analyze_note_durations
)

from tonic_estimation import (
    estimate_tonic
)

from indian_swara import (
    perform_swara_mapping
)

from melody_phrase_analysis import (
    analyze_melodic_phrases
)

from notation_alignment import (
    align_western_indian_notation
)


# ============================================================
# PHASE 8 - RHYTHM ANALYSIS
# ============================================================

from rhythm_analysis import (
    analyze_rhythm
)

from beat_structure_analysis import (
    analyze_beat_structure
)

from tempo_stability import (
    analyze_tempo_stability
)

from rhythm_pattern_analysis import (
    analyze_rhythm_pattern
)

from taal_estimation import (
    estimate_taal
)


# ============================================================
# PHASE 9 - RAGA ANALYSIS
# ============================================================

from raga_estimation import (
    estimate_raga
)

# PHASE 9.2 - GENERAL RAGA RECOGNITION & VALIDATION
from raga_recognition import (
    recognize_and_validate_raga
)


# ============================================================
# PHASE 11 - ADVANCED MUSICAL ANALYSIS
# ============================================================

from musical_element_analysis import (
    analyze_musical_elements
)

from advanced_melody_rhythm import (
    run_phase_11_2
)


# ============================================================
# PHASE 12 - MOTIF ANALYSIS HELPERS
# ============================================================

def run_phase_file_from_main(phase_filename, phase_title):
    """
    Run a file-based phase from main.py.

    Phase 12.1, 12.2 and 12.3 are intentionally kept as standalone
    modules. They read/write their own CSV, JSON and graph outputs.
    main.py only controls their execution order.
    """
    phase_file = os.path.join(
        BASE_DIR,
        phase_filename,
    )

    if not os.path.isfile(phase_file):
        raise FileNotFoundError(
            f"{phase_title} file was not found:\n{phase_file}"
        )

    print("\n")
    print("=" * 80)
    print(phase_title)
    print("=" * 80)
    print()
    print(f"Running {phase_filename}...")

    result = runpy.run_path(
        phase_file,
        run_name="__main__",
    )

    print()
    print(f"{phase_title} completed successfully.")

    return result


def run_phase_12_1_from_main():
    return run_phase_file_from_main(
        "phase_12_1.py",
        "PHASE 12.1 - MUSICAL PATTERN & MOTIF ANALYSIS",
    )


def run_phase_12_2_from_main():
    return run_phase_file_from_main(
        "phase_12_2.py",
        "PHASE 12.2 - MOTIF TRANSFORMATION & VARIATION ANALYSIS",
    )


def run_phase_12_3_from_main():
    return run_phase_file_from_main(
        "phase_12_3.py",
        "PHASE 12.3 - MOTIF DEVELOPMENT & STRUCTURAL ROLE ANALYSIS",
    )


# ============================================================
# ============================================================
# PHASE 13 - LYRIC / MUSICAL NOTATION PIPELINE
# ============================================================

def run_phase_13_1_from_main():
    return run_phase_file_from_main(
        "phase_13_1.py",
        "PHASE 13.1 - LYRICS ACQUISITION",
    )


def run_phase_13_2_from_main():
    return run_phase_file_from_main(
        "phase_13_2.py",
        "PHASE 13.2 - LYRICS + MUSICAL NOTATION ALIGNMENT",
    )


def run_phase_13_3_from_main():
    return run_phase_file_from_main(
        "phase_13_3.py",
        "PHASE 13.3 - FULL SONG LYRIC / MUSICAL NOTATION ALIGNMENT",
    )


# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 80)
    print("AI POWERED MUSIC ANALYSIS SYSTEM")
    print("=" * 80)

    print(
        "Complete Audio -> DSP -> Pitch -> Notes -> Swara"
    )

    print(
        "-> Rhythm -> Taal -> Raga Analysis"
    )

    print(
        "-> Phase 11.1 -> Phase 11.2 -> Phase 12.1 -> Phase 12.2 -> Phase 12.3"
    )
    print(
        "-> Phase 13.1 -> Phase 13.2 -> Phase 13.3"
    )

    print("=" * 80)


    # ========================================================
    # PHASE 3 - AUDIO INPUT
    # ========================================================

    print("\n")
    print("=" * 70)
    print("PHASE 3 - AUDIO INPUT")
    print("=" * 70)

    print("\nChoose input method:")
    print("1. Local audio file")
    print("2. Direct audio URL")

    choice = input(
        "\nEnter your choice (1 or 2): "
    ).strip()


    audio_path = None


    if choice == "1":

        print(
            "\nOpening audio file selector..."
        )

        audio_path = select_local_audio()

        if not audio_path:

            print(
                "\nNo audio file selected."
            )

            return

        print(
            "\nSelected audio file:"
        )

        print(
            audio_path
        )


    elif choice == "2":

        url = input(
            "\nEnter direct audio URL: "
        ).strip()

        if not url:

            print(
                "\nNo URL entered."
            )

            return

        audio_path = (
            download_audio_from_url(
                url
            )
        )


    else:

        print(
            "\nInvalid choice."
        )

        return


    # ========================================================
    # LOAD AUDIO
    # ========================================================

    print("\n")
    print("=" * 70)
    print("LOADING AUDIO")
    print("=" * 70)

    print(
        f"\nAudio file:\n{audio_path}"
    )


    # Use the centralized project audio loader.
    # librosa.load() is kept inside audio_loader.py.
    audio, sample_rate = load_audio(audio_path)


    # ========================================================
    # PHASE 4 - PREPROCESSING
    # ========================================================

    print("\n")
    print("=" * 70)
    print("PHASE 4 - AUDIO PREPROCESSING")
    print("=" * 70)


    processed_audio = (
        preprocess_audio(
            audio,
            sample_rate
        )
    )


    # ========================================================
    # PHASE 5.1 - WAVEFORM
    # ========================================================

    print("\n")
    print("=" * 70)
    print("PHASE 5.1 - WAVEFORM")
    print("=" * 70)


    plot_waveform(
        processed_audio,
        sample_rate
    )


    # ========================================================
    # PHASE 5.2 - FFT
    # ========================================================

    print("\n")
    print("=" * 70)
    print("PHASE 5.2 - FFT SPECTRUM")
    print("=" * 70)


    plot_fft(
        processed_audio,
        sample_rate
    )


    # ========================================================
    # PHASE 5.3 - SPECTROGRAM
    # ========================================================

    print("\n")
    print("=" * 70)
    print("PHASE 5.3 - SPECTROGRAM")
    print("=" * 70)


    plot_spectrogram(
        processed_audio,
        sample_rate
    )


    # ========================================================
    # PHASE 6.1 - MELODY EXTRACTION
    # ========================================================

    print("\n")
    print("=" * 70)
    print("PHASE 6.1 - MELODY EXTRACTION")
    print("=" * 70)


    # ========================================================
    # OPTION A - BYPASS HPSS IN PHASE 6.1
    # ========================================================
    #
    # The current Windows environment blocks the compiled
    # scikit-learn component required by librosa.effects.hpss().
    #
    # We therefore bypass HPSS and pass the already preprocessed
    # audio directly to PYIN in Phase 6.2.
    #
    # All downstream melody-analysis phases remain unchanged.
    # ========================================================

    print("\nOption A selected:")
    print("Bypassing HPSS for Phase 6.1.")
    print("Using preprocessed audio directly for pitch detection.")
    print("Reason: current Windows environment blocks the")
    print("compiled component required by the HPSS dependency.")
    print("\nPhase 6.1 bypass completed successfully.")

    melodic_component = processed_audio


    # ========================================================
    # PHASE 6.2 - PITCH / F0 DETECTION
    # ========================================================

    print("\n")
    print("=" * 70)
    print("PHASE 6.2 - PITCH / F0 DETECTION")
    print("=" * 70)


    (
        f0,
        times,
        voiced_flag,
        voiced_prob
    ) = detect_pitch(
        melodic_component,
        sample_rate
    )


    # ========================================================
    # PHASE 6.3 - PITCH CLEANING
    # ========================================================

    print("\n")
    print("=" * 70)
    print("PHASE 6.3 - PITCH CLEANING")
    print("=" * 70)


    (
        cleaned_f0,
        cleaned_times
    ) = clean_pitch(
        f0,
        times,
        voiced_flag,
        voiced_prob
    )


    # ========================================================
    # PHASE 6.4 - PITCH SMOOTHING
    # ========================================================

    print("\n")
    print("=" * 70)
    print("PHASE 6.4 - PITCH SMOOTHING + OCTAVE CORRECTION")
    print("=" * 70)


    (
        smoothed_f0,
        smoothed_times
    ) = smooth_and_correct_pitch(
        cleaned_f0,
        cleaned_times
    )


    # ========================================================
    # PHASE 6.5 - FINAL MELODY CONTOUR
    # ========================================================

    print("\n")
    print("=" * 70)
    print("PHASE 6.5 - FINAL MELODY CONTOUR")
    print("=" * 70)


    (
        final_f0,
        final_times
    ) = create_final_melody_contour(
        smoothed_f0,
        smoothed_times
    )


    # ========================================================
    # PHASE 6.6 - WESTERN NOTE CONVERSION
    # ========================================================

    print("\n")
    print("=" * 70)
    print("PHASE 6.6 - WESTERN NOTE CONVERSION")
    print("=" * 70)


    western_note_data = (
        convert_f0_to_western_notes(
            final_f0,
            final_times
        )
    )


    # ========================================================
    # PHASE 7.1 - NOTE SEGMENTATION
    # ========================================================

    print("\n")
    print("=" * 70)
    print("PHASE 7.1 - NOTE SEGMENTATION")
    print("=" * 70)


    note_data = (
        segment_melody_notes(
            final_f0,
            final_times
        )
    )


    # ========================================================
    # PHASE 7.2 - NOTE QUANTIZATION
    # ========================================================

    print("\n")
    print("=" * 70)
    print("PHASE 7.2 - NOTE STABILIZATION / QUANTIZATION")
    print("=" * 70)


    quantized_note_data = (
        quantize_notes(
            note_data
        )
    )


    # ========================================================
    # PHASE 7.3 - NOTE DURATION
    # ========================================================

    print("\n")
    print("=" * 70)
    print("PHASE 7.3 - NOTE DURATION ANALYSIS")
    print("=" * 70)


    duration_data = (
        analyze_note_durations(
            note_data
        )
    )


    # ========================================================
    # PHASE 7.4 - TONIC ESTIMATION
    # ========================================================

    print("\n")
    print("=" * 70)
    print("PHASE 7.4 - TONIC ESTIMATION")
    print("=" * 70)


    tonic_data = (
        estimate_tonic(
            quantized_note_data
        )
    )


    # ========================================================
    # PHASE 7.5 - INDIAN SWARA MAPPING
    # ========================================================

    print("\n")
    print("=" * 70)
    print("PHASE 7.5 - INDIAN SWARA MAPPING")
    print("=" * 70)


    swara_data = (
        perform_swara_mapping()
    )


    # ========================================================
    # PHASE 7.6 - MELODIC PHRASE ANALYSIS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("PHASE 7.6 - MELODIC PHRASE ANALYSIS")
    print("=" * 70)


    phrase_data = (
        analyze_melodic_phrases()
    )


    # ========================================================
    # PHASE 7.7 - WESTERN + INDIAN NOTATION ALIGNMENT
    # ========================================================
    #
    # IMPORTANT:
    #
    # This phase is file-driven.
    #
    # notation_alignment.py loads:
    #
    # data/output/melody_phrase_note_data.csv
    #
    # generated by Phase 7.6.
    #
    # Therefore we call:
    #
    # align_western_indian_notation()
    #
    # NOT:
    #
    # create_notation_alignment(swara_data)
    #
    # ========================================================

    print("\n")
    print("=" * 70)
    print("PHASE 7.7 - WESTERN + INDIAN NOTATION ALIGNMENT")
    print("=" * 70)


    notation_data = (
        align_western_indian_notation()
    )


    # ========================================================
    # PHASE 8.1 - RHYTHM / BPM
    # ========================================================

    print("\n")
    print("=" * 70)
    print("PHASE 8.1 - RHYTHM / BPM ANALYSIS")
    print("=" * 70)


    rhythm_data = (
        analyze_rhythm(
            processed_audio,
            sample_rate
        )
    )


    # ========================================================
    # PHASE 8.2 - BEAT / DOWNBEAT STRUCTURE
    # ========================================================
    #
    # IMPORTANT:
    #
    # The actual function requires:
    #
    # analyze_beat_structure(audio, sample_rate)
    #
    # ========================================================

    print("\n")
    print("=" * 70)
    print("PHASE 8.2 - BEAT / DOWNBEAT STRUCTURE")
    print("=" * 70)


    beat_structure_data = (
        analyze_beat_structure(
            processed_audio,
            sample_rate
        )
    )


    # ========================================================
    # PHASE 8.3 - TEMPO STABILITY
    # ========================================================

    print("\n")
    print("=" * 70)
    print("PHASE 8.3 - TEMPO STABILITY")
    print("=" * 70)


    tempo_stability_data = (
        analyze_tempo_stability()
    )


    # ========================================================
    # PHASE 8.4 - RHYTHM PATTERN
    # ========================================================

    print("\n")
    print("=" * 70)
    print("PHASE 8.4 - RHYTHM PATTERN")
    print("=" * 70)


    rhythm_pattern_data = (
        analyze_rhythm_pattern(
            processed_audio,
            sample_rate
        )
    )


    # ========================================================
    # PHASE 8.5 - TAAL ESTIMATION
    # ========================================================

    print("\n")
    print("=" * 70)
    print("PHASE 8.5 - TAAL ESTIMATION")
    print("=" * 70)


    taal_data = (
        estimate_taal()
    )


    # ========================================================
    # PHASE 9.1 - RAGA ESTIMATION
    # ========================================================

    print("\n")
    print("=" * 70)
    print("PHASE 9.1 - RAGA ESTIMATION")
    print("=" * 70)


    raga_data = (
        estimate_raga()
    )


    # ========================================================
    # PHASE 9.2 - GENERAL RAGA RECOGNITION & VALIDATION
    # ========================================================
    #
    # Phase 9.2 is divided into:
    #
    # 9.2.1 - Raga Feature Analysis
    # 9.2.2 - Raga Candidate Recognition & Validation
    #
    # The function internally executes both sub-phases.
    #
    # It is general-purpose and does not depend on
    # a specific song being hard-coded.
    #
    # ========================================================

    print("\n")
    print("=" * 70)
    print("PHASE 9.2 - GENERAL RAGA RECOGNITION & VALIDATION")
    print("=" * 70)


    raga_recognition_data = (
        recognize_and_validate_raga()
    )


    # ========================================================
    # PHASE 10 - LYRICS / WORD-TO-NOTATION
    # ========================================================
    # PHASE 10 IS CURRENTLY ON HOLD.
    # Lyrics acquisition and word-to-notation alignment are
    # intentionally skipped. The rest of the musical-analysis
    # pipeline continues normally.
    # ========================================================

    print("\n")
    print("=" * 70)
    print("PHASE 10 - LYRICS / WORD-TO-NOTATION")
    print("=" * 70)

    print("Status: ON HOLD")
    print("Lyrics acquisition and word-to-notation alignment skipped.")
    print("Continuing to Phase 11.")


    # ========================================================
    # PHASE 11.1 - MUSICAL ELEMENT ANALYSIS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("PHASE 11.1 - MUSICAL ELEMENT ANALYSIS")
    print("=" * 70)

    phase_11_1_data = analyze_musical_elements(
        processed_audio,
        sample_rate
    )

    print("\nPHASE 11.1 completed successfully.")


    # ========================================================
    # PHASE 11.2 - ADVANCED MELODY & RHYTHM ANALYSIS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("PHASE 11.2 - ADVANCED MELODY & RHYTHM ANALYSIS")
    print("=" * 70)

    phase_11_2_data = run_phase_11_2()

    print("\nPHASE 11.2 completed successfully.")


    # ========================================================
    # PHASE 12.1 - MUSICAL PATTERN & MOTIF ANALYSIS
    # ========================================================
    # Phase 12.1 is file-driven. It reads the completed Phase 11.2
    # melodic phrase output and writes repeated_melodic_motifs.csv
    # plus its own motif reports and graphs.
    # ========================================================

    phase_12_1_data = run_phase_12_1_from_main()

    print("\nPHASE 12.1 completed successfully.")


    # ========================================================
    # PHASE 12.2 - MOTIF TRANSFORMATION & VARIATION ANALYSIS
    # ========================================================
    # Phase 12.2 reads the Phase 12.1 swara-based motif output.
    # It does not require an audio path.
    # ========================================================

    phase_12_2_data = run_phase_12_2_from_main()

    print("\nPHASE 12.2 completed successfully.")


    # ========================================================
    # PHASE 12.3 - MOTIF DEVELOPMENT & STRUCTURAL ROLE ANALYSIS
    # ========================================================
    # Phase 12.3 reads the completed Phase 12.1 and Phase 12.2
    # outputs. It does not require an audio path.
    # ========================================================

    phase_12_3_data = run_phase_12_3_from_main()

    print("\nPHASE 12.3 completed successfully.")


    # ========================================================
    # ========================================================
    # PHASE 13.1 - LYRICS ACQUISITION
    # ========================================================
    phase_13_1_data = run_phase_13_1_from_main()
    print("\nPHASE 13.1 completed successfully.")
    
    # ========================================================
    # PHASE 13.2 - LYRICS + MUSICAL NOTATION ALIGNMENT
    # ========================================================
    phase_13_2_data = run_phase_13_2_from_main()
    print("\nPHASE 13.2 completed successfully.")
    
    # ========================================================
    # PHASE 13.3 - FULL SONG ALIGNMENT / DISPLAY
    # ========================================================
    phase_13_3_data = run_phase_13_3_from_main()
    print("\nPHASE 13.3 completed successfully.")


    # FINAL SUMMARY
    # ========================================================

    print("\n")
    print("=" * 80)
    print("AI MUSIC ANALYSIS COMPLETED")
    print("=" * 80)


    print("\nGenerated analysis components:")

    print(
        "  [✓] Audio loading"
    )

    print(
        "  [✓] Audio preprocessing"
    )

    print(
        "  [✓] Waveform analysis"
    )

    print(
        "  [✓] FFT spectrum analysis"
    )

    print(
        "  [✓] Spectrogram analysis"
    )

    print(
        "  [✓] Melody/pitch extraction pipeline (HPSS bypassed)"
    )

    print(
        "  [✓] Fundamental frequency (F0)"
    )

    print(
        "  [✓] Pitch cleaning"
    )

    print(
        "  [✓] Pitch smoothing"
    )

    print(
        "  [✓] Final melody contour"
    )

    print(
        "  [✓] Western note conversion"
    )

    print(
        "  [✓] Note segmentation"
    )

    print(
        "  [✓] Note quantization"
    )

    print(
        "  [✓] Note duration analysis"
    )

    print(
        "  [✓] Tonic estimation"
    )

    print(
        "  [✓] Indian swara mapping"
    )

    print(
        "  [✓] Melodic phrase analysis"
    )

    print(
        "  [✓] Western + Indian notation alignment"
    )

    print(
        "  [✓] BPM / rhythm analysis"
    )

    print(
        "  [✓] Beat / downbeat analysis"
    )

    print(
        "  [✓] Tempo stability analysis"
    )

    print(
        "  [✓] Rhythm pattern analysis"
    )

    print(
        "  [✓] Taal estimation"
    )

    print(
        "  [✓] Initial raga estimation"
    )

    print(
        "  [✓] Raga recognition & validation"
    )

    print(
        "  [✓] Phase 13.1 lyrics acquisition"
    )

    print(
        "  [✓] Phase 13.2 lyrics + Indian swara + Western notation alignment"
    )

    print(
        "  [✓] Phase 13.3 complete-song alignment + notation display"
    )

    print(
        "  [✓] Musical element analysis (Phase 11.1)"
    )

    print(
        "  [✓] Advanced melody & rhythm analysis (Phase 11.2)"
    )

    print(
        "  [✓] Musical pattern & motif analysis (Phase 12.1)"
    )

    print(
        "  [✓] Motif transformation & variation analysis (Phase 12.2)"
    )

    print(
        "  [✓] Motif development & structural role analysis (Phase 12.3)"
    )


    print("\n")
    print("=" * 80)
    print("OUTPUT FOLDERS")
    print("=" * 80)

    print(
        "\nData files:"
    )

    print(
        "  data/output/"
    )

    print(
        "\nGraphs:"
    )

    print(
        "  graphs/"
    )


    print("\n")
    print("=" * 80)
    print("ALL AVAILABLE PHASES COMPLETED")
    print("=" * 80)

    print(
        "\nProject execution finished."
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print("\n")
        print("=" * 70)
        print("PROGRAM INTERRUPTED BY USER")
        print("=" * 70)

    except Exception as error:

        print("\n")
        print("=" * 70)
        print("ANALYSIS STOPPED BECAUSE OF AN ERROR")
        print("=" * 70)

        print(
            f"\nError type    : "
            f"{type(error).__name__}"
        )

        print(
            f"Error message : "
            f"{error}"
        )

        print("\nTraceback:")

        traceback.print_exc()

        print("\n")
        print("=" * 70)
        print("Please send the complete error output.")
        print("=" * 70)
