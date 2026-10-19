"""
PHASE 13.2
LYRICS -> COMPLETE MUSICAL NOTATION ALIGNMENT

IMPORTANT DESIGN RULES
----------------------

1. Lyrics come ONLY from:
       data/output/phase_13/phase_13_lyrics.txt

2. The following are NEVER treated as lyrics:
       - song title
       - artist name
       - album name
       - filename
       - audio metadata

3. Musical timing comes primarily from:
       note_segments.csv

4. Indian swaras are loaded independently from:
       indian_swara_data.csv

5. Western notes can come from:
       note_segments.csv
       OR western_note_data.csv

6. If the project exists in both Documents and OneDrive,
   this program searches both locations.

7. The COMPLETE lyric word stream is aligned.
   The preview only shows the first 15 words.

8. This is a deterministic first-pass alignment.
   It does NOT claim that every syllable equals one note.
"""

from __future__ import annotations

import json
import math
import re
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# ============================================================================
# PROJECT PATH DISCOVERY
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPT_PROJECT_DIR = SCRIPT_DIR.parent

HOME = Path.home()

PROJECT_NAME = "AI Powered Music Analysis System 1"

CANDIDATE_PROJECT_ROOTS = [
    SCRIPT_PROJECT_DIR,
    HOME / "OneDrive" / "Documents" / PROJECT_NAME,
    HOME / "Documents" / PROJECT_NAME,
]


def unique_existing_paths(paths: List[Path]) -> List[Path]:
    result = []
    seen = set()

    for path in paths:
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path

        key = str(resolved).lower()

        if key in seen:
            continue

        seen.add(key)

        if resolved.exists():
            result.append(resolved)

    return result


PROJECT_ROOTS = unique_existing_paths(
    CANDIDATE_PROJECT_ROOTS
)


# ============================================================================
# OUTPUT LOCATIONS
# ============================================================================

PRIMARY_PROJECT_ROOT = (
    PROJECT_ROOTS[0]
    if PROJECT_ROOTS
    else SCRIPT_PROJECT_DIR
)

PRIMARY_OUTPUT_DIR = (
    PRIMARY_PROJECT_ROOT /
    "data" /
    "output"
)

PHASE_13_DIR = (
    PRIMARY_OUTPUT_DIR /
    "phase_13"
)

PHASE_13_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================================
# FILE SEARCH
# ============================================================================

def find_file_anywhere(
    filename: str,
    preferred_roots: Optional[List[Path]] = None,
) -> Optional[Path]:

    roots = preferred_roots or PROJECT_ROOTS

    # ------------------------------------------------------------------
    # First: normal expected paths
    # ------------------------------------------------------------------

    for root in roots:
        direct_candidates = [
            root / "data" / "output" / filename,
            root / "data" / "output" / "phase_13" / filename,
        ]

        for candidate in direct_candidates:
            if candidate.is_file():
                return candidate

    # ------------------------------------------------------------------
    # Second: recursive search
    # ------------------------------------------------------------------

    for root in roots:

        if not root.exists():
            continue

        try:
            matches = list(
                root.rglob(filename)
            )
        except Exception:
            matches = []

        if matches:
            # Prefer files under data/output.
            preferred = [
                item
                for item in matches
                if "data" in item.parts
                and "output" in item.parts
            ]

            if preferred:
                return preferred[0]

            return matches[0]

    return None


# ============================================================================
# RESOLVE ALL MUSICAL FILES
# ============================================================================

def resolve_musical_files() -> Dict[str, Optional[Path]]:

    names = {
        "note_segments": "note_segments.csv",
        "quantized_notes": "quantized_notes.csv",
        "indian_swara": "indian_swara_data.csv",
        "western_indian_alignment":
            "western_indian_notation_alignment.csv",
        "western_note_data": "western_note_data.csv",
        "melody_phrase":
            "melody_phrase_note_data.csv",
        "note_duration":
            "note_duration_analysis.csv",
    }

    resolved = {}

    for key, filename in names.items():
        resolved[key] = find_file_anywhere(
            filename
        )

    return resolved


# ============================================================================
# PHASE 13 OUTPUT FILES
# ============================================================================

LYRICS_FILE = (
    PHASE_13_DIR /
    "phase_13_lyrics.txt"
)

WORD_OUTPUT = (
    PHASE_13_DIR /
    "phase_13_2_word_notation.csv"
)

LINE_OUTPUT = (
    PHASE_13_DIR /
    "phase_13_2_line_notation.csv"
)

JSON_OUTPUT = (
    PHASE_13_DIR /
    "phase_13_2_notation.json"
)

SUMMARY_OUTPUT = (
    PHASE_13_DIR /
    "phase_13_2_summary.txt"
)


# ============================================================================
# TEXT HELPERS
# ============================================================================

def clean_text(value: Any) -> str:

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    return str(value).strip()


def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:

    try:
        number = float(value)

        if math.isfinite(number):
            return number

    except Exception:
        pass

    return default


def normalize_column_name(
    name: Any
) -> str:

    text = clean_text(name).lower()

    text = re.sub(
        r"[^a-z0-9]+",
        "_",
        text
    )

    return text.strip("_")


def find_column(
    frame: pd.DataFrame,
    keywords: List[str],
) -> Optional[str]:

    if frame.empty:
        return None

    normalized = {
        normalize_column_name(column): column
        for column in frame.columns
    }

    # Exact match
    for keyword in keywords:

        key = normalize_column_name(
            keyword
        )

        if key in normalized:
            return normalized[key]

    # Partial match
    for normalized_name, original in normalized.items():

        for keyword in keywords:

            key = normalize_column_name(
                keyword
            )

            if key in normalized_name:
                return original

    return None


# ============================================================================
# INDIAN SWARA NORMALIZATION
# ============================================================================

SWARA_ALIASES = {
    "s": "Sa",
    "sa": "Sa",

    "r": "Re",
    "re": "Re",
    "ri": "Re",

    "g": "Ga",
    "ga": "Ga",

    "m": "Ma",
    "ma": "Ma",

    "p": "Pa",
    "pa": "Pa",

    "d": "Dha",
    "dha": "Dha",

    "n": "Ni",
    "ni": "Ni",
}


def normalize_swara(
    value: Any
) -> str:

    text = clean_text(value)

    if not text:
        return ""

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    lower = text.lower()

    modifier = ""

    if lower.startswith("komal "):

        modifier = "Komal "
        text = text[6:].strip()

    elif lower.startswith("tivra "):

        modifier = "Tivra "
        text = text[6:].strip()

    elif lower.startswith("shuddha "):

        modifier = "Shuddha "
        text = text[8:].strip()

    key = text.lower()

    if key in SWARA_ALIASES:

        return (
            modifier +
            SWARA_ALIASES[key]
        )

    return modifier + text


# ============================================================================
# CSV READER
# ============================================================================

def read_csv_safe(
    path: Optional[Path]
) -> pd.DataFrame:

    if path is None:
        return pd.DataFrame()

    if not path.is_file():
        return pd.DataFrame()

    for encoding in (
        "utf-8-sig",
        "utf-8",
        "cp1252",
        "latin-1",
    ):

        try:

            return pd.read_csv(
                path,
                encoding=encoding
            )

        except Exception:
            continue

    return pd.DataFrame()


# ============================================================================
# LYRIC METADATA FILTERING
# ============================================================================

def looks_like_metadata_header(
    line: str,
    line_number: int,
) -> bool:

    text = clean_text(line)

    if not text:
        return False

    if line_number > 3:
        return False

    lower = text.lower()

    metadata_prefixes = (
        "title:",
        "song:",
        "artist:",
        "singer:",
        "album:",
        "lyrics:",
        "composer:",
        "filename:",
        "duration:",
    )

    if lower.startswith(
        metadata_prefixes
    ):
        return True

    # Very conservative title-artist detection.
    if (
        line_number == 1
        and " - " in text
    ):

        parts = [
            item.strip()
            for item in text.split(" - ")
        ]

        if len(parts) == 2:

            right = parts[1].lower()

            artist_words = (
                "pt.",
                "pandit",
                "smt.",
                "shri",
                "dr.",
                "artist",
                "singer",
            )

            if any(
                item in right
                for item in artist_words
            ):
                return True

            # English title + person name
            if len(parts[1].split()) >= 2:
                return True

    return False


# ============================================================================
# LYRIC CLEANING
# ============================================================================

def clean_lyric_line(
    line: str
) -> str:

    text = clean_text(line)

    if not text:
        return ""

    # Remove LRC timestamps
    text = re.sub(
        r"\[\d{1,2}:\d{2}(?:[.:]\d{1,3})?\]",
        " ",
        text
    )

    # Remove subtitle timing
    text = re.sub(
        r"\d{2}:\d{2}:\d{2}[,.]\d{3}"
        r"\s*-->\s*"
        r"\d{2}:\d{2}:\d{2}[,.]\d{3}",
        " ",
        text
    )

    # Remove HTML-like tags
    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================================
# LOAD ACTUAL LYRICS
# ============================================================================

def load_actual_lyrics(
) -> List[Dict[str, Any]]:

    print()
    print("=" * 80)
    print("READING PHASE 13.1 LYRICS")
    print("=" * 80)

    print()
    print(
        "Lyrics file:"
    )
    print(LYRICS_FILE)

    if not LYRICS_FILE.is_file():

        raise FileNotFoundError(
            "Phase 13.1 lyrics file was not found:\n"
            f"{LYRICS_FILE}\n\n"
            "Run Phase 13.1 first."
        )

    raw_text = None

    for encoding in (
        "utf-8-sig",
        "utf-8",
        "utf-16",
        "cp1252",
        "latin-1",
    ):

        try:

            raw_text = LYRICS_FILE.read_text(
                encoding=encoding
            )

            break

        except UnicodeDecodeError:
            continue

    if raw_text is None:

        raise ValueError(
            "Could not decode phase_13_lyrics.txt."
        )

    raw_text = raw_text.replace(
        "\r\n",
        "\n"
    )

    raw_text = raw_text.replace(
        "\r",
        "\n"
    )

    lyric_lines = []

    for raw_index, raw_line in enumerate(
        raw_text.splitlines(),
        start=1
    ):

        line = clean_lyric_line(
            raw_line
        )

        if not line:
            continue

        if looks_like_metadata_header(
            line,
            raw_index
        ):

            print(
                f"[metadata ignored] {line}"
            )

            continue

        lyric_lines.append({
            "line_number": len(lyric_lines) + 1,
            "text": line,
        })

    if not lyric_lines:

        raise ValueError(
            "No actual lyrics were found "
            "after metadata filtering."
        )

    word_records = []

    for line_record in lyric_lines:

        words = re.findall(
            r"\S+",
            line_record["text"],
            flags=re.UNICODE
        )

        for word_index, word in enumerate(words):

            word = word.strip()

            if not word:
                continue

            word_records.append({
                "line_number":
                    line_record["line_number"],

                "word_index":
                    word_index,

                "text":
                    word,
            })

    print()
    print(
        f"Lyrics lines detected : "
        f"{len(lyric_lines)}"
    )

    print(
        f"Lyrics words detected : "
        f"{len(word_records)}"
    )

    print(
        f"Lyrics characters     : "
        f"{sum(len(x['text']) for x in lyric_lines)}"
    )

    print()
    print(
        "First lyric lines used by Phase 13.2:"
    )

    for item in lyric_lines[:10]:

        print(
            f"  {item['line_number']}: "
            f"{item['text']}"
        )

    return word_records


# ============================================================================
# DISCOVER MUSICAL FILES
# ============================================================================

def discover_musical_files():

    print()
    print("=" * 80)
    print("DISCOVERING MUSICAL NOTE DATA")
    print("=" * 80)

    resolved = resolve_musical_files()

    for key, path in resolved.items():

        if path is None:
            continue

        frame = read_csv_safe(path)

        if frame.empty:
            continue

        print()
        print(
            f"[✓] {path.name}"
        )

        print(
            f"    Records: {len(frame)}"
        )

        print(
            f"    Location: {path}"
        )

    return resolved


def resolve_musical_files():

    filenames = {
        "note_segments":
            "note_segments.csv",

        "quantized_notes":
            "quantized_notes.csv",

        "indian_swara":
            "indian_swara_data.csv",

        "western_alignment":
            "western_indian_notation_alignment.csv",

        "western_note_data":
            "western_note_data.csv",

        "melody_phrase":
            "melody_phrase_note_data.csv",

        "note_duration":
            "note_duration_analysis.csv",
    }

    result = {}

    for key, filename in filenames.items():

        result[key] = find_file_anywhere(
            filename
        )

    return result


# ============================================================================
# FILE DISCOVERY
# ============================================================================

def find_file_anywhere(
    filename: str
) -> Optional[Path]:

    for root in PROJECT_ROOTS:

        if not root.exists():
            continue

        # Normal locations first
        candidates = [
            root / "data" / "output" / filename,
            root / "data" / "output" /
            "phase_13" / filename,
        ]

        for candidate in candidates:

            if candidate.is_file():
                return candidate

        # Recursive fallback
        try:

            matches = list(
                root.rglob(filename)
            )

        except Exception:
            matches = []

        if matches:

            # Prefer data/output locations.
            preferred = [
                item
                for item in matches
                if "data" in item.parts
                and "output" in item.parts
            ]

            if preferred:
                return preferred[0]

            return matches[0]

    return None


# ============================================================================
# PRIMARY MUSICAL TIMELINE
# ============================================================================

def load_primary_timeline(
    paths: Dict[str, Optional[Path]]
) -> pd.DataFrame:

    primary_path = paths.get(
        "note_segments"
    )

    if primary_path is not None:

        frame = read_csv_safe(
            primary_path
        )

        if not frame.empty:

            print()
            print(
                "Primary musical source:"
            )
            print(
                f"  {primary_path}"
            )

            return frame

    # Fallback to quantized notes.
    fallback_path = paths.get(
        "quantized_notes"
    )

    if fallback_path is not None:

        frame = read_csv_safe(
            fallback_path
        )

        if not frame.empty:

            print()
            print(
                "Fallback musical source:"
            )
            print(
                f"  {fallback_path}"
            )

            return frame

    # Fallback to western note data.
    western_path = paths.get(
        "western_note_data"
    )

    if western_path is not None:

        frame = read_csv_safe(
            western_path
        )

        if not frame.empty:

            print()
            print(
                "Fallback musical source:"
            )
            print(
                f"  {western_path}"
            )

            return frame

    raise FileNotFoundError(
        "\nNo usable musical-note source was found.\n\n"
        "Phase 13.2 searched both project locations:\n"
        + "\n".join(
            f"  {root}"
            for root in PROJECT_ROOTS
        )
        + "\n\nExpected one of:\n"
        "  note_segments.csv\n"
        "  quantized_notes.csv\n"
        "  western_note_data.csv\n\n"
        "Check that Phase 7 generated the musical CSV files."
    )


# ============================================================================
# BUILD BASE NOTE TIMELINE
# ============================================================================

def build_base_note_stream(
    frame: pd.DataFrame
) -> List[Dict[str, Any]]:

    print()
    print("=" * 80)
    print("BUILDING UNIFIED MUSICAL NOTE STREAM")
    print("=" * 80)

    if frame.empty:
        return []

    western_column = find_column(
        frame,
        [
            "western_note",
            "western",
            "note_name",
            "pitch_name",
            "note",
        ]
    )

    start_column = find_column(
        frame,
        [
            "start_time_seconds",
            "start",
            "start_time",
            "onset",
            "start_seconds",
        ]
    )

    end_column = find_column(
        frame,
        [
            "end_time_seconds",
            "end",
            "end_time",
            "offset",
            "end_seconds",
        ]
    )

    duration_column = find_column(
        frame,
        [
            "duration_seconds",
            "duration",
            "length",
        ]
    )

    print()
    print(
        f"Western note : "
        f"{western_column or '[not found]'}"
    )

    print(
        f"Start        : "
        f"{start_column or '[not found]'}"
    )

    print(
        f"End          : "
        f"{end_column or '[not found]'}"
    )

    print(
        f"Duration     : "
        f"{duration_column or '[not found]'}"
    )

    notes = []

    current_time = 0.0

    for row_number, (_, row) in enumerate(
        frame.iterrows()
    ):

        western = clean_text(
            row.get(
                western_column
            )
            if western_column
            else ""
        )

        if start_column:
            start = safe_float(
                row.get(
                    start_column
                ),
                current_time
            )
        else:
            start = current_time

        if duration_column:
            duration = safe_float(
                row.get(
                    duration_column
                ),
                0.5
            )
        else:
            duration = 0.5

        if duration <= 0:
            duration = 0.05

        if end_column:
            end = safe_float(
                row.get(
                    end_column
                ),
                start + duration
            )
        else:
            end = start + duration

        if end <= start:
            end = start + 0.05

        # Keep rows even if Western note is absent
        # because the Indian swara table may provide it.
        notes.append({
            "note_index":
                len(notes),

            "source_row":
                row_number,

            "swara":
                "",

            "western":
                western,

            "start":
                start,

            "end":
                end,

            "duration":
                end - start,
        })

        current_time = max(
            current_time,
            end
        )

    return notes


# ============================================================================
# ATTACH INDIAN SWARAS
# ============================================================================

def attach_indian_swaras(
    notes: List[Dict[str, Any]],
    swara_frame: pd.DataFrame,
) -> None:

    if not notes:
        return

    if swara_frame.empty:
        return

    swara_column = find_column(
        swara_frame,
        [
            "swara",
            "indian_swara",
            "indian",
            "swar",
            "svara",
        ]
    )

    if swara_column is None:

        print()
        print(
            "[!] indian_swara_data.csv found "
            "but no swara column was detected."
        )

        print(
            "    Columns found:"
        )

        print(
            "    "
            + ", ".join(
                str(x)
                for x in swara_frame.columns
            )
        )

        return

    swaras = []

    for _, row in swara_frame.iterrows():

        swara = normalize_swara(
            row.get(
                swara_column
            )
        )

        swaras.append(
            swara
        )

    # ------------------------------------------------------------
    # BEST CASE: same number of records
    # ------------------------------------------------------------

    if len(swaras) == len(notes):

        for index, swara in enumerate(
            swaras
        ):

            notes[index]["swara"] = swara

        return

    # ------------------------------------------------------------
    # If counts differ, proportional mapping.
    # ------------------------------------------------------------

    if len(swaras) > 0:

        for index, note in enumerate(notes):

            source_index = min(
                len(swaras) - 1,
                int(
                    index *
                    len(swaras) /
                    len(notes)
                )
            )

            note["swara"] = swaras[
                source_index
            ]


# ============================================================================
# ATTACH WESTERN NOTES FROM WESTERN NOTE DATA
# ============================================================================

def attach_western_fallback(
    notes: List[Dict[str, Any]],
    western_frame: pd.DataFrame,
) -> None:

    if not notes:
        return

    if western_frame.empty:
        return

    western_column = find_column(
        western_frame,
        [
            "western_note",
            "western",
            "note_name",
            "pitch_name",
            "note",
        ]
    )

    if western_column is None:
        return

    western_values = [
        clean_text(row.get(western_column))
        for _, row in western_frame.iterrows()
    ]

    western_values = [
        value
        for value in western_values
        if value
    ]

    if not western_values:
        return

    # Exact record count
    if len(western_values) == len(notes):

        for index, value in enumerate(
            western_values
        ):

            if not notes[index]["western"]:
                notes[index]["western"] = value

        return

    # Proportional fallback
    for index, note in enumerate(notes):

        if note["western"]:
            continue

        source_index = min(
            len(western_values) - 1,
            int(
                index *
                len(western_values) /
                len(notes)
            )
        )

        note["western"] = western_values[
            source_index
        ]


# ============================================================================
# BUILD COMPLETE MUSICAL STREAM
# ============================================================================

def build_complete_musical_stream(
    paths: Dict[str, Optional[Path]]
) -> List[Dict[str, Any]]:

    primary = load_primary_timeline(
        paths
    )

    notes = build_base_note_stream(
        primary
    )

    if not notes:
        raise ValueError(
            "The primary musical timeline "
            "contains no usable note records."
        )

    swara_path = paths.get(
        "indian_swara"
    )

    swara_frame = read_csv_safe(
        swara_path
    )

    if not swara_frame.empty:

        print()
        print(
            "Indian swara source:"
        )
        print(
            f"  {swara_path}"
        )

        print(
            f"  Records: "
            f"{len(swara_frame)}"
        )

        attach_indian_swaras(
            notes,
            swara_frame
        )

    western_path = paths.get(
        "western_note_data"
    )

    western_frame = read_csv_safe(
        western_path
    )

    attach_western_fallback(
        notes,
        western_frame
    )

    print()
    print(
        f"Unified musical notes: "
        f"{len(notes)}"
    )

    swara_count = sum(
        1
        for item in notes
        if item["swara"]
    )

    western_count = sum(
        1
        for item in notes
        if item["western"]
    )

    print(
        f"Indian swara records  : "
        f"{swara_count}"
    )

    print(
        f"Western note records  : "
        f"{western_count}"
    )

    if swara_count:

        print(
            "[✓] Indian swara "
            "information available."
        )

    else:

        print(
            "[!] Indian swara "
            "information unavailable."
        )

    return notes


# ============================================================================
# COMPLETE SONG ALIGNMENT
# ============================================================================

def align_complete_lyrics(
    lyrics: List[Dict[str, Any]],
    notes: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    print()
    print("=" * 80)
    print("ALIGNING COMPLETE LYRICS WITH MUSICAL NOTES")
    print("=" * 80)

    if not lyrics:
        return []

    if not notes:
        return []

    lyric_count = len(lyrics)
    note_count = len(notes)

    result = []

    for lyric_index, lyric in enumerate(
        lyrics
    ):

        # ------------------------------------------------------------
        # Proportional distribution across COMPLETE song.
        #
        # Example:
        # 92 lyric words
        # 329 musical notes
        #
        # Every lyric word gets its corresponding note range.
        # ------------------------------------------------------------

        start_ratio = (
            lyric_index /
            lyric_count
        )

        end_ratio = (
            (lyric_index + 1) /
            lyric_count
        )

        note_start = int(
            math.floor(
                start_ratio *
                note_count
            )
        )

        note_end = int(
            math.floor(
                end_ratio *
                note_count
            )
        )

        note_start = max(
            0,
            min(
                note_count - 1,
                note_start
            )
        )

        note_end = max(
            note_start + 1,
            min(
                note_count,
                note_end
            )
        )

        grouped_notes = notes[
            note_start:note_end
        ]

        if not grouped_notes:

            grouped_notes = [
                notes[
                    min(
                        lyric_index,
                        note_count - 1
                    )
                ]
            ]

        indian = [
            note["swara"]
            for note in grouped_notes
            if note["swara"]
        ]

        western = [
            note["western"]
            for note in grouped_notes
            if note["western"]
        ]

        first_note = grouped_notes[0]
        last_note = grouped_notes[-1]

        result.append({
            "lyric_index":
                lyric_index,

            "line_number":
                lyric["line_number"],

            "word_index":
                lyric["word_index"],

            "lyric":
                lyric["text"],

            "indian_swara":
                " ".join(indian)
                if indian
                else "-",

            "western_note":
                " ".join(western)
                if western
                else "-",

            "note_start_index":
                first_note["note_index"],

            "note_end_index":
                last_note["note_index"],

            "start":
                first_note["start"],

            "end":
                last_note["end"],

            "duration":
                last_note["end"]
                - first_note["start"],

            "notes_used":
                len(grouped_notes),
        })

    return result


# ============================================================================
# LINE OUTPUT
# ============================================================================

def build_line_output(
    aligned: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:

    grouped = {}

    for item in aligned:

        line_number = item[
            "line_number"
        ]

        if line_number not in grouped:

            grouped[line_number] = {
                "line_number":
                    line_number,

                "lyrics": [],

                "indian_swaras": [],

                "western_notes": [],
            }

        grouped[
            line_number
        ]["lyrics"].append(
            item["lyric"]
        )

        grouped[
            line_number
        ]["indian_swaras"].append(
            item["indian_swara"]
        )

        grouped[
            line_number
        ]["western_notes"].append(
            item["western_note"]
        )

    result = []

    for line_number in sorted(
        grouped
    ):

        item = grouped[
            line_number
        ]

        result.append({
            "line_number":
                line_number,

            "lyrics":
                " ".join(
                    item["lyrics"]
                ),

            "indian_swaras":
                " | ".join(
                    item["indian_swaras"]
                ),

            "western_notes":
                " | ".join(
                    item["western_notes"]
                ),
        })

    return result


# ============================================================================
# SAVE WORD CSV
# ============================================================================

def save_word_output(
    aligned: List[Dict[str, Any]]
):

    pd.DataFrame(
        aligned
    ).to_csv(
        WORD_OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )


# ============================================================================
# SAVE LINE CSV
# ============================================================================

def save_line_output(
    lines: List[Dict[str, Any]]
):

    pd.DataFrame(
        lines
    ).to_csv(
        LINE_OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )


# ============================================================================
# SAVE JSON
# ============================================================================

def save_json(
    lyrics: List[Dict[str, Any]],
    notes: List[Dict[str, Any]],
    aligned: List[Dict[str, Any]],
    lines: List[Dict[str, Any]],
):

    data = {
        "phase":
            "13.2",

        "description":
            "Complete lyrics to musical notation alignment",

        "lyric_source":
            str(LYRICS_FILE),

        "lyrics":
            lyrics,

        "musical_note_count":
            len(notes),

        "musical_notes":
            notes,

        "aligned_words":
            aligned,

        "aligned_lines":
            lines,
    }

    JSON_OUTPUT.write_text(
        json.dumps(
            data,
            indent=4,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )


# ============================================================================
# SAVE SUMMARY
# ============================================================================

def save_summary(
    lyrics: List[Dict[str, Any]],
    notes: List[Dict[str, Any]],
    aligned: List[Dict[str, Any]],
):

    swara_words = sum(
        1
        for item in aligned
        if item["indian_swara"] != "-"
    )

    summary = []

    summary.append(
        "PHASE 13.2 - SUMMARY"
    )

    summary.append(
        "=" * 70
    )

    summary.append("")

    summary.append(
        "LYRIC SOURCE"
    )

    summary.append(
        str(LYRICS_FILE)
    )

    summary.append("")

    summary.append(
        f"Lyric words loaded : "
        f"{len(lyrics)}"
    )

    summary.append(
        f"Musical notes used : "
        f"{len(notes)}"
    )

    summary.append(
        f"Words aligned      : "
        f"{len(aligned)}"
    )

    summary.append(
        f"Words with swaras  : "
        f"{swara_words}"
    )

    summary.append("")

    summary.append(
        "IMPORTANT:"
    )

    summary.append(
        "Song title and artist "
        "are excluded from lyrics."
    )

    summary.append(
        "The complete lyric word stream "
        "is aligned across the complete "
        "musical note stream."
    )

    summary.append(
        "Indian swaras are loaded "
        "independently from "
        "indian_swara_data.csv."
    )

    summary.append(
        "This is a deterministic "
        "first-pass alignment."
    )

    SUMMARY_OUTPUT.write_text(
        "\n".join(summary),
        encoding="utf-8"
    )


# ============================================================================
# PREVIEW
# ============================================================================

def print_preview(
    aligned: List[Dict[str, Any]]
):

    print()
    print("=" * 80)
    print(
        "LYRICS + INDIAN SWARA + WESTERN NOTATION PREVIEW"
    )
    print("=" * 80)

    print()

    if not aligned:

        print(
            "No lyric/notation alignment "
            "was generated."
        )

        return

    # Preview only.
    # The CSV/JSON contain the COMPLETE song.
    preview_count = min(
        15,
        len(aligned)
    )

    for item in aligned[
        :preview_count
    ]:

        print(
            f"{item['lyric_index'] + 1}. "
            f"{item['lyric']}"
        )

        print(
            f"   Indian : "
            f"{item['indian_swara']}"
        )

        print(
            f"   Western: "
            f"{item['western_note']}"
        )

        print()


# ============================================================================
# MAIN PHASE
# ============================================================================

def run_phase_13_2():

    print()
    print("=" * 80)
    print(
        "PHASE 13.2 - COMPLETE LYRICS "
        "TO MUSICAL NOTATION ALIGNMENT"
    )
    print("=" * 80)

    print()
    print(
        "Phase 13.2 uses ONLY the actual lyrics "
        "saved by Phase 13.1."
    )

    print()
    print(
        "Song title, artist name, filename and "
        "audio metadata are NOT used as lyrics."
    )

    # ------------------------------------------------------------------
    # 1. LYRICS
    # ------------------------------------------------------------------

    lyrics = load_actual_lyrics()

    # ------------------------------------------------------------------
    # 2. DISCOVER FILES
    # ------------------------------------------------------------------

    paths = discover_musical_files()

    # ------------------------------------------------------------------
    # 3. MUSICAL TIMELINE
    # ------------------------------------------------------------------

    notes = build_complete_musical_stream(
        paths
    )

    # ------------------------------------------------------------------
    # 4. COMPLETE ALIGNMENT
    # ------------------------------------------------------------------

    aligned = align_complete_lyrics(
        lyrics,
        notes
    )

    if not aligned:

        raise ValueError(
            "No lyric-to-note alignment "
            "could be generated."
        )

    # ------------------------------------------------------------------
    # 5. LINE DATA
    # ------------------------------------------------------------------

    line_data = build_line_output(
        aligned
    )

    # ------------------------------------------------------------------
    # 6. SAVE EVERYTHING
    # ------------------------------------------------------------------

    save_word_output(
        aligned
    )

    save_line_output(
        line_data
    )

    save_json(
        lyrics,
        notes,
        aligned,
        line_data
    )

    save_summary(
        lyrics,
        notes,
        aligned
    )

    # ------------------------------------------------------------------
    # 7. PREVIEW
    # ------------------------------------------------------------------

    print_preview(
        aligned
    )

    # ------------------------------------------------------------------
    # 8. REPORT
    # ------------------------------------------------------------------

    swara_words = sum(
        1
        for item in aligned
        if item["indian_swara"] != "-"
    )

    print()
    print("=" * 80)
    print(
        "PHASE 13.2 COMPLETED SUCCESSFULLY"
    )
    print("=" * 80)

    print()

    print(
        "[✓] Actual lyrics loaded "
        "from Phase 13.1"
    )

    print(
        f"    Lyrics words: "
        f"{len(lyrics)}"
    )

    print(
        f"[✓] Complete musical timeline loaded"
    )

    print(
        f"    Musical notes: "
        f"{len(notes)}"
    )

    print(
        f"[✓] Complete lyric stream aligned"
    )

    print(
        f"    Words aligned: "
        f"{len(aligned)}"
    )

    print(
        f"[✓] Indian swara information attached"
    )

    print(
        f"    Words with Indian swaras: "
        f"{swara_words}"
    )

    print()

    print(
        "Generated files:"
    )

    print(
        f"  {WORD_OUTPUT}"
    )

    print(
        f"  {LINE_OUTPUT}"
    )

    print(
        f"  {JSON_OUTPUT}"
    )

    print(
        f"  {SUMMARY_OUTPUT}"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "The preview above shows only the "
        "first 15 lyric words."
    )

    print(
        "The CSV and JSON files contain "
        "the COMPLETE song."
    )

    print()

    print("=" * 80)

    return aligned


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():

    try:

        run_phase_13_2()

    except KeyboardInterrupt:

        print()
        print("=" * 80)
        print(
            "PHASE 13.2 INTERRUPTED BY USER"
        )
        print("=" * 80)

    except Exception as error:

        print()
        print("=" * 80)
        print(
            "PHASE 13.2 STOPPED BECAUSE OF AN ERROR"
        )
        print("=" * 80)

        print()

        print(
            f"Error type    : "
            f"{type(error).__name__}"
        )

        print(
            f"Error message : "
            f"{error}"
        )

        print()

        print(
            "Traceback:"
        )

        traceback.print_exc()

        print()
        print("=" * 80)


if __name__ == "__main__":
    main()