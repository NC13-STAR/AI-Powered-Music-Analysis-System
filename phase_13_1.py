import os
import re
import traceback
import tkinter as tk
from tkinter import filedialog


# ============================================================
# PHASE 13.1
# LYRICS ACQUISITION & PREPARATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "data",
    "output",
    "phase_13",
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

LYRICS_OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "phase_13_lyrics.txt",
)

METADATA_OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "phase_13_lyrics_metadata.txt",
)


# ============================================================
# FILE SELECTOR
# ============================================================

def select_lyrics_file():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    path = filedialog.askopenfilename(
        title="Select Lyrics Text File",
        filetypes=[
            ("Text files", "*.txt"),
            ("Lyrics files", "*.txt;*.lrc;*.srt"),
            ("All files", "*.*"),
        ],
    )

    root.destroy()

    if not path:
        return None

    return os.path.abspath(path)


# ============================================================
# READ FILE
# ============================================================

def read_text_file(path):

    encodings = [
        "utf-8",
        "utf-8-sig",
        "utf-16",
        "cp1252",
        "latin-1",
    ]

    for encoding in encodings:
        try:
            with open(
                path,
                "r",
                encoding=encoding,
            ) as f:
                return f.read()
        except UnicodeDecodeError:
            continue

    raise ValueError(
        "Could not decode the selected lyrics file."
    )


# ============================================================
# NORMALIZE
# ============================================================

def normalize_line(line):

    line = line.replace("\ufeff", "")
    line = line.replace("\r", "")
    line = line.strip()

    return line


# ============================================================
# REMOVE LRC TIMESTAMPS
# ============================================================

def remove_lrc_timestamps(line):

    pattern = re.compile(
        r"\[\d{1,2}:\d{2}(?:[.:]\d{1,3})?\]"
    )

    return pattern.sub("", line).strip()


# ============================================================
# REMOVE SRT TIMESTAMPS
# ============================================================

def is_srt_timestamp(line):

    pattern = re.compile(
        r"^\d{2}:\d{2}:\d{2},\d{3}"
        r"\s*-->\s*"
        r"\d{2}:\d{2}:\d{2},\d{3}"
    )

    return bool(pattern.match(line.strip()))


# ============================================================
# DETECT METADATA HEADER
# ============================================================

def looks_like_metadata(line):

    text = line.strip()

    if not text:
        return True

    lowered = text.lower()

    metadata_patterns = [
        r"^title\s*:",
        r"^song\s*:",
        r"^artist\s*:",
        r"^singer\s*:",
        r"^album\s*:",
        r"^lyrics\s*:",
        r"^composer\s*:",
        r"^music\s*:",
        r"^singer\s*-",
        r"^artist\s*-",
    ]

    for pattern in metadata_patterns:
        if re.match(pattern, lowered):
            return True

    return False


# ============================================================
# IDENTIFY TITLE / ARTIST HEADER
# ============================================================

def detect_header(lines):

    """
    Handles common files such as:

        Vande Mataram
        Pt. Chandrashekhar Vaze

        Vande Mataram
        Sujalam suphalam...

    The first two lines are treated as metadata only when
    followed by a blank line and then actual lyrics.
    """

    cleaned = [
        normalize_line(line)
        for line in lines
    ]

    while cleaned and not cleaned[0]:
        cleaned.pop(0)

    if not cleaned:
        return [], []

    header = []

    # Explicit metadata format
    index = 0

    while index < len(cleaned):

        line = cleaned[index]

        if looks_like_metadata(line):
            header.append(line)
            index += 1
        else:
            break

    if header:
        return header, cleaned[index:]

    # Common two-line format:
    #
    # Song title
    # Singer
    #
    # blank
    #
    # lyrics

    if len(cleaned) >= 4:

        first = cleaned[0]
        second = cleaned[1]

        # Look for a blank line after first two lines
        if (
            first
            and second
            and not cleaned[2]
        ):
            return (
                [first, second],
                cleaned[3:],
            )

    return [], cleaned


# ============================================================
# CLEAN LYRICS
# ============================================================

def extract_lyrics(raw_text):

    raw_lines = raw_text.splitlines()

    # Normalize lines
    lines = [
        normalize_line(line)
        for line in raw_lines
    ]

    # Remove SRT numbering/timestamps
    processed = []

    for line in lines:

        if not line:
            processed.append("")
            continue

        if line.isdigit():
            continue

        if is_srt_timestamp(line):
            continue

        line = remove_lrc_timestamps(line)

        if line:
            processed.append(line)

    header, body = detect_header(processed)

    # Remove blank lines around header
    while body and not body[0]:
        body.pop(0)

    # Remove empty trailing lines
    while body and not body[-1]:
        body.pop()

    # Remove obvious metadata from beginning
    while body and looks_like_metadata(body[0]):
        body.pop(0)

    lyrics_lines = []

    for line in body:

        line = normalize_line(line)

        if not line:
            continue

        # Never allow obvious metadata into lyrics
        if looks_like_metadata(line):
            continue

        lyrics_lines.append(line)

    return header, lyrics_lines


# ============================================================
# VALIDATE LYRICS
# ============================================================

def validate_lyrics(lyrics_lines):

    if not lyrics_lines:
        raise ValueError(
            "No actual lyric lines were detected."
        )

    text = "\n".join(lyrics_lines)

    if len(text.strip()) < 3:
        raise ValueError(
            "The detected lyrics are too short."
        )

    return text


# ============================================================
# SAVE
# ============================================================

def save_results(
    lyrics,
    source_file,
    metadata,
):

    with open(
        LYRICS_OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(lyrics)
        f.write("\n")

    with open(
        METADATA_OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "PHASE 13.1 - LYRICS METADATA\n"
        )
        f.write("=" * 70)
        f.write("\n\n")

        f.write(
            f"Source file : {source_file}\n"
        )

        for key, value in metadata.items():

            f.write(
                f"{key} : {value}\n"
            )

    return LYRICS_OUTPUT_FILE


# ============================================================
# DISPLAY
# ============================================================

def display_preview(
    lyrics_lines,
):

    print()
    print("=" * 80)
    print("EXTRACTED LYRICS")
    print("=" * 80)
    print()

    for index, line in enumerate(
        lyrics_lines,
        start=1,
    ):

        print(
            f"{index:03d}. {line}"
        )

    print()
    print("=" * 80)


# ============================================================
# MAIN PHASE
# ============================================================

def run_phase_13_1():

    print()
    print("=" * 80)
    print(
        "PHASE 13.1 - LYRICS ACQUISITION "
        "& PREPARATION"
    )
    print("=" * 80)

    print()
    print(
        "Select the TXT/LRC/SRT file containing the lyrics."
    )

    print(
        "The first song-title / singer header will NOT "
        "be treated as lyrics."
    )

    print()
    print(
        "Opening lyrics file selector..."
    )

    source_file = select_lyrics_file()

    if not source_file:

        print()
        print(
            "No lyrics file selected."
        )

        return False

    print()
    print("Selected lyrics file:")
    print(source_file)

    raw_text = read_text_file(
        source_file
    )

    header, lyrics_lines = extract_lyrics(
        raw_text
    )

    lyrics = validate_lyrics(
        lyrics_lines
    )

    metadata = {
        "Header lines ignored": len(header),
        "Lyric lines": len(lyrics_lines),
        "Lyric characters": len(lyrics),
    }

    save_results(
        lyrics,
        source_file,
        metadata,
    )

    print()

    if header:

        print(
            "Metadata/header ignored:"
        )

        for line in header:
            print(
                f"  [IGNORED] {line}"
            )

    display_preview(
        lyrics_lines
    )

    print()
    print("=" * 80)
    print("PHASE 13.1 COMPLETED SUCCESSFULLY")
    print("=" * 80)

    print()
    print(
        "Actual lyrics saved to:"
    )

    print(
        LYRICS_OUTPUT_FILE
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Only the extracted lyric lines will be "
        "passed to Phase 13.2."
    )

    return True


# ============================================================
# ENTRY
# ============================================================

def main():

    try:

        success = run_phase_13_1()

        if not success:
            return

    except KeyboardInterrupt:

        print()
        print(
            "Phase 13.1 interrupted by user."
        )

    except Exception as error:

        print()
        print("=" * 80)
        print(
            "PHASE 13.1 STOPPED BECAUSE OF AN ERROR"
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
        traceback.print_exc()


if __name__ == "__main__":
    main()