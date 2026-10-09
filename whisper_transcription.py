
import argparse
import difflib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import (
    OpenAI,
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
)
from pydub import AudioSegment

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL = "whisper-1"            # whisper-1 returns segment timestamps (verbose_json)
AUDIO_DIR = Path("audio")
CHUNK_DIR = Path("chunks")
OUTPUT_DIR = Path("outputs")
SUPPORTED = {".mp3", ".wav", ".m4a", ".mp4", ".mpeg", ".mpga", ".webm", ".ogg", ".flac"}
MAX_FILE_MB = 25               # Whisper API upload limit
BASIC_CLIP_SECONDS = 60        # Step 3/4/5 use only the first minute
PRICE_PER_MINUTE = 0.006       # USD, whisper-1 (check openai.com/api/pricing)
MAX_RETRIES = 3

# Default context prompt for the guided approach. Replace it with names and
# technical terms that actually occur in YOUR recording (or use --prompt).
DEFAULT_PROMPT = (
    "Recorded interview in American English. The speaker talks about local "
    "traditions, places and tools. Use correct spelling for names and "
    "regional terms, and include punctuation."
)

minutes_transcribed = 0.0      # running total for cost tracking


# ---------------------------------------------------------------------------
# Step 1: Environment
# ---------------------------------------------------------------------------
def setup_environment() -> OpenAI:
    """Check API key + ffmpeg, create folders, return the OpenAI client."""
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("✗ OPENAI_API_KEY is not set. Put it in a .env file (never commit it).")
    if shutil.which("ffmpeg") is None:
        sys.exit("✗ ffmpeg not found. Install it:  brew install ffmpeg  "
                 "(or: conda install -c conda-forge ffmpeg)")
    for d in (AUDIO_DIR, CHUNK_DIR, OUTPUT_DIR):
        d.mkdir(exist_ok=True)
    client = OpenAI(timeout=120)
    print(f"✓ OpenAI client ready (model: {MODEL}, key loaded from environment)")
    print(f"✓ Folders ready: {AUDIO_DIR}/, {CHUNK_DIR}/, {OUTPUT_DIR}/")
    return client


# ---------------------------------------------------------------------------
# Step 2: Audio file
# ---------------------------------------------------------------------------
def find_audio_file(path_arg: str | None) -> Path:
    """Use --audio if given, otherwise the first supported file in audio/."""
    if path_arg:
        path = Path(path_arg)
    else:
        files = sorted(p for p in AUDIO_DIR.iterdir() if p.suffix.lower() in SUPPORTED)
        if not files:
            sys.exit(f"✗ No audio found. Put an .mp3/.wav file into {AUDIO_DIR}/")
        path = files[0]
    if not path.exists():
        sys.exit(f"✗ Audio file not found: {path}")
    if path.suffix.lower() not in SUPPORTED:
        sys.exit(f"✗ Unsupported format {path.suffix}. Use one of {sorted(SUPPORTED)}")
    return path


def audio_info(path: Path) -> AudioSegment:
    """Load the file and print its properties (Step 2 checkpoint)."""
    audio = AudioSegment.from_file(path)
    size_mb = path.stat().st_size / 1_048_576
    print(f"\nAudio file : {path}")
    print(f"Duration   : {fmt_clock(len(audio) / 1000)} ({len(audio) / 1000:.1f} s)")
    print(f"Channels   : {audio.channels}   Sample rate: {audio.frame_rate} Hz")
    print(f"File size  : {size_mb:.2f} MB  "
          f"({'OK' if size_mb <= MAX_FILE_MB else 'too large – chunking required'})")
    return audio


# ---------------------------------------------------------------------------
# Core API call (used by Steps 3–7)
# ---------------------------------------------------------------------------
def transcribe(client: OpenAI, path: Path, prompt: str | None = None,
               language: str | None = "en") -> dict:
    """
    Transcribe one file. Returns {"text", "language", "duration", "segments"}.
    verbose_json gives us segment-level start/end times (= timestamps).
    Rate limits / timeouts are retried with exponential backoff.
    """
    global minutes_transcribed
    size_mb = path.stat().st_size / 1_048_576
    if size_mb > MAX_FILE_MB:
        raise ValueError(f"{path.name} is {size_mb:.1f} MB (> {MAX_FILE_MB} MB) – chunk it first")

    params = {"model": MODEL, "response_format": "verbose_json",
              "timestamp_granularities": ["segment"]}
    if prompt:
        params["prompt"] = prompt
    if language:
        params["language"] = language

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with open(path, "rb") as f:
                r = client.audio.transcriptions.create(file=f, **params)
            segments = [{"start": round(s.start, 2), "end": round(s.end, 2),
                         "text": s.text.strip()} for s in (r.segments or [])]
            minutes_transcribed += (r.duration or 0) / 60
            return {"text": r.text.strip(), "language": r.language,
                    "duration": r.duration, "segments": segments}
        except (AuthenticationError, BadRequestError):
            raise                      # retrying will not help
        except (RateLimitError, APITimeoutError, APIConnectionError) as e:
            if attempt == MAX_RETRIES:
                raise
            wait = 2 ** attempt
            print(f"   ⚠ {type(e).__name__} – retrying in {wait}s")
            time.sleep(wait)


# ---------------------------------------------------------------------------
# Step 3: Basic transcription (no chunking) of a short clip
# ---------------------------------------------------------------------------
def make_clip(audio: AudioSegment, seconds: int, name: str) -> Path:
    clip_path = CHUNK_DIR / name
    audio[: seconds * 1000].export(clip_path, format="mp3", bitrate="64k")
    return clip_path


def basic_transcription(client: OpenAI, audio: AudioSegment) -> dict:
    print("\n" + "=" * 60 + "\nSTEP 3 – BASIC TRANSCRIPTION (first "
          f"{BASIC_CLIP_SECONDS} s, no prompt, no chunking)\n" + "=" * 60)
    clip = make_clip(audio, BASIC_CLIP_SECONDS, "basic_clip.mp3")
    result = transcribe(client, clip)
    print(f"Language detected : {result['language']}")
    print(f"Clip duration     : {result['duration']:.1f} s")
    print(f"Segments          : {len(result['segments'])}")
    print(f"\nText:\n{result['text']}\n")
    print("First segments with timestamps:")
    for s in result["segments"][:5]:
        print(f"  [{fmt_clock(s['start'])} → {fmt_clock(s['end'])}] {s['text']}")
    save_json(result, OUTPUT_DIR / "basic_transcription.json")
    return result


# ---------------------------------------------------------------------------
# Steps 4 + 5: Guided (prompt) vs. unguided (no prompt) + comparison
# ---------------------------------------------------------------------------
def words(text: str) -> list[str]:
    return re.findall(r"[\w'’-]+", text.lower())


def compare_transcriptions(client: OpenAI, audio: AudioSegment, prompt: str) -> dict:
    print("\n" + "=" * 60 + "\nSTEPS 4 + 5 – PROMPTED vs. UNPROMPTED\n" + "=" * 60)
    clip = make_clip(audio, BASIC_CLIP_SECONDS, "compare_clip.mp3")

    print(f"Prompt used:\n  \"{prompt}\"\n")
    guided = transcribe(client, clip, prompt=prompt)
    unguided = transcribe(client, clip, prompt=None)

    gw, uw = words(guided["text"]), words(unguided["text"])
    similarity = difflib.SequenceMatcher(None, uw, gw).ratio()

    # word-level differences: what changed between the two versions
    changes = []
    sm = difflib.SequenceMatcher(None, uw, gw)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            changes.append({"type": tag,
                            "unprompted": " ".join(uw[i1:i2]),
                            "prompted": " ".join(gw[j1:j2])})

    # which prompt terms (words of 4+ letters) show up in each version
    prompt_terms = sorted({w for w in words(prompt) if len(w) >= 4})
    in_guided = [t for t in prompt_terms if t in gw]
    in_unguided = [t for t in prompt_terms if t in uw]

    comparison = {
        "prompt": prompt,
        "prompted": guided, "unprompted": unguided,
        "metrics": {
            "words_prompted": len(gw), "words_unprompted": len(uw),
            "word_similarity": round(similarity, 3),
            "punctuation_prompted": sum(guided["text"].count(c) for c in ".,?!"),
            "punctuation_unprompted": sum(unguided["text"].count(c) for c in ".,?!"),
            "prompt_terms_in_prompted": in_guided,
            "prompt_terms_in_unprompted": in_unguided,
        },
        "differences": changes,
    }

    print(f"UNPROMPTED ({len(uw)} words):\n{unguided['text']}\n")
    print(f"PROMPTED   ({len(gw)} words):\n{guided['text']}\n")
    print(f"Word similarity : {similarity:.1%}")
    print(f"Punctuation     : unprompted {comparison['metrics']['punctuation_unprompted']}"
          f"  |  prompted {comparison['metrics']['punctuation_prompted']}")
    print(f"Differences     : {len(changes)}")
    for c in changes[:15]:
        print(f"  - '{c['unprompted']}'  →  '{c['prompted']}'")

    save_json(comparison, OUTPUT_DIR / "comparison_prompted_vs_unprompted.json")
    with open(OUTPUT_DIR / "comparison_prompted_vs_unprompted.txt", "w", encoding="utf-8") as f:
        f.write(f"PROMPT:\n{prompt}\n\nUNPROMPTED:\n{unguided['text']}\n\n"
                f"PROMPTED:\n{guided['text']}\n\nWord similarity: {similarity:.1%}\n"
                "Differences (unprompted → prompted):\n")
        f.writelines(f"- '{c['unprompted']}' → '{c['prompted']}'\n" for c in changes)
    return comparison


# ---------------------------------------------------------------------------
# Step 6: Chunking
# ---------------------------------------------------------------------------
def chunk_audio(audio: AudioSegment, chunk_minutes: float, stem: str) -> list[dict]:
    """
    Split audio into fixed-length chunks and export each as a compact mp3.
    Every chunk remembers its start offset – needed to fix timestamps later.
    Chunks are mono/16 kHz/64 kbps, so even a 20-minute chunk stays far below 25 MB.
    """
    print("\n" + "=" * 60 + f"\nSTEP 6 – CHUNKING ({chunk_minutes:g}-minute chunks)\n" + "=" * 60)
    chunk_ms = int(chunk_minutes * 60 * 1000)
    audio = audio.set_channels(1).set_frame_rate(16000)
    chunks = []
    for i, start_ms in enumerate(range(0, len(audio), chunk_ms)):
        piece = audio[start_ms:start_ms + chunk_ms]
        path = CHUNK_DIR / f"{stem}_chunk_{i:03d}.mp3"
        piece.export(path, format="mp3", bitrate="64k")
        chunks.append({"index": i, "path": path, "offset": start_ms / 1000,
                       "duration": len(piece) / 1000})
        print(f"  chunk {i:03d}: {fmt_clock(start_ms / 1000)} → "
              f"{fmt_clock((start_ms + len(piece)) / 1000)}  "
              f"({path.stat().st_size / 1024:.0f} KB)")
    print(f"✓ {len(chunks)} chunks created in {CHUNK_DIR}/")
    return chunks


# ---------------------------------------------------------------------------
# Step 7: Transcribe chunks + shift timestamps
# ---------------------------------------------------------------------------
def transcribe_chunks(client: OpenAI, chunks: list[dict], prompt: str | None) -> dict:
    """
    Transcribe each chunk, then add the chunk's offset to every segment so the
    timestamps refer to the ORIGINAL recording, not to the chunk.
    The end of the previous chunk's text is passed as context in the prompt,
    which helps Whisper keep spelling and style consistent across chunks.
    """
    print("\n" + "=" * 60 + "\nSTEP 7 – TRANSCRIBING CHUNKS WITH TIMESTAMPS\n" + "=" * 60)
    all_segments, texts, previous_text = [], [], ""
    for c in chunks:
        chunk_prompt = " ".join(p for p in (prompt, previous_text[-300:]) if p) or None
        print(f"  [{c['index'] + 1}/{len(chunks)}] {c['path'].name} "
              f"(offset {fmt_clock(c['offset'])})...", end=" ", flush=True)
        try:
            r = transcribe(client, c["path"], prompt=chunk_prompt)
        except AuthenticationError:
            raise
        except Exception as e:                 # one bad chunk must not stop the rest
            print(f"✗ {type(e).__name__}: {e}")
            texts.append(f"[chunk {c['index']} failed]")
            continue
        for s in r["segments"]:
            all_segments.append({"start": round(s["start"] + c["offset"], 2),
                                 "end": round(s["end"] + c["offset"], 2),
                                 "text": s["text"], "chunk": c["index"]})
        texts.append(r["text"])
        previous_text = r["text"]
        print(f"✓ {len(r['segments'])} segments")
        time.sleep(0.5)                        # gentle on rate limits

    result = {"text": " ".join(texts), "segments": all_segments,
              "chunks": len(chunks), "prompt": prompt}
    print(f"\n✓ Combined transcript: {len(all_segments)} segments, "
          f"{len(words(result['text']))} words")
    if all_segments:
        print("Timestamp check (first segment of each chunk, shifted by its offset):")
        for c in chunks:
            first = next((s for s in all_segments if s["chunk"] == c["index"]), None)
            if first:
                print(f"  chunk {c['index']:03d} starts at {fmt_clock(c['offset'])} → "
                      f"first segment [{fmt_clock(first['start'])}] {first['text'][:55]}")
    return result


# ---------------------------------------------------------------------------
# Step 8: Export (TXT, SRT, JSON)
# ---------------------------------------------------------------------------
def fmt_clock(seconds: float) -> str:
    """Seconds → HH:MM:SS for the human-readable transcript."""
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def fmt_srt(seconds: float) -> str:
    """Seconds → HH:MM:SS,mmm (SRT uses a comma before milliseconds)."""
    ms = int(round(seconds * 1000))
    return f"{ms // 3_600_000:02d}:{ms % 3_600_000 // 60_000:02d}:" \
           f"{ms % 60_000 // 1000:02d},{ms % 1000:03d}"


def export_txt(result: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for s in result["segments"]:
            f.write(f"[{fmt_clock(s['start'])}] {s['text']}\n")


def export_srt(result: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for i, s in enumerate(result["segments"], start=1):
            f.write(f"{i}\n{fmt_srt(s['start'])} --> {fmt_srt(s['end'])}\n{s['text']}\n\n")


def save_json(data: dict, path: Path) -> None:
    def default(o):
        return str(o)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=default)


def export_all(result: dict, stem: str) -> list[Path]:
    print("\n" + "=" * 60 + "\nSTEP 8 – EXPORT\n" + "=" * 60)
    paths = [OUTPUT_DIR / f"{stem}_transcript.txt",
             OUTPUT_DIR / f"{stem}_transcript.srt",
             OUTPUT_DIR / f"{stem}_transcript.json"]
    export_txt(result, paths[0])
    export_srt(result, paths[1])
    save_json(result, paths[2])
    for p in paths:
        print(f"  ✓ {p}  ({p.stat().st_size / 1024:.1f} KB)")
    print("\nPreview (TXT):")
    print("".join(open(paths[0], encoding="utf-8").readlines()[:5]))
    return paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Whisper STT lab")
    ap.add_argument("--audio", help="path to an audio file (default: first file in audio/)")
    ap.add_argument("--chunk-minutes", type=float, default=10.0,
                    help="chunk length in minutes (default 10; use 1 for short demo audio)")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT, help="context prompt for guided mode")
    ap.add_argument("--step", choices=["all", "info", "basic", "compare", "chunk"],
                    default="all")
    args = ap.parse_args()

    client = setup_environment()
    path = find_audio_file(args.audio)
    audio = audio_info(path)
    stem = path.stem

    try:
        if args.step in ("all", "basic"):
            basic_transcription(client, audio)
        if args.step in ("all", "compare"):
            compare_transcriptions(client, audio, args.prompt)
        if args.step in ("all", "chunk"):
            chunks = chunk_audio(audio, args.chunk_minutes, stem)
            result = transcribe_chunks(client, chunks, args.prompt)
            export_all(result, stem)
    except AuthenticationError:
        sys.exit("✗ Invalid API key – check OPENAI_API_KEY in .env")

    if minutes_transcribed:
        print(f"\nAudio minutes sent to Whisper: {minutes_transcribed:.1f} "
              f"→ estimated cost ${minutes_transcribed * PRICE_PER_MINUTE:.3f}")


if __name__ == "__main__":
    main()
