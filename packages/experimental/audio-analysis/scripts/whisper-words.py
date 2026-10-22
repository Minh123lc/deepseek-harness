"""Word-level timestamps for lyric alignment.

Faster-Whisper is not a dependency of the plugin: it is an optional, offline evidence generator.
Point it at an audio file and it writes JSON `{"words": [{start, end, text}], "segments": [...]}`
that `lyrics_timestamp` can consume as external timing evidence (see scripts/timestamp-from-words.ts).

Usage: python whisper-words.py <audio> <out.json> [model] [language]
"""

import json
import subprocess
import sys

import numpy as np
from faster_whisper import WhisperModel

audio = sys.argv[1]
out = sys.argv[2]
model_size = sys.argv[3] if len(sys.argv) > 3 else "small.en"
language = sys.argv[4] if len(sys.argv) > 4 else "en"


def decode_16k_mono(path: str) -> np.ndarray:
    """Decode with FFmpeg instead of PyAV: the bundled PyAV build can disagree with faster-whisper's API."""
    command = [
        "ffmpeg", "-v", "error", "-nostdin", "-i", path,
        "-map", "0:a:0", "-vn", "-f", "f32le", "-ac", "1", "-ar", "16000", "pipe:1",
    ]
    raw = subprocess.run(command, check=True, stdout=subprocess.PIPE).stdout
    return np.frombuffer(raw, dtype=np.float32)


samples = decode_16k_mono(audio)

# CPU int8 keeps this usable without a GPU; word timestamps are what the aligner needs.
model = WhisperModel(model_size, device="cpu", compute_type="int8")
segments_iter, info = model.transcribe(
    samples,
    word_timestamps=True,
    vad_filter=False,          # singing is not speech; VAD would discard most of it
    language=language,
    beam_size=5,
    condition_on_previous_text=False,
)

words = []
segments = []
for segment in segments_iter:
    segments.append({"start": round(segment.start, 3), "end": round(segment.end, 3), "text": segment.text.strip()})
    for word in segment.words or []:
        text = word.word.strip()
        if not text:
            continue
        words.append({"start": round(word.start, 3), "end": round(word.end, 3), "text": text})

with open(out, "w", encoding="utf-8") as handle:
    json.dump({"audio": audio, "model": model_size, "language": info.language,
               "duration": round(info.duration, 3), "words": words, "segments": segments}, handle, ensure_ascii=False)

print(f"model={model_size} language={info.language} duration={info.duration:.2f}s words={len(words)} segments={len(segments)}")
