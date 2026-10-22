"""Forced alignment of known lyrics against the recording.

Unlike ASR, this does not guess the words: it Viterbi-aligns the official character sequence to the
acoustic emissions of a wav2vec2 CTC model, so every line gets a frame-level (20 ms) span even where
recognition would hallucinate or fail.

Usage: python forced-align.py <audio> <lyrics> <out.json>
"""

import json
import re
import subprocess
import sys

import numpy as np
import torch
import torchaudio

SAMPLE_RATE = 16000
# wav2vec2-base strides: 5x conv with 320-sample total hop.
FRAME_SAMPLES = 320
WINDOW_SECONDS = 20.0
OVERLAP_SECONDS = 1.0


def decode_mono(path: str) -> torch.Tensor:
    """Decode with FFmpeg so PyAV version drift cannot break the pipeline."""
    command = [
        "ffmpeg", "-v", "error", "-nostdin", "-i", path,
        "-map", "0:a:0", "-vn", "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "pipe:1",
    ]
    raw = subprocess.run(command, check=True, stdout=subprocess.PIPE).stdout
    return torch.from_numpy(np.frombuffer(raw, dtype=np.float32).copy())


def lyric_lines(path: str) -> list[str]:
    """Read the file and drop the title/credit banner, mirroring the TypeScript extractor."""
    raw = [line.strip() for line in open(path, encoding="utf-8")]
    start = 0
    while start < len(raw):
        line = raw[start]
        if not line:
            start += 1
            continue
        if re.search(r";\s*$", line) or re.search(r"\b(song|lyrics?|music)\s+by\b", line, re.I):
            start += 1
            continue
        if len(line.split()) <= 3 and re.fullmatch(r"[A-Za-z][A-Za-z'.\- ]*", line):
            start += 1
            continue
        break
    return [line for line in raw[start:] if line]


def normalize_line(text: str) -> str:
    """Map a lyric line to the model's alphabet (uppercase letters, '|' between words)."""
    return "|".join(re.findall(r"[a-z]+", text.lower())).upper()


def emit(model: torch.nn.Module, waveform: torch.Tensor) -> torch.Tensor:
    """Return frame log-probabilities laid out at absolute frame positions.

    Windows overlap, so each chunk is written at the frame index its starting sample maps to and any
    already-written frames are skipped. Assuming a fixed overlap count instead drifts, and a drift
    here compresses the whole timeline.
    """
    total_frames = int(round(waveform.shape[0] / FRAME_SAMPLES))
    step = int(WINDOW_SECONDS * SAMPLE_RATE)
    overlap = int(OVERLAP_SECONDS * SAMPLE_RATE)
    emission = None
    written = 0
    for offset in range(0, waveform.shape[0], step - overlap):
        chunk = waveform[offset:offset + step]
        if chunk.shape[0] < FRAME_SAMPLES * 4:
            continue
        with torch.inference_mode():
            logits, _ = model(chunk.unsqueeze(0))
        log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)
        if emission is None:
            emission = torch.full((total_frames, log_probs.shape[1]), float(log_probs.min()) - 1.0)
        target = int(round(offset / FRAME_SAMPLES))
        skip = max(0, written - target)
        piece = log_probs[skip:]
        start = target + skip
        end = min(total_frames, start + piece.shape[0])
        if end > start:
            emission[start:end] = piece[:end - start]
            written = end
    if emission is None:
        raise SystemExit("no audio frames produced")
    return emission


def trim_silence(waveform: torch.Tensor) -> torch.Tensor:
    """Drop leading/trailing silence.

    CTC alignment happily spends lyric characters on silence. A quiet intro or outro then pulls the
    whole timeline toward it, which shows up as an every-line offset rather than a local error.
    """
    frame = int(0.1 * SAMPLE_RATE)
    frames = waveform.shape[0] // frame
    if frames == 0:
        return waveform
    rms = waveform[:frames * frame].reshape(frames, frame).pow(2).mean(dim=1).sqrt()
    threshold = max(0.005, float(rms.median()) * 0.12)
    loud = torch.nonzero(rms > threshold).flatten()
    if loud.numel() == 0:
        return waveform
    start = max(0, int(loud[0]) * frame - frame * 2)
    end = min(waveform.shape[0], (int(loud[-1]) + 1) * frame + frame * 2)
    return waveform[start:end]


def main() -> None:
    audio_path, lyrics_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    lines = lyric_lines(lyrics_path)
    if not lines:
        raise SystemExit("no lyric lines after metadata stripping")

    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().eval()
    # The 960h bundle exposes its alphabet as labels; index 0 is the CTC blank.
    dictionary = {label: index for index, label in enumerate(bundle.get_labels())}

    waveform = trim_silence(decode_mono(audio_path))
    emission = emit(model, waveform).unsqueeze(0)
    frame_seconds = FRAME_SAMPLES / SAMPLE_RATE
    duration = waveform.shape[0] / SAMPLE_RATE
    print(f"emission frames={emission.shape[1]} expected={duration / frame_seconds:.0f} duration={duration:.2f}s")

    targets, owner = [], []
    for index, line in enumerate(lines):
        for char in normalize_line(line):
            symbol = dictionary.get(char)
            if symbol is None:
                continue
            targets.append(symbol)
            owner.append(index)
    if not targets:
        raise SystemExit("no alignable characters in the lyrics")
    target = torch.tensor([targets], dtype=torch.int32)
    input_lengths = torch.tensor([emission.shape[1]], dtype=torch.int32)
    target_lengths = torch.tensor([target.shape[1]], dtype=torch.int32)
    token_spans, token_scores = torchaudio.functional.forced_align(
        emission, target, input_lengths, target_lengths, blank=0)
    merged = torchaudio.functional.merge_tokens(token_spans[0], token_scores[0], blank=0)

    per_line: dict[int, dict] = {}
    for span, score, line_index in zip(merged, token_scores[0].tolist(), owner):
        entry = per_line.setdefault(line_index, {"start": span.start, "end": span.end, "scores": []})
        entry["start"] = min(entry["start"], span.start)
        entry["end"] = max(entry["end"], span.end)
        entry["scores"].append(float(score))

    timed = []
    for index, line in enumerate(lines):
        entry = per_line.get(index)
        timed.append({
            "index": index,
            "text": line,
            "start": None if entry is None else round(entry["start"] * frame_seconds, 3),
            "end": None if entry is None else round(entry["end"] * frame_seconds, 3),
            "confidence": 0.0 if entry is None else round(float(np.mean(entry["scores"])), 4),
        })

    duration = waveform.shape[0] / SAMPLE_RATE
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump({"frameSeconds": frame_seconds, "duration": round(duration, 3), "lines": timed},
                  handle, ensure_ascii=False, indent=1)
    aligned = sum(1 for entry in timed if entry["start"] is not None)
    print(f"lines={len(lines)} aligned={aligned} duration={duration:.2f}s frame={frame_seconds:.3f}s")


if __name__ == "__main__":
    main()
