---
description: "Offline waveform/tempo analysis and chunked, provider-routed transcription tools."
kind: "package-reference"
---

# @deepseek-ai/dsh-experimental-audio-analysis

## Summary

Adds three Host tools that let an agent inspect a workspace audio file without sending it anywhere, then transcribe it through one explicitly chosen speech provider, then place supplied lyrics on a timeline. Analysis decodes locally with FFmpeg and never retains the whole recording: it streams 16 kHz mono PCM through a frame accumulator that keeps only per-frame peak/RMS values. Transcription decodes and submits bounded chunks, so a multi-minute file works under a provider's per-request cap.

## Use this package

Install this package together with its `cordis.patch.yml` bundle in the current profile. The bundle inserts the tools row and the Host rows the tools need: `subprocess-local`, `working-directory`, `fs-observation-policy`, and the `speech-to-text` registry with the local `sensevoice-local` recognizer. The Host must provide FFmpeg and ffprobe on `PATH`; they are not bundled. SenseVoice model weights are not downloaded until a transcription requests preparation.

Shipped tools:

- `audio_analyze` — exact duration, bounded peak/RMS waveform, and candidate BPM plus candidate beat grids. Local decode only; no model, network request, or external service. BPM is heuristic: speech, rubato, and syncopation make it unreliable, and half/double-time ambiguity is common, so candidates are reported rather than one asserted tempo. Beat times are one plausible grid, not confirmed downbeats.
- `audio_transcribe` — requires an exact `provider_id`, so no provider or cloud destination is chosen implicitly. Decodes chunk by chunk (`chunk_seconds`, 5–120, default 20) and submits each chunk separately, returning the joined text plus ordered segments carrying real `start`/`end` offsets. `prepare: true` downloads/verifies/loads the selected provider first. Boundaries are processing boundaries, not detected phrase boundaries. A cloud provider would upload the recording, so its use needs explicit confirmation.
- `lyrics_align` — matches supplied lyrics against timestamped segments (accepting `start`/`end` or `startSeconds`/`endSeconds`, so `audio_transcribe` output feeds it directly) and reports per-line confidence, `matched`/`review`/`unmatched` status, and optional character-weighted per-word times. With `duration_seconds` alone it only spreads lines evenly and says so; that is a placeholder, not detected timing.
- `lyrics_timestamp` — timestamps an official lyrics file against its audio. It reads the lyrics file (title/credit metadata dropped), transcribes the track in chunks as timing evidence, runs a monotonic word-level alignment over the whole lyric sequence, splits each recognition chunk among the lines it matched, and interpolates lines with no evidence. Every line reports `source: 'transcript'` (anchored) or `interpolated`/`extrapolated` (a guess), so measured and estimated timing stay separable. It also returns a ready-to-save LRC document.

Input files are capped at 64 MiB and 15 minutes of audio. Each temporary decoded file is removed after the call.

## Validated results on audio.m4a

Measured through these tools on the provided `audio.m4a` (AAC in M4A, 44.1 kHz stereo):

- Duration **212.323 s** (3:32.32); plugin decodes it to 16 kHz mono.
- Waveform: 4247 frames at 0.05 s hops, summarized to the requested point count.
- Leading BPM candidate **133.33** (novelty-flux autocorrelation 0.687), then **66.67** (0.421), **63.16** (0.298), **120** (0.06). Leading period ≈ 0.45 s. These are candidates, not a confirmed tempo.
- Transcription ran through the local SenseVoiceSmall INT8 provider with `chunk_seconds: 20`, producing 11 chunks (last 12.323 s) and 4 non-empty segments at 0–20 s, 20–40 s, 40–60 s and 100–120 s; silent chunks were omitted rather than padded.
- The song is not intelligible speech for a speech recognizer, so the transcript is visibly noisy and partially wrong. Treat it as a rough draft to verify by listening, never as known lyrics.

Sample artifacts from that run are written to the user's workspace as `audio-analysis-output/audio-analysis.json` (analysis, transcription and alignment together) and `audio-analysis-output/transcript-as-lyrics.lrc` (the aligned lines with a header stating the model and the chunk resolution). Reproduce them with:

```sh
node --import tsx packages/experimental/audio-analysis/scripts/validate-pipeline.ts \
  <audio-file> <absolute-model-dir> <output-dir> [chunkSeconds]
```

## Official-lyrics timestamping

`lyrics_timestamp` closes the gap that `audio_transcribe` + `lyrics_align` cannot: it aligns the whole lyric sequence at once instead of matching line by line against a lossy transcript.

Run against the recorded `lyrics.txt` (92 sung lines after dropping the title, `Song by`, blank and `Mili` credit) and `audio.m4a`:

```sh
node --import tsx packages/experimental/audio-analysis/scripts/timestamp-lyrics.ts \
  <audio-file> <lyrics-file> <output-dir> <absolute-model-dir> [chunkSeconds] [vadThreshold]
```

Result at `chunkSeconds: 6`, `vadThreshold: 0.2`: **35 of 92 lines anchored by recognition**, 57 interpolated/extrapolated, written to `audio-analysis-output/lyrics-timed.lrc` and `audio-analysis-output/lyrics-timed.json`.

### Calibration: why the first pass mis-timed a section

A listened review found the block "Execution, execution" starting at 2:18 instead of ~2:27. Mapping the recording at 3 s resolution around that point gave the ground truth:

| Window | Recognition |
| --- | --- |
| 132 +3s | `arguments.` (last line of the section) |
| 135…144 +3s | `The.` `The.` `The.` (instrumental) |
| 147 +3s | `You.` |
| 150 +3s | `Ex.` (singing resumes) |
| 153 +3s | no speech |
| 156 +3s | `The.` |

Three defects were involved, and all three were fixed rather than patching the one timestamp:

1. **Function words counted as full evidence.** `The.` scored a perfect 1.0 match against `the` and pinned the next line at 2:18, cramming nine lines into four seconds. Matches now carry an informativeness weight; a line needs at least one informative match to become an anchor, and a lone stop word is treated as no evidence at all.
2. **Chunk spans were trusted as line spans.** A drifted chunk could stretch a two-word line over 12 s. Spans are now clamped to a plausible sung length.
3. **Interpolation ignored breaks.** Lines were spread over the raw span, so an instrumental passage swallowed them. Placement now subtracts measured breaks, keeps short weak stretches (recognizer failure inside continuous singing), and estimates the resume point as the midpoint of the final weak chunk — unbiased, and converging as chunks shrink.

With those changes the block lands at **2:27.00**. Across the region a reviewer had already accepted, mean drift from the previous output is ~1 s (largest 4.5 s), and spot checks of the movers agreed with the audio to within a chunk.

Finer 3 s chunks make anchors weaker (25 anchored vs 35) because shorter windows truncate words, so 6 s remains the default and covers the break with the midpoint estimate.

### Accuracy ceiling: why 0.2 s is not reachable with these engines

Line timing is recognition-anchored, not forced alignment, so it cannot be certified to 0.2 s. Three engines were tried on this recording:

| Engine | Result |
| --- | --- |
| SenseVoice chunks (shipped) | Accurate text, chunk-quantised timing. Gets the reviewed anchors right (132.00 s and 147.00 s) but bounds starts to the chunk grid. |
| Faster-Whisper `small.en`, word timestamps | Precise word spans, but it hallucinated unrelated lyrics over this vocal ("If I can live, I can hear a sound of endless fragments" where the real lines are 65–68), so its words cannot be trusted as anchors. |
| torchaudio wav2vec2-960h CTC forced alignment | Aligned all 92 lines at 20 ms frames, but it is trained on read speech: 10 s global drift, 22 s after trailing-silence trimming. Unstable, so not shipped. |

`lyrics_timestamp` therefore accepts external evidence via `evidence_path` (a `{"words":[…]}` JSON file). A singing-capable forced aligner — WhisperX with a large model, or a dedicated lyric aligner — can drive the same tool without changing it, and word-level evidence is what would tighten starts to word boundaries. Until then, treat anchored lines as accurate to about one recognition chunk (±3 s at the 6 s default) and interpolated lines as estimates.

`scripts/whisper-words.py` (Faster-Whisper word/segment JSON) and `scripts/forced-align.py` (torchaudio CTC alignment) are the optional evidence generators behind that experiment; neither is required by the plugin.

A lower VAD threshold is what makes this work at all: at the stock `0.5`, the local recognizer discarded most of the song and reported under a third of the lines; at `0.2` it recovered verse and chorus. The bundle therefore ships `vadThreshold: 0.2` with matching `minSpeechSeconds`/`minSilenceSeconds`. Raise it again if the same profile is used for dictation, where permissive VAD admits background noise.

`scripts/validate-real.ts` exercises analyze plus chunking with a stub provider, and `scripts/validate-transcribe.ts` transcribes only. Those scripts are development aids outside the package's TypeScript project; the shipped entry point is `lib/index.js`.

## Model Experience

### audio_analyze

#### What the model sees

One `audio_path` in the workspace plus optional `minimum_bpm`, `maximum_bpm`, `waveform_points` (integer 1–1500, default 600) and `bpm_step_cents`. The result reports `durationSeconds`, a bounded `waveform` of `{time, peak, rms}` points, the leading `estimatedBpm`, and up to five `bpmCandidates`, each carrying the full beat grid for its period.

#### Token effect

One tool schema per mounted Agent and one result per call. The waveform costs one small object per requested point and every candidate repeats its whole beat grid — about 2400 numbers, near ten thousand tokens, for a 3.5-minute track at the default 600 points; pass a small `waveform_points` when only duration or tempo is needed. Audio bytes never enter model messages.

#### KV Cache effect

The four tool schemas stay static for the mount lifetime, and a result appends to the conversation without rewriting the prompt prefix.

### audio_transcribe and lyrics_timestamp

#### What the model sees

`audio_transcribe` requires an exact `provider_id` plus `audio_path`, with optional `language`, `chunk_seconds` (5–120, default 20) and `prepare`, and returns joined text with ordered `start`/`end` segments. `lyrics_timestamp` takes `audio_path` plus `lyrics_path` or inline lyrics and returns per-line times whose `source` is `transcript`, `interpolated` or `extrapolated`, with a ready-to-save LRC document.

#### Token effect

One schema and one result per call: a transcript scales with recognized phrases, a lyric timeline with lyric lines plus one `source` marker each. Recordings stay on the Host, so only recognized text enters model messages.

#### KV Cache effect

Both schemas stay static for the mount lifetime; results append as new conversation content, so the cached prompt prefix survives each call.

### lyrics_align

#### What the model sees

`lyrics` plus either `transcript_segments` in the shape `audio_transcribe` returns or `duration_seconds`; optional `bpm` and `word_level` annotate times without inventing data. Each line reports `startSeconds`, `endSeconds`, `confidence`, `source`, and optional character-weighted word times.

#### Token effect

One schema and one result per call, sized by the number of lyric lines. Passing `duration_seconds` alone returns an evenly spread placeholder, which costs a full result without providing measured timing.

#### KV Cache effect

The schema stays static for the mount lifetime and results append without rewriting the prompt cache prefix.

## Known Limitations and Deferred Work

- Decoding depends on Host FFmpeg and ffprobe; both are required, and the plugin fails with one clear message when either is missing.
- Transcription quality is the chosen provider's; SenseVoice is a speech model and handles sung vocals poorly. Chunk boundaries can split words, and adjacent chunks may duplicate or drop short phrases.
- Segment times are chunk-resolution. There is no word-level acoustic forced alignment; `lyrics_align` word times are character-weight estimates within a matched segment.
- The local recognizer rejects one request above 4 MiB of WAV, which is why chunking exists; a 120 s chunk is 3.84 MB and stays under it.
- `audio_transcribe` refuses to run against a provider that is not ready unless `prepare: true` is passed, so an unattended call cannot start a multi-hundred-megabyte download by accident.
- `audio_analyze` reports every candidate's complete beat grid, so a long track spends far more tokens on beat times than on the waveform; a bounded or on-demand grid is deferred.
