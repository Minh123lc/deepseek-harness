/** Agent tools for offline waveform/tempo analysis and chunked, provider-routed transcription. */
import type { Context } from '@deepseek-ai/cordis'
import { defineTool } from '@deepseek-ai/dsh-tools'
import type { ToolExecution } from '@deepseek-ai/dsh-tools'
import type { JsonValue } from '@deepseek-ai/dsh-util-values'
import type {} from '@deepseek-ai/dsh-working-directory'
import type {} from '@deepseek-ai/dsh-fs'
import type {} from '@deepseek-ai/dsh-experimental-speech-to-text'
import type { SubprocessHandle, SubprocessOutcome, SubprocessRuntime } from '@deepseek-ai/dsh-subprocess'
import type { SpeechPreparationState, SpeechProviderId } from '@deepseek-ai/dsh-experimental-speech-to-text/types'
import type SpeechToText from '@deepseek-ai/dsh-experimental-speech-to-text'
import { FrameAccumulator, analyzeFrames, type AudioAnalysis } from '@deepseek-ai/dsh-experimental-speech-to-text/analysis'
import { alignLyricsToEvidence, extractLyricLines, toLrc, type TranscriptSegment } from './align.ts'
import { deadline, MAX_TIMER_DELAY_MS } from '@deepseek-ai/dsh-timeout'
import { PassThrough } from 'node:stream'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { mkdtemp, rm, writeFile } from 'node:fs/promises'

export { analyzeWav, analyzeFrames, FrameAccumulator } from '@deepseek-ai/dsh-experimental-speech-to-text/analysis'
export type {
  AudioAnalysis,
  AudioAnalysisOptions,
  BeatEstimate,
  FrameSummary,
  WaveformPoint,
} from '@deepseek-ai/dsh-experimental-speech-to-text/analysis'

export const name = 'experimental-audio-analysis'
export const inject = ['tools', 'fs', 'workingDirectory', 'subprocess']

const MAX_INPUT_BYTES = 64 * 1024 * 1024
const MAX_TOOL_DURATION_SECONDS = 15 * 60
const MAX_WAVEFORM_POINTS = 1500
const DEFAULT_WAVEFORM_POINTS = 600
/** Per-frame cap; 0.05 s frames over 15 minutes is 18000 frames. */
const MAX_ANALYSIS_FRAMES = 20_000
const MIN_BPM = 30
const MAX_BPM = 300
const PROCESS_GRACE_MS = 2_000
const AUDIO_PROCESS_TIMEOUT_MS = 120_000
const PROVIDER_PREPARE_TIMEOUT_MS = 30 * 60_000
/** The local recognizer refuses a WAV above 4 MiB; 120 s of 16 kHz mono PCM16 is 3.84 MB. */
const MAX_CHUNK_SECONDS = 120

// Type aliases, not interfaces: a JsonValue-returning tool result needs an implicit index signature.
type WaveformSample = { readonly time: number; readonly peak: number; readonly rms: number }
type AudioRange = { readonly start: number; readonly end: number; readonly text: string }

/** Resolve one audio file through the session filesystem, bounded by the plugin's input cap. */
async function readAudio(ctx: Context, exec: ToolExecution, audioPath: string): Promise<Uint8Array> {
  const agent = exec.agent
  if (agent === undefined) throw new Error('Audio tools require an Agent-backed workspace session')
  const cwd = await ctx.workingDirectory.ensure(agent, exec.signal)
  const target = await ctx.fs.resolve(audioPath, { cwd, signal: exec.signal })
  const info = await ctx.fs.stat(target, exec.signal)
  if (info === undefined) {
    ctx.emit('fs/observed', target, { kind: 'absent' }, exec)
    throw new Error(`Audio file was not found: ${target.displayPath}`)
  }
  if (info.type !== 'file') throw new Error(`Audio path is not a regular file: ${target.displayPath}`)
  ctx.emit('fs/observed', target, { kind: 'present', version: info.version }, exec)
  return await ctx.fs.readBytes(target, exec.signal, MAX_INPUT_BYTES)
}

/** Await one child process, converting deadline/caller cancellation into real termination. */
async function waitForProcess(handle: SubprocessHandle, signal: AbortSignal): Promise<SubprocessOutcome> {
  if (signal.aborted) {
    handle.terminate()
    throw signal.reason ?? new Error('Audio processing was cancelled')
  }
  const aborted = Promise.withResolvers<never>()
  const onAbort = (): void => {
    handle.terminate()
    aborted.reject(signal.reason ?? new Error('Audio processing was cancelled'))
  }
  signal.addEventListener('abort', onAbort, { once: true })
  try {
    return await Promise.race([handle.done, aborted.promise])
  } finally { signal.removeEventListener('abort', onAbort) }
}

function stderrOf(handle: SubprocessHandle): string {
  return (handle.collected.stderr?.readFrom(0).text ?? '').trim().slice(0, 1000)
}

interface AudioTools {
  readonly runtime: SubprocessRuntime
  readonly ffmpeg: string
  readonly ffprobe: string
}

/** Resolve FFmpeg/ffprobe once per call so a missing installation fails with one clear message. */
async function resolveTools(ctx: Context, signal: AbortSignal): Promise<AudioTools> {
  const runtime = ctx.get('subprocess')
  if (runtime === undefined) throw new Error('Audio decoding requires the Host subprocess service (`ctx.subprocess`)')
  const ffmpeg = await runtime.resolveExecutable('ffmpeg', undefined, signal).catch(() => undefined)
  if (ffmpeg === undefined) {
    throw new Error('FFmpeg is not available on the Harness Host. Install FFmpeg (including ffprobe) on the Host PATH.')
  }
  const ffprobe = await runtime.resolveExecutable('ffprobe', undefined, signal).catch(() => undefined)
  if (ffprobe === undefined) throw new Error('ffprobe is not available on the Harness Host. Install the full FFmpeg package.')
  return { runtime, ffmpeg, ffprobe }
}

/** Materialize the input in a private temp file, then always delete it. */
async function withSourceFile<T>(source: Uint8Array, body: (sources: { file: string; cwd: string }) => Promise<T>): Promise<T> {
  const cwd = await mkdtemp(join(tmpdir(), 'dsh-audio-'))
  const file = join(cwd, 'input-audio')
  try {
    await writeFile(file, source, { flag: 'wx' })
    return await body({ file, cwd })
  } finally {
    await rm(cwd, { recursive: true, force: true })
  }
}

async function probeDuration(tools: AudioTools, sourcePath: string, cwd: string, signal: AbortSignal): Promise<number> {
  const probe = tools.runtime.spawn({
    argv: [tools.ffprobe, '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', sourcePath],
    cwd,
    stdio: { stdin: 'ignore', stdout: { maxBytes: 4096 }, stderr: { maxBytes: 16_384 } },
    graceMs: PROCESS_GRACE_MS,
    signal,
  })
  const outcome = await waitForProcess(probe, signal)
  if (outcome.exitCode !== 0) throw new Error(`ffprobe could not read this audio file: ${stderrOf(probe)}`)
  const seconds = Number((probe.collected.stdout?.readFrom(0).text ?? '').trim())
  if (!Number.isFinite(seconds) || seconds <= 0) throw new Error('ffprobe reported an invalid audio duration')
  if (seconds > MAX_TOOL_DURATION_SECONDS) {
    throw new Error(`Audio duration ${seconds.toFixed(1)}s exceeds the ${MAX_TOOL_DURATION_SECONDS}s limit`)
  }
  return seconds
}

interface PcmStreamRequest {
  readonly sourcePath: string
  readonly cwd: string
  readonly signal: AbortSignal
  readonly onData: (bytes: Uint8Array) => void
  readonly startSeconds?: number
  readonly durationSeconds?: number
}

/** Stream 16 kHz mono PCM into a sink without ever retaining the whole recording. */
async function streamPcm(tools: AudioTools, request: PcmStreamRequest): Promise<void> {
  const { sourcePath, cwd, signal, onData, startSeconds, durationSeconds } = request
  // Output-side -ss trims sample-accurately; the tool reports the requested offsets to the model.
  const trim = startSeconds === undefined ? [] : ['-ss', startSeconds.toFixed(3), '-t', (durationSeconds ?? 0).toFixed(3)]
  const conversion = tools.runtime.spawn({
    argv: [tools.ffmpeg, '-v', 'error', '-nostdin', '-i', sourcePath, ...trim,
      '-map', '0:a:0', '-vn', '-f', 's16le', '-ac', '1', '-ar', '16000', 'pipe:1'],
    cwd,
    stdio: { stdin: 'ignore', stdout: 'pipe', stderr: { maxBytes: 16_384 } },
    graceMs: PROCESS_GRACE_MS,
    signal,
  })
  if (conversion.stdout === undefined) throw new Error('The subprocess provider did not expose FFmpeg stdout')
  const sink = new PassThrough()
  conversion.stdout.pipe(sink)
  sink.on('data', (chunk: Uint8Array) => { onData(Uint8Array.from(chunk)) })
  const [outcome] = await Promise.all([
    waitForProcess(conversion, signal),
    new Promise<void>((resolve, reject) => {
      sink.on('end', resolve)
      sink.on('error', reject)
      sink.resume()
    }),
  ])
  if (outcome.exitCode !== 0) throw new Error(`ffmpeg could not decode this audio file: ${stderrOf(conversion)}`)
}

/** Decode one bounded range into a canonical 16 kHz mono PCM16 WAV for the speech provider. */
async function decodeRange(
  tools: AudioTools,
  sourcePath: string,
  cwd: string,
  startSeconds: number,
  durationSeconds: number,
  signal: AbortSignal,
): Promise<Uint8Array> {
  const chunks: Uint8Array[] = []
  let total = 0
  await streamPcm(tools, { sourcePath, cwd, signal, startSeconds, durationSeconds, onData: (bytes) => {
    total += bytes.byteLength
    if (total > 4 * 1024 * 1024) throw new Error('Decoded chunk exceeded the speech provider’s 4 MiB request limit')
    chunks.push(bytes)
  } })
  const pcm = new Uint8Array(total)
  let offset = 0
  for (const chunk of chunks) { pcm.set(chunk, offset); offset += chunk.byteLength }
  if (pcm.byteLength < 2 || pcm.byteLength % 2 !== 0) throw new Error('FFmpeg returned no usable PCM16 samples for this range')
  return wrapCanonicalWav(pcm)
}

/** Start or join provider preparation, then wait for a state a transcription can actually use. */
async function prepareProvider(speech: SpeechToText, providerId: SpeechProviderId, exec: ToolExecution): Promise<void> {
  const stateOf = (): SpeechPreparationState | undefined =>
    speech.snapshot().providers.find(candidate => candidate.id === providerId)?.preparation
  const phase = stateOf()?.phase
  if (phase === 'ready' || phase === 'standby') return
  speech.prepare(providerId)
  const timebox = deadline(exec.signal, PROVIDER_PREPARE_TIMEOUT_MS, 'AUDIO_PREPARE_TIMEOUT')
  try {
    for (;;) {
      const current = stateOf()?.phase
      if (current === 'ready' || current === 'standby') return
      if (current === 'failed') {
        const failed = stateOf() as { message?: string } | undefined
        throw new Error(`Speech provider preparation failed: ${failed?.message ?? 'unknown failure'}`)
      }
      if (current === 'cancelled') throw new Error('Speech provider preparation was cancelled')
      if (timebox.signal.aborted) throw timebox.signal.reason ?? new Error('Speech provider preparation timed out')
      await new Promise<void>((resolve) => { setTimeout(resolve, 250) })
    }
  } finally { timebox[Symbol.dispose]() }
}

function downsampleWaveform(points: readonly WaveformSample[], limit: number): WaveformSample[] {
  if (points.length <= limit) return points.map(point => ({ time: point.time, peak: point.peak, rms: point.rms }))
  const stride = points.length / limit
  return Array.from({ length: limit }, (_empty, bucket) => {
    const from = Math.floor(bucket * stride), to = Math.min(points.length, Math.floor((bucket + 1) * stride))
    let peak = 0, rmsSquares = 0
    for (let index = from; index < to; index++) {
      const point = points[index]
      if (point === undefined) continue
      peak = Math.max(peak, point.peak)
      rmsSquares += point.rms ** 2
    }
    const first = points[from], last = points[to - 1]
    const time = ((first?.time ?? 0) + (last?.time ?? 0)) / 2
    const count = Math.max(1, to - from)
    return { time: Number(time.toFixed(3)), peak: Number(peak.toFixed(5)), rms: Number(Math.sqrt(rmsSquares / count).toFixed(5)) }
  })
}

function wrapCanonicalWav(pcm: Uint8Array): Uint8Array {
  const header = new Uint8Array(44)
  const view = new DataView(header.buffer)
  const text = (offset: number, value: string): void => {
    for (let index = 0; index < value.length; index++) header[offset + index] = value.charCodeAt(index)
  }
  text(0, 'RIFF')
  view.setUint32(4, 36 + pcm.byteLength, true)
  text(8, 'WAVE')
  text(12, 'fmt ')
  view.setUint32(16, 16, true)
  view.setUint16(20, 1, true)
  view.setUint16(22, 1, true)
  view.setUint32(24, 16000, true)
  view.setUint32(28, 32000, true)
  view.setUint16(32, 2, true)
  view.setUint16(34, 16, true)
  text(36, 'data')
  view.setUint32(40, pcm.byteLength, true)
  const wave = new Uint8Array(header.byteLength + pcm.byteLength)
  wave.set(header)
  wave.set(pcm, header.byteLength)
  return wave
}

interface AlignInput {
  readonly lyrics: string
  readonly transcript_segments?: readonly AudioRange[]
  readonly duration_seconds?: number
  readonly bpm?: number
  readonly word_level?: boolean
}

/** Character weight of one word, using code points so combined characters are not split. */
function wordWeight(word: string): number {
  return Math.max(1, Array.from(word).length)
}

function alignLyrics(input: AlignInput): { alignment: Record<string, JsonValue>[]; reviewWarning: string } {
  const lines = input.lyrics.split(/\r?\n/).map(line => line.trim()).filter(Boolean)
  if (!lines.length) return { alignment: [], reviewWarning: 'No non-empty lyric lines were supplied.' }
  const segments = input.transcript_segments?.filter(segment =>
    Number.isFinite(segment.start) && Number.isFinite(segment.end)
    && segment.start >= 0 && segment.end >= segment.start && segment.text.trim()) ?? []
  const duration = segments.length ? Math.max(...segments.map(segment => segment.end)) : input.duration_seconds ?? 0
  if (!segments.length && (!Number.isFinite(duration) || duration <= 0)) {
    throw new Error('duration_seconds is required when transcript_segments are omitted')
  }
  const normalized = (text: string): string => text.toLocaleLowerCase().replace(/[^\p{L}\p{N}]+/gu, '')
  const output: Record<string, JsonValue>[] = []
  let priorEnd = 0, segmentIndex = 0
  for (const line of lines) {
    const lineWords = line.split(/\s+/u)
    const needle = lineWords.map(normalized).filter(Boolean).join('')
    let match: { start: number; end: number; confidence: number; recognition: string } | undefined
    while (segmentIndex < segments.length && match === undefined) {
      const segment = segments[segmentIndex]
      segmentIndex += 1
      if (segment === undefined) break
      const joined = segment.text.split(/\s+/u).map(normalized).join('')
      if (needle.length > 0 && joined.includes(needle)) {
        match = { start: segment.start, end: segment.end, confidence: joined === needle ? 0.72 : 0.58, recognition: segment.text }
      }
    }
    if (match === undefined && !segments.length) {
      const index = output.length
      const start = index === 0 ? 0 : priorEnd
      const end = index === lines.length - 1 ? duration : start + (duration - start) / (lines.length - index)
      match = { start, end, confidence: 0.1, recognition: '' }
    }
    if (match === undefined) {
      output.push({ line, status: 'unmatched', confidence: 0 })
      continue
    }
    const start = Math.max(priorEnd, match.start), end = Math.max(start, match.end)
    const confidence = match.confidence
    priorEnd = end
    const record: Record<string, JsonValue> = {
      line,
      startSeconds: Number(start.toFixed(3)),
      endSeconds: Number(end.toFixed(3)),
      confidence,
      status: confidence < 0.6 ? 'review' : 'matched',
      recognizedText: match.recognition,
    }
    if (input.bpm !== undefined && input.bpm > 0) {
      const beatPeriod = 60 / input.bpm
      record.nearestBeatStartSeconds = Number((Math.round(start / beatPeriod) * beatPeriod).toFixed(3))
    }
    if (input.word_level === true) {
      const totalWeight = lineWords.reduce((sum, word) => sum + wordWeight(word), 0)
      let cursor = start
      record.words = lineWords.map((word, index) => {
        const wordStart = cursor
        cursor = index === lineWords.length - 1
          ? end
          : cursor + (end - start) * wordWeight(word) / totalWeight
        return {
          text: word,
          startSeconds: Number(wordStart.toFixed(3)),
          endSeconds: Number(cursor.toFixed(3)),
          confidence: Number((confidence * 0.55).toFixed(3)),
          status: 'approximate—review timing',
        }
      })
    }
    output.push(record)
  }
  return {
    alignment: output,
    reviewWarning: segments.length
      ? 'Approximate sequential text matching only. Unmatched/repeated words, transcription errors and sung/overlapped '
        + 'phrases need a human listening pass. Word timestamps are character-weighted guesses, not acoustic forced alignment.'
      : 'No transcript was provided: line times are evenly spread across duration and contain no acoustic evidence. '
        + 'This is a placeholder, not detected lyrics timing.',
  }
}

const ANALYSIS_DESCRIPTION = 'Inspect a workspace audio file for exact duration, a compact peak/RMS waveform and '
  + 'candidate BPM/beat grids. Decodes locally with Host FFmpeg: no model, network request or external service is '
  + 'involved, and the whole recording is never retained in memory. Report BPM as an estimate — speech, rubato and '
  + 'syncopation make it unreliable, and a doubled or halved value is a common alternative worth checking against the '
  + 'reported candidates. Beat timestamps are one plausible grid, not confirmed downbeats.'

const TRANSCRIBE_DESCRIPTION = 'Transcribe a workspace audio file through one explicitly named registered speech '
  + 'provider, decoding and submitting bounded chunks so a multi-minute file works under the provider request cap. '
  + 'Returns full text plus ordered segments with real start/end offsets; boundaries are chunk boundaries, not '
  + 'detected phrase boundaries. This installs no model: the voice-input composition must be mounted, and '
  + 'prepare=true lets this tool download/prepare the selected provider when it is not ready. Call only after the '
  + 'user asks for or confirms transcription, and confirm first if the provider is a cloud service that uploads the '
  + 'recording. Output is a noisy model suggestion: verify by listening, never present coarse chunk offsets as '
  + 'forced-alignment-grade lyric timing, and never invent lyrics from metadata.'

const ALIGN_DESCRIPTION = 'Produce approximate per-line or per-word times by matching user-supplied lyrics against '
  + 'timestamped segments, which audio_transcribe returns in exactly the accepted shape. Passing duration_seconds '
  + 'alone only spreads lines evenly and yields a placeholder, not detected timing. Never invent words or claim '
  + 'confidence the output does not report; repeated or misrecognized lines stay unmatched for human review. These '
  + 'times help editing but are not phoneme-level forced alignment.'

const TIMESTAMP_DESCRIPTION = 'Timestamp an official lyrics file against its audio track, which is stronger than '
  + 'audio_transcribe plus lyrics_align because it aligns the whole lyric sequence at once instead of line by line. '
  + 'It decodes the audio, reads the lyrics file, transcribes the track in chunks through the named provider as '
  + 'evidence, runs a monotonic word-level alignment, splits each chunk among the lines it matched, and interpolates '
  + 'lines with no evidence. Each returned line reports source `transcript` (anchored by recognition) or '
  + '`interpolated`/`extrapolated` (a guess between anchors), so measured and estimated timing stay separable. Also '
  + 'returns a ready-to-save LRC document. Recognition on sung vocals is lossy: check interpolated lines by ear. '
  + 'Sung material needs a lower provider VAD threshold than dictation, so the bundled speech row sets one. Call '
  + 'only when the user asks for lyric timing, and use prepare=true when the provider is not yet ready.'


/** Register the three model-facing audio tools. */
export function apply(ctx: Context): void {
  ctx.tools.register(defineTool({
    name: 'audio_analyze',
    description: ANALYSIS_DESCRIPTION,
    parameters: {
      audio_path: { type: 'string', required: true, description: 'Path to a local audio file in the active session workspace, e.g. audio.m4a.' },
      minimum_bpm: { type: 'number', description: 'Lowest BPM candidate considered (30–300, default 55).' },
      maximum_bpm: { type: 'number', description: 'Highest BPM candidate considered (above minimum_bpm, up to 300, default 200).' },
      waveform_points: { type: 'number', description: 'Returned waveform summary points (integer 1–1500, default 600).' },
      bpm_step_cents: { type: 'number', description: 'Minimum BPM separation between reported candidates in cents (0.5–10, default 2).' },
    },
    output: { schema: { type: 'object', additionalProperties: true }, render: (_args, value) => [{ type: 'text', text: JSON.stringify(value, null, 2) }] },
    async execute(args, exec) {
      const minimumBpm = args.minimum_bpm ?? 55
      const maximumBpm = args.maximum_bpm ?? 200
      if (minimumBpm < MIN_BPM || minimumBpm >= maximumBpm || maximumBpm > MAX_BPM) {
        throw new Error(`BPM range must satisfy ${MIN_BPM} <= minimum_bpm < maximum_bpm <= ${MAX_BPM}`)
      }
      const waveformPoints = args.waveform_points ?? DEFAULT_WAVEFORM_POINTS
      if (!Number.isInteger(waveformPoints) || waveformPoints < 1 || waveformPoints > MAX_WAVEFORM_POINTS) {
        throw new Error(`waveform_points must be an integer from 1 to ${MAX_WAVEFORM_POINTS}`)
      }
      const source = await readAudio(ctx, exec, args.audio_path)
      const timebox = deadline(exec.signal, AUDIO_PROCESS_TIMEOUT_MS, 'AUDIO_ANALYZE_TIMEOUT')
      try {
        const analysis = await withSourceFile(source, async ({ file, cwd }) => {
          const tools = await resolveTools(ctx, timebox.signal)
          const durationSeconds = await probeDuration(tools, file, cwd, timebox.signal)
          const frames = new FrameAccumulator({ maxPeaks: MAX_ANALYSIS_FRAMES })
          await streamPcm(tools, { sourcePath: file, cwd, signal: timebox.signal, onData: (bytes) => { frames.push(bytes) } })
          return analyzeFrames(frames.finish(), durationSeconds, {
            minimumBpm,
            maximumBpm,
            ...args.bpm_step_cents === undefined ? {} : { bpmStepCents: args.bpm_step_cents },
          })
        })
        return presentAnalysis(args.audio_path, analysis, waveformPoints)
      } finally { timebox[Symbol.dispose]() }
    },
  }))

  ctx.tools.register(defineTool({
    name: 'audio_transcribe',
    description: TRANSCRIBE_DESCRIPTION,
    parameters: {
      audio_path: { type: 'string', required: true, description: 'Path to a speech/audio file in the active workspace.' },
      provider_id: { type: 'string', required: true, description: 'Exact id of a registered speech provider, such as sensevoice-local. Required so no provider or cloud destination is chosen implicitly.' },
      language: { type: 'string', description: 'Language hint accepted by the provider: auto, zh, en, yue, ja or ko for SenseVoice.' },
      chunk_seconds: { type: 'number', description: 'Maximum audio seconds per provider request (5–120, default 20). Smaller values give finer offsets but more boundary artifacts.' },
      prepare: { type: 'boolean', description: 'Prepare (download/verify/load) the selected provider before transcribing when it is not ready (default false).' },
    },
    output: { schema: { type: 'object', additionalProperties: true }, render: (_args, value) => [{ type: 'text', text: JSON.stringify(value, null, 2) }] },
    async execute(args, exec) {
      const speech = ctx.get('speechToText')
      if (speech === undefined) throw new Error('Transcription is not configured: mount the speech-to-text registry and a recognizer provider.')
      const providerId = args.provider_id as SpeechProviderId
      const provider = speech.listProviders().find(candidate => candidate.id === providerId)
      if (provider === undefined) throw new Error(`Requested speech provider is unavailable: ${args.provider_id}`)
      const chunkSeconds = args.chunk_seconds ?? 20
      if (!Number.isFinite(chunkSeconds) || chunkSeconds < 5 || chunkSeconds > MAX_CHUNK_SECONDS) {
        throw new Error(`chunk_seconds must be between 5 and ${MAX_CHUNK_SECONDS}`)
      }
      const source = await readAudio(ctx, exec, args.audio_path)
      const timebox = deadline(exec.signal, Math.max(AUDIO_PROCESS_TIMEOUT_MS, chunkSeconds * 6000), 'AUDIO_TRANSCRIBE_TIMEOUT')
      try {
        if (args.prepare === true) await prepareProvider(speech, providerId, exec)
        const readiness = speech.snapshot().providers.find(candidate => candidate.id === providerId)?.preparation.phase
        if (readiness !== undefined && readiness !== 'ready' && readiness !== 'standby') {
          throw new Error(`Selected speech provider is not ready (state: ${readiness}). Retry with prepare=true.`)
        }
        const result = await withSourceFile(source, async ({ file, cwd }) => {
          const tools = await resolveTools(ctx, timebox.signal)
          const durationSeconds = await probeDuration(tools, file, cwd, timebox.signal)
          const segments: AudioRange[] = []
          for (let start = 0; start < durationSeconds; start += chunkSeconds) {
            exec.signal.throwIfAborted()
            const length = Math.min(chunkSeconds, durationSeconds - start)
            const audio = await decodeRange(tools, file, cwd, start, length, timebox.signal)
            const spec = speech.resolve({ audio, providerId, ...args.language === undefined ? {} : { language: args.language } })
            const transcript = await speech.transcribe(spec, exec.signal)
            const text = transcript.text.trim()
            if (text.length > 0) {
              segments.push({ start: Number(start.toFixed(3)), end: Number((start + length).toFixed(3)), text })
            }
          }
          return { durationSeconds, segments }
        })
        return {
          audioPath: args.audio_path,
          provider: { id: provider.id, name: provider.name, location: provider.location },
          audioDurationSeconds: Number(result.durationSeconds.toFixed(3)),
          chunkSeconds,
          chunkCount: Math.ceil(result.durationSeconds / chunkSeconds),
          transcript: result.segments.map(segment => segment.text).join(' ').trim(),
          segments: result.segments,
          timingNote: 'Segment boundaries are processing-chunk boundaries, not detected phrase boundaries. Pass these '
            + 'segments to lyrics_align for reference-lyric matching; verify everything by listening.',
        }
      } finally { timebox[Symbol.dispose]() }
    },
  }))

  ctx.tools.register(defineTool({
    name: 'lyrics_align',
    description: ALIGN_DESCRIPTION,
    parameters: {
      lyrics: { type: 'string', required: true, description: 'Reference lyrics to timestamp; line breaks define the units.' },
      transcript_segments: { type: 'array', items: { type: 'object', additionalProperties: false, properties: {
        start: { type: 'number' },
        startSeconds: { type: 'number' },
        end: { type: 'number' },
        endSeconds: { type: 'number' },
        text: { type: 'string', required: true },
        index: { type: 'number' },
      } }, description: 'Timestamped segments, as returned by audio_transcribe (start/end or startSeconds/endSeconds).' },
      duration_seconds: { type: 'number', description: 'Audio duration; only used when transcript_segments is omitted, producing an even-spread placeholder.' },
      bpm: { type: 'number', description: 'Candidate tempo; only annotates the nearest beat-grid time and never moves a lyric onset.' },
      word_level: { type: 'boolean', description: 'Add per-word times by character weight within the matched segment (default false).' },
    },
    output: { schema: { type: 'object', additionalProperties: true }, render: (_args, value) => [{ type: 'text', text: JSON.stringify(value, null, 2) }] },
    execute(args) {
      const segments = args.transcript_segments?.map((segment) => {
        const start = segment.start ?? segment.startSeconds
        const end = segment.end ?? segment.endSeconds
        if (start === undefined || end === undefined) {
          throw new Error('Each transcript segment needs start/end (or startSeconds/endSeconds) and text')
        }
        return { start, end, text: segment.text }
      })
      const value = alignLyrics({
        lyrics: args.lyrics,
        ...segments === undefined ? {} : { transcript_segments: segments },
        ...args.duration_seconds === undefined ? {} : { duration_seconds: args.duration_seconds },
        ...args.bpm === undefined ? {} : { bpm: args.bpm },
        ...args.word_level === undefined ? {} : { word_level: args.word_level },
      })
      return Promise.resolve(value as unknown as Record<string, JsonValue>)
    },
  }))
  ctx.tools.register(defineTool({
    name: 'lyrics_timestamp',
    description: TIMESTAMP_DESCRIPTION,
    parameters: {
      audio_path: { type: 'string', required: true, description: 'Path to the audio file in the active workspace.' },
      lyrics_path: { type: 'string', description: 'Path to the lyrics file, e.g. lyrics.txt. Title and credit lines are dropped automatically.' },
      lyrics: { type: 'string', description: 'Inline lyrics text; used only when lyrics_path is omitted.' },
      provider_id: { type: 'string', description: 'Exact id of the speech provider used as timing evidence, such as sensevoice-local. Optional when evidence_path supplies timing instead.' },
      evidence_path: { type: 'string', description: 'JSON file of external timing evidence: either {"words":[{start,end,text}]} or a bare [{start,end,text}] array. Replaces provider recognition, which is how a word-timestamping aligner (for example Faster-Whisper output) drives this tool. Word-level evidence is what achieves sub-second line starts.' },
      chunk_seconds: { type: 'number', description: 'Recognition chunk length in seconds (3–30, default 6). Smaller chunks tighten break boundaries at the cost of weaker phrases and proportionally more recognizer calls.' },
      language: { type: 'string', description: 'Language hint accepted by the provider: auto, zh, en, yue, ja or ko.' },
      prepare: { type: 'boolean', description: 'Prepare the selected provider first when it is not ready (default false).' },
      bpm: { type: 'number', description: 'Candidate tempo from audio_analyze; annotates each line with its nearest beat.' },
    },
    output: { schema: { type: 'object', additionalProperties: true }, render: (_args, value) => [{ type: 'text', text: JSON.stringify(value, null, 2) }] },
    async execute(args, exec) {
      const speech = ctx.get('speechToText')
      const chunkSeconds = args.chunk_seconds ?? 6
      if (!Number.isFinite(chunkSeconds) || chunkSeconds < 3 || chunkSeconds > 30) throw new Error('chunk_seconds must be between 3 and 30')
      if (args.lyrics_path === undefined && args.lyrics === undefined) throw new Error('Supply lyrics_path or inline lyrics')
      if (args.provider_id === undefined && args.evidence_path === undefined) {
        throw new Error('Supply provider_id for local recognition, or evidence_path for external word-level timing')
      }
      let providerId: SpeechProviderId | undefined
      let providerName = 'external evidence'
      if (args.provider_id !== undefined) {
        if (speech === undefined) throw new Error('Transcription is not configured: mount the speech-to-text registry and a recognizer provider.')
        providerId = args.provider_id as SpeechProviderId
        const found = speech.listProviders().find(candidate => candidate.id === providerId)
        if (found === undefined) throw new Error(`Requested speech provider is unavailable: ${args.provider_id}`)
        providerName = found.name
      }

      const rawLyrics = args.lyrics_path === undefined
        ? args.lyrics ?? ''
        : new TextDecoder().decode(await readAudio(ctx, exec, args.lyrics_path))
      const lines = extractLyricLines(rawLyrics)
      if (lines.length === 0) throw new Error('No sung lyric lines were found after dropping title/credit metadata')
      const source = await readAudio(ctx, exec, args.audio_path)
      const external = args.evidence_path === undefined
        ? undefined
        : parseEvidence(new TextDecoder().decode(await readAudio(ctx, exec, args.evidence_path)))
      const timebox = deadline(exec.signal, AUDIO_PROCESS_TIMEOUT_MS, 'LYRICS_PROBE_TIMEOUT')
      try {
        if (external === undefined) {
          if (speech === undefined || providerId === undefined) throw new Error('Local recognition needs a mounted speech provider')
          if (args.prepare === true) await prepareProvider(speech, providerId, exec)
          const readiness = speech.snapshot().providers.find(candidate => candidate.id === providerId)?.preparation.phase
          if (readiness !== undefined && readiness !== 'ready' && readiness !== 'standby') {
            throw new Error(`Selected speech provider is not ready (state: ${readiness}). Retry with prepare=true.`)
          }
        }
        const result = await withSourceFile(source, async ({ file, cwd }) => {
          const tools = await resolveTools(ctx, timebox.signal)
          const durationSeconds = await probeDuration(tools, file, cwd, timebox.signal)
          if (external !== undefined) return { durationSeconds, segments: external }
          if (speech === undefined || providerId === undefined) throw new Error('Local recognition needs a mounted speech provider')
          // Recognition cost scales with chunk count, so the work deadline is sized from the real duration.
          const chunkCount = Math.ceil(durationSeconds / chunkSeconds)
          const budget = Math.min(MAX_TIMER_DELAY_MS, 60_000 + chunkCount * 4_000)
          const work = deadline(exec.signal, budget, 'LYRICS_TIMESTAMP_TIMEOUT')
          try {
            const segments = await collectEvidence(speech, providerId, tools, file, cwd, durationSeconds,
              chunkSeconds, args.language, work.signal, exec.signal)
            return { durationSeconds, segments }
          } finally { work[Symbol.dispose]() }
        })
        const timed = alignLyricsToEvidence(lines, result.segments, { durationSeconds: result.durationSeconds })
        const anchoredCount = timed.filter(line => line.source === 'transcript').length
        const notes = [
          `audio: ${args.audio_path}`,
          `evidence: ${providerName}`,
          external === undefined
            ? `recognition chunk: ${chunkSeconds}s, ${result.segments.length} chunks produced evidence`
            : `external segments: ${result.segments.length}`,
          `line timing: ${anchoredCount} anchored, ${timed.length - anchoredCount} interpolated/extrapolated`,
          'interpolated lines are guesses between anchors: verify by listening',
        ]
        return {
          audioPath: args.audio_path,
          durationSeconds: round(result.durationSeconds),
          evidenceSource: external === undefined ? { kind: 'provider', name: providerName } : { kind: 'external', name: args.evidence_path ?? '' },
          chunkSeconds: external === undefined ? chunkSeconds : null,
          lineCount: lines.length,
          anchoredLines: anchoredCount,
          interpolatedLines: timed.length - anchoredCount,
          lines: timed,
          lrc: toLrc(timed, notes),
          warnings: [
            'Anchored lines are placed at their first matched word/phrase; interpolated lines have no direct evidence.',
            'Timings are not phoneme-level forced alignment.',
          ],
        }
      } finally { timebox[Symbol.dispose]() }
    },
  }))
}

/**
 * Parse external timing evidence.
 * Accepts `{"words":[...]}` (Faster-Whisper style) or a bare array, and tolerates the common
 * `start`/`end` or `startSeconds`/`endSeconds` field spellings.
 * @param text - file contents.
 * @returns normalized evidence segments.
 */
function parseEvidence(text: string): TranscriptSegment[] {
  const parsed: unknown = JSON.parse(text)
  const raw = Array.isArray(parsed)
    ? parsed
    : typeof parsed === 'object' && parsed !== null && Array.isArray((parsed as { words?: unknown }).words)
      ? (parsed as { words: unknown[] }).words
      : []
  const segments: TranscriptSegment[] = []
  for (const entry of raw) {
    if (typeof entry !== 'object' || entry === null) continue
    const record = entry as { start?: unknown; end?: unknown; startSeconds?: unknown; endSeconds?: unknown; text?: unknown }
    const start = typeof record.start === 'number' ? record.start : typeof record.startSeconds === 'number' ? record.startSeconds : undefined
    const end = typeof record.end === 'number' ? record.end : typeof record.endSeconds === 'number' ? record.endSeconds : undefined
    const text = typeof record.text === 'string' ? record.text.trim() : ''
    if (start === undefined || end === undefined || text.length === 0 || end <= start) continue
    segments.push({ start, end, text })
  }
  if (segments.length === 0) throw new Error('External evidence contained no usable {start,end,text} entries')
  segments.sort((a, b) => a.start - b.start)
  return segments
}

/** Transcribe the whole track in bounded chunks and keep only the non-empty results as timing evidence. */
async function collectEvidence(
  speech: SpeechToText,
  providerId: SpeechProviderId,
  tools: AudioTools,
  file: string,
  cwd: string,
  durationSeconds: number,
  chunkSeconds: number,
  language: string | undefined,
  signal: AbortSignal,
  cancel: AbortSignal,
): Promise<TranscriptSegment[]> {
  const segments: TranscriptSegment[] = []
  for (let start = 0; start < durationSeconds; start += chunkSeconds) {
    cancel.throwIfAborted()
    const length = Math.min(chunkSeconds, durationSeconds - start)
    const audio = await decodeRange(tools, file, cwd, start, length, signal)
    const spec = speech.resolve({ audio, providerId, ...language === undefined ? {} : { language } })
    const transcript = await speech.transcribe(spec, cancel)
    const text = transcript.text.trim()
    if (text.length > 0) segments.push({ start: round(start), end: round(start + length), text })
  }
  return segments
}

function round(value: number): number {
  return Number(value.toFixed(3))
}

function presentAnalysis(audioPath: string, analysis: AudioAnalysis, waveformPoints: number): Record<string, JsonValue> {
  return {
    audioPath,
    durationSeconds: analysis.durationSeconds,
    sampleRate: analysis.sampleRate,
    channels: analysis.channels,
    waveform: downsampleWaveform(analysis.waveform, waveformPoints),
    estimatedBpm: analysis.estimatedBpm === null ? null : {
      bpm: analysis.estimatedBpm.bpm,
      confidence: analysis.estimatedBpm.confidence,
      beatPeriodSeconds: analysis.estimatedBpm.beatPeriodSeconds,
      beatTimes: [...analysis.estimatedBpm.beatTimes],
    },
    bpmCandidates: analysis.bpmCandidates.map(candidate => ({
      bpm: candidate.bpm,
      confidence: candidate.confidence,
      beatPeriodSeconds: candidate.beatPeriodSeconds,
      beatTimes: [...candidate.beatTimes],
    })),
  }
}
