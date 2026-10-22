/** Bounded, dependency-free audio probes for the speech plugin and offline consumers. */
import { validateWave } from './wave.ts'

/** Tuning for frame summarization and tempo search; every field has a bounded default. */
export interface AudioAnalysisOptions {
  /** Analysis frame length in seconds. */
  readonly windowSeconds?: number
  /** Distance between analysis frames in seconds; never greater than the window. */
  readonly hopSeconds?: number
  /** Maximum retained waveform frames; extra frames stop being summarized but still drive tempo. */
  readonly maxPeaks?: number
  /** Lowest tempo candidate considered, in BPM. */
  readonly minimumBpm?: number
  /** Highest tempo candidate considered, in BPM. */
  readonly maximumBpm?: number
  /** Minimum separation between reported tempo candidates, in cents. */
  readonly bpmStepCents?: number
}

/** One summarized analysis frame: its offset and its peak and RMS loudness. */
export interface WaveformPoint {
  readonly time: number
  readonly peak: number
  readonly rms: number
}

/** One candidate tempo with its equally plausible beat grid. */
export interface BeatEstimate {
  readonly bpm: number
  readonly confidence: number
  readonly beatPeriodSeconds: number
  readonly beatTimes: readonly number[]
}

/** Duration, waveform summary, and candidate tempo/beat grids for one recording. */
export interface AudioAnalysis {
  readonly durationSeconds: number
  readonly sampleRate: 16000
  readonly channels: 1
  readonly waveform: readonly WaveformPoint[]
  readonly estimatedBpm: BeatEstimate | null
  readonly bpmCandidates: readonly BeatEstimate[]
}

/** Ordered per-frame loudness summary; the only state retained from decoded audio. */
export interface FrameSummary {
  readonly points: readonly WaveformPoint[]
  readonly energy: readonly number[]
  readonly times: readonly number[]
  readonly hopSeconds: number
}

/** Read one byte without an unchecked index expression. */
function byteAt(bytes: Uint8Array, index: number): number {
  return bytes[index] ?? 0
}

/** Read one little-endian signed 16-bit sample without an unchecked index expression. */
function sampleAt(samples: DataView, index: number): number {
  return samples.getInt16(index * 2, true)
}

/** Verify one complete canonical RIFF/WAVE PCM16 mono file without trusting its header alone. */
function waveBytes(audio: Uint8Array): { readonly data: Uint8Array; readonly samples: DataView; readonly durationSeconds: number } {
  const durationSeconds = validateWave(audio, Number.MAX_SAFE_INTEGER)
  const chunkEnd = audio.byteLength
  let offset = 12
  let dataOffset = -1, dataLength = 0, formatOffset = -1, formatLength = 0
  while (offset + 8 <= chunkEnd) {
    const name = String.fromCharCode(byteAt(audio, offset), byteAt(audio, offset + 1), byteAt(audio, offset + 2), byteAt(audio, offset + 3))
    const length = new DataView(audio.buffer, audio.byteOffset + offset + 4, 4).getUint32(0, true)
    const body = offset + 8
    if (body + length > chunkEnd) throw new Error('WAV contains a truncated chunk')
    if (name === 'fmt ') { formatOffset = body; formatLength = length }
    if (name === 'data') { dataOffset = body; dataLength = length }
    offset = body + length + (length & 1)
  }
  if (offset !== chunkEnd || formatOffset < 0 || formatLength < 16 || dataOffset < 0 || dataLength < 2 || dataLength % 2 !== 0) {
    throw new Error('WAV is missing a well-formed PCM format or data chunk')
  }
  const view = new DataView(audio.buffer, audio.byteOffset, audio.byteLength)
  if (view.getUint16(formatOffset, true) !== 1 || view.getUint16(formatOffset + 2, true) !== 1
    || view.getUint32(formatOffset + 4, true) !== 16000 || view.getUint32(formatOffset + 8, true) !== 32000
    || view.getUint16(formatOffset + 12, true) !== 2 || view.getUint16(formatOffset + 14, true) !== 16
    || dataLength !== audio.byteLength - dataOffset) {
    throw new Error('WAV must contain one complete 16 kHz mono signed PCM16 data chunk')
  }
  return {
    data: audio.subarray(dataOffset),
    samples: new DataView(audio.buffer, audio.byteOffset + dataOffset, dataLength),
    durationSeconds,
  }
}

function finiteOption(value: number | undefined, name: string, fallback: number, minimum: number, maximum: number): number {
  const result = value ?? fallback
  if (!Number.isFinite(result) || result < minimum || result > maximum) throw new Error(`${name} must be between ${minimum} and ${maximum}`)
  return result
}

/** Build frame summaries incrementally so arbitrarily long audio never needs whole-file retention. */
export class FrameAccumulator {
  private readonly windowSamples: number
  private readonly hopSamples: number
  private readonly maxPoints: number
  private readonly hopSeconds: number
  private readonly points: WaveformPoint[] = []
  private readonly energy: number[] = []
  private readonly times: number[] = []
  /** Decoded but not yet framed samples, plus the absolute frame origin. Bounded by one window. */
  private pending: number[] = []
  private startSample = 0

  constructor(options: AudioAnalysisOptions = {}) {
    const windowSeconds = Math.max(0.005, Math.min(1, options.windowSeconds ?? 0.05))
    this.hopSeconds = Math.max(0.005, Math.min(windowSeconds, options.hopSeconds ?? 0.05))
    this.windowSamples = Math.max(1, Math.round(windowSeconds * 16000))
    this.hopSamples = Math.max(1, Math.round(this.hopSeconds * 16000))
    this.maxPoints = Math.floor(finiteOption(options.maxPeaks, 'maxPeaks', 20000, 1, 20000))
  }

  /**
   * Consume one decoded PCM16 little-endian chunk and frame every complete window it completes.
   * @param bytes - 16 kHz mono signed PCM16 samples; a trailing odd byte is ignored until the next push.
   */
  push(bytes: Uint8Array): void {
    for (let index = 0; index + 1 < bytes.byteLength; index += 2) {
      this.pending.push((byteAt(bytes, index) | (byteAt(bytes, index + 1) << 8)) << 16 >> 16)
    }
    while (this.pending.length >= this.windowSamples) {
      let peak = 0, sumSquares = 0
      for (let offset = 0; offset < this.windowSamples; offset++) {
        const sample = (this.pending[offset] ?? 0) / 32768
        const magnitude = Math.abs(sample)
        if (magnitude > peak) peak = magnitude
        sumSquares += sample * sample
      }
      const time = this.startSample / 16000
      if (this.points.length < this.maxPoints) {
        const rms = Number(Math.sqrt(sumSquares / this.windowSamples).toFixed(5))
        this.points.push({ time: Number(time.toFixed(3)), peak: Number(peak.toFixed(5)), rms })
      }
      this.times.push(Number(time.toFixed(3)))
      this.energy.push(sumSquares / this.windowSamples)
      this.pending = this.pending.slice(this.hopSamples)
      this.startSample += this.hopSamples
    }
  }

  /**
   * Read the summary retained so far; further pushes continue appending to it.
   * @returns the retained per-frame summary.
   */
  finish(): FrameSummary {
    return { points: this.points, energy: this.energy, times: this.times, hopSeconds: this.hopSeconds }
  }
}

/**
 * Derive waveform, tempo candidates and beat grids from an already-summarized frame stream.
 * @param frames - the retained frame summary produced by {@link FrameAccumulator}.
 * @param durationSeconds - real recording duration; beat grids stop at its end.
 * @param options - optional tempo-search tuning, validated against the same bounds as the accumulator.
 * @returns duration, waveform summary, and the leading candidate plus up to five tempo candidates.
 */
export function analyzeFrames(frames: FrameSummary, durationSeconds: number, options: AudioAnalysisOptions = {}): AudioAnalysis {
  const minimumBpm = finiteOption(options.minimumBpm, 'minimumBpm', 55, 30, 300)
  const maximumBpm = finiteOption(options.maximumBpm, 'maximumBpm', 200, minimumBpm, 300)
  const cents = finiteOption(options.bpmStepCents, 'bpmStepCents', 2, 0.5, 10)
  const candidates = estimateBpm(frames.energy, frames.hopSeconds, frames.times, { minBpm: minimumBpm, maxBpm: maximumBpm, cents })
  const best = candidates.estimates[0]
  const toEstimate = (item: { readonly bpm: number; readonly confidence: number; readonly period: number }): BeatEstimate => ({
    bpm: Number(item.bpm.toFixed(2)),
    confidence: Number(item.confidence.toFixed(3)),
    beatPeriodSeconds: Number(item.period.toFixed(4)),
    beatTimes: candidates.beatTimesFor(item.period),
  })
  return {
    durationSeconds: Number(durationSeconds.toFixed(3)), sampleRate: 16000, channels: 1,
    waveform: frames.points,
    estimatedBpm: best === undefined ? null : toEstimate(best),
    bpmCandidates: candidates.estimates.slice(0, 5).map(toEstimate),
  }
}

/**
 * Analyze a canonical 16 kHz mono PCM16 WAV. No shell, network, model, or retained audio state is used.
 * @param audio - complete RIFF/WAVE PCM16 mono 16 kHz file bytes.
 * @param options - optional frame and tempo-search tuning.
 * @returns duration, waveform summary, and candidate tempo/beat grids.
 */
export function analyzeWav(audio: Uint8Array, options: AudioAnalysisOptions = {}): AudioAnalysis {
  const { data, samples, durationSeconds } = waveBytes(audio)
  const accumulator = new FrameAccumulator(options)
  const pcm = new Uint8Array(data.byteLength)
  for (let index = 0; index < data.byteLength / 2; index++) {
    const value = sampleAt(samples, index)
    pcm[index * 2] = value & 0xff
    pcm[index * 2 + 1] = (value >> 8) & 0xff
  }
  accumulator.push(pcm)
  return analyzeFrames(accumulator.finish(), durationSeconds, options)
}

interface InternalEstimate {
  readonly bpm: number
  readonly period: number
  readonly confidence: number
  readonly periodLag: number
}

interface TempoSearch {
  readonly minBpm: number
  readonly maxBpm: number
  readonly cents: number
}

function estimateBpm(energy: readonly number[], hopSeconds: number, times: readonly number[], search: TempoSearch): {
  estimates: readonly InternalEstimate[]
  beatTimesFor(period: number): number[]
} {
  const { minBpm, maxBpm, cents } = search
  if (energy.length < 8) return { estimates: [], beatTimesFor: () => [] }
  // Novelty flux removes most sustained-tone/drone energy before autocorrelation.
  const flux = energy.map((value, index) => Math.max(0, value - (index === 0 ? 0 : energy[index - 1] ?? 0)))
  const mean = flux.reduce((sum, value) => sum + value, 0) / flux.length
  const centered = flux.map(value => value - mean)
  const minLag = Math.max(1, Math.floor(60 / (maxBpm * hopSeconds)))
  const maxLag = Math.min(centered.length - 2, Math.ceil(60 / (minBpm * hopSeconds)))
  if (maxLag <= minLag) return { estimates: [], beatTimesFor: () => [] }
  const scores: Array<{ lag: number; score: number }> = []
  for (let lag = minLag; lag <= maxLag; lag++) {
    let sum = 0, leftPower = 0, rightPower = 0
    for (let index = 0; index + lag < centered.length; index++) {
      const left = centered[index] ?? 0
      const right = centered[index + lag] ?? 0
      sum += left * right; leftPower += left * left; rightPower += right * right
    }
    scores.push({ lag, score: sum / Math.sqrt((leftPower * rightPower) || 1) })
  }
  scores.sort((a, b) => b.score - a.score)
  const selected: InternalEstimate[] = []
  for (const item of scores) {
    if (item.score <= 0 || selected.some(other => Math.abs(1200 * Math.log2(item.lag / other.periodLag)) < cents)) continue
    selected.push({ bpm: 60 / (item.lag * hopSeconds), period: item.lag * hopSeconds, confidence: item.score, periodLag: item.lag })
    if (selected.length >= 5) break
  }
  const beatTimesFor = (period: number): number[] => {
    if (period <= 0 || times.length === 0) return []
    // Align the grid to the strongest positive novelty event, then expose its neighboring plausible pulses.
    let peakIndex = 0
    for (let index = 1; index < flux.length; index++) {
      if ((flux[index] ?? 0) > (flux[peakIndex] ?? 0)) peakIndex = index
    }
    const origin = times[peakIndex] ?? 0
    const duration = times.at(-1) ?? 0
    const firstIndex = Math.floor(-origin / period)
    const lastIndex = Math.ceil((duration - origin) / period)
    const count = Math.min(2000, Math.max(0, lastIndex - firstIndex + 1))
    return Array.from({ length: count }, (_value, index) => Number((origin + (firstIndex + index) * period).toFixed(3)))
      .filter(time => time >= 0 && time <= duration + hopSeconds)
  }
  return { estimates: selected, beatTimesFor }
}
