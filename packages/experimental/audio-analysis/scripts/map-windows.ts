/**
 * Map where the recognizer hears words across a time range at fixed window size.
 * Usage: node --import tsx scripts/map-windows.ts <audio> <absolute-model-dir> <fromSec> <toSec> <windowSec> [vadThreshold]
 * Prints one line per window: start, duration, text. Use it to calibrate chunk size against real onsets.
 */
import { Context } from '@deepseek-ai/cordis'
import { isAbsolute } from 'node:path'
import { execFile } from 'node:child_process'
import { promisify } from 'node:util'
import LocalSubprocessRuntime from '@deepseek-ai/dsh-subprocess-local'
import { provideWorkingDirectoryFixture } from '@deepseek-ai/dsh-agent-loop-testkit'
import SpeechToText from '@deepseek-ai/dsh-experimental-speech-to-text'
import * as SenseVoice from '@deepseek-ai/dsh-experimental-speech-to-text-sensevoice'
import type { SpeechProviderId } from '@deepseek-ai/dsh-experimental-speech-to-text/types'

const [target, modelDir, fromArg, toArg, windowArg, vadArg] = process.argv.slice(2)
if (target === undefined || modelDir === undefined || !isAbsolute(modelDir)) {
  throw new Error('usage: map-windows.ts <audio> <absolute-model-dir> <fromSec> <toSec> <windowSec> [vadThreshold]')
}
const from = Number(fromArg), to = Number(toArg), window = Number(windowArg), vadThreshold = Number(vadArg ?? 0.2)
const run = promisify(execFile)
const ffmpeg = process.platform === 'win32' ? 'C:/Universal include/FFmpeg/bin/ffmpeg.exe' : 'ffmpeg'

const ctx = new Context()
provideWorkingDirectoryFixture(ctx, process.cwd())
await ctx.plugin(LocalSubprocessRuntime)
await ctx.plugin(SpeechToText, { defaultProvider: 'sensevoice-local', language: 'en' })
await ctx.plugin(SenseVoice, {
  dataRoot: modelDir,
  modelDirectory: modelDir,
  vadModelPath: `${modelDir}/silero_vad.onnx`,
  precision: 'int8',
  threads: 4,
  idleTimeoutMs: 0,
  vadThreshold,
  minSpeechSeconds: 0.15,
  minSilenceSeconds: 0.25,
})
const speech = ctx.get('speechToText')
if (speech === undefined) throw new Error('speech registry missing')
const provider = speech.listProviders()[0]
if (provider === undefined) throw new Error('no provider registered')
speech.prepare(provider.id as SpeechProviderId)
const started = Date.now()
while (speech.snapshot().providers[0]?.preparation.phase !== 'ready' && Date.now() - started < 600_000) {
  await new Promise<void>((resolve) => { setTimeout(resolve, 250) })
}

/** Wrap raw PCM16 mono 16 kHz as a canonical WAV. */
function wrap(pcm: Uint8Array): Uint8Array {
  const header = new Uint8Array(44)
  const view = new DataView(header.buffer)
  const put = (offset: number, value: string): void => {
    for (let index = 0; index < value.length; index++) header[offset + index] = value.charCodeAt(index)
  }
  put(0, 'RIFF'); view.setUint32(4, 36 + pcm.length, true); put(8, 'WAVE'); put(12, 'fmt ')
  view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true)
  view.setUint32(24, 16000, true); view.setUint32(28, 32000, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true)
  put(36, 'data'); view.setUint32(40, pcm.length, true)
  const wave = new Uint8Array(44 + pcm.length)
  wave.set(header); wave.set(pcm, 44)
  return wave
}

for (let start = from; start < to; start += window) {
  const { stdout } = await run(ffmpeg, ['-v', 'error', '-nostdin', '-i', target, '-ss', String(start), '-t', String(window),
    '-map', '0:a:0', '-vn', '-f', 's16le', '-ac', '1', '-ar', '16000', 'pipe:1'], { encoding: 'buffer', maxBuffer: 64 * 1024 * 1024 })
  const spec = speech.resolve({ audio: wrap(new Uint8Array(stdout)), providerId: provider.id as SpeechProviderId, language: 'en' })
  const transcript = await speech.transcribe(spec, new AbortController().signal)
  const text = transcript.text.trim()
  console.log(`${start.toFixed(2).padStart(7)} +${window}s ${text.length === 0 ? '(no speech)' : text}`)
}
await ctx.fiber.dispose()
