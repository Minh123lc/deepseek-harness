/**
 * Probe how the local recognizer's VAD settings affect detected speech.
 * Usage: node --import tsx scripts/probe-vad.ts <audio> <absolute-model-dir> <startSeconds> <durationSeconds> <vadThreshold>
 */
import { Context } from '@deepseek-ai/cordis'
import { isAbsolute } from 'node:path'
import LocalSubprocessRuntime from '@deepseek-ai/dsh-subprocess-local'
import { provideWorkingDirectoryFixture } from '@deepseek-ai/dsh-agent-loop-testkit'
import SpeechToText from '@deepseek-ai/dsh-experimental-speech-to-text'
import * as SenseVoice from '@deepseek-ai/dsh-experimental-speech-to-text-sensevoice'
import type { SpeechProviderId } from '@deepseek-ai/dsh-experimental-speech-to-text/types'
import { FrameAccumulator } from '../src/index.ts'

const [target, modelDir, startArg, durationArg, thresholdArg] = process.argv.slice(2)
if (target === undefined || modelDir === undefined || !isAbsolute(modelDir)) {
  throw new Error('usage: probe-vad.ts <audio> <absolute-model-dir> <startSeconds> <durationSeconds> <vadThreshold>')
}
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
  vadThreshold: Number(thresholdArg ?? 0.5),
  minSpeechSeconds: 0.15,
  minSilenceSeconds: 0.25,
})

// Decode the requested range, then submit it as one canonical WAV.
const { execFile } = await import('node:child_process')
const { promisify } = await import('node:util')
const run = promisify(execFile)
const ffmpeg = process.platform === 'win32' ? 'C:/Universal include/FFmpeg/bin/ffmpeg.exe' : 'ffmpeg'
const { stdout } = await run(ffmpeg, ['-v', 'error', '-nostdin', '-i', target, '-ss', String(startArg), '-t', String(durationArg),
  '-map', '0:a:0', '-vn', '-f', 's16le', '-ac', '1', '-ar', '16000', 'pipe:1'], { encoding: 'buffer', maxBuffer: 64 * 1024 * 1024 })
const pcm = new Uint8Array(stdout)
const header = new Uint8Array(44)
const view = new DataView(header.buffer)
const put = (offset: number, value: string): void => {
  for (let index = 0; index < value.length; index++) header[offset + index] = value.charCodeAt(index)
}
put(0, 'RIFF'); view.setUint32(4, 36 + pcm.length, true); put(8, 'WAVE'); put(12, 'fmt '); view.setUint32(16, 16, true)
view.setUint16(20, 1, true); view.setUint16(22, 1, true); view.setUint32(24, 16000, true); view.setUint32(28, 32000, true)
view.setUint16(32, 2, true); view.setUint16(34, 16, true); put(36, 'data'); view.setUint32(40, pcm.length, true)
const wave = new Uint8Array(44 + pcm.length)
wave.set(header); wave.set(pcm, 44)

const speech = ctx.get('speechToText')
if (speech === undefined) throw new Error('speech registry missing')
const provider = speech.listProviders()[0]
if (provider === undefined) throw new Error('no provider registered')
speech.prepare(provider.id as SpeechProviderId)
const started = Date.now()
while (speech.snapshot().providers[0]?.preparation.phase !== 'ready' && Date.now() - started < 600_000) {
  await new Promise<void>((resolve) => { setTimeout(resolve, 250) })
}
const frames = new FrameAccumulator({ maxPeaks: 1 })
frames.push(pcm)
console.log(`rms=${Math.sqrt(frames.finish().energy.reduce((sum, value) => sum + value, 0) / Math.max(1, frames.finish().energy.length)).toFixed(4)}`)
const spec = speech.resolve({ audio: wave, providerId: provider.id as SpeechProviderId, language: 'en' })
const transcript = await speech.transcribe(spec, new AbortController().signal)
console.log(`threshold=${thresholdArg} range=${startArg}+${durationArg} text="${transcript.text.trim()}"`)
await ctx.fiber.dispose()
