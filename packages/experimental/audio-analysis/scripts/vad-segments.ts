/**
 * Extract speech/phrase segments with Silero VAD at 32 ms frame resolution.
 * Usage: node --import tsx scripts/vad-segments.ts <audio> <absolute-model-dir> [threshold] [minSilence] [maxSpeech]
 * Prints segment count, durations and the segments inside a review window.
 */
import { createRequire } from 'node:module'
import { execFile } from 'node:child_process'
import { promisify } from 'node:util'
import { isAbsolute } from 'node:path'

const [target, modelDir, thresholdArg, silenceArg, maxArg] = process.argv.slice(2)
if (target === undefined || modelDir === undefined || !isAbsolute(modelDir)) {
  throw new Error('usage: vad-segments.ts <audio> <absolute-model-dir> [threshold] [minSilence] [maxSpeech]')
}
const threshold = Number(thresholdArg ?? 0.2)
const minSilence = Number(silenceArg ?? 0.25)
const maxSpeech = Number(maxArg ?? 30)

interface Detector {
  acceptWaveform(samples: Float32Array): void
  isEmpty(): boolean
  front(externalBuffer: false): { start: number; samples: Float32Array }
  pop(): void
  reset(): void
  flush(): void
}
const require = createRequire(new URL('../../speech-to-text-sensevoice/package.json', import.meta.url))
const sherpa = require('sherpa-onnx-node') as { Vad: new (config: object, bufferSeconds: number) => Detector }

const run = promisify(execFile)
const ffmpeg = process.platform === 'win32' ? 'C:/Universal include/FFmpeg/bin/ffmpeg.exe' : 'ffmpeg'
const { stdout } = await run(ffmpeg, ['-v', 'error', '-nostdin', '-i', target, '-map', '0:a:0', '-vn',
  '-f', 's16le', '-ac', '1', '-ar', '16000', 'pipe:1'], { encoding: 'buffer', maxBuffer: 512 * 1024 * 1024 })
const pcm = new DataView(stdout.buffer, stdout.byteOffset, stdout.byteLength)
const samples = Float32Array.from({ length: pcm.byteLength / 2 }, (_v, i) => pcm.getInt16(i * 2, true) / 32768)

const detector = new sherpa.Vad({
  sileroVad: { model: `${modelDir}/silero_vad.onnx`, threshold, minSilenceDuration: minSilence,
    minSpeechDuration: 0.1, maxSpeechDuration: maxSpeech, windowSize: 512 },
  sampleRate: 16000, numThreads: 4, provider: 'cpu', debug: 0,
}, maxSpeech + minSilence + 1)

const segments: { start: number; end: number; seconds: number }[] = []
const drain = (): void => {
  while (!detector.isEmpty()) {
    const segment = detector.front(false)
    segments.push({
      start: Number((segment.start / 16000).toFixed(3)),
      end: Number(((segment.start + segment.samples.length) / 16000).toFixed(3)),
      seconds: Number((segment.samples.length / 16000).toFixed(3)),
    })
    detector.pop()
  }
}
for (let offset = 0; offset < samples.length; offset += 512) {
  detector.acceptWaveform(samples.subarray(offset, offset + 512))
  drain()
}
detector.flush()
drain()

const durations = segments.map(segment => segment.seconds).sort((a, b) => a - b)
const median = durations[Math.floor(durations.length / 2)] ?? 0
console.log(`threshold=${threshold} minSilence=${minSilence} segments=${segments.length} median=${median.toFixed(2)}s totalVoiced=${segments.reduce((s, x) => s + x.seconds, 0).toFixed(1)}s`)
const review = segments.filter(segment => segment.start >= 118 && segment.end <= 180)
console.log('review window 118-180s:')
for (const segment of review) console.log(`  ${segment.start.toFixed(2)} -> ${segment.end.toFixed(2)} (${segment.seconds.toFixed(2)}s)`)
console.log('first 12:')
for (const segment of segments.slice(0, 12)) console.log(`  ${segment.start.toFixed(2)} -> ${segment.end.toFixed(2)} (${segment.seconds.toFixed(2)}s)`)
