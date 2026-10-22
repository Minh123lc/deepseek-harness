/**
 * Profile one audio file: per-window RMS/peak plus vocal-activity regions.
 * Usage: node --import tsx scripts/profile-audio.ts <audio> [windowSeconds]
 */
import { Context } from '@deepseek-ai/cordis'
import { execFile } from 'node:child_process'
import { readFile } from 'node:fs/promises'
import { promisify } from 'node:util'
import { FrameAccumulator } from '../src/index.ts'

const run = promisify(execFile)
const target = process.argv[2]
if (target === undefined) throw new Error('usage: profile-audio.ts <audio> [windowSeconds]')
const windowSeconds = Number(process.argv[3] ?? 2)
const ffmpeg = process.platform === 'win32' ? 'C:/Universal include/FFmpeg/bin/ffmpeg.exe' : 'ffmpeg'

// Decode to 16 kHz mono PCM16 through FFmpeg, exactly as the plugin does.
const { stdout } = await run(ffmpeg, ['-v', 'error', '-nostdin', '-i', target, '-map', '0:a:0', '-vn',
  '-f', 's16le', '-ac', '1', '-ar', '16000', 'pipe:1'], { encoding: 'buffer', maxBuffer: 256 * 1024 * 1024 })
const pcm = new Uint8Array(stdout)
const accumulator = new FrameAccumulator({ windowSeconds: 0.05, hopSeconds: 0.05, maxPeaks: 20000 })
accumulator.push(pcm)
const frames = accumulator.finish()

const buckets: { t: number; rms: number; peak: number }[] = []
const perBucket = Math.round(windowSeconds / frames.hopSeconds)
for (let index = 0; index < frames.energy.length; index += perBucket) {
  const slice = frames.points.slice(index, index + perBucket)
  const window = frames.energy.slice(index, index + perBucket)
  const rms = Math.sqrt(window.reduce((sum, value) => sum + value, 0) / Math.max(1, window.length))
  const peak = Math.max(...slice.map(point => point.peak))
  buckets.push({
    t: Number((index * frames.hopSeconds).toFixed(1)),
    rms: Number(rms.toFixed(4)),
    peak: Number(peak.toFixed(3)),
  })
}
const active = buckets.filter(bucket => bucket.peak > 0.05)
console.log(`duration=${(pcm.length / 32_000).toFixed(2)}s frames=${frames.energy.length} activeBuckets=${active.length}/${buckets.length}`)
console.log(buckets.map(bucket => `${bucket.t}:${bucket.rms.toFixed(3)}/${bucket.peak.toFixed(2)}`).join(' '))

// Vocal-ish regions: sustained energy above an adaptive floor, with short gaps bridged.
const floor = 0.02
const regions: { start: number; end: number }[] = []
let open: { start: number; end: number } | null = null
for (let index = 0; index < frames.energy.length; index++) {
  const loud = Math.sqrt(frames.energy[index] ?? 0) > floor
  if (loud) {
    if (open === null) open = { start: index * frames.hopSeconds, end: (index + 1) * frames.hopSeconds }
    else open.end = (index + 1) * frames.hopSeconds
  } else if (open !== null && (index + 1) * frames.hopSeconds - open.end > 0.45) {
    regions.push(open)
    open = null
  }
}
if (open !== null) regions.push(open)
const merged = regions.filter(region => region.end - region.start >= 0.2)
console.log(`regions=${merged.length}`)
console.log(merged.map(region => `${region.start.toFixed(2)}-${region.end.toFixed(2)}`).join(' '))
void Context
void readFile
