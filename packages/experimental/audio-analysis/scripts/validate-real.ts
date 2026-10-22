/**
 * Manual validation of the audio-analysis plugin against a real file.
 * Usage: node --import tsx packages/experimental/audio-analysis/tests/validate-real.ts <audio> [chunkSeconds]
 * Prints the real decoded duration, waveform summary and BPM candidates.
 */
import { Context } from '@deepseek-ai/cordis'
import { dirname, basename } from 'node:path'
import { ToolCallId } from '@deepseek-ai/dsh-llm'
import SystemPrompt from '@deepseek-ai/dsh-system-prompt'
import ToolRuntime from '@deepseek-ai/dsh-tools'
import { LocalFileSystem } from '@deepseek-ai/dsh-fs-local'
import LocalSubprocessRuntime from '@deepseek-ai/dsh-subprocess-local'
import { provideWorkingDirectoryFixture } from '@deepseek-ai/dsh-agent-loop-testkit'
import SpeechToText from '@deepseek-ai/dsh-experimental-speech-to-text'
import type { SpeechProvider, SpeechProviderId } from '@deepseek-ai/dsh-experimental-speech-to-text/types'
import * as AudioAnalysis from '../src/index.ts'

const target = process.argv[2]
const chunkSeconds = process.argv[3] === undefined ? 20 : Number(process.argv[3])
if (target === undefined) throw new Error('usage: validate-real.ts <audio-file> [chunkSeconds]')
const dir = dirname(target), file = basename(target)

const provider: SpeechProvider = {
  info: { id: 'echo-local' as SpeechProviderId, name: 'echo-local', location: 'host-local', languages: ['auto'] },
  transcribe: input => Promise.resolve({ text: `[${((input.audio.byteLength - 44) / 32_000).toFixed(1)}s]`, audioSeconds: (input.audio.byteLength - 44) / 32_000, inferenceSeconds: 0 }),
}

const ctx = new Context()
await ctx.plugin(SystemPrompt)
await ctx.plugin(ToolRuntime)
await ctx.plugin(LocalFileSystem, { cwd: dir })
await ctx.plugin(LocalSubprocessRuntime)
provideWorkingDirectoryFixture(ctx, dir)
await ctx.plugin(SpeechToText, { defaultProvider: 'echo-local', language: 'auto' })
ctx.get('speechToText')!.register(provider)
await ctx.plugin(AudioAnalysis)

const agent = { session: { header: { cwd: dir } } }
const call = (name: string, args: unknown) => ctx.tools.execute({ signal: new AbortController().signal, callId: ToolCallId('v1'), name, arguments: args, agent: agent as never })
const textOf = (result: { content: { type: string; text?: string }[] }) => result.content.filter(block => block.type === 'text').map(block => block.text).join('')

type AnalyzeResult = {
  durationSeconds: number
  sampleRate: number
  channels: number
  estimatedBpm: { bpm: number; confidence: number } | null
  bpmCandidates: { bpm: number; confidence: number }[]
  waveform: { time: number; peak: number; rms: number }[]
}
const analysis = JSON.parse(textOf(await call('audio_analyze', { audio_path: file, waveform_points: 24 }))) as AnalyzeResult
console.log('ANALYZE:', JSON.stringify({
  file, durationSeconds: analysis.durationSeconds, sampleRate: analysis.sampleRate, channels: analysis.channels,
  estimatedBpm: analysis.estimatedBpm, candidates: analysis.bpmCandidates.map(candidate => `${candidate.bpm} (${candidate.confidence})`),
  waveformPoints: analysis.waveform.length, firstWaveform: analysis.waveform.slice(0, 3), lastWaveform: analysis.waveform.at(-1),
}, null, 2))

type TranscribeResult = {
  chunkCount: number
  segments: { start: number; end: number; text: string }[]
}
const transcription = JSON.parse(textOf(await call('audio_transcribe', { audio_path: file, provider_id: 'echo-local', chunk_seconds: chunkSeconds }))) as TranscribeResult
console.log('TRANSCRIBE:', JSON.stringify({
  chunkSeconds, chunkCount: transcription.chunkCount,
  firstSegments: transcription.segments.slice(0, 3),
  lastSegment: transcription.segments.at(-1),
  segmentCount: transcription.segments.length,
}, null, 2))

await ctx.fiber.dispose()
