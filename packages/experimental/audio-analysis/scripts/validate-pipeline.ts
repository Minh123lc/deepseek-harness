/**
 * Full pipeline over one real audio file: analyze → transcribe (real local SenseVoice) → align.
 * Usage: node --import tsx tests/validate-pipeline.ts <audio> <absolute-model-dir> <out-dir> [chunkSeconds]
 * Writes <out>/audio-analysis.json and <out>/transcript-as-lyrics.lrc, and prints a summary.
 */
import { Context } from '@deepseek-ai/cordis'
import { mkdir, writeFile } from 'node:fs/promises'
import { dirname, basename, isAbsolute, join } from 'node:path'
import { ToolCallId } from '@deepseek-ai/dsh-llm'
import SystemPrompt from '@deepseek-ai/dsh-system-prompt'
import ToolRuntime from '@deepseek-ai/dsh-tools'
import { LocalFileSystem } from '@deepseek-ai/dsh-fs-local'
import LocalSubprocessRuntime from '@deepseek-ai/dsh-subprocess-local'
import { provideWorkingDirectoryFixture } from '@deepseek-ai/dsh-agent-loop-testkit'
import SpeechToText from '@deepseek-ai/dsh-experimental-speech-to-text'
import * as SenseVoice from '@deepseek-ai/dsh-experimental-speech-to-text-sensevoice'
import * as AudioAnalysis from '../src/index.ts'

const [target, modelDir, outDir, chunkArg] = process.argv.slice(2)
if (target === undefined || modelDir === undefined || outDir === undefined || !isAbsolute(modelDir)) {
  throw new Error('usage: validate-pipeline.ts <audio> <absolute-model-dir> <out-dir> [chunkSeconds]')
}
const chunkSeconds = chunkArg === undefined ? 20 : Number(chunkArg)
const dir = dirname(target), file = basename(target)

const ctx = new Context()
await ctx.plugin(SystemPrompt)
await ctx.plugin(ToolRuntime)
await ctx.plugin(LocalFileSystem, { cwd: dir })
await ctx.plugin(LocalSubprocessRuntime)
provideWorkingDirectoryFixture(ctx, dir)
await ctx.plugin(SpeechToText, { defaultProvider: 'sensevoice-local', language: 'auto' })
await ctx.plugin(SenseVoice, { dataRoot: modelDir, modelDirectory: modelDir, vadModelPath: `${modelDir}/silero_vad.onnx`, precision: 'int8', threads: 4, idleTimeoutMs: 0 })
await ctx.plugin(AudioAnalysis)

const agent = { session: { header: { cwd: dir } } }
let counter = 0
async function tool(name: string, args: unknown): Promise<Record<string, never>> {
  const result = await ctx.tools.execute({ signal: new AbortController().signal, callId: ToolCallId(`p${++counter}`), name, arguments: args, agent: agent as never })
  const text = result.content.filter(block => block.type === 'text').map(block => block.text).join('')
  if (result.isError === true) throw new Error(`${name} failed: ${text.slice(0, 800)}`)
  return JSON.parse(text) as Record<string, never>
}

const analysis = await tool('audio_analyze', { audio_path: file, waveform_points: 200 })
const transcription = await tool('audio_transcribe', { audio_path: file, provider_id: 'sensevoice-local', chunk_seconds: chunkSeconds, prepare: true })
const segments = transcription.segments as unknown as { startSeconds: number; endSeconds: number; text: string }[]
const bpm = (analysis.estimatedBpm as unknown as { bpm: number } | null)?.bpm
const lyrics = segments.map(segment => segment.text).join('\n')
const alignment = await tool('lyrics_align', { lyrics, transcript_segments: segments, ...bpm === undefined ? {} : { bpm }, word_level: false })

await mkdir(outDir, { recursive: true })
await writeFile(join(outDir, 'audio-analysis.json'), JSON.stringify({ analysis, transcription, alignment }, null, 2), 'utf8')

const stamp = (seconds: number): string => {
  const minutes = Math.floor(seconds / 60)
  const rest = seconds - minutes * 60
  return `[${String(minutes).padStart(2, '0')}:${rest.toFixed(2).padStart(5, '0')}]`
}
type AlignedLine = {
  line: string
  status: string
  startSeconds?: number
  nearestBeatStartSeconds?: number
  confidence: number
}
const lines = alignment.alignment as unknown as AlignedLine[]
const lrc = [
  '[by] dsh-experimental-audio-analysis',
  `[alg] chunk-level transcript, voice model: SenseVoiceSmall INT8, chunk=${chunkSeconds}s`,
  '[alg] approximate: segment boundaries are processing boundaries, not phrase boundaries',
  ...lines.flatMap(entry => entry.startSeconds === undefined ? [] : [`${stamp(entry.startSeconds)}${entry.line}`]),
].join('\n')
await writeFile(join(outDir, 'transcript-as-lyrics.lrc'), `${lrc}\n`, 'utf8')

console.log(JSON.stringify({
  file,
  durationSeconds: analysis.durationSeconds,
  estimatedBpm: analysis.estimatedBpm,
  bpmCandidates: (analysis.bpmCandidates as unknown as { bpm: number; confidence: number }[]).map(candidate => `${candidate.bpm} (${candidate.confidence})`),
  chunkSeconds,
  segmentCount: segments.length,
  transcriptChars: (transcription.transcript as unknown as string).length,
  alignedLines: lines.filter(entry => entry.status === 'matched').length,
  unmatchedLines: lines.filter(entry => entry.status === 'unmatched').length,
  reviewWarning: alignment.reviewWarning,
  artifacts: [join(outDir, 'audio-analysis.json'), join(outDir, 'transcript-as-lyrics.lrc')],
}, null, 2))

await ctx.fiber.dispose()
