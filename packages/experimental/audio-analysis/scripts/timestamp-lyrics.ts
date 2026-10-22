/**
 * Timestamp an official lyrics file against its audio track, writing LRC and JSON.
 * Usage: node --import tsx scripts/timestamp-lyrics.ts <audio> <lyrics> <out-dir> <absolute-model-dir> [chunkSeconds] [vadThreshold]
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

const [target, lyricsPath, outDir, modelDir, chunkArg, vadArg] = process.argv.slice(2)
if (target === undefined || lyricsPath === undefined || outDir === undefined || modelDir === undefined || !isAbsolute(modelDir)) {
  throw new Error('usage: timestamp-lyrics.ts <audio> <lyrics> <out-dir> <absolute-model-dir> [chunkSeconds] [vadThreshold]')
}
const chunkSeconds = chunkArg === undefined ? 6 : Number(chunkArg)
const vadThreshold = vadArg === undefined ? 0.2 : Number(vadArg)
const dir = dirname(target)

const ctx = new Context()
await ctx.plugin(SystemPrompt)
await ctx.plugin(ToolRuntime)
await ctx.plugin(LocalFileSystem, { cwd: dir })
await ctx.plugin(LocalSubprocessRuntime)
provideWorkingDirectoryFixture(ctx, dir)
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
await ctx.plugin(AudioAnalysis)

const agent = { session: { header: { cwd: dir } } }
const result = await ctx.tools.execute({
  signal: new AbortController().signal,
  callId: ToolCallId('lyrics-1'),
  name: 'lyrics_timestamp',
  arguments: {
    audio_path: basename(target),
    lyrics_path: basename(lyricsPath),
    provider_id: 'sensevoice-local',
    chunk_seconds: chunkSeconds,
    prepare: true,
  },
  agent: agent as never,
})
type TimestampPayload = {
  lrc: string
  lineCount: number
  anchoredLines: number
  interpolatedLines: number
  durationSeconds: number
}

const text = result.content.filter(block => block.type === 'text').map(block => block.text).join('')
if (result.isError === true) {
  console.log('TIMESTAMP_ERROR:', text.slice(0, 1200))
} else {
  const payload = JSON.parse(text) as TimestampPayload
  await mkdir(outDir, { recursive: true })
  await writeFile(join(outDir, 'lyrics-timed.lrc'), payload.lrc, 'utf8')
  await writeFile(join(outDir, 'lyrics-timed.json'), text, 'utf8')
  console.log(JSON.stringify({
    durationSeconds: payload.durationSeconds,
    lineCount: payload.lineCount,
    anchoredLines: payload.anchoredLines,
    interpolatedLines: payload.interpolatedLines,
    chunkSeconds,
    vadThreshold,
    artifacts: [join(outDir, 'lyrics-timed.lrc'), join(outDir, 'lyrics-timed.json')],
  }, null, 2))
}
await ctx.fiber.dispose()
