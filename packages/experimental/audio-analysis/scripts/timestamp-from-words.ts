/**
 * Timestamp lyrics from external word-level evidence (for example whisper-words.py output).
 * No speech provider is mounted: this exercises the evidence_path path end to end.
 * Usage: node --import tsx scripts/timestamp-from-words.ts <audio> <lyrics> <evidence.json> <out-dir>
 */
import { Context } from '@deepseek-ai/cordis'
import { mkdir, writeFile } from 'node:fs/promises'
import { basename, dirname, join } from 'node:path'
import { ToolCallId } from '@deepseek-ai/dsh-llm'
import SystemPrompt from '@deepseek-ai/dsh-system-prompt'
import ToolRuntime from '@deepseek-ai/dsh-tools'
import { LocalFileSystem } from '@deepseek-ai/dsh-fs-local'
import LocalSubprocessRuntime from '@deepseek-ai/dsh-subprocess-local'
import { provideWorkingDirectoryFixture } from '@deepseek-ai/dsh-agent-loop-testkit'
import * as AudioAnalysis from '../src/index.ts'

const [target, lyricsPath, evidencePath, outDir] = process.argv.slice(2)
if (target === undefined || lyricsPath === undefined || evidencePath === undefined || outDir === undefined) {
  throw new Error('usage: timestamp-from-words.ts <audio> <lyrics> <evidence.json> <out-dir>')
}
const dir = dirname(target)

const ctx = new Context()
await ctx.plugin(SystemPrompt)
await ctx.plugin(ToolRuntime)
await ctx.plugin(LocalFileSystem, { cwd: dir })
await ctx.plugin(LocalSubprocessRuntime)
provideWorkingDirectoryFixture(ctx, dir)
await ctx.plugin(AudioAnalysis)

const agent = { session: { header: { cwd: dir } } }
const result = await ctx.tools.execute({
  signal: new AbortController().signal,
  callId: ToolCallId('words-1'),
  name: 'lyrics_timestamp',
  arguments: {
    audio_path: basename(target),
    lyrics_path: basename(lyricsPath),
    evidence_path: basename(evidencePath),
  },
  agent: agent as never,
})
const text = result.content.filter(block => block.type === 'text').map(block => block.text).join('')
if (result.isError === true) {
  console.log('TIMESTAMP_ERROR:', text.slice(0, 1500))
} else {
  type Payload = { lrc: string; lineCount: number; anchoredLines: number; interpolatedLines: number; durationSeconds: number }
  const payload = JSON.parse(text) as Payload
  await mkdir(outDir, { recursive: true })
  await writeFile(join(outDir, 'lyrics-timed.lrc'), payload.lrc, 'utf8')
  await writeFile(join(outDir, 'lyrics-timed.json'), text, 'utf8')
  console.log(JSON.stringify({
    mode: 'external word evidence',
    durationSeconds: payload.durationSeconds,
    lineCount: payload.lineCount,
    anchoredLines: payload.anchoredLines,
    interpolatedLines: payload.interpolatedLines,
    artifacts: [join(outDir, 'lyrics-timed.lrc'), join(outDir, 'lyrics-timed.json')],
  }, null, 2))
}
await ctx.fiber.dispose()
