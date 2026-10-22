/**
 * Real-end-to-end transcription of a user audio file through the local SenseVoice provider.
 * Usage: node --import tsx tests/validate-transcribe.ts <audio> <modelDir> [chunkSeconds] [language]
 */
import { Context } from '@deepseek-ai/cordis'
import { dirname, basename, isAbsolute } from 'node:path'
import { ToolCallId } from '@deepseek-ai/dsh-llm'
import SystemPrompt from '@deepseek-ai/dsh-system-prompt'
import ToolRuntime from '@deepseek-ai/dsh-tools'
import { LocalFileSystem } from '@deepseek-ai/dsh-fs-local'
import LocalSubprocessRuntime from '@deepseek-ai/dsh-subprocess-local'
import { provideWorkingDirectoryFixture } from '@deepseek-ai/dsh-agent-loop-testkit'
import SpeechToText from '@deepseek-ai/dsh-experimental-speech-to-text'
import * as SenseVoice from '@deepseek-ai/dsh-experimental-speech-to-text-sensevoice'
import * as AudioAnalysis from '../src/index.ts'

const [target, modelDir, chunkArg, language] = process.argv.slice(2)
if (target === undefined || modelDir === undefined || !isAbsolute(modelDir)) throw new Error('usage: validate-transcribe.ts <audio> <absolute-model-dir> [chunkSeconds] [language]')
const chunkSeconds = chunkArg === undefined ? 20 : Number(chunkArg)
const dir = dirname(target), file = basename(target)

const ctx = new Context()
await ctx.plugin(SystemPrompt)
await ctx.plugin(ToolRuntime)
await ctx.plugin(LocalFileSystem, { cwd: dir })
await ctx.plugin(LocalSubprocessRuntime)
provideWorkingDirectoryFixture(ctx, dir)
await ctx.plugin(SpeechToText, { defaultProvider: 'sensevoice-local', language: language ?? 'auto' })
await ctx.plugin(SenseVoice, {
  dataRoot: modelDir,
  modelDirectory: modelDir,
  vadModelPath: `${modelDir}/silero_vad.onnx`,
  precision: 'int8',
  threads: 4,
  idleTimeoutMs: 0,
})
await ctx.plugin(AudioAnalysis)

const agent = { session: { header: { cwd: dir } } }
const result = await ctx.tools.execute({ signal: new AbortController().signal, callId: ToolCallId('t1'), name: 'audio_transcribe', arguments: {
  audio_path: file, provider_id: 'sensevoice-local', chunk_seconds: chunkSeconds, prepare: true,
  ...language === undefined ? {} : { language },
}, agent: agent as never })
const text = result.content.filter(block => block.type === 'text').map(block => block.text).join('')
if (result.isError === true) { console.log('TRANSCRIBE_ERROR:', text.slice(0, 1000)) }
else console.log('TRANSCRIBE_RESULT:', text.slice(0, 6000))
await ctx.fiber.dispose()
