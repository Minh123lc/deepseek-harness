/**
 * End-to-end audio tool tests over the real local filesystem, the real FFmpeg
 * decode path and the real tool registry. The speech recognizer is a double so
 * the test needs no model download while still proving chunk offsets, provider
 * routing and the returned segment shape.
 */
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import { existsSync } from 'node:fs'
import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { execFile } from 'node:child_process'
import { promisify } from 'node:util'
import { ToolCallId } from '@deepseek-ai/dsh-llm'
import { provideWorkingDirectoryFixture } from '@deepseek-ai/dsh-agent-loop-testkit'
import SystemPrompt from '@deepseek-ai/dsh-system-prompt'
import ToolRuntime from '@deepseek-ai/dsh-tools'
import { LocalFileSystem } from '@deepseek-ai/dsh-fs-local'
import LocalSubprocessRuntime from '@deepseek-ai/dsh-subprocess-local'
import SpeechToText from '@deepseek-ai/dsh-experimental-speech-to-text'
import type { SpeechProvider, SpeechProviderId } from '@deepseek-ai/dsh-experimental-speech-to-text/types'
import * as AudioAnalysis from '../src/index.ts'

const run = promisify(execFile)

/** A recognizer double that reports the audio duration it was handed. */
function fakeProvider(id: string): SpeechProvider {
  let calls = 0
  return {
    info: { id: id as SpeechProviderId, name: id, location: 'host-local', languages: ['auto', 'en'] },
    transcribe(input) {
      calls += 1
      const seconds = (input.audio.byteLength - 44) / 32_000
      return Promise.resolve({ text: `chunk${calls} ${seconds.toFixed(1)}s`, audioSeconds: seconds, inferenceSeconds: 0.01 })
    },
  }
}

let dir: string
let ctx: Context
let fiber: Awaited<ReturnType<Context['plugin']>>
let callCounter = 0

function call(name: string, args: unknown) {
  return ctx.tools.execute({
    signal: new AbortController().signal,
    callId: ToolCallId(`audio-${++callCounter}`),
    name,
    arguments: args,
    agent: { session: { header: { cwd: dir } } } as never,
  })
}

function json(result: { content: { type: string; text?: string }[] }): Record<string, never> {
  const text = result.content.filter(block => block.type === 'text').map(block => block.text).join('')
  return JSON.parse(text) as Record<string, never>
}

/** Tool failures surface as `isError` results carrying the message, not rejected promises. */
function errorText(result: { isError?: boolean; content: { type: string; text?: string }[] }): string {
  expect(result.isError).toBe(true)
  return result.content.filter(block => block.type === 'text').map(block => block.text).join('')
}

const ffmpeg = existsSync('C:/Universal include/FFmpeg/bin/ffmpeg.exe') ? 'C:/Universal include/FFmpeg/bin/ffmpeg.exe' : 'ffmpeg'

beforeAll(async () => {
  dir = await mkdtemp(join(tmpdir(), 'dsh-audio-it-'))
  // 6 s sine tone as a real M4A container, exercising the same decoder a user file uses.
  await run(ffmpeg, ['-v', 'error', '-f', 'lavfi', '-i', 'sine=frequency=440:duration=6', '-c:a', 'aac', '-y', join(dir, 'tone.m4a')])
  await writeFile(join(dir, 'lyrics.txt'), 'Title Song ;\nSong by\n\nArtist\nfirst lyric line\nsecond lyric line\nthird lyric line\n', 'utf8')
  ctx = new Context()
  await ctx.plugin(SystemPrompt)
  await ctx.plugin(ToolRuntime)
  await ctx.plugin(LocalFileSystem, { cwd: dir })
  await ctx.plugin(LocalSubprocessRuntime)
  provideWorkingDirectoryFixture(ctx, dir)
  fiber = await ctx.plugin(SpeechToText, { defaultProvider: 'fake-local', language: 'auto' })
  ctx.get('speechToText')!.register(fakeProvider('fake-local'))
  await ctx.plugin(AudioAnalysis)
})

afterAll(async () => {
  await fiber?.dispose()
  await rm(dir, { recursive: true, force: true })
})

describe('audio_analyze', () => {
  it('decodes a real M4A and reports duration plus waveform', async () => {
    const result = json(await call('audio_analyze', { audio_path: 'tone.m4a', waveform_points: 40 }))
    expect(result.durationSeconds as unknown as number).toBeCloseTo(6, 0)
    expect(result.sampleRate as unknown as number).toBe(16_000)
    expect(result.channels as unknown as number).toBe(1)
    expect((result.waveform as unknown as unknown[]).length).toBe(40)
    const peaks = (result.waveform as unknown as { peak: number }[]).map(point => point.peak)
    expect(Math.max(...peaks)).toBeGreaterThan(0.05)
  })

  it('rejects an out-of-range BPM window', async () => {
    expect(errorText(await call('audio_analyze', { audio_path: 'tone.m4a', minimum_bpm: 400, maximum_bpm: 10 }))).toMatch(/BPM range/)
  })
})

describe('audio_transcribe', () => {
  it('splits a longer file into sequential provider-sized chunks at real offsets', async () => {
    const result = json(await call('audio_transcribe', { audio_path: 'tone.m4a', provider_id: 'fake-local', chunk_seconds: 5 }))
    const duration = result.audioDurationSeconds as unknown as number
    expect(duration).toBeCloseTo(6, 0)
    expect(result.chunkSeconds as unknown as number).toBe(5)
    expect(result.chunkCount as unknown as number).toBe(2)
    const segments = result.segments as unknown as { start: number; end: number; text: string }[]
    expect(segments).toHaveLength(2)
    // Every segment starts where the previous one ended: no gaps, no overlaps.
    expect(segments[0]!.start).toBe(0)
    expect(segments[0]!.end).toBe(5)
    expect(segments[1]!.start).toBe(segments[0]!.end)
    expect(segments[1]!.end).toBeCloseTo(duration, 1)
    expect(segments[0]!.text).toContain('chunk1')
    expect(segments[1]!.text).toContain('chunk2')
    expect(result.transcript as unknown as string).toContain('chunk1')
  })

  it('accepts audio_transcribe segments verbatim in lyrics_align', async () => {
    const transcription = json(await call('audio_transcribe', { audio_path: 'tone.m4a', provider_id: 'fake-local', chunk_seconds: 5 }))
    const segments = transcription.segments as unknown as { start: number; end: number; text: string }[]
    const alignment = json(await call('lyrics_align', {
      lyrics: segments.map(segment => segment.text).join('\n'),
      transcript_segments: transcription.segments,
    }))
    const lines = alignment.alignment as unknown as { status: string; startSeconds?: number }[]
    expect(lines).toHaveLength(segments.length)
    expect(lines.every(line => line.status === 'matched')).toBe(true)
    expect(lines[0]!.startSeconds).toBe(0)
  })

  it('refuses an unknown provider instead of silently choosing one', async () => {
    expect(errorText(await call('audio_transcribe', { audio_path: 'tone.m4a', provider_id: 'missing' }))).toMatch(/unavailable/)
  })

  it('refuses an unusable chunk size', async () => {
    expect(errorText(await call('audio_transcribe', { audio_path: 'tone.m4a', provider_id: 'fake-local', chunk_seconds: 1 }))).toMatch(/chunk_seconds/)
  })
})

describe('lyrics_timestamp', () => {
  it('drops lyric metadata, times every sung line and returns saveable LRC', async () => {
    const result = json(await call('lyrics_timestamp', {
      audio_path: 'tone.m4a', lyrics_path: 'lyrics.txt', provider_id: 'fake-local', chunk_seconds: 3,
    }))
    expect(result.lineCount as unknown as number).toBe(3)
    // The stub transcript never matches lyrics, so every line is honestly reported as a guess.
    expect(result.anchoredLines as unknown as number).toBe(0)
    expect(result.interpolatedLines as unknown as number).toBe(3)
    const lines = result.lines as unknown as { line: string; startSeconds: number; source: string }[]
    expect(lines.map(line => line.line)).toEqual(['first lyric line', 'second lyric line', 'third lyric line'])
    expect(lines[0]!.startSeconds).toBe(0)
    expect(lines.every(line => line.source !== 'transcript')).toBe(true)
    expect(result.lrc as unknown as string).toContain('second lyric line')
    expect(result.lrc as unknown as string).toContain('[note] audio: tone.m4a')
  })

  it('requires lyrics text', async () => {
    expect(errorText(await call('lyrics_timestamp', { audio_path: 'tone.m4a', provider_id: 'fake-local' })))
      .toMatch(/lyrics_path or inline lyrics/)
  })

  it('rejects an unusable recognition chunk', async () => {
    expect(errorText(await call('lyrics_timestamp', { audio_path: 'tone.m4a', lyrics_path: 'lyrics.txt', provider_id: 'fake-local', chunk_seconds: 1 })))
      .toMatch(/chunk_seconds/)
  })
})

describe('lyrics_align', () => {
  it('matches supplied lyrics to provided segments and flags the rest', async () => {
    const result = json(await call('lyrics_align', {
      lyrics: 'first line\nsecond line',
      transcript_segments: [{ start: 0, end: 2, text: 'first line' }],
      word_level: true,
    }))
    const alignment = result.alignment as unknown as { line: string; status: string; words?: unknown[] }[]
    expect(alignment[0]!.status).toBe('matched')
    expect(alignment[0]!.words).toHaveLength(2)
    expect(alignment[1]!.status).toBe('unmatched')
    expect(result.reviewWarning as unknown as string).toMatch(/Approximate/)
  })

  it('marks an even-spread timeline as a placeholder', async () => {
    const result = json(await call('lyrics_align', { lyrics: 'a\nb\nc', duration_seconds: 30 }))
    expect(result.reviewWarning as unknown as string).toMatch(/placeholder/)
  })
})
