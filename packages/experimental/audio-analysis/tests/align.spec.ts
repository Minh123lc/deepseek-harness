/** Reference-lyric metadata stripping and monotonic evidence alignment. */
import { describe, expect, it } from 'vitest'
import { alignLyricsToEvidence, extractLyricLines, lrcTimestamp, toLrc } from '../src/align.ts'

const LYRICS_FILE = `World.Execute (Me) ;
Song by

Mili
Switch on the power line
Remember to put on protection
Lay down your pieces
And let's begin object creation
`

describe('extractLyricLines', () => {
  it('drops title, credit banner and bare artist from a real lyrics header', () => {
    expect(extractLyricLines(LYRICS_FILE)).toEqual([
      'Switch on the power line',
      'Remember to put on protection',
      'Lay down your pieces',
      "And let's begin object creation",
    ])
  })

  it('keeps a lyric file that starts directly with content', () => {
    expect(extractLyricLines('First line\nSecond line\n')).toEqual(['First line', 'Second line'])
  })

  it('returns nothing for a file of only metadata', () => {
    expect(extractLyricLines('Title ;\nSong by\nArtist\n')).toEqual([])
  })
})

describe('alignLyricsToEvidence', () => {
  it('anchors matched lines and interpolates the ones between them', () => {
    const lines = ['alpha beta', 'gamma delta', 'epsilon zeta', 'eta theta', 'iota kappa']
    const segments = [
      { start: 0, end: 5, text: 'alpha beta' },
      { start: 20, end: 25, text: 'iota kappa' },
    ]
    const timed = alignLyricsToEvidence(lines, segments, { durationSeconds: 30 })
    expect(timed).toHaveLength(5)
    expect(timed[0]!.source).toBe('transcript')
    expect(timed[4]!.source).toBe('transcript')
    // The three lines with no evidence are interpolated between the two anchors.
    expect(timed.slice(1, 4).every(line => line.source === 'interpolated')).toBe(true)
    expect(timed[0]!.startSeconds).toBe(0)
    expect(timed[4]!.endSeconds).toBeLessThanOrEqual(30)
  })

  it('keeps every line ordered and non-overlapping', () => {
    const lines = ['one one', 'two two', 'three three', 'four four', 'five five']
    const timed = alignLyricsToEvidence(lines, [{ start: 1, end: 4, text: 'raw two' }], { durationSeconds: 12 })
    for (let index = 1; index < timed.length; index++) {
      const previous = timed[index - 1]!
      const current = timed[index]!
      expect(current.startSeconds).toBeGreaterThanOrEqual(previous.startSeconds)
      expect(previous.endSeconds).toBeLessThanOrEqual(current.startSeconds + 0.001)
      expect(current.endSeconds).toBeGreaterThan(current.startSeconds)
    }
  })

  it('tolerates misspelled recognition and split chunks', () => {
    // Recognition mangles endings; a fuzzy match must still anchor the line.
    const timed = alignLyricsToEvidence(
      ['object creation', 'simulation now'],
      [{ start: 2, end: 8, text: 'object creations' }],
      { durationSeconds: 10 },
    )
    expect(timed[0]!.source).toBe('transcript')
    expect(timed[0]!.confidence).toBeGreaterThan(timed[1]!.confidence)
  })

  it('falls back to a full even distribution without any evidence', () => {
    const timed = alignLyricsToEvidence(['a b', 'c d'], [], { durationSeconds: 10 })
    expect(timed).toHaveLength(2)
    expect(timed.every(line => line.source === 'interpolated')).toBe(true)
    expect(timed[0]!.startSeconds).toBe(0)
  })

  it('ignores blank lines but keeps one timed entry per non-empty line', () => {
    const timed = alignLyricsToEvidence(
      ['alpha beta', '', 'gamma delta'],
      [{ start: 0, end: 4, text: 'alpha beta gamma delta' }],
      { durationSeconds: 6 },
    )
    expect(timed.map(line => line.line)).toEqual(['alpha beta', 'gamma delta'])
    expect(timed.every(line => line.source === 'transcript')).toBe(true)
  })

  it('returns nothing for an empty lyric list', () => {
    expect(alignLyricsToEvidence([], [{ start: 0, end: 2, text: 'x' }], { durationSeconds: 5 })).toEqual([])
  })

  // Regression: a real song ends a section at 2:12, goes instrumental for ~15 s, and resumes with
  // "Execution, execution" at ~2:27. A stray "The." in the silent stretch used to score a perfect
  // function-word match and anchor the next block at 2:18, jamming nine lines into four seconds.
  it('refuses a function-word-only chunk as an anchor', () => {
    const lines = ['illegal arguments', 'execution execution', 'execution execution', 'execution execution', 'give them all the execution']
    const segments = [
      { start: 132, end: 135, text: 'arguments.' },
      { start: 135, end: 138, text: 'The.' },
      { start: 138, end: 141, text: 'The.' },
      { start: 141, end: 144, text: 'The.' },
      { start: 144, end: 147, text: 'The.' },
      { start: 147, end: 150, text: 'You.' },
      { start: 150, end: 153, text: 'Exec.' },
      { start: 165, end: 168, text: 'give them all the execution' },
    ]
    const timed = alignLyricsToEvidence(lines, segments, { durationSeconds: 180 })
    const resumed = timed[1]!
    // The run resumes after the silent stretch, not at 138.
    expect(resumed.startSeconds).toBeGreaterThanOrEqual(147)
    expect(resumed.startSeconds).toBeLessThan(160)
    // And it stays before the next anchored line.
    expect(resumed.startSeconds).toBeLessThan(timed[4]!.startSeconds)
  })

  it('treats a lone stop word as no evidence at all', () => {
    const timed = alignLyricsToEvidence(['a b c', 'd e f'], [{ start: 10, end: 16, text: 'The' }], { durationSeconds: 20 })
    expect(timed.every(line => line.source !== 'transcript')).toBe(true)
  })

  it('clamps a drifted chunk span to a plausible line length', () => {
    const timed = alignLyricsToEvidence(['execution'], [{ start: 100, end: 112, text: 'execution' }], { durationSeconds: 120 })
    expect(timed[0]!.startSeconds).toBe(100)
    expect(timed[0]!.endSeconds).toBeLessThanOrEqual(109)
  })
})

describe('LRC rendering', () => {
  it('formats timestamps past the one-minute mark', () => {
    expect(lrcTimestamp(0)).toBe('[00:00.00]')
    expect(lrcTimestamp(65.5)).toBe('[01:05.50]')
    expect(lrcTimestamp(212.323)).toBe('[03:32.32]')
  })

  it('writes notes before the timed lines', () => {
    const lrc = toLrc([{ index: 0, line: 'hello', startSeconds: 1.5, endSeconds: 2, confidence: 1, source: 'transcript' }], ['audio: a.m4a'])
    expect(lrc).toContain('[note] audio: a.m4a')
    expect(lrc).toContain('[00:01.50]hello')
    expect(lrc.endsWith('\n')).toBe(true)
  })
})
