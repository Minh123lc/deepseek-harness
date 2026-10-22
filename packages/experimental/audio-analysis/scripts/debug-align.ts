/**
 * Offline diagnosis of the aligner against a hand-built evidence list.
 * Usage: node --import tsx scripts/debug-align.ts <lyrics> [chunkSeconds]
 */
import { readFileSync } from 'node:fs'
import { alignLyricsToEvidence, extractLyricLines } from '../src/align.ts'

const lyricsPath = process.argv[2]
const chunkSeconds = Number(process.argv[3] ?? 3)
if (lyricsPath === undefined) throw new Error('usage: debug-align.ts <lyrics> [chunkSeconds]')
const lines = extractLyricLines(readFileSync(lyricsPath, 'utf8'))

// Evidence measured from the real recording around the reported break (135 s → 165 s).
const measured: [number, string][] = [
  [129, 'need so.'], [132, 'arguments.'], [135, 'The.'], [138, 'The.'], [141, 'The.'], [144, 'The.'],
  [147, 'You.'], [150, 'Ex.'], [153, ''], [156, 'The.'], [159, 'I.'], [162, 'you I can get.'],
  [165, 'And execution I can have you got problem accept you show.'],
]
const segments = measured.flatMap(([start, text]) => text.length === 0 ? [] : [{ start, end: start + chunkSeconds, text }])
const timed = alignLyricsToEvidence(lines, segments, { durationSeconds: 212.323 })

console.log(`lines=${lines.length} chunk=${chunkSeconds}s segments=${segments.length}`)
for (let index = 60; index < 82; index++) {
  const line = timed[index]
  if (line === undefined) continue
  console.log(String(index).padStart(2), line.startSeconds.toFixed(2).padStart(7), line.source.padEnd(12), line.line.slice(0, 34))
}
