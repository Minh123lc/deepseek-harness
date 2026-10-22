/**
 * Monotonic alignment of reference lyrics to timestamped recognition evidence.
 *
 * Recognition on sung vocals is noisy and lossy, so this trusts no single phrase. It runs a global
 * word-level sequence alignment (Needleman–Wunsch with fuzzy word equality) between lyric tokens and
 * transcript tokens, splits each recognition chunk among the lines it matched, then interpolates the
 * lines that got no evidence. Every emitted line records whether recognition anchored it, so a
 * consumer can tell measured timing from a guess.
 * @module @deepseek-ai/dsh-experimental-audio-analysis/align
 */

/** One time-stamped recognition result, as produced by chunked transcription. */
export interface TranscriptSegment {
  readonly start: number
  readonly end: number
  readonly text: string
}

/** How a line's timing was obtained. */
export type LineTimingSource = 'transcript' | 'interpolated' | 'extrapolated'

// Type alias, not an interface: a JsonValue-returning tool result needs an implicit index signature.
/** One lyric line with its time range, provenance, and the recognition text that anchored it. */
export type TimedLine = {
  readonly index: number
  readonly line: string
  readonly startSeconds: number
  readonly endSeconds: number
  /** 1.0 for a directly anchored line, lower for interpolated or extrapolated ones. */
  readonly confidence: number
  readonly source: LineTimingSource
  /** Recognition text supporting an anchored line. */
  readonly evidence?: string
}

/** Whole-track facts the alignment needs beyond the measured evidence. */
export interface AlignOptions {
  /** Audio duration; extrapolated tail lines stop here. */
  readonly durationSeconds: number
}

interface LineToken {
  readonly text: string
  readonly lineIndex: number
}

interface EvidenceToken {
  readonly text: string
  readonly segmentIndex: number
}

/** Normalize one word for fuzzy comparison. */
function normalize(word: string): string {
  return word.toLocaleLowerCase().replace(/[^\p{L}\p{N}]+/gu, '')
}

/**
 * Function words that occur in almost every line. A match on these alone proves nothing: recognition
 * emits them while hearing only breath or bleed, so treating one as evidence invents an anchor.
 */
const STOP_WORDS = new Set([
  'the', 'a', 'an', 'and', 'or', 'but', 'if', 'then', 'to', 'of', 'in', 'on', 'at', 'for', 'with',
  'my', 'me', 'i', 'you', 'your', 'we', 'so', 'can', 'be', 'is', 'are', 'was', 'will', 'it', 'that',
  'this', 'as', 'do', 'not', 'no', 'oh', 'from', 'by', 'up', 'down', 'all', 'just', 'have', 'has',
  'he', 'she', 'they', 'them', 'our', 'us', 'yeah', 'the',
])

/** A token long and specific enough to prove its lyric line was actually sung. */
function isInformative(token: string): boolean {
  return token.length >= 3 && !STOP_WORDS.has(token)
}

/** Split text into comparable words; nothing survives an empty normalization. */
function tokenize(text: string): string[] {
  return text.split(/\s+/u).map(normalize).filter(token => token.length > 0)
}

/** Codepoint length, so a combined character counts once. */
function charLength(word: string): number {
  return Array.from(word).length
}

/** Levenshtein distance over codepoints. */
function editDistance(a: string, b: string): number {
  const left = Array.from(a), right = Array.from(b)
  if (left.length === 0) return right.length
  if (right.length === 0) return left.length
  let previous = Array.from({ length: right.length + 1 }, (_value, index) => index)
  for (let row = 1; row <= left.length; row++) {
    const current = [row]
    for (let column = 1; column <= right.length; column++) {
      const cost = left[row - 1] === right[column - 1] ? 0 : 1
      const best = Math.min((current[column - 1] ?? 0) + 1, (previous[column] ?? 0) + 1, (previous[column - 1] ?? 0) + cost)
      current[column] = best
    }
    previous = current
  }
  return previous[right.length] ?? Math.max(left.length, right.length)
}

/**
 * Similarity in [0, 1] between two normalized words.
 * Recognition mangles endings and merges syllables, so a bounded edit ratio beats exact equality.
 */
function similarity(a: string, b: string): number {
  if (a === b) return 1
  if (a.length === 0 || b.length === 0) return 0
  const shorter = Math.min(a.length, b.length)
  const longer = Math.max(a.length, b.length)
  // Recognition often truncates a word to a short fragment ("Ex" for "execution"). Accept a fragment
  // only when it is not a function word, so "the" and "in" cannot manufacture evidence.
  if (shorter >= 2 && longer >= 4 && (a.startsWith(b) || b.startsWith(a))) {
    const fragment = a.length < b.length ? a : b
    if (!STOP_WORDS.has(fragment)) return 0.7
  }
  const ratio = 1 - editDistance(a, b) / longer
  if (shorter >= 4 && (a.startsWith(b) || b.startsWith(a))) return Math.max(ratio, 0.75)
  return ratio >= 0.6 ? ratio : 0
}

interface MatchedToken {
  readonly lineIndex: number
  readonly segmentIndex: number
  readonly score: number
  /** False when the match rests on a function word and therefore proves little. */
  readonly informative: boolean
}

/** Align lyric tokens to transcript tokens monotonically, returning matched pairs in order. */
function alignTokens(lyrics: readonly LineToken[], evidence: readonly EvidenceToken[]): MatchedToken[] {
  const rows = lyrics.length, columns = evidence.length
  if (rows === 0 || columns === 0) return []
  // Bound the quadratic table; a song and its transcript stay far below this.
  if (rows * columns > 6_000_000) return []
  const width = columns + 1
  const score = new Float64Array((rows + 1) * width)
  const move = new Int8Array((rows + 1) * width)
  for (let row = 1; row <= rows; row++) score[row * width] = (score[(row - 1) * width] ?? 0) - 1
  for (let column = 1; column <= columns; column++) score[column] = (score[column - 1] ?? 0) - 1
  for (let row = 1; row <= rows; row++) {
    const lyric = lyrics[row - 1]
    if (lyric === undefined) continue
    for (let column = 1; column <= columns; column++) {
      const token = evidence[column - 1]
      if (token === undefined) continue
      const match = similarity(lyric.text, token.text)
      // A function-word match still guides the sequence, but must not outbid a real word.
      const bonus = match > 0 ? match * 2 * (isInformative(lyric.text) ? 1 : 0.3) : -1.2
      const diagonal = (score[(row - 1) * width + column - 1] ?? 0) + bonus
      const up = (score[(row - 1) * width + column] ?? 0) - 0.7
      const left = (score[row * width + column - 1] ?? 0) - 0.4
      const best = Math.max(diagonal, up, left)
      score[row * width + column] = best
      move[row * width + column] = best === diagonal ? 1 : best === up ? 2 : 3
    }
  }
  const matches: MatchedToken[] = []
  let row = rows, column = columns
  while (row > 0 && column > 0) {
    const direction = move[row * width + column] ?? 0
    if (direction === 1) {
      const lyric = lyrics[row - 1]
      const token = evidence[column - 1]
      if (lyric !== undefined && token !== undefined) {
        const match = similarity(lyric.text, token.text)
        if (match > 0) {
          matches.push({ lineIndex: lyric.lineIndex, segmentIndex: token.segmentIndex, score: match,
            informative: isInformative(lyric.text) && !STOP_WORDS.has(token.text) })
        }
      }
      row -= 1
      column -= 1
    } else if (direction === 2) row -= 1
    else column -= 1
  }
  return matches.reverse()
}

/** Shortest plausible sung duration for one line, so zero-length entries never appear. */
function minimalLineDuration(line: string): number {
  return Math.min(4, Math.max(0.35, tokenize(line).length * 0.22))
}

function round(value: number): number {
  return Number(value.toFixed(3))
}

/** Relative length of a lyric line, used to split a shared span between lines. */
function lineWeight(line: string): number {
  return Math.max(1, tokenize(line).reduce((sum, token) => sum + charLength(token), 0))
}

interface Putative {
  readonly index: number
  readonly text: string
  readonly start: number
  readonly end: number
  readonly confidence: number
  readonly evidence: string
  /** End shown to the consumer, clamped to a plausible sung length. */
  readonly visibleEnd: number
}

/** Plausible sung length of a line, used to reject chunk spans that clearly drifted. */
function plausibleSpanSeconds(line: string): number {
  return Math.min(9, Math.max(2, tokenize(line).length * 0.75 + 1.5))
}

/**
 * Turn matched tokens into provisional line spans: each recognition chunk is divided among the
 * lines it matched, in lyric order and proportional to line length.
 *
 * A line only becomes an anchor when its match carries real evidence. Recognition routinely emits a
 * stray function word over an instrumental passage, and honouring that as an anchor pins a whole run
 * of lines to the wrong place, so matches are weighted by informativeness and a line with no
 * informative match is left for interpolation instead.
 */
function spansFromMatches(
  lines: readonly string[],
  segments: readonly TranscriptSegment[],
  matches: readonly MatchedToken[],
): Putative[] {
  // line index -> segment index -> accumulated score of informative matches only
  const perLine = new Map<number, Map<number, number>>()
  const massPerLine = new Map<number, number>()
  for (const match of matches) {
    massPerLine.set(match.lineIndex, (massPerLine.get(match.lineIndex) ?? 0) + (match.informative ? match.score : 0.1))
    if (!match.informative) continue
    const segmentsForLine = perLine.get(match.lineIndex) ?? new Map<number, number>()
    segmentsForLine.set(match.segmentIndex, Math.max(segmentsForLine.get(match.segmentIndex) ?? 0, match.score))
    perLine.set(match.lineIndex, segmentsForLine)
  }
  // segment index -> ordered line indices that matched inside it
  const linesPerSegment = new Map<number, number[]>()
  for (const [lineIndex, segmentScores] of perLine) {
    for (const segmentIndex of segmentScores.keys()) {
      const list = linesPerSegment.get(segmentIndex) ?? []
      list.push(lineIndex)
      linesPerSegment.set(segmentIndex, list)
    }
  }
  // Split each segment span among its lines, then union per line across segments.
  const spans = new Map<number, { start: number; end: number; confidence: number; evidence: string }>()
  for (const [segmentIndex, lineIndices] of linesPerSegment) {
    const segment = segments[segmentIndex]
    if (segment === undefined) continue
    const ordered = [...new Set(lineIndices)].sort((a, b) => a - b)
    const weights = ordered.map(lineIndex => lineWeight(lines[lineIndex] ?? ''))
    const total = weights.reduce((sum, weight) => sum + weight, 0)
    let cursor = segment.start
    for (let position = 0; position < ordered.length; position++) {
      const lineIndex = ordered[position]
      if (lineIndex === undefined) continue
      const share = (segment.end - segment.start) * (weights[position] ?? 1) / total
      const start = cursor
      const end = position === ordered.length - 1 ? segment.end : Math.min(segment.end, cursor + share)
      cursor = end
      const score = perLine.get(lineIndex)?.get(segmentIndex) ?? 0.6
      const existing = spans.get(lineIndex)
      const wider = existing !== undefined && existing.end - existing.start >= end - start
      spans.set(lineIndex, existing === undefined
        ? { start, end, confidence: score, evidence: segment.text }
        : {
          start: Math.min(existing.start, start),
          end: Math.max(existing.end, end),
          confidence: Math.max(existing.confidence, score),
          evidence: wider ? existing.evidence : segment.text,
        })
    }
  }
  const result: Putative[] = []
  for (const [lineIndex, span] of [...spans].sort((a, b) => a[0] - b[0])) {
    const text = lines[lineIndex]?.trim() ?? ''
    if (text.length === 0) continue
    // Require at least one informative match before trusting this line's position.
    if ((massPerLine.get(lineIndex) ?? 0) < 1) continue
    // A chunk span wider than the line could plausibly be sung means the match drifted; keep the onset.
    const limit = plausibleSpanSeconds(text)
    result.push({
      index: lineIndex, text,
      // Keep the raw chunk end: the following placement window must start after the chunk that
      // produced this evidence, not at the clamped end of the visible line.
      start: span.start, end: span.end,
      confidence: span.confidence, evidence: span.evidence,
      visibleEnd: Math.min(span.end, span.start + limit),
    })
  }
  return result
}

/** True when a chunk carries a word that could only come from the sung lyric. */
function isVoiced(text: string): boolean {
  return tokenize(text).some(isInformative)
}

/**
 * Split a placement window into the parts a lyric run can occupy, subtracting genuine breaks.
 *
 * Recognition failure and an instrumental break look alike for a few seconds, so only a stretch long
 * relative to the recognition chunks counts as a break. Short weak stretches stay inside the window:
 * sung lyric is continuous there and the missing words are a recognizer limitation, not silence.
 * @param segments - recognition chunks used as the voiced/weak mask.
 * @param from - window start.
 * @param to - window end.
 * @returns ordered intervals totalling at most the window, covering everything but measured breaks.
 */
function placementRanges(segments: readonly TranscriptSegment[], from: number, to: number): { start: number; end: number }[] {
  const inside = segments.filter(segment => segment.end > from && segment.start < to)
  if (inside.length === 0 || to <= from) return [{ start: from, end: to }]
  const durations = inside.map(segment => segment.end - segment.start).sort((a, b) => a - b)
  const median = durations[Math.floor(durations.length / 2)] ?? 6
  const minBreak = Math.max(8, median * 1.6)

  // Group consecutive weak chunks (only function words or nothing) into candidate breaks.
  const breaks: { start: number; end: number; lastStart: number; lastEnd: number }[] = []
  let run: { start: number; end: number; lastStart: number; lastEnd: number } | undefined
  for (const segment of inside) {
    const clippedStart = Math.max(from, segment.start), clippedEnd = Math.min(to, segment.end)
    if (clippedEnd <= clippedStart) continue
    if (isVoiced(segment.text)) {
      if (run !== undefined) { breaks.push(run); run = undefined }
      continue
    }
    run = run === undefined || clippedStart > run.end + 0.05
      ? { start: clippedStart, end: clippedEnd, lastStart: clippedStart, lastEnd: clippedEnd }
      : { start: run.start, end: Math.max(run.end, clippedEnd), lastStart: clippedStart, lastEnd: clippedEnd }
  }
  const openAtEnd = run !== undefined
  if (run !== undefined) breaks.push(run)

  const ranges: { start: number; end: number }[] = []
  let cursor = from
  for (let index = 0; index < breaks.length; index++) {
    const gap = breaks[index]
    if (gap === undefined) continue
    if (gap.end - gap.start < minBreak) continue
    if (gap.start > cursor) ranges.push({ start: cursor, end: gap.start })
    // A break that a sung chunk follows does not end where that chunk begins: the phrase was already
    // under way during the last weak chunk. Its true onset is unknown inside that chunk, so take the
    // midpoint — an unbiased estimate that tightens as chunks get smaller instead of quantising the
    // whole block one chunk late.
    const followed = index < breaks.length - 1 || !openAtEnd
    cursor = Math.max(cursor, followed ? (gap.lastStart + gap.lastEnd) / 2 : gap.end)
  }
  if (cursor < to) ranges.push({ start: cursor, end: to })
  return ranges.length > 0 ? ranges : [{ start: from, end: to }]
}

/**
 * Place lines across the usable parts of a span, skipping measured breaks.
 *
 * Distributing over the raw span assumes every second carries lyric, so an instrumental passage
 * swallows lines and pins the next sung block far too early.
 * @param group - lines to place, in order.
 * @param ranges - intervals the run may occupy, in order.
 * @param source - reported provenance of the placement.
 * @param confidence - reported confidence of the placement.
 * @returns timed lines, one per input line.
 */
function distributeAcrossRanges(
  group: readonly { index: number; text: string }[],
  ranges: readonly { start: number; end: number }[],
  source: LineTimingSource,
  confidence: number,
): TimedLine[] {
  if (group.length === 0) return []
  const total = ranges.reduce((sum, range) => sum + (range.end - range.start), 0)
  if (total <= 0) {
    const first = ranges[0]?.start ?? 0
    return distributeEvenly(group, first, first + group.length * 0.4, source, confidence)
  }
  const weights = group.map(entry => Math.max(1, lineWeight(entry.text)))
  const totalWeight = weights.reduce((sum, weight) => sum + weight, 0)
  /** Map elapsed usable seconds to an absolute time. */
  const timeAt = (elapsed: number): number => {
    let remaining = Math.max(0, elapsed)
    for (const range of ranges) {
      const length = range.end - range.start
      if (remaining <= length) return range.start + remaining
      remaining -= length
    }
    return ranges.at(-1)?.end ?? 0
  }
  const timed: TimedLine[] = []
  let consumed = 0
  for (let position = 0; position < group.length; position++) {
    const entry = group[position]
    if (entry === undefined) continue
    const start = timeAt(consumed)
    consumed += total * (weights[position] ?? 1) / totalWeight
    const rawEnd = position === group.length - 1 ? timeAt(total) : timeAt(consumed)
    const end = Math.max(start + minimalLineDuration(entry.text), rawEnd)
    timed.push({
      index: entry.index, line: entry.text,
      startSeconds: round(start), endSeconds: round(end),
      confidence, source,
    })
  }
  return timed
}

/** Spread lines over a span, weighted by line length. */
function distributeEvenly(
  group: readonly { index: number; text: string }[],
  fromSeconds: number,
  toSeconds: number,
  source: LineTimingSource,
  confidence: number,
): TimedLine[] {
  if (group.length === 0) return []
  const start = fromSeconds
  const end = Math.max(toSeconds, start + group.length * 0.4)
  const weights = group.map(entry => Math.max(1, tokenize(entry.text).reduce((sum, token) => sum + charLength(token), 0)))
  const total = weights.reduce((sum, weight) => sum + weight, 0)
  const timed: TimedLine[] = []
  let cursor = start
  for (let position = 0; position < group.length; position++) {
    const entry = group[position]
    if (entry === undefined) continue
    const share = (end - start) * (weights[position] ?? 1) / total
    const lineStart = cursor
    const lineEnd = position === group.length - 1 ? end : Math.min(end, cursor + share)
    cursor = lineEnd
    timed.push({
      index: entry.index, line: entry.text,
      startSeconds: round(lineStart),
      endSeconds: round(Math.max(lineStart + minimalLineDuration(entry.text), lineEnd)),
      confidence, source,
    })
  }
  return timed
}

/**
 * Time every non-empty reference line against recognition evidence.
 * @param lines - reference lyric lines, in performance order.
 * @param segments - timestamped recognition results; may be sparse, empty, or partly wrong.
 * @param options - audio duration for the interpolated head and tail.
 * @returns one timed line per non-empty input line, in order.
 */
export function alignLyricsToEvidence(
  lines: readonly string[],
  segments: readonly TranscriptSegment[],
  options: AlignOptions,
): TimedLine[] {
  const duration = Math.max(0.1, options.durationSeconds)
  const kept: { index: number; text: string }[] = []
  for (const line of lines) {
    const text = line.trim()
    if (text.length > 0) kept.push({ index: kept.length, text })
  }
  if (kept.length === 0) return []
  const texts = kept.map(entry => entry.text)

  const lineTokens: LineToken[] = []
  for (const entry of kept) {
    for (const token of tokenize(entry.text)) lineTokens.push({ text: token, lineIndex: entry.index })
  }
  const usable = segments.filter(segment =>
    Number.isFinite(segment.start) && Number.isFinite(segment.end) && segment.end > segment.start && segment.text.trim().length > 0)
  const evidenceTokens: EvidenceToken[] = []
  for (let segmentIndex = 0; segmentIndex < usable.length; segmentIndex++) {
    const segment = usable[segmentIndex]
    if (segment === undefined) continue
    for (const token of tokenize(segment.text)) evidenceTokens.push({ text: token, segmentIndex })
  }

  const matches = alignTokens(lineTokens, evidenceTokens)
  const anchored = spansFromMatches(texts, usable, matches)
  if (anchored.length === 0) {
    return distributeEvenly(kept, 0, duration, 'interpolated', 0.1)
  }

  const timed: TimedLine[] = []
  let cursorLine = 0
  let cursorTime = 0
  for (const anchor of anchored) {
    if (anchor.index > cursorLine) {
      const gap = kept.slice(cursorLine, anchor.index)
      timed.push(...distributeAcrossRanges(gap, placementRanges(usable, cursorTime, anchor.start), 'interpolated', 0.28))
      const last = timed.at(-1)
      if (last !== undefined) cursorTime = Math.max(cursorTime, last.endSeconds)
    }
    const start = Math.max(cursorTime, anchor.start)
    const visibleEnd = Math.min(duration, Math.max(start + minimalLineDuration(anchor.text), anchor.visibleEnd))
    timed.push({
      index: anchor.index, line: anchor.text,
      startSeconds: round(start), endSeconds: round(visibleEnd),
      confidence: round(0.55 + anchor.confidence * 0.45),
      source: 'transcript', evidence: anchor.evidence,
    })
    cursorLine = anchor.index + 1
    // Advance past the whole evidence chunk, so the next window cannot reuse its tail as singing time.
    cursorTime = Math.max(cursorTime, anchor.end)
  }
  const tail = kept.slice(cursorLine)
  if (tail.length > 0) {
    timed.push(...distributeAcrossRanges(tail, placementRanges(usable, cursorTime, duration), 'extrapolated', 0.18))
  }

  // A line may not outlast the next line's start.
  for (let position = 0; position < timed.length; position++) {
    const current = timed[position]
    const next = timed[position + 1]
    if (current === undefined || next === undefined) continue
    if (current.endSeconds > next.startSeconds) {
      timed[position] = {
        ...current,
        endSeconds: round(Math.max(current.startSeconds + minimalLineDuration(current.line), next.startSeconds)),
      }
    }
  }
  return timed
}

/** A bare performer name: short, capitalization only, no sentence punctuation. */
function isBareName(line: string): boolean {
  return /^[\p{L}][\p{L}'’.\- ]*$/u.test(line) && line.split(/\s+/u).length <= 3 && !/[.,!?;]/.test(line)
}

/** First plausible non-empty line index in a raw lyrics file. */
function firstContentLine(lines: readonly string[]): number {
  const credit = lines.findIndex(line => /\b(?:song|lyrics?|music)\s+by\b/iu.test(line))
  if (credit >= 0) {
    let start = credit + 1
    while (start < lines.length && (lines[start]?.trim() ?? '') === '') start += 1
    // Exactly one performer line follows the credit; consuming more would eat the first lyric line.
    if (start < lines.length && isBareName(lines[start]?.trim() ?? '')) start += 1
    while (start < lines.length && (lines[start]?.trim() ?? '') === '') start += 1
    return start
  }
  // Without an explicit credit line, a title ending in punctuation such as ';' is still metadata.
  let start = 0
  while (start < lines.length) {
    const line = lines[start]?.trim() ?? ''
    if (line.length === 0) { start += 1; continue }
    if (/;\s*$/u.test(line) || /\(\s*(?:feat|ft)\b/iu.test(line)) { start += 1; continue }
    break
  }
  return start
}

/**
 * Split a raw lyrics file into sung lines, dropping title and credit metadata.
 * @param raw - complete lyrics file text.
 * @returns non-empty lyric lines in performance order.
 */
export function extractLyricLines(raw: string): string[] {
  const lines = raw.split(/\r?\n/).map(line => line.trim())
  const start = firstContentLine(lines)
  return lines.slice(start).map(line => line.trim()).filter(line => line.length > 0)
}

/**
 * Format seconds as an LRC `[mm:ss.xx]` stamp.
 * @param seconds - offset from the start of the track; negative or non-finite input clamps to zero.
 * @returns the LRC time tag, including its surrounding brackets.
 */
export function lrcTimestamp(seconds: number): string {
  const safe = Number.isFinite(seconds) && seconds > 0 ? seconds : 0
  const minutes = Math.floor(safe / 60)
  const rest = safe - minutes * 60
  return `[${String(minutes).padStart(2, '0')}:${rest.toFixed(2).padStart(5, '0')}]`
}

/**
 * Render timed lines as synced LRC.
 * @param lines - timed lyric lines in performance order.
 * @param notes - metadata lines written as LRC `[tag]` comments before the lyrics.
 * @returns the complete LRC document.
 */
export function toLrc(lines: readonly TimedLine[], notes: readonly string[] = []): string {
  const header = [
    '[by] dsh-experimental-audio-analysis',
    ...notes.map(note => `[note] ${note}`),
  ]
  return [
    ...header,
    ...lines.map(line => `${lrcTimestamp(line.startSeconds)}${line.line}`),
  ].join('\n') + '\n'
}

