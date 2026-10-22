/** Deterministic PCM framing, waveform summaries and BPM tests. */
import { describe, expect, it } from 'vitest';
import { analyzeWav } from "../src/../../speech-to-text/src/analysis.js";
import { validateWave } from "../src/../../speech-to-text/src/wave.js";
function wave(durationSeconds, onsetsSeconds = []) {
    const sampleCount = Math.round(durationSeconds * 16_000);
    const pcm = new Uint8Array(sampleCount * 2);
    const view = new DataView(pcm.buffer);
    for (const onset of onsetsSeconds) {
        const start = Math.floor(onset * 16_000);
        for (let i = start; i < Math.min(start + 1600, sampleCount); i++) {
            const sample = Math.round(0.7 * Math.sin(2 * Math.PI * 220 * (i - start) / 16_000) * 32767);
            view.setInt16(i * 2, sample, true);
        }
    }
    const header = new Uint8Array(44);
    const wavView = new DataView(header.buffer);
    const text = (offset, value) => { for (let i = 0; i < value.length; i++)
        header[offset + i] = value.charCodeAt(i); };
    text(0, 'RIFF');
    wavView.setUint32(4, 36 + pcm.length, true);
    text(8, 'WAVE');
    text(12, 'fmt ');
    wavView.setUint32(16, 16, true);
    wavView.setUint16(20, 1, true);
    wavView.setUint16(22, 1, true);
    wavView.setUint32(24, 16_000, true);
    wavView.setUint32(28, 32_000, true);
    wavView.setUint16(32, 2, true);
    wavView.setUint16(34, 16, true);
    text(36, 'data');
    wavView.setUint32(40, pcm.length, true);
    const bytes = new Uint8Array(header.length + pcm.length);
    bytes.set(header);
    bytes.set(pcm, header.length);
    return bytes;
}
describe('offline audio analysis', () => {
    it('summarizes PCM waveform and duration without retaining unbounded frames', () => {
        const result = analyzeWav(wave(1, [0.25]), { maxPeaks: 4 });
        expect(result.durationSeconds).toBe(1);
        expect(result.sampleRate).toBe(16_000);
        expect(result.channels).toBe(1);
        expect(result.waveform).toHaveLength(4);
        expect(result.waveform[0]).toEqual({ time: 0, peak: 0, rms: 0 });
        expect(result.waveform.some(point => point.peak > 0.5)).toBe(true);
    });
    it('recognizes a steady 120 BPM click-like onset train', () => {
        const beatOnsets = Array.from({ length: 8 }, (_, index) => index * 0.5);
        const result = analyzeWav(wave(5, beatOnsets), { minimumBpm: 90, maximumBpm: 150 });
        expect(result.estimatedBpm?.bpm).toBeCloseTo(120, 0);
        expect(result.estimatedBpm?.beatTimes.length).toBeGreaterThan(5);
        expect(result.bpmCandidates.length).toBeGreaterThan(0);
    });
    it('rejects malformed WAV data', () => {
        expect(() => analyzeWav(new Uint8Array(48))).toThrow(/canonical/);
        const valid = wave(1);
        expect(validateWave(valid, 2)).toBe(1);
        valid[0] = 0;
        expect(() => analyzeWav(valid)).toThrow(/canonical/);
    });
    it('bounds configuration and yields no BPM for silence', () => {
        expect(() => analyzeWav(wave(1), { maxPeaks: 50_000 })).toThrow(/maxPeaks/);
        const silent = analyzeWav(wave(2));
        expect(silent.estimatedBpm).toBeNull();
    });
});
//# sourceMappingURL=analysis.spec.js.map